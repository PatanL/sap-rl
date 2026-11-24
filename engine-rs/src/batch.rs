//! A vector of persistent games behind one call: reset selected games, step
//! all of them, results written into arrays the caller owns.
//!
//! Semantics (also in `python/sap_env_rs.py`):
//! - `reset`: the selected games start a new match from their seed
//!   (`ParallelSelfPlayEnv().reset(seed, paired_tape=True)`); the others are
//!   not touched, nor are their output rows.
//! - `step`: every running game takes one `step_decisions` call. Per seat one
//!   `[type, source, target]` triple (type < 0 means "no decision", which is
//!   what a seat that has ended its turn must send) and one END freeze mask.
//! - No automatic reset. A finished game is skipped (status 1): its
//!   observations, `terminated`, `winner`, lives and wins stay as the last
//!   step left them; its rewards and battle flags are written as zero.
//! - Illegal input is an error for that game only (status -1): nothing in
//!   the game changes and its observation rows stay valid. An internal error
//!   after the step began (status -2) or a game that was never reset
//!   (status -3) needs a reset. The message is kept per game.
//! - Games are stepped on `threads` OS threads; every game is independent,
//!   so the thread count never changes a result.
//!
//! Search and ghost primitives (second half of `impl BatchEnv`; `reset` and
//! `step` are as they were): `step_which`, `observe`, `copy`, `rekey`,
//! `set_team`, `set_gold`, `set_shop_item`, `battle_events`, `eval_vs`. The contract is in
//! `python/sap_env_rs.py`.

use crate::env::{BoardRows, Game, BOARD_FIELDS, NS};
use crate::obs::{encode, Obs, OBS_KEYS};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub const STATUS_OK: i32 = 0;
pub const STATUS_FINISHED: i32 = 1;
pub const STATUS_ILLEGAL: i32 = -1;
pub const STATUS_BROKEN: i32 = -2;
pub const STATUS_NOT_RESET: i32 = -3;

/// Caller-owned output arrays; every pointer addresses `n` rows.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Outputs {
    /// The fifteen observation arrays in `obs::OBS_KEYS` order, each `[n, 2, ...]`.
    pub obs: [*mut u8; 15],
    /// `[n, 2]` float64.
    pub rewards: *mut f64,
    /// `[n]` int8.
    pub battle_resolved: *mut i8,
    /// `[n]` int8: 0 / 1 / 2, or -1 when no battle was resolved.
    pub battle_result: *mut i8,
    /// `[n]` int8.
    pub terminated: *mut i8,
    /// `[n]` int8: `_completed_match_result` once terminated, else -1.
    pub winner: *mut i8,
    /// `[n, 2]` int32.
    pub lives: *mut i32,
    /// `[n, 2]` int32.
    pub wins: *mut i32,
    /// `[n, 2]` int8: the seat still has to act this turn.
    pub active: *mut i8,
    /// `[n]` int32: see the `STATUS_*` constants.
    pub status: *mut i32,
}

unsafe impl Send for Outputs {}
unsafe impl Sync for Outputs {}

struct Slot {
    game: Option<Game>,
    error: String,
}

pub struct BatchEnv {
    slots: Vec<Slot>,
    threads: usize,
    pool: Option<Pool>,
}

// ----------------------------------------------------------------------
// A small persistent thread pool (std only). `run` hands the same closure to
// every worker and to the calling thread and returns when all are done;
// workers pull index ranges from a shared counter, so a batch with many
// finished (skipped) games stays balanced.
// ----------------------------------------------------------------------

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

type JobFn = dyn Fn() + Sync;

struct PoolState {
    generation: u64,
    job: Option<&'static JobFn>,
    pending: usize,
    shutdown: bool,
    panicked: bool,
}

struct PoolShared {
    state: Mutex<PoolState>,
    start: Condvar,
    done: Condvar,
    /// Mirrors `state.generation` so idle workers can spin briefly before sleeping.
    generation: AtomicUsize,
    busy: AtomicBool,
}

struct Pool {
    shared: Arc<PoolShared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn new(workers: usize) -> Pool {
        let shared = Arc::new(PoolShared {
            state: Mutex::new(PoolState { generation: 0, job: None, pending: 0, shutdown: false, panicked: false }),
            start: Condvar::new(),
            done: Condvar::new(),
            generation: AtomicUsize::new(0),
            busy: AtomicBool::new(false),
        });
        let handles = (0..workers)
            .map(|_| {
                let shared = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        // Spin a little: batch calls usually follow each other closely.
                        let mut spins = 0;
                        while shared.generation.load(Ordering::Acquire) as u64 == seen && spins < 4000 {
                            std::hint::spin_loop();
                            spins += 1;
                        }
                        let job = {
                            let mut state = shared.state.lock().unwrap();
                            while state.generation == seen && !state.shutdown {
                                state = shared.start.wait(state).unwrap();
                            }
                            if state.shutdown {
                                return;
                            }
                            seen = state.generation;
                            state.job
                        };
                        let ok = match job {
                            Some(job) => catch_unwind(AssertUnwindSafe(job)).is_ok(),
                            None => true,
                        };
                        let mut state = shared.state.lock().unwrap();
                        state.panicked |= !ok;
                        state.pending -= 1;
                        if state.pending == 0 {
                            shared.done.notify_all();
                        }
                    }
                })
            })
            .collect();
        Pool { shared, workers: handles }
    }

    /// Run `job` on every worker and on the calling thread; returns when all have finished.
    fn run<F: Fn() + Sync>(&self, job: &F) {
        assert!(!self.shared.busy.swap(true, Ordering::AcqRel), "the pool is not re-entrant");
        // The reference only lives until this function returns, and it does
        // not return before every worker has dropped it.
        let erased: &(dyn Fn() + Sync + '_) = job;
        let erased: &'static JobFn = unsafe { std::mem::transmute(erased) };
        {
            let mut state = self.shared.state.lock().unwrap();
            state.job = Some(erased);
            state.pending = self.workers.len();
            state.generation += 1;
            self.shared.generation.store(state.generation as usize, Ordering::Release);
            self.shared.start.notify_all();
        }
        let mine = catch_unwind(AssertUnwindSafe(job));
        let mut state = self.shared.state.lock().unwrap();
        while state.pending > 0 {
            state = self.shared.done.wait(state).unwrap();
        }
        state.job = None;
        let panicked = std::mem::replace(&mut state.panicked, false);
        drop(state);
        self.shared.busy.store(false, Ordering::Release);
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        assert!(!panicked, "a pool worker panicked");
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap();
            state.shutdown = true;
            self.shared.start.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// `work(i)` for every `i < count` on up to `threads` threads (the caller's included).
fn parallel_for<F: Fn(usize) + Sync>(pool: Option<&Pool>, count: usize, work: F) {
    let pool = match pool {
        Some(p) if count > 1 => p,
        _ => {
            for i in 0..count {
                work(i);
            }
            return;
        }
    };
    let grain = (count / ((pool.workers.len() + 1) * 8)).clamp(1, 64);
    let next = AtomicUsize::new(0);
    pool.run(&|| loop {
        let begin = next.fetch_add(grain, Ordering::Relaxed);
        if begin >= count {
            break;
        }
        for i in begin..(begin + grain).min(count) {
            work(i);
        }
    });
}

/// Write everything that describes game `i` after a reset or a step.
///
/// Safety: `out` must address at least `i + 1` rows of every array.
unsafe fn write_state(game: &mut Game, i: usize, out: &Outputs) {
    let mut obs = Obs::zeroed();
    for side in 0..2 {
        encode(game, side, &mut obs);
        let row = i * 2 + side;
        let parts: [(*const u8, usize); 15] = [
            (obs.item_ids.as_ptr() as *const u8, std::mem::size_of_val(&obs.item_ids)),
            (obs.kind_ids.as_ptr() as *const u8, std::mem::size_of_val(&obs.kind_ids)),
            (obs.status_ids.as_ptr() as *const u8, std::mem::size_of_val(&obs.status_ids)),
            (obs.positions.as_ptr() as *const u8, std::mem::size_of_val(&obs.positions)),
            (obs.numeric.as_ptr() as *const u8, std::mem::size_of_val(&obs.numeric)),
            (obs.entity_mask.as_ptr() as *const u8, std::mem::size_of_val(&obs.entity_mask)),
            (obs.type_mask.as_ptr() as *const u8, std::mem::size_of_val(&obs.type_mask)),
            (obs.source_mask.as_ptr() as *const u8, std::mem::size_of_val(&obs.source_mask)),
            (obs.target_mask.as_ptr() as *const u8, std::mem::size_of_val(&obs.target_mask)),
            (obs.history_item_ids.as_ptr() as *const u8, std::mem::size_of_val(&obs.history_item_ids)),
            (obs.history_status_ids.as_ptr() as *const u8, std::mem::size_of_val(&obs.history_status_ids)),
            (obs.history_positions.as_ptr() as *const u8, std::mem::size_of_val(&obs.history_positions)),
            (obs.history_numeric.as_ptr() as *const u8, std::mem::size_of_val(&obs.history_numeric)),
            (obs.history_mask.as_ptr() as *const u8, std::mem::size_of_val(&obs.history_mask)),
            (obs.history_board_mask.as_ptr() as *const u8, std::mem::size_of_val(&obs.history_board_mask)),
        ];
        for (k, (src, bytes)) in parts.iter().enumerate() {
            debug_assert_eq!(*bytes, OBS_KEYS[k].1 * OBS_KEYS[k].2);
            std::ptr::copy_nonoverlapping(*src, out.obs[k].add(row * bytes), *bytes);
        }
        *out.lives.add(row) = game.seats[side].lives;
        *out.wins.add(row) = game.seats[side].wins;
        *out.active.add(row) = (!game.seats[side].turn_ended && !game.terminated) as i8;
    }
    *out.terminated.add(i) = game.terminated as i8;
    *out.winner.add(i) = if game.terminated { game.match_result() as i8 } else { -1 };
}

unsafe fn write_idle(i: usize, out: &Outputs) {
    *out.rewards.add(i * 2) = 0.0;
    *out.rewards.add(i * 2 + 1) = 0.0;
    *out.battle_resolved.add(i) = 0;
    *out.battle_result.add(i) = -1;
}

fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        format!("panic: {s}")
    } else if let Some(s) = p.downcast_ref::<String>() {
        format!("panic: {s}")
    } else {
        "panic".to_string()
    }
}

impl Slot {
    /// Safety: as `write_state`.
    unsafe fn reset(&mut self, i: usize, seed: i64, out: &Outputs) {
        write_idle(i, out);
        let made = if seed < 0 {
            Err("Paired world requires an explicit seed in [0, 2**63)".to_string())
        } else {
            match catch_unwind(|| Game::new(seed as u64, false)) {
                Ok(Ok(g)) => Ok(g),
                Ok(Err(e)) => Err(e.0),
                Err(p) => Err(panic_text(p)),
            }
        };
        // Encoding runs inside the guard too: nothing here may unwind into the caller.
        let made = made.and_then(|mut game| {
            match catch_unwind(AssertUnwindSafe(|| unsafe { write_state(&mut game, i, out) })) {
                Ok(()) => Ok(game),
                Err(p) => Err(panic_text(p)),
            }
        });
        match made {
            Ok(game) => {
                self.game = Some(game);
                self.error.clear();
                *out.status.add(i) = STATUS_OK;
            }
            Err(e) => {
                // The previous match (if any) stays; the caller sees why the reset was refused.
                self.error = e;
                *out.status.add(i) = STATUS_ILLEGAL;
            }
        }
    }

    /// Safety: as `write_state`; `actions` has 6 and `masks` 2 readable elements.
    unsafe fn step(&mut self, i: usize, actions: *const i64, masks: *const i64, out: &Outputs) {
        write_idle(i, out);
        let game = match self.game.as_mut() {
            Some(g) => g,
            None => {
                self.error = "the game was never reset".to_string();
                *out.status.add(i) = STATUS_NOT_RESET;
                return;
            }
        };
        if game.poisoned {
            *out.status.add(i) = STATUS_BROKEN;
            return;
        }
        if game.terminated {
            *out.status.add(i) = STATUS_FINISHED;
            return;
        }
        let mut acts: [Option<[i64; 3]>; 2] = [None, None];
        let mut mask_ids = [0u32; 2];
        let mut bad = None;
        for side in 0..2 {
            let a = [*actions.add(side * 3), *actions.add(side * 3 + 1), *actions.add(side * 3 + 2)];
            if a[0] >= 0 {
                acts[side] = Some(a);
            }
            let m = *masks.add(side);
            if !(0..1024).contains(&m) {
                bad = Some("Invalid freeze subset ID".to_string());
            }
            mask_ids[side] = m as u32;
        }
        if let Some(e) = bad {
            self.error = e;
            *out.status.add(i) = STATUS_ILLEGAL;
            return;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            let step = game.step(acts, mask_ids)?;
            unsafe { write_state(game, i, out) };
            Ok::<_, crate::engine::EngineError>(step)
        }));
        match result {
            Ok(Ok(step)) => {
                *out.rewards.add(i * 2) = step.rewards[0];
                *out.rewards.add(i * 2 + 1) = step.rewards[1];
                *out.battle_resolved.add(i) = step.battle_resolved as i8;
                *out.battle_result.add(i) = step.battle_result.map(|r| r as i8).unwrap_or(-1);
                self.error.clear();
                *out.status.add(i) = STATUS_OK;
            }
            Ok(Err(e)) => {
                self.error = e.0;
                *out.status.add(i) = if game.poisoned { STATUS_BROKEN } else { STATUS_ILLEGAL };
            }
            Err(p) => {
                game.poisoned = true;
                self.error = panic_text(p);
                *out.status.add(i) = STATUS_BROKEN;
            }
        }
    }
}

impl BatchEnv {
    pub fn new(n: usize, threads: usize) -> BatchEnv {
        // Parse the tables once, outside any worker.
        crate::envdata::Tables::global();
        let mut env = BatchEnv {
            slots: (0..n).map(|_| Slot { game: None, error: String::new() }).collect(),
            threads: 1,
            pool: None,
        };
        env.set_threads(threads);
        env
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Number of threads that step games (the calling thread counts as one).
    pub fn set_threads(&mut self, threads: usize) {
        let threads = threads.max(1).min(self.slots.len().max(1));
        if threads != self.threads || (threads > 1) != self.pool.is_some() {
            self.pool = if threads > 1 { Some(Pool::new(threads - 1)) } else { None };
            self.threads = threads;
        }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    pub fn error(&self, i: usize) -> &str {
        self.slots.get(i).map(|s| s.error.as_str()).unwrap_or("no such game")
    }

    pub fn game(&self, i: usize) -> Option<&Game> {
        self.slots.get(i).and_then(|s| s.game.as_ref())
    }

    pub fn game_mut(&mut self, i: usize) -> Option<&mut Game> {
        self.slots.get_mut(i).and_then(|s| s.game.as_mut())
    }

    /// `work(i)` for every `i < count` on the environment's threads. For
    /// callers that prepare their inputs per game (the benchmark's action
    /// choice); `work` must only touch what belongs to index `i`.
    pub fn parallel<F: Fn(usize) + Sync>(&self, count: usize, work: F) {
        parallel_for(self.pool.as_ref(), count, work);
    }

    /// Run `work(slot, index)` over all slots on the configured threads.
    fn for_each<F: Fn(&mut Slot, usize) + Sync>(&mut self, work: F) {
        struct Base(*mut Slot);
        unsafe impl Sync for Base {}
        let base = Base(self.slots.as_mut_ptr());
        let base = &base;
        // Every index is handed out exactly once, so the `&mut Slot`s never alias.
        parallel_for(self.pool.as_ref(), self.slots.len(), |i| work(unsafe { &mut *base.0.add(i) }, i));
    }

    /// Start new matches in the games with a non-zero entry in `which`
    /// (all games when `which` is `None`). Returns the number of refused resets.
    ///
    /// Safety: `seeds` holds `len()` values, `which` (if given) `len()` bytes,
    /// and `out` addresses `len()` rows of every array.
    pub unsafe fn reset(&mut self, which: Option<&[u8]>, seeds: &[i64], out: &Outputs) -> usize {
        let out = *out;
        self.for_each(|slot, i| {
            if which.map(|w| w[i] != 0).unwrap_or(true) {
                unsafe { slot.reset(i, seeds[i], &out) };
            }
        });
        (0..self.slots.len())
            .filter(|&i| which.map(|w| w[i] != 0).unwrap_or(true) && unsafe { *out.status.add(i) } < 0)
            .count()
    }

    /// One `step_decisions` per running game. `actions` is `[n, 2, 3]`,
    /// `masks` is `[n, 2]`. Returns the number of games with a negative status.
    ///
    /// Safety: `out` addresses `len()` rows of every array.
    pub unsafe fn step(&mut self, actions: &[i64], masks: &[i64], out: &Outputs) -> usize {
        let out = *out;
        let (a, m) = (actions.as_ptr() as usize, masks.as_ptr() as usize);
        self.for_each(|slot, i| unsafe {
            slot.step(i, (a as *const i64).add(i * 6), (m as *const i64).add(i * 2), &out);
        });
        (0..self.slots.len()).filter(|&i| unsafe { *out.status.add(i) } < 0).count()
    }

    // ------------------------------------------------------------------
    // Search and ghost primitives (additions; `reset` and `step` above are
    // unchanged).
    // ------------------------------------------------------------------

    /// `step` for the games with a non-zero byte in `which` only; the other
    /// games and their output rows are not touched at all. Returns the number
    /// of selected games with a negative status.
    ///
    /// Safety: as `step`.
    pub unsafe fn step_which(&mut self, which: &[u8], actions: &[i64], masks: &[i64], out: &Outputs) -> usize {
        let out = *out;
        let (a, m) = (actions.as_ptr() as usize, masks.as_ptr() as usize);
        self.for_each(|slot, i| unsafe {
            if which[i] != 0 {
                slot.step(i, (a as *const i64).add(i * 6), (m as *const i64).add(i * 2), &out);
            }
        });
        (0..self.slots.len()).filter(|&i| which[i] != 0 && unsafe { *out.status.add(i) } < 0).count()
    }

    /// Write the observation rows, lives, wins, active, terminated and winner
    /// of the selected games again from their current state (after `set_team`
    /// or `set_gold`, which do not). Like every observation of the reference
    /// this registers the seats' current team orders as seen. Rewards, battle
    /// flags and status are not touched. Returns the number of games written.
    ///
    /// Safety: as `step`.
    pub unsafe fn observe(&mut self, which: &[u8], out: &Outputs) -> usize {
        let out = *out;
        let done = AtomicUsize::new(0);
        self.for_each(|slot, i| {
            if which[i] == 0 {
                return;
            }
            if let Some(game) = slot.game.as_mut() {
                if game.poisoned {
                    return;
                }
                if catch_unwind(AssertUnwindSafe(|| unsafe { write_state(game, i, &out) })).is_ok() {
                    done.fetch_add(1, Ordering::Relaxed);
                } else {
                    game.poisoned = true;
                    slot.error = "panic while encoding".to_string();
                }
            }
        });
        done.into_inner()
    }

    /// Game `dst[k]` becomes an exact, independent copy of game `src[k]`
    /// (the whole `Game`: both seats, shops, hidden shop state, streams and
    /// tape cursors, counters, histories, the last battle event), and the
    /// output rows of `dst[k]` (observations, lives, wins, active, terminated,
    /// winner, and the rewards and battle flags of the source's last step,
    /// which go with the copied battle event) become copies of the rows of
    /// `src[k]` as they are in `out`; its status is written as 0 (-2 if the
    /// source is broken). The call is refused as a
    /// whole, with nothing changed, when an index is out of range (-11), a
    /// source was never reset (-12), or a game appears twice in `dst` or in
    /// both lists (-13): every copy then reads a game no copy writes.
    ///
    /// Safety: `out` addresses `len()` rows of every array.
    pub unsafe fn copy(&mut self, src: &[i32], dst: &[i32], out: &Outputs) -> i32 {
        let n = self.slots.len();
        let mut role = vec![0u8; n];
        for &s in src {
            if s < 0 || s as usize >= n {
                return -11;
            }
            if self.slots[s as usize].game.is_none() {
                return -12;
            }
            role[s as usize] = 1;
        }
        for &d in dst {
            if d < 0 || d as usize >= n {
                return -11;
            }
            if role[d as usize] != 0 {
                return -13;
            }
            role[d as usize] = 2;
        }
        let out = *out;
        struct Base(*mut Slot);
        unsafe impl Sync for Base {}
        let base = Base(self.slots.as_mut_ptr());
        let base = &base;
        // Sources are only read, every destination is written by exactly one task.
        let out = &out;
        parallel_for(self.pool.as_ref(), src.len(), |k| unsafe {
            let (s, d) = (src[k] as usize, dst[k] as usize);
            let from = &*(base.0.add(s) as *const Slot);
            let to = &mut *base.0.add(d);
            match (to.game.as_mut(), from.game.as_ref()) {
                (Some(old), Some(new)) => old.clone_from(new),
                (_, new) => to.game = new.cloned(),
            }
            to.error.clear();
            for (key, row) in OBS_KEYS.iter().enumerate() {
                let bytes = row.1 * row.2 * 2;
                std::ptr::copy_nonoverlapping(out.obs[key].add(s * bytes), out.obs[key].add(d * bytes), bytes);
            }
            std::ptr::copy_nonoverlapping(out.lives.add(s * 2), out.lives.add(d * 2), 2);
            std::ptr::copy_nonoverlapping(out.wins.add(s * 2), out.wins.add(d * 2), 2);
            std::ptr::copy_nonoverlapping(out.active.add(s * 2), out.active.add(d * 2), 2);
            *out.terminated.add(d) = *out.terminated.add(s);
            *out.winner.add(d) = *out.winner.add(s);
            std::ptr::copy_nonoverlapping(out.rewards.add(s * 2), out.rewards.add(d * 2), 2);
            *out.battle_resolved.add(d) = *out.battle_resolved.add(s);
            *out.battle_result.add(d) = *out.battle_result.add(s);
            let broken = from.game.as_ref().map(|g| g.poisoned).unwrap_or(false);
            *out.status.add(d) = if broken { STATUS_BROKEN } else { STATUS_OK };
        });
        0
    }

    /// `Game::rekey` for the listed games (see there for what is replaced).
    /// Refused as a whole with nothing changed: -11 index out of range, -12
    /// never reset, -14 a seed outside `[0, 2**63)` or a broken game.
    pub fn rekey(&mut self, games: &[i32], seeds: &[i64]) -> i32 {
        let n = self.slots.len();
        for (&g, &seed) in games.iter().zip(seeds) {
            if g < 0 || g as usize >= n {
                return -11;
            }
            match &self.slots[g as usize].game {
                None => return -12,
                Some(game) if game.poisoned || seed < 0 => return -14,
                _ => {}
            }
        }
        for (&g, &seed) in games.iter().zip(seeds) {
            if let Some(game) = self.slots[g as usize].game.as_mut() {
                if game.rekey(seed as u64).is_err() {
                    return -14;
                }
            }
        }
        0
    }

    /// `Game::set_team`. 0, or -11 / -12 as above, or -14 with the reason in
    /// `error(game)` (the game is unchanged).
    pub fn set_team(&mut self, game: i32, seat: i32, rows: &BoardRows) -> i32 {
        self.edit(game, seat, |g, side| g.set_team(side, rows))
    }

    /// `Game::set_gold`; codes as `set_team`.
    pub fn set_gold(&mut self, game: i32, seat: i32, gold: i32) -> i32 {
        self.edit(game, seat, |g, side| g.set_gold(side, gold))
    }

    /// `Game::set_shop_item`; codes as `set_team`. `item_type` 0: `item` is a
    /// pet kind id (`sap_env_kind`); 1: a food id (`sap_env_food`).
    pub fn set_shop_item(&mut self, game: i32, seat: i32, slot: i32, item_type: i32, item: i32, frozen: bool) -> i32 {
        self.edit(game, seat, |g, side| {
            if slot < 0 {
                return Err(crate::engine::EngineError(format!("shop slot {slot} does not exist")));
            }
            let item = match (item_type, usize::try_from(item)) {
                (0, Ok(kind)) => crate::env::ShopItem::Pet(kind),
                (1, Ok(kind)) => crate::env::ShopItem::Food(kind),
                (0, _) => return Err(crate::engine::EngineError("unknown pet".into())),
                (1, _) => return Err(crate::engine::EngineError("unknown food".into())),
                _ => return Err(crate::engine::EngineError(format!("item type {item_type} is neither 0 (pet) nor 1 (food)"))),
            };
            g.set_shop_item(side, slot as usize, item, frozen)
        })
    }

    fn edit<F: FnOnce(&mut Game, usize) -> crate::env::R<()>>(&mut self, game: i32, seat: i32, change: F) -> i32 {
        if game < 0 || game as usize >= self.slots.len() || !(0..2).contains(&seat) {
            return -11;
        }
        let slot = &mut self.slots[game as usize];
        let g = match slot.game.as_mut() {
            Some(g) => g,
            None => return -12,
        };
        match catch_unwind(AssertUnwindSafe(|| change(g, seat as usize))) {
            Ok(Ok(())) => 0,
            Ok(Err(e)) => {
                slot.error = e.0;
                -14
            }
            Err(p) => {
                g.poisoned = true;
                slot.error = panic_text(p);
                -14
            }
        }
    }

    /// The battle resolved by each game's last accepted step: `turns[i]` is
    /// the one-based turn (0: that step resolved no battle, or the game was
    /// never reset; a refused or skipped step leaves the game, and with it
    /// the event, as it was, so a finished game keeps its last battle),
    /// `seeds[i]` the battle seed, `teams[i]` both teams in
    /// board rows `[2, 5, 7]` (seat 0 first; all rows empty when there was no
    /// battle), and, if `carried` is not null, `[2, 5, 2]` per game: each
    /// slot's `friends_hurt_counter` and `_hurt`, which the battle reads and
    /// the rows cannot carry. Returns the number of games with a battle.
    ///
    /// Safety: `turns` and `seeds` address `len()` elements, `teams`
    /// `len() * 70`, `carried` (if not null) `len() * 20`.
    pub unsafe fn battle_events(&self, turns: *mut i32, seeds: *mut i64, teams: *mut i32, carried: *mut i32) -> usize {
        let per_game = 2 * NS * BOARD_FIELDS;
        let mut count = 0;
        for (i, slot) in self.slots.iter().enumerate() {
            let event = slot.game.as_ref().and_then(|g| g.last_battle.as_ref());
            let rows = std::slice::from_raw_parts_mut(teams.add(i * per_game), per_game);
            match event {
                Some(event) => {
                    *turns.add(i) = event.turn;
                    *seeds.add(i) = event.seed as i64;
                    for (cell, value) in rows.iter_mut().zip(event.teams.iter().flatten().flatten()) {
                        *cell = *value;
                    }
                    if !carried.is_null() {
                        let extra = std::slice::from_raw_parts_mut(carried.add(i * 2 * NS * 2), 2 * NS * 2);
                        for (cell, value) in extra.iter_mut().zip(event.carried.iter().flatten().flatten()) {
                            *cell = *value;
                        }
                    }
                    count += 1;
                }
                None => {
                    *turns.add(i) = 0;
                    *seeds.add(i) = 0;
                    for (k, cell) in rows.iter_mut().enumerate() {
                        *cell = crate::env::EMPTY_ROW[k % BOARD_FIELDS];
                    }
                    if !carried.is_null() {
                        std::slice::from_raw_parts_mut(carried.add(i * 2 * NS * 2), 2 * NS * 2).fill(0);
                    }
                }
            }
        }
        count
    }

    /// `Game::eval_vs` for `games.len()` (game, seat) pairs against `m`
    /// boards each: `boards` is `[k, m, 5, 7]`, `seeds` one per pair, `counts`
    /// `[k, m, 3]` (wins, draws, losses of the seat). The fights of all pairs
    /// run on the environment's threads. No game is changed. Refused as a
    /// whole: -11 index out of range, -12 never reset, -14 a bad board, seed
    /// or broken game (the reason is in `error` of the pair's game).
    ///
    /// Safety: `boards` addresses `k * m * 35` and `counts` `k * m * 3` elements.
    pub unsafe fn eval_vs(
        &mut self,
        games: &[i32],
        seats: &[i32],
        boards: *const i32,
        m: usize,
        samples: u32,
        seeds: &[i64],
        counts: *mut i32,
    ) -> i32 {
        let n = self.slots.len();
        let per_board = NS * BOARD_FIELDS;
        struct In(*const i32);
        unsafe impl Sync for In {}
        let source = In(boards);
        let source = &source;
        let rows_of = |k: usize, b: usize| -> BoardRows {
            let mut rows = [[0i32; BOARD_FIELDS]; NS];
            let cells = unsafe { std::slice::from_raw_parts(source.0.add((k * m + b) * per_board), per_board) };
            for (row, chunk) in rows.iter_mut().zip(cells.chunks(BOARD_FIELDS)) {
                row.copy_from_slice(chunk);
            }
            rows
        };
        // Everything that can be refused is checked before any fight.
        let mut own = Vec::with_capacity(games.len());
        for k in 0..games.len() {
            let (g, seat) = (games[k], seats[k]);
            if g < 0 || g as usize >= n || !(0..2).contains(&seat) {
                return -11;
            }
            let slot = &mut self.slots[g as usize];
            let game = match slot.game.as_ref() {
                Some(game) => game,
                None => return -12,
            };
            if game.poisoned || seeds[k] < 0 {
                slot.error = "eval_vs: broken game or a seed outside [0, 2**63)".to_string();
                return -14;
            }
            let prepared = catch_unwind(AssertUnwindSafe(|| -> crate::env::R<_> {
                let team = game.team_if_ended(seat as usize)?;
                for b in 0..m {
                    crate::env::check_board(&rows_of(k, b)).map_err(|e| crate::engine::EngineError(format!("board {b}: {}", e.0)))?;
                }
                Ok(team)
            }));
            match prepared {
                Ok(Ok(team)) => own.push(team),
                Ok(Err(e)) => {
                    slot.error = e.0;
                    return -14;
                }
                Err(p) => {
                    slot.error = panic_text(p);
                    return -14;
                }
            }
        }
        struct Out(*mut i32);
        unsafe impl Sync for Out {}
        let target = Out(counts);
        let target = &target;
        let failed = AtomicBool::new(false);
        let slots = &self.slots;
        let own = &own;
        parallel_for(self.pool.as_ref(), games.len() * m.max(1), |unit| {
            if m == 0 {
                return;
            }
            let (k, b) = (unit / m, unit % m);
            let game = slots[games[k] as usize].game.as_ref().unwrap();
            let result = catch_unwind(AssertUnwindSafe(|| -> crate::env::R<[i32; 3]> {
                let other = game.board_team(&rows_of(k, b))?;
                game.eval_board(seats[k] as usize, &own[k], &other, b, samples, seeds[k] as u64)
            }));
            let cell = unsafe { std::slice::from_raw_parts_mut(target.0.add((k * m + b) * 3), 3) };
            match result {
                Ok(Ok(c)) => cell.copy_from_slice(&c),
                _ => {
                    cell.fill(-1);
                    failed.store(true, Ordering::Relaxed);
                }
            }
        });
        if failed.into_inner() {
            -15
        } else {
            0
        }
    }
}
