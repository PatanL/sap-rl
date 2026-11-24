//! C interface for calling the battle engine from Python (ctypes).
//!
//! `sap_engine_init(data_dir)` loads the tables once. `sap_kind(name)` and
//! `sap_status(name)` translate names to the ids used in the pet records.
//! `sap_battle` resolves one seeded fight and returns 0 (first team wins),
//! 1 (second team wins), 2 (draw) or a negative error code.
//!
//! A pet record is seven i32: kind, attack, health, total experience,
//! status, per-turn triggers already used, and the Parrot copy
//! (-2: derive from the friend ahead, -1: copied nothing, k: copied kind k).

use crate::data::Data;
use crate::engine::{simulate, PetSpec};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::OnceLock;

static DATA: OnceLock<Data> = OnceLock::new();
pub const PET_FIELDS: usize = 7;
/// Status ids used in pet records.
const STATUSES: [&str; 13] = [
    "none",
    "status-bread",
    "status-cake",
    "status-bone-attack",
    "status-coconut-shield",
    "status-extra-life",
    "status-garlic-armor",
    "status-honey-bee",
    "status-melon-armor",
    "status-poison-attack",
    "status-splash-attack",
    "status-steak-attack",
    "status-weak",
];

fn text(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }.to_str().ok().map(|s| s.to_string())
}

/// Load the tables. Returns 0 on success, -1 on failure. Safe to call twice.
#[no_mangle]
pub extern "C" fn sap_engine_init(data_dir: *const c_char) -> i32 {
    if DATA.get().is_some() {
        return 0;
    }
    let dir = match text(data_dir) {
        Some(d) => d,
        None => return -1,
    };
    match Data::load_dir(std::path::Path::new(&dir)) {
        Ok(d) => {
            let _ = DATA.set(d);
            0
        }
        Err(_) => -1,
    }
}

/// Id of a pet by name ("pet-ant"), or -1.
#[no_mangle]
pub extern "C" fn sap_kind(name: *const c_char) -> i32 {
    match (DATA.get(), text(name)) {
        (Some(d), Some(n)) => d.by_name.get(&n).map(|k| *k as i32).unwrap_or(-1),
        _ => -1,
    }
}

/// Id of a status by name ("none", "status-garlic-armor"), or -1.
#[no_mangle]
pub extern "C" fn sap_status(name: *const c_char) -> i32 {
    match text(name) {
        Some(n) => STATUSES.iter().position(|s| *s == n).map(|i| i as i32).unwrap_or(-1),
        None => -1,
    }
}

fn specs(d: &Data, raw: *const i32, count: i32) -> Option<Vec<PetSpec>> {
    if count < 0 || count > 5 || (count > 0 && raw.is_null()) {
        return None;
    }
    let cells = unsafe { std::slice::from_raw_parts(raw, count as usize * PET_FIELDS) };
    let mut out = Vec::with_capacity(count as usize);
    for pet in cells.chunks(PET_FIELDS) {
        let kind = d.pets.get(usize::try_from(pet[0]).ok()?)?;
        let status = *STATUSES.get(usize::try_from(pet[4]).ok()?)?;
        let parrot = match pet[6] {
            -2 => None,
            -1 => Some(None),
            k => Some(Some(d.pets.get(usize::try_from(k).ok()?)?.name.clone())),
        };
        out.push(PetSpec {
            id: kind.name.clone(),
            attack: pet[1],
            health: pet[2],
            experience: pet[3],
            status: status.to_string(),
            triggers_consumed: pet[5],
            parrot_copies: parrot,
        });
    }
    Some(out)
}

/// Resolve one fight. Teams are front first, at most five pets each.
#[no_mangle]
pub extern "C" fn sap_battle(first: *const i32, n_first: i32, second: *const i32, n_second: i32, seed: u32) -> i32 {
    let d = match DATA.get() {
        Some(d) => d,
        None => return -2,
    };
    let (a, b) = match (specs(d, first, n_first), specs(d, second, n_second)) {
        (Some(a), Some(b)) => (a, b),
        _ => return -3,
    };
    match simulate(d, &a, &b, seed, false) {
        Ok(result) => result.outcome as i32,
        Err(_) => -4,
    }
}

// ----------------------------------------------------------------------
// Batch interface for the training environment (see `batch.rs`).
// ----------------------------------------------------------------------

use crate::batch::{BatchEnv, Outputs};

/// Create `n` games stepped on `threads` threads. Returns null on bad arguments.
#[no_mangle]
pub extern "C" fn sap_env_create(n: i32, threads: i32) -> *mut BatchEnv {
    if n <= 0 || threads <= 0 {
        return std::ptr::null_mut();
    }
    match std::panic::catch_unwind(|| BatchEnv::new(n as usize, threads as usize)) {
        Ok(env) => Box::into_raw(Box::new(env)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free an environment made by `sap_env_create`.
#[no_mangle]
pub extern "C" fn sap_env_destroy(env: *mut BatchEnv) {
    if !env.is_null() {
        drop(unsafe { Box::from_raw(env) });
    }
}

/// Number of games, or -1.
#[no_mangle]
pub extern "C" fn sap_env_len(env: *const BatchEnv) -> i32 {
    match unsafe { env.as_ref() } {
        Some(e) => e.len() as i32,
        None => -1,
    }
}

#[no_mangle]
pub extern "C" fn sap_env_set_threads(env: *mut BatchEnv, threads: i32) -> i32 {
    match unsafe { env.as_mut() } {
        Some(e) if threads > 0 => {
            e.set_threads(threads as usize);
            0
        }
        _ => -1,
    }
}

fn outputs_ok(out: *const Outputs) -> Option<Outputs> {
    let out = unsafe { out.as_ref() }?;
    let any_null = out.obs.iter().any(|p| p.is_null())
        || out.rewards.is_null()
        || out.battle_resolved.is_null()
        || out.battle_result.is_null()
        || out.terminated.is_null()
        || out.winner.is_null()
        || out.lives.is_null()
        || out.wins.is_null()
        || out.active.is_null()
        || out.status.is_null();
    if any_null {
        None
    } else {
        Some(*out)
    }
}

/// Reset the games whose byte in `which` is non-zero (all games when `which`
/// is null) with `seeds[i]` (each in `[0, 2**63)`). `seeds` holds one value
/// per game. Returns the number of refused resets, or a negative code for a
/// bad call (-10: null pointer).
#[no_mangle]
pub extern "C" fn sap_env_reset(env: *mut BatchEnv, which: *const u8, seeds: *const i64, out: *const Outputs) -> i32 {
    let (env, out) = match (unsafe { env.as_mut() }, outputs_ok(out)) {
        (Some(e), Some(o)) if !seeds.is_null() => (e, o),
        _ => return -10,
    };
    let n = env.len();
    let seeds = unsafe { std::slice::from_raw_parts(seeds, n) };
    let which = if which.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts(which, n) }) };
    guarded(|| unsafe { env.reset(which, seeds, &out) as i32 })
}

/// No panic may cross the C boundary: -100 instead.
fn guarded<F: FnOnce() -> i32>(call: F) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).unwrap_or(-100)
}

/// Step every game: `actions` is int64 `[n, 2, 3]` (type < 0: no decision
/// for that seat), `masks` int64 `[n, 2]`. Returns the number of games whose
/// status is negative, or -10 for a null pointer.
#[no_mangle]
pub extern "C" fn sap_env_step(env: *mut BatchEnv, actions: *const i64, masks: *const i64, out: *const Outputs) -> i32 {
    let (env, out) = match (unsafe { env.as_mut() }, outputs_ok(out)) {
        (Some(e), Some(o)) if !actions.is_null() && !masks.is_null() => (e, o),
        _ => return -10,
    };
    let n = env.len();
    let actions = unsafe { std::slice::from_raw_parts(actions, n * 6) };
    let masks = unsafe { std::slice::from_raw_parts(masks, n * 2) };
    guarded(|| unsafe { env.step(actions, masks, &out) as i32 })
}

fn copy_text(text: &str, buf: *mut c_char, len: i32) -> i32 {
    if buf.is_null() || len <= 0 {
        return text.len() as i32;
    }
    let n = text.len().min(len as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
    text.len() as i32
}

/// Copy the last error message of game `game` into `buf` (NUL-terminated,
/// truncated to `len`). Returns the full length, or -1.
#[no_mangle]
pub extern "C" fn sap_env_error(env: *const BatchEnv, game: i32, buf: *mut c_char, len: i32) -> i32 {
    match unsafe { env.as_ref() } {
        Some(e) if game >= 0 && (game as usize) < e.len() => copy_text(e.error(game as usize), buf, len),
        _ => -1,
    }
}

/// Flat legal-action mask (197 int8) of one seat of one game. Returns the
/// number of legal actions, or -1.
#[no_mangle]
pub extern "C" fn sap_env_legal(env: *mut BatchEnv, game: i32, seat: i32, mask: *mut i8) -> i32 {
    let env = match unsafe { env.as_mut() } {
        Some(e) => e,
        None => return -1,
    };
    if game < 0 || !(0..2).contains(&seat) || mask.is_null() {
        return -1;
    }
    guarded(|| match env.game_mut(game as usize) {
        Some(g) => {
            // Pure after any observation: the current team order is already registered.
            let legal = g.seats[seat as usize].avail_actions();
            let mut count = 0;
            for a in 0..crate::env::NUM_ACTIONS {
                let on = legal.has(a);
                unsafe { *mask.add(a) = on as i8 };
                count += on as i32;
            }
            count
        }
        None => -1,
    })
}

/// The canonical state dump of one game as JSON (the layout of
/// `tools/gen_env_golden.py --ext`), for parity checks. Returns the full
/// length, or -1.
#[no_mangle]
pub extern "C" fn sap_env_state_json(env: *const BatchEnv, game: i32, buf: *mut c_char, len: i32) -> i32 {
    match unsafe { env.as_ref() } {
        Some(e) if game >= 0 => guarded(|| match e.game(game as usize) {
            Some(g) => copy_text(&crate::canon::canonical(g, true).to_string(), buf, len),
            None => -1,
        }),
        _ => -1,
    }
}

/// The observation layout as JSON: `[[name, dtype, [shape...]], ...]` in key order.
#[no_mangle]
pub extern "C" fn sap_env_obs_layout(buf: *mut c_char, len: i32) -> i32 {
    let rows: Vec<serde_json::Value> = crate::obs::OBS_KEYS
        .iter()
        .map(|(name, _, _, dtype, shape)| serde_json::json!([name, dtype, shape]))
        .collect();
    copy_text(&serde_json::Value::Array(rows).to_string(), buf, len)
}

// ----------------------------------------------------------------------
// Search and ghost primitives (additions; nothing above changed).
//
// Boards are int32 `[5, 7]` rows front to back: kind id, attack, health,
// total experience, perk id, consumed triggers, copied species, with the ids
// and Parrot sentinels of `sap_battle` above (see `env::BOARD_FIELDS`); an
// empty slot is a row with kind -1. `sap_env_kind` / `sap_env_status` give
// the same ids as `sap_kind` / `sap_status` without `sap_engine_init`.
//
// Call-level return codes: 0 ok, -10 null pointer or bad count, -11 index out
// of range, -12 a game that was never reset, -13 overlapping copy lists,
// -14 refused value (reason in `sap_env_error` of that game), -15 a fight
// failed inside the engine, -100 panic.
// ----------------------------------------------------------------------

use crate::env::{BoardRows, BOARD_FIELDS, BOARD_STATUSES, NS};

/// Id of a pet by name in the tables compiled into the library, or -1.
#[no_mangle]
pub extern "C" fn sap_env_kind(name: *const c_char) -> i32 {
    match text(name) {
        Some(n) => crate::envdata::Tables::global().d.by_name.get(&n).map(|k| *k as i32).unwrap_or(-1),
        None => -1,
    }
}

/// Name of pet kind `kind` copied into `buf`; returns its length, or -1.
#[no_mangle]
pub extern "C" fn sap_env_kind_name(kind: i32, buf: *mut c_char, len: i32) -> i32 {
    match usize::try_from(kind).ok().and_then(|k| crate::envdata::Tables::global().d.pets.get(k)) {
        Some(pet) => copy_text(&pet.name, buf, len),
        None => -1,
    }
}

/// Perk id of a status by name (same ids as `sap_status`), or -1.
#[no_mangle]
pub extern "C" fn sap_env_status(name: *const c_char) -> i32 {
    match text(name) {
        Some(n) => BOARD_STATUSES.iter().position(|s| s.name() == n).map(|i| i as i32).unwrap_or(-1),
        None => -1,
    }
}

/// `sap_env_step` for the games with a non-zero byte in `which` only; the
/// other games and their output rows are not touched.
#[no_mangle]
pub extern "C" fn sap_env_step_which(
    env: *mut BatchEnv,
    which: *const u8,
    actions: *const i64,
    masks: *const i64,
    out: *const Outputs,
) -> i32 {
    let (env, out) = match (unsafe { env.as_mut() }, outputs_ok(out)) {
        (Some(e), Some(o)) if !actions.is_null() && !masks.is_null() && !which.is_null() => (e, o),
        _ => return -10,
    };
    let n = env.len();
    let which = unsafe { std::slice::from_raw_parts(which, n) };
    let actions = unsafe { std::slice::from_raw_parts(actions, n * 6) };
    let masks = unsafe { std::slice::from_raw_parts(masks, n * 2) };
    guarded(|| unsafe { env.step_which(which, actions, masks, &out) as i32 })
}

/// Write the observation, lives, wins, active, terminated and winner rows of
/// the selected games again from their current state. Returns the number written.
#[no_mangle]
pub extern "C" fn sap_env_observe(env: *mut BatchEnv, which: *const u8, out: *const Outputs) -> i32 {
    let (env, out) = match (unsafe { env.as_mut() }, outputs_ok(out)) {
        (Some(e), Some(o)) if !which.is_null() => (e, o),
        _ => return -10,
    };
    let which = unsafe { std::slice::from_raw_parts(which, env.len()) };
    guarded(|| unsafe { env.observe(which, &out) as i32 })
}

/// Game `dst[k]` becomes an exact, independent copy of game `src[k]` for
/// `k < count`, output rows included (see `BatchEnv::copy`).
#[no_mangle]
pub extern "C" fn sap_env_copy(env: *mut BatchEnv, src: *const i32, dst: *const i32, count: i32, out: *const Outputs) -> i32 {
    let (env, out) = match (unsafe { env.as_mut() }, outputs_ok(out)) {
        (Some(e), Some(o)) if count >= 0 && (count == 0 || (!src.is_null() && !dst.is_null())) => (e, o),
        _ => return -10,
    };
    if count == 0 {
        return 0;
    }
    let src = unsafe { std::slice::from_raw_parts(src, count as usize) };
    let dst = unsafe { std::slice::from_raw_parts(dst, count as usize) };
    guarded(|| unsafe { env.copy(src, dst, &out) })
}

/// From now on every random draw of the listed games comes from the tape of
/// `seeds[k]` (each in `[0, 2**63)`); see `Game::rekey`.
#[no_mangle]
pub extern "C" fn sap_env_rekey(env: *mut BatchEnv, games: *const i32, seeds: *const i64, count: i32) -> i32 {
    let env = match unsafe { env.as_mut() } {
        Some(e) if count >= 0 && (count == 0 || (!games.is_null() && !seeds.is_null())) => e,
        _ => return -10,
    };
    if count == 0 {
        return 0;
    }
    let games = unsafe { std::slice::from_raw_parts(games, count as usize) };
    let seeds = unsafe { std::slice::from_raw_parts(seeds, count as usize) };
    guarded(|| env.rekey(games, seeds))
}

fn board_rows(raw: *const i32) -> BoardRows {
    let mut rows = [[0i32; BOARD_FIELDS]; NS];
    let cells = unsafe { std::slice::from_raw_parts(raw, NS * BOARD_FIELDS) };
    for (row, chunk) in rows.iter_mut().zip(cells.chunks(BOARD_FIELDS)) {
        row.copy_from_slice(chunk);
    }
    rows
}

/// Replace the team of `seat` of `game` by `board` (int32 `[5, 7]`); see `Seat::set_team`.
#[no_mangle]
pub extern "C" fn sap_env_set_team(env: *mut BatchEnv, game: i32, seat: i32, board: *const i32) -> i32 {
    match unsafe { env.as_mut() } {
        Some(e) if !board.is_null() => {
            let rows = board_rows(board);
            guarded(|| e.set_team(game, seat, &rows))
        }
        _ => -10,
    }
}

/// Set the gold of `seat` of `game`.
#[no_mangle]
pub extern "C" fn sap_env_set_gold(env: *mut BatchEnv, game: i32, seat: i32, gold: i32) -> i32 {
    match unsafe { env.as_mut() } {
        Some(e) => guarded(|| e.set_gold(game, seat, gold)),
        None => -10,
    }
}

/// Put a pet or a food into shop slot `slot` of `seat` of `game`, as if the
/// shop had just rolled it there; see `Seat::set_shop_item`. `item_type` 0:
/// `item` is a pet kind id (`sap_env_kind`); 1: a food id (`sap_env_food`).
/// `frozen` non-zero sets the slot's frozen flag, zero clears it. Returns 0,
/// -10 / -11 / -12 as the other edits, or -14 (refused, game unchanged, the
/// reason in `sap_env_error`): a slot index outside the shop list, a pet for
/// a food slot or a food for a pet slot, an unknown id, or an item the
/// Turtle pack shop cannot roll.
#[no_mangle]
pub extern "C" fn sap_env_set_shop_item(
    env: *mut BatchEnv,
    game: i32,
    seat: i32,
    slot: i32,
    item_type: i32,
    item: i32,
    frozen: i32,
) -> i32 {
    match unsafe { env.as_mut() } {
        Some(e) => guarded(|| e.set_shop_item(game, seat, slot, item_type, item, frozen != 0)),
        None => -10,
    }
}

/// Id of a food by name ("food-apple") in the tables compiled into the
/// library, or -1. The id space is separate from the pet kinds.
#[no_mangle]
pub extern "C" fn sap_env_food(name: *const c_char) -> i32 {
    match text(name) {
        Some(n) => crate::envdata::Tables::global().e.food_by_name.get(&n).map(|k| *k as i32).unwrap_or(-1),
        None => -1,
    }
}

/// Name of food `food` copied into `buf`; returns its length, or -1.
#[no_mangle]
pub extern "C" fn sap_env_food_name(food: i32, buf: *mut c_char, len: i32) -> i32 {
    match usize::try_from(food).ok().and_then(|k| crate::envdata::Tables::global().e.foods.get(k)) {
        Some(row) => copy_text(&row.name, buf, len),
        None => -1,
    }
}

/// The battle each game's last accepted step resolved: `turns` int32 `[n]`
/// (0: none), `seeds` int64 `[n]`, `teams` int32 `[n, 2, 5, 7]`, and
/// optionally `carried` int32 `[n, 2, 5, 2]` (may be null). Returns the
/// number of games with a battle.
#[no_mangle]
pub extern "C" fn sap_env_battle_events(
    env: *const BatchEnv,
    turns: *mut i32,
    seeds: *mut i64,
    teams: *mut i32,
    carried: *mut i32,
) -> i32 {
    match unsafe { env.as_ref() } {
        Some(e) if !turns.is_null() && !seeds.is_null() && !teams.is_null() => {
            guarded(|| unsafe { e.battle_events(turns, seeds, teams, carried) as i32 })
        }
        _ => -10,
    }
}

/// For each of `count` (game, seat) pairs: the team the seat would take into
/// battle if it ended its turn now, fought `samples` times against each of
/// `m` boards (`boards` int32 `[count, m, 5, 7]`), battle seed of sample `s`
/// against board `b` = `int(RandomTape(seeds[k]).uniform("eval/{b}") * (2**32-1))`
/// at cursor `s`. `counts` int32 `[count, m, 3]`: wins, draws, losses of the
/// seat. No game is changed.
#[no_mangle]
pub extern "C" fn sap_env_eval_vs(
    env: *mut BatchEnv,
    games: *const i32,
    seats: *const i32,
    count: i32,
    boards: *const i32,
    m: i32,
    samples: i32,
    seeds: *const i64,
    counts: *mut i32,
) -> i32 {
    let env = match unsafe { env.as_mut() } {
        Some(e) if count >= 0 && m >= 0 && samples >= 0 => e,
        _ => return -10,
    };
    if count == 0 || m == 0 {
        return 0;
    }
    if games.is_null() || seats.is_null() || boards.is_null() || seeds.is_null() || counts.is_null() {
        return -10;
    }
    let games = unsafe { std::slice::from_raw_parts(games, count as usize) };
    let seats = unsafe { std::slice::from_raw_parts(seats, count as usize) };
    let seeds = unsafe { std::slice::from_raw_parts(seeds, count as usize) };
    guarded(|| unsafe { env.eval_vs(games, seats, boards, m as usize, samples as u32, seeds, counts) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The board layout's perk ids are the battle entry point's.
    #[test]
    fn board_status_ids_match_the_battle_abi() {
        assert_eq!(STATUSES.len(), BOARD_STATUSES.len());
        for (i, name) in STATUSES.iter().enumerate() {
            assert_eq!(BOARD_STATUSES[i].name(), *name);
            assert_eq!(BOARD_STATUSES[i] as usize, i);
            assert_eq!(crate::data::Status::parse(name), Some(BOARD_STATUSES[i]));
        }
    }
}
