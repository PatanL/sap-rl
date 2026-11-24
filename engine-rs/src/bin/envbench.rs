//! Speed of the environment port through the batch interface (`batch.rs`),
//! observations written on every step.
//!
//!   envbench replay [--threads T] [--repeat R] <trace.jsonl[.gz]>...
//!       Replays the recorded decisions of every match in the traces. All
//!       matches run side by side as one batch (R copies of each with
//!       --repeat); finished games are skipped. Every winner is checked.
//!   envbench random [--threads T] [--matches M] [--first SEED] [--batch B]
//!       Uniform random legal play (END freeze masks as in gen_env_golden.py:
//!       each freezable slot with probability 0.2 at zero gold), chosen from
//!       the observation masks. M complete matches from consecutive seeds, B
//!       games in the batch at a time; every match is played to its end, so
//!       short games are not favoured.
//!
//! Prints decisions, steps (batch calls per game), matches, battles, wall
//! time and the rates. `random --digest` also hashes everything written on
//! every step (slow; the digest is the same for any thread count and batch
//! size); `replay` hashes the final rows and checks every winner.

use sap_engine::batch::{BatchEnv, Outputs};
use sap_engine::obs::OBS_KEYS;
use serde_json::Value;
use std::io::{BufRead, BufReader};

struct Buffers {
    obs: Vec<Vec<u8>>,
    rewards: Vec<f64>,
    battle_resolved: Vec<i8>,
    battle_result: Vec<i8>,
    terminated: Vec<i8>,
    winner: Vec<i8>,
    lives: Vec<i32>,
    wins: Vec<i32>,
    active: Vec<i8>,
    status: Vec<i32>,
}

impl Buffers {
    fn new(n: usize) -> Buffers {
        Buffers {
            obs: OBS_KEYS.iter().map(|k| vec![0u8; n * 2 * k.1 * k.2]).collect(),
            rewards: vec![0.0; n * 2],
            battle_resolved: vec![0; n],
            battle_result: vec![0; n],
            terminated: vec![0; n],
            winner: vec![0; n],
            lives: vec![0; n * 2],
            wins: vec![0; n * 2],
            active: vec![0; n * 2],
            status: vec![0; n],
        }
    }

    fn outputs(&mut self) -> Outputs {
        let mut obs = [std::ptr::null_mut(); 15];
        for (k, v) in self.obs.iter_mut().enumerate() {
            obs[k] = v.as_mut_ptr();
        }
        Outputs {
            obs,
            rewards: self.rewards.as_mut_ptr(),
            battle_resolved: self.battle_resolved.as_mut_ptr(),
            battle_result: self.battle_result.as_mut_ptr(),
            terminated: self.terminated.as_mut_ptr(),
            winner: self.winner.as_mut_ptr(),
            lives: self.lives.as_mut_ptr(),
            wins: self.wins.as_mut_ptr(),
            active: self.active.as_mut_ptr(),
            status: self.status.as_mut_ptr(),
        }
    }

    /// FNV-1a over one game's rows (observations, rewards, flags).
    fn digest_game(&self, i: usize, mut h: u64) -> u64 {
        let mut eat = |bytes: &[u8]| {
            for b in bytes {
                h ^= *b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        for (k, v) in self.obs.iter().enumerate() {
            let row = 2 * OBS_KEYS[k].1 * OBS_KEYS[k].2;
            eat(&v[i * row..(i + 1) * row]);
        }
        eat(&self.rewards[i * 2].to_le_bytes());
        eat(&self.rewards[i * 2 + 1].to_le_bytes());
        eat(&[self.battle_resolved[i] as u8, self.battle_result[i] as u8, self.terminated[i] as u8, self.winner[i] as u8]);
        h
    }
}

fn open(path: &str) -> Box<dyn BufRead> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("cannot open {path}: {e}");
        std::process::exit(2);
    });
    if path.ends_with(".gz") {
        Box::new(BufReader::with_capacity(1 << 20, flate2::read::MultiGzDecoder::new(BufReader::new(file))))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, file))
    }
}

struct Tape {
    seed: i64,
    /// Per step: six action values and two masks.
    steps: Vec<([i64; 6], [i64; 2])>,
    winner: i8,
}

fn load(files: &[String]) -> Vec<Tape> {
    let mut tapes: Vec<Tape> = Vec::new();
    for path in files {
        for line in open(path).lines() {
            let line = line.expect("read");
            if line.is_empty() {
                continue;
            }
            // Only the small fields are needed; skip the observation payloads cheaply.
            let v: Value = serde_json::from_str(&line).expect("json");
            if let Some(seed) = v.get("match").and_then(|s| s.as_i64()) {
                tapes.push(Tape { seed, steps: Vec::new(), winner: -1 });
                continue;
            }
            let tape = tapes.last_mut().expect("header first");
            let mut a = [-1i64; 6];
            let mut m = [0i64; 2];
            for side in 0..2 {
                if let Some(triple) = v["act"][side].as_array() {
                    for k in 0..3 {
                        a[side * 3 + k] = triple[k].as_i64().unwrap();
                    }
                }
                m[side] = v["mask"][side].as_i64().unwrap_or(0);
            }
            tape.steps.push((a, m));
            if let Some(w) = v.get("winner").and_then(|w| w.as_i64()) {
                tape.winner = w as i8;
            }
        }
    }
    tapes
}

fn report(what: &str, threads: usize, matches: u64, steps: u64, decisions: u64, battles: u64, secs: f64, digest: u64) {
    println!(
        "{what}: threads {threads}  matches {matches}  steps {steps}  decisions {decisions}  battles {battles}  \
         {secs:.3}s  ->  {:.0} decisions/s  {:.0} steps/s  {:.1} matches/s  ({:.1} decisions/match)  digest {digest:016x}",
        decisions as f64 / secs,
        steps as f64 / secs,
        matches as f64 / secs,
        decisions as f64 / matches as f64,
    );
}

fn replay(files: &[String], threads: usize, repeat: usize) {
    let mut tapes = load(files);
    // `--repeat R`: every match R times in the batch (longer, steadier timing; same work per match).
    let unique = tapes.len();
    for r in 1..repeat {
        let _ = r;
        for k in 0..unique {
            let copy = Tape { seed: tapes[k].seed, steps: tapes[k].steps.clone(), winner: tapes[k].winner };
            tapes.push(copy);
        }
    }
    let n = tapes.len();
    let mut env = BatchEnv::new(n, threads);
    let mut buf = Buffers::new(n);
    let out = buf.outputs();
    let seeds: Vec<i64> = tapes.iter().map(|t| t.seed).collect();
    let longest = tapes.iter().map(|t| t.steps.len()).max().unwrap_or(0);
    let mut actions = vec![-1i64; n * 6];
    let mut masks = vec![0i64; n * 2];
    let started = std::time::Instant::now();
    let refused = unsafe { env.reset(None, &seeds, &out) };
    assert_eq!(refused, 0, "a reset was refused");
    let (mut decisions, mut steps, mut digest) = (0u64, 0u64, 0xcbf29ce484222325u64);
    for s in 0..longest {
        for (i, tape) in tapes.iter().enumerate() {
            match tape.steps.get(s) {
                Some((a, m)) => {
                    actions[i * 6..i * 6 + 6].copy_from_slice(a);
                    masks[i * 2..i * 2 + 2].copy_from_slice(m);
                    decisions += (a[0] >= 0) as u64 + (a[3] >= 0) as u64;
                    steps += 1;
                }
                None => {
                    actions[i * 6..i * 6 + 6].copy_from_slice(&[-1; 6]);
                    masks[i * 2..i * 2 + 2].copy_from_slice(&[0, 0]);
                }
            }
        }
        let bad = unsafe { env.step(&actions, &masks, &out) };
        if bad != 0 {
            let i = (0..n).find(|&i| buf.status[i] < 0).unwrap();
            panic!("game {i} (seed {}) step {s}: status {} {}", seeds[i], buf.status[i], env.error(i));
        }
    }
    let secs = started.elapsed().as_secs_f64();
    let mut battles = 0;
    for (i, tape) in tapes.iter().enumerate() {
        assert_eq!(buf.terminated[i], 1, "match {} did not finish", tape.seed);
        assert_eq!(buf.winner[i], tape.winner, "match {} winner", tape.seed);
        battles += env.game(i).unwrap().battles;
        digest = buf.digest_game(i, digest);
    }
    report("replay", threads, n as u64, steps, decisions, battles, secs, digest);
}

#[derive(Clone, Copy)]
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        ((self.next() >> 11) as u128 * n as u128 >> 53) as u64
    }
    fn chance(&mut self, p: f64) -> bool {
        ((self.next() >> 11) as f64) < p * 9007199254740992.0
    }
}

/// One uniformly random legal decision for a seat, from its observation row.
fn choose(target_mask: &[u8], numeric0: &[u8], rng: &mut XorShift) -> ([i64; 3], i64) {
    // The mask bytes are 0 or 1: count and select eight at a time.
    let mut words = [0u64; 75];
    let mut legal = 0u32;
    let mut padded = [0u8; 600];
    padded[..594].copy_from_slice(&target_mask[..594]);
    for (k, w) in words.iter_mut().enumerate() {
        *w = u64::from_le_bytes(padded[k * 8..k * 8 + 8].try_into().unwrap());
        legal += w.count_ones();
    }
    assert!(legal > 0, "no legal action");
    let mut pick = rng.below(legal as u64) as u32;
    let mut flat = 0usize;
    for (k, w) in words.iter().enumerate() {
        let here = w.count_ones();
        if pick < here {
            let mut word = *w;
            for _ in 0..pick {
                word &= word - 1;
            }
            flat = k * 8 + (word.trailing_zeros() / 8) as usize;
            break;
        }
        pick -= here;
    }
    let (ty, source, target) = (flat / 66, (flat / 6) % 11, flat % 6);
    let mut mask = 0i64;
    if ty == 0 {
        let num = |k: usize| f32::from_le_bytes(numeric0[k * 4..k * 4 + 4].try_into().unwrap());
        if num(11) == 0.0 {
            let count = (num(5) * 50.0).round() as i64;
            let mut bits = 0;
            for slot in 0..10 {
                let freezable = target_mask[6 * 66 + slot * 6 + 5] != 0 || target_mask[7 * 66 + slot * 6 + 5] != 0;
                if freezable && rng.chance(0.2) {
                    mask |= 1 << slot;
                    bits += 1;
                }
            }
            if bits > (50 - count).max(0) {
                mask = 0;
            }
        }
    }
    ([ty as i64, source as i64, target as i64], mask)
}

/// Shared mutable rows for the parallel action choice: index `i` only ever touches its own rows.
struct Rows<T>(*mut T);
unsafe impl<T> Sync for Rows<T> {}

fn random(threads: usize, matches: usize, first: i64, batch: usize, with_digest: bool) {
    let n = batch.min(matches);
    let mut env = BatchEnv::new(n, threads);
    let mut buf = Buffers::new(n);
    let out = buf.outputs();
    let mut seeds: Vec<i64> = (0..n as i64).map(|i| first + i).collect();
    let mut rngs: Vec<XorShift> = seeds.iter().map(|s| XorShift(0x9e3779b97f4a7c15 ^ (*s as u64).wrapping_mul(0xbf58476d1ce4e5b9))).collect();
    let mut next_seed = first + n as i64;
    let last_seed = first + matches as i64;
    let mut live: Vec<bool> = vec![true; n];
    let mut which = vec![1u8; n];
    let mut actions = vec![-1i64; n * 6];
    let mut masks = vec![0i64; n * 2];
    let (mut decisions, mut steps, mut finished, mut battles) = (0u64, 0u64, 0u64, 0u64);
    // Per-seed digests, combined in seed order at the end (independent of batching).
    let mut digests: Vec<(i64, u64)> = Vec::with_capacity(matches);
    let mut running: Vec<u64> = vec![0xcbf29ce484222325; n];
    let started = std::time::Instant::now();
    assert_eq!(unsafe { env.reset(Some(&which), &seeds, &out) }, 0);
    let target_row = 9 * 11 * 6;
    let numeric_row = 21 * 16 * 4;
    loop {
        let live_now = live.iter().filter(|l| **l).count() as u64;
        if live_now == 0 {
            break;
        }
        steps += live_now;
        {
            let (a, m, r) = (Rows(actions.as_mut_ptr()), Rows(masks.as_mut_ptr()), Rows(rngs.as_mut_ptr()));
            let (a, m, r) = (&a, &m, &r);
            let (live, buf) = (&live, &buf);
            let chosen = std::sync::atomic::AtomicU64::new(0);
            env.parallel(n, |i| unsafe {
                let act = std::slice::from_raw_parts_mut(a.0.add(i * 6), 6);
                let msk = std::slice::from_raw_parts_mut(m.0.add(i * 2), 2);
                act.copy_from_slice(&[-1; 6]);
                msk.copy_from_slice(&[0, 0]);
                if !live[i] {
                    return;
                }
                let rng = &mut *r.0.add(i);
                let mut count = 0;
                for side in 0..2 {
                    let row = i * 2 + side;
                    if buf.active[row] != 0 {
                        let tm = &buf.obs[8][row * target_row..(row + 1) * target_row];
                        let nm = &buf.obs[4][row * numeric_row..row * numeric_row + 64];
                        let (choice, mask) = choose(tm, nm, rng);
                        act[side * 3..side * 3 + 3].copy_from_slice(&choice);
                        msk[side] = mask;
                        count += 1;
                    }
                }
                chosen.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
            });
            decisions += chosen.into_inner();
        }
        let bad = unsafe { env.step(&actions, &masks, &out) };
        if bad != 0 {
            let i = (0..n).find(|&i| buf.status[i] < 0).unwrap();
            panic!("game {i} (seed {}): status {} {}", seeds[i], buf.status[i], env.error(i));
        }
        let mut reset_any = false;
        for i in 0..n {
            which[i] = 0;
            if !live[i] {
                continue;
            }
            if with_digest {
                running[i] = buf.digest_game(i, running[i]);
            }
            if buf.terminated[i] != 0 {
                finished += 1;
                battles += env.game(i).unwrap().battles;
                digests.push((seeds[i], running[i]));
                if next_seed < last_seed {
                    seeds[i] = next_seed;
                    rngs[i] = XorShift(0x9e3779b97f4a7c15 ^ (next_seed as u64).wrapping_mul(0xbf58476d1ce4e5b9));
                    running[i] = 0xcbf29ce484222325;
                    next_seed += 1;
                    which[i] = 1;
                    reset_any = true;
                } else {
                    live[i] = false;
                }
            }
        }
        if reset_any {
            assert_eq!(unsafe { env.reset(Some(&which), &seeds, &out) }, 0);
        }
    }
    let secs = started.elapsed().as_secs_f64();
    digests.sort();
    let mut digest = 0xcbf29ce484222325u64;
    for (_, d) in &digests {
        digest = (digest ^ d).wrapping_mul(0x100000001b3);
    }
    report("random", threads, finished, steps, decisions, battles, secs, digest);
}

/// A small sampling profiler (perf is not available to unprivileged users
/// here): SIGPROF every millisecond of CPU time, the handler records the
/// program counter and the frame-pointer chain. Build with
/// `RUSTFLAGS="-C force-frame-pointers=yes"` and symbolise the output with
/// `tools/profile_report.py`.
mod sampler {
    use std::sync::atomic::{AtomicUsize, Ordering};
    const DEPTH: usize = 20;
    const MAX: usize = 400_000;
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    static mut SAMPLES: *mut [usize; DEPTH] = std::ptr::null_mut();

    extern "C" fn on_tick(_sig: libc::c_int, _info: *mut libc::siginfo_t, context: *mut libc::c_void) {
        let slot = COUNT.fetch_add(1, Ordering::Relaxed);
        if slot >= MAX {
            return;
        }
        unsafe {
            let uc = context as *const libc::ucontext_t;
            let row = &mut *SAMPLES.add(slot);
            #[cfg(target_arch = "aarch64")]
            let (pc, mut fp, sp) = ((*uc).uc_mcontext.pc as usize, (*uc).uc_mcontext.regs[29] as usize, (*uc).uc_mcontext.sp as usize);
            #[cfg(target_arch = "x86_64")]
            let (pc, mut fp, sp) = (
                (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] as usize,
                (*uc).uc_mcontext.gregs[libc::REG_RBP as usize] as usize,
                (*uc).uc_mcontext.gregs[libc::REG_RSP as usize] as usize,
            );
            row[0] = pc;
            let mut depth = 1;
            while depth < DEPTH && fp >= sp && fp < sp + (8 << 20) && fp % 8 == 0 {
                let frame = fp as *const usize;
                let ret = *frame.add(1);
                let next = *frame;
                if ret == 0 {
                    break;
                }
                row[depth] = ret;
                depth += 1;
                if next <= fp {
                    break;
                }
                fp = next;
            }
            while depth < DEPTH {
                row[depth] = 0;
                depth += 1;
            }
        }
    }

    pub fn start() {
        unsafe {
            let layout = std::alloc::Layout::array::<[usize; DEPTH]>(MAX).unwrap();
            SAMPLES = std::alloc::alloc_zeroed(layout) as *mut [usize; DEPTH];
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_tick as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigaction(libc::SIGPROF, &action, std::ptr::null_mut());
            let tick = libc::timeval { tv_sec: 0, tv_usec: 1000 };
            let timer = libc::itimerval { it_interval: tick, it_value: tick };
            libc::setitimer(libc::ITIMER_PROF, &timer, std::ptr::null_mut());
        }
    }

    pub fn finish(path: &str) {
        use std::io::Write;
        unsafe {
            let off = libc::itimerval {
                it_interval: libc::timeval { tv_sec: 0, tv_usec: 0 },
                it_value: libc::timeval { tv_sec: 0, tv_usec: 0 },
            };
            libc::setitimer(libc::ITIMER_PROF, &off, std::ptr::null_mut());
        }
        let n = COUNT.load(Ordering::Relaxed).min(MAX);
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default();
        let base = maps
            .lines()
            .find(|l| l.ends_with(&exe))
            .and_then(|l| usize::from_str_radix(l.split('-').next().unwrap_or("0"), 16).ok())
            .unwrap_or(0);
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).expect("sample file"));
        writeln!(out, "# exe {exe} base {base:x} samples {n}").unwrap();
        for i in 0..n {
            let row = unsafe { &*SAMPLES.add(i) };
            let frames: Vec<String> = row.iter().take_while(|a| **a != 0).map(|a| format!("{:x}", a.wrapping_sub(base))).collect();
            writeln!(out, "{}", frames.join(" ")).unwrap();
        }
        eprintln!("{n} samples written to {path}");
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: envbench replay|random [options] (see the top of src/bin/envbench.rs)");
        std::process::exit(2);
    }
    let mode = args.remove(0);
    let (mut threads, mut matches, mut first, mut batch) = (1usize, 2000usize, 990_000_000i64, 1024usize);
    let mut with_digest = false;
    let mut repeat = 1usize;
    let mut sample: Option<String> = None;
    let mut files = Vec::new();
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--threads" => threads = it.next().unwrap().parse().unwrap(),
            "--matches" => matches = it.next().unwrap().parse().unwrap(),
            "--first" => first = it.next().unwrap().parse().unwrap(),
            "--batch" => batch = it.next().unwrap().parse().unwrap(),
            "--digest" => with_digest = true,
            "--repeat" => repeat = it.next().unwrap().parse().unwrap(),
            "--sample" => sample = it.next(),
            _ => files.push(arg),
        }
    }
    if sample.is_some() {
        sampler::start();
    }
    match mode.as_str() {
        "replay" => replay(&files, threads, repeat),
        "random" => random(threads, matches, first, batch, with_digest),
        _ => {
            eprintln!("unknown mode {mode}");
            std::process::exit(2);
        }
    }
    if let Some(path) = sample {
        sampler::finish(&path);
    }
}
