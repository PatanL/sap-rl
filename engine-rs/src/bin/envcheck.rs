//! Checker for the environment port: replays every match of the reference
//! traces (`tools/gen_env_golden.py`) with the recorded actions and compares,
//! at every step, the legal action list before the step, the executed plan,
//! rewards, battle flag and result, `terminated`, `winner`, the canonical
//! state and both observations (bytes for full-observation matches, SHA-256
//! otherwise). Exits non-zero on any mismatch and reports the match, the step
//! index and the first differing field.
//!
//! Optional keys of a step line (`tools/gen_env_ghost.py`; a line without
//! them is checked exactly as before):
//!   pre         state edits made before the step, in order, applied here through the same
//!               `Game` methods the C interface uses: ["gold", seat, value] (`set_gold`),
//!               ["team", seat, board] (`set_team`), ["rekey", seed] (`rekey`); ["legal"] marks a
//!               point between two edits where the reference computed the legal actions
//!               (`tools/gen_env_shopset.py`) ["shop", seat, slot, item name, frozen]
//!               (`set_shop_item`); ["state", dump] compares the canonical state at that point,
//!               ["obs", [seat 0, seat 1]] both observations at that point (bytes)
//!   teams       both teams as the reference's `Battle.__init__` received them, compared with
//!               the battle event of the step; `battleTurn` and `carried` belong to it
//! A board is five rows or nulls, a row [name, attack, health, total experience, status name,
//! consumed triggers, copies] with copies false (not stated), null (a Parrot that copied
//! nothing) or a pet name.
//!
//! Usage:
//!   envcheck [--threads N] [--report K] [--coverage out.json] [--no-obs] <trace.jsonl[.gz]>...

use sap_engine::canon::{canonical, first_difference};
use sap_engine::env::{decode_action, BoardRows, Coverage, Game, ShopItem, BOARD_STATUSES, EMPTY_ROW, NS};
use sap_engine::obs::{encode, Obs, OBS_BYTES, OBS_KEYS};
use sap_engine::tape::sha256;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read};
use std::sync::{mpsc, Arc, Mutex};

struct Options {
    threads: usize,
    report: usize,
    coverage: Option<String>,
    check_obs: bool,
}

#[derive(Default)]
struct MatchResult {
    seed: u64,
    file: String,
    steps: u64,
    steps_ok: u64,
    decisions: u64,
    battles: u64,
    obs_full: u64,
    obs_hash: u64,
    extended: u64,
    /// Counts of the optional keys: gold, team and rekey edits, battle events
    /// compared, events with a gap in a team, events with a carried counter.
    extra: [u64; 6],
    /// `set_shop_item` edits replayed (pets, foods), state dumps compared
    /// right after an edit, observations byte-compared right after an edit.
    shop_edits: [u64; 4],
    mismatch: Option<String>,
    coverage: Coverage,
}

/// A board of the trace (names) as board rows (ids).
fn board_from_json(game: &Game, board: &Value) -> Result<BoardRows, String> {
    let list = board.as_array().filter(|a| a.len() == NS).ok_or("a board needs five rows")?;
    let d = &game.t.d;
    let mut rows = [EMPTY_ROW; NS];
    for (i, row) in list.iter().enumerate() {
        if row.is_null() {
            continue;
        }
        let cell = |k: usize| row.get(k).ok_or(format!("row {i} is short"));
        let number = |k: usize| -> Result<i32, String> {
            cell(k)?.as_i64().map(|v| v as i32).ok_or(format!("row {i} field {k} is not a number"))
        };
        let kind = cell(0)?.as_str().and_then(|n| d.by_name.get(n)).ok_or(format!("row {i}: unknown pet"))?;
        let status = cell(4)?
            .as_str()
            .and_then(|n| BOARD_STATUSES.iter().position(|s| s.name() == n))
            .ok_or(format!("row {i}: unknown status"))?;
        let copies = match cell(6)? {
            Value::Bool(false) => -2,
            Value::Null => -1,
            Value::String(n) => *d.by_name.get(n.as_str()).ok_or(format!("row {i}: unknown copied pet"))? as i32,
            _ => return Err(format!("row {i}: bad copies field")),
        };
        rows[i] = [*kind as i32, number(1)?, number(2)?, number(3)?, status as i32, number(5)?, copies];
    }
    Ok(rows)
}

/// Apply the `pre` edits of a step line.
fn apply_pre(line: &Value, game: &mut Game, result: &mut MatchResult) -> Result<(), String> {
    let list = match line.get("pre") {
        None => return Ok(()),
        Some(v) => v.as_array().ok_or("pre is not a list")?,
    };
    for op in list {
        let name = op.get(0).and_then(|v| v.as_str()).unwrap_or("");
        let int = |k: usize| op.get(k).and_then(|v| v.as_i64()).ok_or(format!("pre {name}: field {k}"));
        match name {
            "gold" => {
                game.set_gold(int(1)? as usize, int(2)? as i32).map_err(|e| format!("set_gold: {e}"))?;
                result.extra[0] += 1;
            }
            "team" => {
                let rows = board_from_json(game, op.get(2).unwrap_or(&Value::Null))?;
                game.set_team(int(1)? as usize, &rows).map_err(|e| format!("set_team: {e}"))?;
                result.extra[1] += 1;
            }
            "rekey" => {
                let seed = op.get(1).and_then(|v| v.as_u64()).ok_or("pre rekey: seed")?;
                game.rekey(seed).map_err(|e| format!("rekey: {e}"))?;
                result.extra[2] += 1;
            }
            // ["shop", seat, slot, item name, frozen] (`set_shop_item`)
            "shop" => {
                let item_name = op.get(3).and_then(|v| v.as_str()).ok_or("pre shop: item name")?;
                let frozen = op.get(4).and_then(|v| v.as_bool()).ok_or("pre shop: frozen flag")?;
                let item = if let Some(kind) = game.t.d.by_name.get(item_name) {
                    ShopItem::Pet(*kind)
                } else if let Some(kind) = game.t.e.food_by_name.get(item_name) {
                    ShopItem::Food(*kind)
                } else {
                    return Err(format!("pre shop: unknown item {item_name:?}"));
                };
                game.set_shop_item(int(1)? as usize, int(2)? as usize, item, frozen)
                    .map_err(|e| format!("set_shop_item: {e}"))?;
                result.shop_edits[match item {
                    ShopItem::Pet(_) => 0,
                    ShopItem::Food(_) => 1,
                }] += 1;
            }
            // ["state", dump]: the reference's canonical state at this point (after the edits so far).
            "state" => {
                let reference = op.get(1).ok_or("pre state: dump")?;
                let extended = reference.get("seats").and_then(|s| s.get(0)).map(|s| s.get("x").is_some()).unwrap_or(false);
                if let Some(d) = first_difference(reference, &canonical(game, extended), "pre.state") {
                    return Err(d);
                }
                result.shop_edits[2] += 1;
            }
            // ["obs", [b64, b64]]: both seats' observations taken by the reference at this point
            // (like every observation, taking one registers the current team order as seen).
            "obs" => {
                let holder = serde_json::json!({ "obs": op.get(1).cloned().unwrap_or(Value::Null) });
                let before = result.obs_full;
                if let Some(d) = compare_obs(&holder, "", game, result, true) {
                    return Err(format!("pre.{d}"));
                }
                result.obs_full = before;
                result.shop_edits[3] += 2;
            }
            // The reference computed the active seats' legal actions between two edits.
            "legal" => {
                legal_value(game);
            }
            other => return Err(format!("unknown pre edit {other:?}")),
        }
    }
    Ok(())
}

/// Compare the battle event of the step with the teams the reference's battle received.
fn compare_teams(line: &Value, game: &Game, result: &mut MatchResult) -> Option<String> {
    let teams = line.get("teams")?;
    let event = match &game.last_battle {
        Some(e) => e,
        None => return Some("teams: the trace has a battle, the port has no battle event".into()),
    };
    if line.get("battleTurn").and_then(|v| v.as_i64()) != Some(event.turn as i64) {
        return Some(format!("battleTurn: reference {}, here {}", line["battleTurn"], event.turn));
    }
    if let Some(seed) = line.get("battleSeed") {
        if seed.as_u64() != Some(event.seed as u64) {
            return Some(format!("battle event seed: reference {seed}, here {}", event.seed));
        }
    }
    for side in 0..2 {
        let want = match board_from_json(game, &teams[side]) {
            Ok(rows) => rows,
            Err(e) => return Some(format!("teams[{side}]: {e}")),
        };
        if want != event.teams[side] {
            return Some(format!("teams[{side}]: reference {:?}, here {:?}", want, event.teams[side]));
        }
        for slot in 0..NS {
            for k in 0..2 {
                let reference = line["carried"][side][slot][k].as_i64();
                if reference != Some(event.carried[side][slot][k] as i64) {
                    return Some(format!(
                        "carried[{side}][{slot}][{k}]: reference {reference:?}, here {}",
                        event.carried[side][slot][k]
                    ));
                }
            }
        }
    }
    result.extra[3] += 1;
    let gap = |rows: &BoardRows| (1..NS).any(|i| rows[i][0] >= 0 && rows[i - 1][0] < 0);
    if event.teams.iter().any(gap) {
        result.extra[4] += 1;
    }
    if event.carried.iter().flatten().flatten().any(|v| *v != 0) {
        result.extra[5] += 1;
    }
    None
}

fn b64_decode(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => return Err("bad base64".into()),
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Name the first element that differs between two observation byte strings.
fn describe_obs_difference(reference: &[u8], ours: &[u8]) -> String {
    if reference.len() != ours.len() {
        return format!("observation is {} bytes in the reference, {} here", reference.len(), ours.len());
    }
    let mut offset = 0usize;
    for (name, size, count, dtype, _) in OBS_KEYS.iter() {
        for element in 0..*count {
            let a = &reference[offset..offset + size];
            let b = &ours[offset..offset + size];
            if a != b {
                let show = |x: &[u8]| -> String {
                    match *dtype {
                        "int64" => i64::from_le_bytes(x.try_into().unwrap()).to_string(),
                        "float32" => {
                            let v = f32::from_le_bytes(x.try_into().unwrap());
                            format!("{v:?} (bits {:08x})", v.to_bits())
                        }
                        _ => (x[0] as i8).to_string(),
                    }
                };
                return format!("{name}[flat index {element}]: reference {}, here {}", show(a), show(b));
            }
            offset += size;
        }
    }
    "observations differ".into()
}

fn compare_obs(line: &Value, suffix: &str, game: &mut Game, result: &mut MatchResult, check: bool) -> Option<String> {
    let full = line.get(format!("obs{suffix}").as_str());
    let hashed = line.get(format!("obs_sha{suffix}").as_str());
    let mut obs = Obs::zeroed();
    for side in 0..2 {
        // Encoding also registers the current team order, as the reference's does.
        encode(game, side, &mut obs);
        if !check {
            continue;
        }
        let ours = obs.to_bytes();
        debug_assert_eq!(ours.len(), OBS_BYTES);
        if let Some(list) = full {
            let text = match list.get(side).and_then(|v| v.as_str()) {
                Some(t) => t,
                None => return Some(format!("obs{suffix}[{side}] missing")),
            };
            let packed = match b64_decode(text) {
                Ok(p) => p,
                Err(e) => return Some(format!("obs{suffix}[{side}]: {e}")),
            };
            let mut reference = Vec::with_capacity(OBS_BYTES);
            if flate2::read::ZlibDecoder::new(&packed[..]).read_to_end(&mut reference).is_err() {
                return Some(format!("obs{suffix}[{side}]: zlib stream is damaged"));
            }
            if reference != ours {
                return Some(format!("obs{suffix}[{side}] {}", describe_obs_difference(&reference, &ours)));
            }
            result.obs_full += 1;
        } else if let Some(list) = hashed {
            let want = list.get(side).and_then(|v| v.as_str()).unwrap_or("");
            let got = hex(&sha256(&ours));
            if want != got {
                return Some(format!("obs_sha{suffix}[{side}]: reference {want}, here {got}"));
            }
            result.obs_hash += 1;
        } else {
            return Some(format!("no obs{suffix} or obs_sha{suffix} in the trace line"));
        }
    }
    None
}

fn legal_value(game: &mut Game) -> Value {
    let active = game.active();
    let mut out = Vec::new();
    for side in 0..2 {
        if active[side] {
            out.push(Value::from(game.seats[side].avail_actions().to_vec()));
        } else {
            out.push(Value::Null);
        }
    }
    Value::Array(out)
}

fn compare_state(line: &Value, key: &str, game: &Game, result: &mut MatchResult) -> Option<String> {
    let reference = match line.get(key) {
        Some(v) => v,
        None => return Some(format!("{key} missing in the trace line")),
    };
    let extended = reference
        .get("seats")
        .and_then(|s| s.get(0))
        .map(|s| s.get("x").is_some())
        .unwrap_or(false);
    if extended {
        result.extended += 1;
    }
    let ours = canonical(game, extended);
    first_difference(reference, &ours, key)
}

fn check_match(file: &str, lines: &[String], opt: &Options) -> MatchResult {
    let mut result = MatchResult { file: file.to_string(), ..Default::default() };
    let header: Value = match serde_json::from_str(&lines[0]) {
        Ok(v) => v,
        Err(e) => {
            result.mismatch = Some(format!("header is not JSON: {e}"));
            return result;
        }
    };
    result.seed = header.get("match").and_then(|v| v.as_u64()).unwrap_or(u64::MAX);
    result.steps = (lines.len() - 1) as u64;
    let mut game = match Game::new(result.seed, opt.coverage.is_some()) {
        Ok(g) => g,
        Err(e) => {
            result.mismatch = Some(format!("reset failed: {e}"));
            return result;
        }
    };
    let finish = |game: &Game, result: &mut MatchResult| {
        result.battles = game.battles;
        for seat in game.seats.iter() {
            if let Some(c) = &seat.cov {
                result.coverage.merge(c);
            }
        }
    };
    let fail = |game: &Game, mut result: MatchResult, step: i64, what: String| -> MatchResult {
        result.mismatch = Some(if step < 0 { format!("reset: {what}") } else { format!("step {step}: {what}") });
        finish(game, &mut result);
        result
    };
    // Reset: observations first (they register the starting order), then state and legal lists.
    if let Some(d) = compare_obs(&header, "0", &mut game, &mut result, opt.check_obs) {
        return fail(&game, result, -1, d);
    }
    if let Some(d) = compare_state(&header, "state0", &game, &mut result) {
        return fail(&game, result, -1, d);
    }
    let legal = legal_value(&mut game);
    if header.get("legal0") != Some(&legal) {
        return fail(&game, result, -1, format!("legal0: reference {}, here {legal}", header["legal0"]));
    }

    for (number, text) in lines[1..].iter().enumerate() {
        let line: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(e) => return fail(&game, result, number as i64, format!("line is not JSON: {e}")),
        };
        let step = line.get("i").and_then(|v| v.as_i64()).unwrap_or(-2);
        if step != number as i64 {
            return fail(&game, result, number as i64, format!("trace step index is {step}"));
        }
        if let Err(e) = apply_pre(&line, &mut game, &mut result) {
            return fail(&game, result, step, format!("pre: {e}"));
        }
        // Legal actions before the step.
        let legal = legal_value(&mut game);
        if line.get("legal") != Some(&legal) {
            return fail(&game, result, step, format!("legal: reference {}, here {legal}", line["legal"]));
        }
        let active = game.active();
        let want_active: Vec<bool> =
            line["active"].as_array().map(|a| a.iter().map(|x| x.as_bool() == Some(true)).collect()).unwrap_or_default();
        if want_active != active {
            return fail(&game, result, step, format!("active: reference {:?}, here {:?}", want_active, active));
        }
        let mut actions: [Option<[i64; 3]>; 2] = [None, None];
        let mut masks = [0u32; 2];
        for side in 0..2 {
            if let Some(triple) = line["act"][side].as_array() {
                let mut a = [0i64; 3];
                for k in 0..3 {
                    a[k] = triple.get(k).and_then(|v| v.as_i64()).unwrap_or(-1);
                }
                actions[side] = Some(a);
                result.decisions += 1;
            }
            masks[side] = line["mask"][side].as_u64().unwrap_or(u64::MAX) as u32;
        }
        // END freeze plans that use up the action budget exactly (counted before the step resets anything).
        if opt.coverage.is_some() {
            for side in 0..2 {
                let toggles = masks[side].count_ones() as i32;
                if toggles > 0 && game.seats[side].action_count + toggles == 50 {
                    result.coverage.hit("flow:end-freeze-plan:uses-the-last-budgeted-actions".into());
                }
                if actions[side].map(|a| a[0]) == Some(0) && game.seats[side].action_count >= 50 {
                    result.coverage.hit("flow:end-turn:at-the-50-action-cap".into());
                }
            }
        }
        let out = match game.step(actions, masks) {
            Ok(o) => o,
            Err(e) => return fail(&game, result, step, format!("the port raised: {e}")),
        };
        let plan: Value = Value::Array(
            out.plans
                .iter()
                .map(|p| Value::Array(p.iter().map(|&flat| Value::from(decode_action(flat).to_vec())).collect()))
                .collect(),
        );
        if line.get("plan") != Some(&plan) {
            return fail(&game, result, step, format!("plan: reference {}, here {plan}", line["plan"]));
        }
        for side in 0..2 {
            // Bit for bit: the reference returns -0.0 for the second seat of a zero swing.
            let want = line["rew"][side].as_f64();
            if want.map(f64::to_bits) != Some(out.rewards[side].to_bits()) {
                return fail(&game, result, step, format!("rew[{side}]: reference {want:?}, here {}", out.rewards[side]));
            }
        }
        if line["battle"].as_bool() != Some(out.battle_resolved) {
            return fail(&game, result, step, format!("battle: reference {}, here {}", line["battle"], out.battle_resolved));
        }
        let want_result = line["result"].as_u64().map(|v| v as u8);
        if want_result != out.battle_result {
            return fail(
                &game,
                result,
                step,
                format!("result: reference {:?}, here {:?} (battle seed {:?})", want_result, out.battle_result, game.battle_seeds.last()),
            );
        }
        if line["term"].as_bool() != Some(out.terminated) {
            return fail(&game, result, step, format!("term: reference {}, here {}", line["term"], out.terminated));
        }
        if let Some(d) = compare_obs(&line, "", &mut game, &mut result, opt.check_obs) {
            return fail(&game, result, step, d);
        }
        if let Some(d) = compare_state(&line, "state", &game, &mut result) {
            return fail(&game, result, step, d);
        }
        if let Some(seed) = line.get("battleSeed") {
            let ours = game.battle_seeds.last().map(|x| x.1 as u64);
            if out.battle_resolved && seed.as_u64() != ours {
                return fail(&game, result, step, format!("battleSeed: reference {seed}, here {ours:?}"));
            }
        }
        if let Some(d) = compare_teams(&line, &game, &mut result) {
            return fail(&game, result, step, d);
        }
        if line.get("teams").is_none() && out.battle_resolved != game.last_battle.is_some() {
            return fail(&game, result, step, "battle event present without a battle, or missing after one".into());
        }
        if opt.coverage.is_some() {
            // Match-flow rules, which the per-rule counters inside the engine do not see.
            let c = &mut result.coverage;
            for side in 0..2 {
                let plan = &out.plans[side];
                if plan.len() > 1 {
                    let freezes = plan.iter().filter(|a| (152..162).contains(*a)).count();
                    let unfreezes = plan.iter().filter(|a| (162..172).contains(*a)).count();
                    c.hit(format!("flow:end-freeze-plan:{}-toggles", plan.len() - 1));
                    if freezes > 0 {
                        c.hit("flow:end-freeze-plan:with-freeze".into());
                    }
                    if unfreezes > 0 {
                        c.hit("flow:end-freeze-plan:with-unfreeze".into());
                    }
                    if game.seats[side].action_count == 50 || (out.battle_resolved && !out.terminated) {
                        // (after a battle the counters are reset; the budget case is counted below)
                    }
                }
                if plan.last() == Some(&0) {
                    c.hit("flow:end-turn".into());
                    if actions[side].is_some() && game.seats[side].gold > 0 && !out.battle_resolved {
                        c.hit("flow:end-turn:forced-by-action-cap-with-gold-left".into());
                    }
                }
            }
            if out.battle_resolved {
                c.hit(format!("flow:battle:result-{}", out.battle_result.unwrap_or(9)));
                c.hit(format!("flow:battle:reward-{:?}", out.rewards[0]));
                if game.turn == 3 {
                    c.hit("flow:battle:turn-two-heart".into());
                }
            }
            if out.terminated {
                let (a, o) = (&game.seats[0], &game.seats[1]);
                if a.lives <= 0 || o.lives <= 0 {
                    c.hit("flow:end:a-seat-ran-out-of-lives".into());
                } else if a.lives != o.lives {
                    c.hit("flow:end:turn-cap:decided-by-lives".into());
                } else if a.wins != o.wins {
                    c.hit("flow:end:turn-cap:decided-by-wins".into());
                } else {
                    c.hit("flow:end:turn-cap:exact-tie".into());
                }
                c.hit(format!("flow:end:winner-{}", game.match_result()));
                c.hit(format!("flow:end:after-turn-{:02}", game.turn - 1));
            }
        }
        if out.terminated {
            let want = line["winner"].as_u64();
            let ours = game.match_result() as u64;
            if want != Some(ours) {
                return fail(&game, result, step, format!("winner: reference {want:?}, here {ours}"));
            }
            if number + 2 != lines.len() {
                return fail(&game, result, step, "the port ended the match before the trace did".into());
            }
        }
        result.steps_ok += 1;
    }
    if !game.terminated {
        return fail(&game, result, lines.len() as i64 - 2, "the trace ended before the match did".into());
    }
    finish(&game, &mut result);
    result
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

fn main() {
    let mut opt = Options { threads: 1, report: 5, coverage: None, check_obs: true };
    let mut files: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--threads" => opt.threads = args.next().and_then(|v| v.parse().ok()).unwrap_or(1).max(1),
            "--report" => opt.report = args.next().and_then(|v| v.parse().ok()).unwrap_or(5),
            "--coverage" => opt.coverage = args.next(),
            "--no-obs" => opt.check_obs = false,
            other if other.starts_with("--") => {
                eprintln!("unknown option {other}");
                std::process::exit(2);
            }
            _ => files.push(arg),
        }
    }
    if files.is_empty() {
        eprintln!("usage: envcheck [--threads N] [--report K] [--coverage out.json] [--no-obs] <trace.jsonl[.gz]>...");
        std::process::exit(2);
    }
    let opt = Arc::new(opt);
    let started = std::time::Instant::now();
    let (job_tx, job_rx) = mpsc::sync_channel::<(String, Vec<String>)>(opt.threads * 4);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let (res_tx, res_rx) = mpsc::channel::<MatchResult>();
    let mut workers = Vec::new();
    for _ in 0..opt.threads {
        let rx = job_rx.clone();
        let tx = res_tx.clone();
        let opt = opt.clone();
        workers.push(std::thread::spawn(move || loop {
            let job = rx.lock().unwrap().recv();
            match job {
                Ok((file, lines)) => {
                    let _ = tx.send(check_match(&file, &lines, &opt));
                }
                Err(_) => break,
            }
        }));
    }
    drop(res_tx);
    let reader_files = files.clone();
    let reader = std::thread::spawn(move || {
        for path in reader_files {
            let mut current: Vec<String> = Vec::new();
            for line in open(&path).lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("{path}: read error: {e}");
                        std::process::exit(2);
                    }
                };
                if line.is_empty() {
                    continue;
                }
                if line.starts_with("{\"match\"") && !current.is_empty() {
                    job_tx.send((path.clone(), std::mem::take(&mut current))).unwrap();
                }
                current.push(line);
            }
            if !current.is_empty() {
                job_tx.send((path.clone(), current)).unwrap();
            }
        }
    });

    let mut total = MatchResult::default();
    let mut matches = 0u64;
    let mut matches_ok = 0u64;
    let mut coverage = Coverage::default();
    let mut failures: Vec<(String, u64, String)> = Vec::new();
    let mut per_file: std::collections::BTreeMap<String, [u64; 4]> = Default::default();
    for r in res_rx {
        matches += 1;
        total.steps += r.steps;
        total.steps_ok += r.steps_ok;
        total.decisions += r.decisions;
        total.battles += r.battles;
        total.obs_full += r.obs_full;
        total.obs_hash += r.obs_hash;
        total.extended += r.extended;
        for k in 0..6 {
            total.extra[k] += r.extra[k];
        }
        for k in 0..4 {
            total.shop_edits[k] += r.shop_edits[k];
        }
        coverage.merge(&r.coverage);
        let entry = per_file.entry(r.file.clone()).or_insert([0; 4]);
        entry[0] += 1;
        entry[2] += r.steps;
        entry[3] += r.steps_ok;
        match r.mismatch {
            None => {
                matches_ok += 1;
                entry[1] += 1;
            }
            Some(m) => failures.push((r.file, r.seed, m)),
        }
    }
    reader.join().unwrap();
    for w in workers {
        w.join().unwrap();
    }
    failures.sort();
    for (file, seed, what) in failures.iter().take(opt.report) {
        println!("MISMATCH {file} match {seed} {what}");
    }
    if failures.len() > opt.report {
        println!("... and {} more mismatching matches", failures.len() - opt.report);
    }
    for (file, c) in &per_file {
        println!("{file}: matches {}/{} steps {}/{}", c[1], c[0], c[3], c[2]);
    }
    if total.extra.iter().any(|v| *v != 0) {
        println!(
            "EDITS gold {} team {} rekey {}; BATTLE EVENTS compared {} (a team with a gap in {}, a carried counter in {})",
            total.extra[0], total.extra[1], total.extra[2], total.extra[3], total.extra[4], total.extra[5]
        );
    }
    if total.shop_edits.iter().any(|v| *v != 0) {
        println!(
            "SHOP EDITS pets {} foods {}; right after an edit: {} state dumps compared, {} observations byte-compared",
            total.shop_edits[0], total.shop_edits[1], total.shop_edits[2], total.shop_edits[3]
        );
    }
    println!(
        "TOTAL matches {matches_ok}/{matches} steps {}/{} decisions {} battles {} observations: {} byte-compared, {} hash-compared, states with extended block {} ({:.1}s)",
        total.steps_ok,
        total.steps,
        total.decisions,
        total.battles,
        total.obs_full,
        total.obs_hash,
        total.extended,
        started.elapsed().as_secs_f64()
    );
    if let Some(path) = &opt.coverage {
        let mut keys: Vec<(&String, &u64)> = coverage.counts.iter().collect();
        keys.sort();
        let map: serde_json::Map<String, Value> = keys.into_iter().map(|(k, v)| (k.clone(), Value::from(*v))).collect();
        std::fs::write(path, serde_json::to_string_pretty(&Value::Object(map)).unwrap()).unwrap();
        println!("coverage written to {path}");
    }
    if matches_ok != matches || matches == 0 {
        std::process::exit(1);
    }
}
