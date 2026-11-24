//! Replay golden rows from the Python reference and require exact equality.
//!
//! Usage: golden [--data DIR] [--golden FILE] [--limit N] [--show N] [--bench]

use sap_engine::{simulate, Data, PetSpec, Status};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn specs(v: &Value) -> Vec<PetSpec> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|p| PetSpec {
                    id: p["id"].as_str().unwrap_or("").to_string(),
                    attack: p["attack"].as_i64().unwrap_or(0) as i32,
                    health: p["health"].as_i64().unwrap_or(0) as i32,
                    experience: p["experience"].as_i64().unwrap_or(0) as i32,
                    status: p["status"].as_str().unwrap_or("none").to_string(),
                    triggers_consumed: p["triggersConsumed"].as_i64().unwrap_or(0) as i32,
                    parrot_copies: p.get("parrotCopies").map(|v| v.as_str().map(|s| s.to_string())),
                })
                .collect()
        })
        .unwrap_or_default()
}

type Slot = Option<(String, i64, i64, String, i64)>;

fn expected_board(v: &Value) -> Vec<Slot> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|s| {
                    if s.is_null() {
                        None
                    } else {
                        Some((
                            s[0].as_str().unwrap_or("").to_string(),
                            s[1].as_i64().unwrap_or(i64::MIN),
                            s[2].as_i64().unwrap_or(i64::MIN),
                            s[3].as_str().unwrap_or("").to_string(),
                            s[4].as_i64().unwrap_or(i64::MIN),
                        ))
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn got_board(d: &Data, b: &[Option<sap_engine::SlotView>; 5]) -> Vec<Slot> {
    b.iter()
        .map(|s| {
            s.map(|v| {
                (
                    d.pets[v.kind].name.clone(),
                    v.attack as i64,
                    v.health as i64,
                    Status::name(v.status).to_string(),
                    v.level as i64,
                )
            })
        })
        .collect()
}

fn show(b: &[Slot]) -> String {
    b.iter()
        .map(|s| match s {
            None => "___".to_string(),
            Some((n, a, h, st, l)) => {
                let st = if st == "none" { String::new() } else { format!("*{}", &st[7..]) };
                format!("{}({}/{}{} L{})", &n[4..], a, h, st, l)
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut data_dir = "data".to_string();
    let mut golden = "golden/v31-340.jsonl".to_string();
    let mut limit = usize::MAX;
    let mut show_n = 5usize;
    let mut bench = false;
    let mut only: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => {
                data_dir = args[i + 1].clone();
                i += 1;
            }
            "--golden" => {
                golden = args[i + 1].clone();
                i += 1;
            }
            "--limit" => {
                limit = args[i + 1].parse().expect("limit");
                i += 1;
            }
            "--show" => {
                show_n = args[i + 1].parse().expect("show");
                i += 1;
            }
            "--only" => {
                only = Some(args[i + 1].clone());
                i += 1;
            }
            "--bench" => bench = true,
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    let data = Data::load_dir(std::path::Path::new(&data_dir)).expect("load data");
    let file = std::fs::File::open(&golden).expect("open golden file");
    let mut rows: Vec<Value> = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line).expect("parse row");
        if let Some(prefix) = &only {
            if !v["battleId"].as_str().unwrap_or("").starts_with(prefix.as_str()) {
                continue;
            }
        }
        rows.push(v);
        if rows.len() >= limit {
            break;
        }
    }

    let mut matched = 0usize;
    let mut errors = 0usize;
    let mut skipped = 0usize;
    let mut by_pet_bad: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_pet_all: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_step: BTreeMap<String, usize> = BTreeMap::new();
    let mut by_error: BTreeMap<String, usize> = BTreeMap::new();
    let mut shown = 0usize;
    let mut inputs = Vec::with_capacity(rows.len());

    for row in &rows {
        if row.get("error").is_some() {
            skipped += 1;
            continue;
        }
        let player = specs(&row["player"]);
        let opponent = specs(&row["opponent"]);
        let seed = row["seed"].as_u64().unwrap_or(0) as u32;
        let mut names: Vec<String> = player.iter().chain(opponent.iter()).map(|p| p.id.clone()).collect();
        names.sort();
        names.dedup();
        for n in &names {
            *by_pet_all.entry(n.clone()).or_default() += 1;
        }
        let expected_trace = row["trace"].as_array().cloned().unwrap_or_default();
        let result = simulate(&data, &player, &opponent, seed, true);
        let mut problem: Option<(String, String)> = None; // (step label, detail)
        match &result {
            Err(e) => {
                errors += 1;
                *by_error.entry(e.0.clone()).or_default() += 1;
                problem = Some(("error".to_string(), e.0.clone()));
            }
            Ok(out) => {
                let steps = expected_trace.len().max(out.trace.len());
                for step in 0..steps {
                    if step >= expected_trace.len() || step >= out.trace.len() {
                        problem = Some((
                            format!("step {step}"),
                            format!("trace length differs: expected {} got {}", expected_trace.len(), out.trace.len()),
                        ));
                        break;
                    }
                    let mut differs = false;
                    let mut detail = String::new();
                    for side in 0..2 {
                        let e = expected_board(&expected_trace[step][side]);
                        let g = got_board(&data, &out.trace[step][side]);
                        if e != g {
                            differs = true;
                        }
                        detail.push_str(&format!(
                            "\n      {} expected: {}\n      {} got     : {}",
                            if side == 0 { "P" } else { "O" },
                            show(&e),
                            if side == 0 { "P" } else { "O" },
                            show(&g)
                        ));
                    }
                    if differs {
                        let mut context = String::new();
                        if step > 0 {
                            context.push_str(&format!(
                                "\n      previous step {}: P {} | O {}",
                                step - 1,
                                show(&expected_board(&expected_trace[step - 1][0])),
                                show(&expected_board(&expected_trace[step - 1][1]))
                            ));
                        }
                        let label = match step {
                            0 => "step 0 (construction)".to_string(),
                            1 => "step 1 (start of battle)".to_string(),
                            _ => "attack round".to_string(),
                        };
                        problem = Some((label, format!("first difference at step {step}{context}{detail}")));
                        break;
                    }
                }
                if problem.is_none() {
                    let want = row["outcome"].as_i64().unwrap_or(-9);
                    if want != out.outcome as i64 {
                        problem = Some(("outcome".to_string(), format!("outcome expected {want} got {}", out.outcome)));
                    } else {
                        let probe = row["probe"].as_f64().unwrap_or(f64::NAN);
                        if probe != out.probe {
                            problem = Some((
                                "probe (draw count)".to_string(),
                                format!("probe expected {probe} got {}", out.probe),
                            ));
                        }
                    }
                }
            }
        }
        match problem {
            None => matched += 1,
            Some((label, detail)) => {
                *by_step.entry(label.clone()).or_default() += 1;
                for n in &names {
                    *by_pet_bad.entry(n.clone()).or_default() += 1;
                }
                if shown < show_n {
                    shown += 1;
                    println!(
                        "MISMATCH {} seed {} [{}]\n    pets: {}\n    {}",
                        row["battleId"].as_str().unwrap_or(""),
                        seed,
                        label,
                        names.iter().map(|n| &n[4..]).collect::<Vec<_>>().join(", "),
                        detail
                    );
                }
            }
        }
        inputs.push((player, opponent, seed));
    }

    let total = rows.len() - skipped;
    println!("rows: {total}  matched exactly: {matched}  mismatched: {}  (engine errors: {errors}, skipped reference errors: {skipped})", total - matched);
    if total != matched {
        println!("first difference by stage: {:?}", by_step);
        if !by_error.is_empty() {
            println!("engine errors: {:?}", by_error);
        }
        let mut rates: Vec<(f64, usize, usize, String)> = by_pet_bad
            .iter()
            .map(|(n, &b)| {
                let all = by_pet_all[n];
                (b as f64 / all as f64, b, all, n.clone())
            })
            .collect();
        rates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("mismatch rate by pet present (bad/all):");
        for (rate, b, all, n) in rates.iter().take(25) {
            println!("  {:<18} {:>5}/{:<5} {:.1}%", &n[4..], b, all, rate * 100.0);
        }
    }
    if bench {
        let start = Instant::now();
        let mut done = 0usize;
        let mut rounds = 0usize;
        for _ in 0..3 {
            for (p, o, seed) in &inputs {
                if let Ok(out) = simulate(&data, p, o, *seed, false) {
                    done += 1;
                    rounds += out.rounds;
                }
            }
        }
        let secs = start.elapsed().as_secs_f64();
        println!(
            "bench: {} battles in {:.3}s on one core = {:.0} battles/s ({:.1} rounds per battle)",
            done,
            secs,
            done as f64 / secs,
            rounds as f64 / done.max(1) as f64
        );
    }
    if total != matched {
        std::process::exit(1);
    }
}
