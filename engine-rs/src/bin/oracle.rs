//! Score the engine against real replay fights.
//!
//! Reads a fixtures file (the format written by
//! runs/replay-oracle-20261122/extract_battles.py), simulates every distinct
//! fight over many seeds and reports how often the engine produces the real
//! outcome and the real surviving board. With the same seed list as
//! run_sapai.py the result must equal the Python reference's.
//!
//! Usage: oracle --fixtures FILE [--data DIR] [--samples N] [--threads K] [--out FILE]

use sap_engine::{simulate, Data, PetSpec};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

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

/// Winner's survivors as a sorted list of (name, attack, health), order-free,
/// exactly as run_sapai.py compares them.
fn real_state(fixture: &Value) -> Vec<(String, i64, i64)> {
    let mut out: Vec<(String, i64, i64)> = fixture["survivors"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|s| {
                    (
                        s["id"].as_str().unwrap_or("").to_string(),
                        s["attack"].as_i64().unwrap_or(0),
                        s["health"].as_i64().unwrap_or(0),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

fn evaluate(data: &Data, fixture: &Value, samples: u32) -> Value {
    let player = specs(&fixture["player"]);
    let opponent = specs(&fixture["opponent"]);
    let real_outcome = match fixture["realOutcome"].as_str().unwrap_or("") {
        "player" => 0u8,
        "opponent" => 1u8,
        _ => 2u8,
    };
    let want = real_state(fixture);
    let mut outcomes = [0u32; 3];
    let mut states: HashMap<(u8, Vec<(String, i64, i64)>), u32> = HashMap::new();
    for index in 0..samples {
        let seed = (1234567u64 + index as u64 * 2654435761u64) % (1u64 << 32);
        let result = match simulate(data, &player, &opponent, seed as u32, false) {
            Ok(r) => r,
            Err(e) => {
                return json!({"battleId": fixture["battleId"], "skipped": "engine error", "error": e.0});
            }
        };
        outcomes[result.outcome as usize] += 1;
        let mut state: Vec<(String, i64, i64)> = Vec::new();
        if result.outcome < 2 {
            for slot in result.final_boards[result.outcome as usize].iter().flatten() {
                state.push((data.pets[slot.kind].name.clone(), slot.attack as i64, slot.health as i64));
            }
        }
        state.sort();
        *states.entry((result.outcome, state)).or_insert(0) += 1;
    }
    let state_hits = *states.get(&(real_outcome, want.clone())).unwrap_or(&0);
    json!({
        "battleId": fixture["battleId"],
        "matchId": fixture["matchId"],
        "turn": fixture["turn"],
        "realOutcome": fixture["realOutcome"],
        "simOutcomes": {"player": outcomes[0], "opponent": outcomes[1], "draw": outcomes[2]},
        "pReal": outcomes[real_outcome as usize] as f64 / samples as f64,
        "pState": state_hits as f64 / samples as f64,
        "distinctStates": states.len(),
        "deterministic": states.len() == 1,
        "outcomePossible": outcomes[real_outcome as usize] > 0,
        "statePossible": state_hits > 0,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };
    let fixtures_path = get("--fixtures", "");
    let samples: u32 = get("--samples", "200").parse().expect("--samples");
    let threads: usize = get("--threads", "3").parse().expect("--threads");
    let out_path = get("--out", "");
    let data = Arc::new(Data::load_dir(std::path::Path::new(&get("--data", "data"))).expect("data tables"));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&fixtures_path).expect("fixtures file"))
        .expect("fixtures json");
    let fixtures: Arc<Vec<Value>> = Arc::new(
        doc["fixtures"]
            .as_array()
            .expect("fixtures array")
            .iter()
            .filter(|f| f.get("mirrorOf").is_none())
            .cloned()
            .collect(),
    );
    let mut handles = Vec::new();
    for worker in 0..threads {
        let data = Arc::clone(&data);
        let fixtures = Arc::clone(&fixtures);
        handles.push(std::thread::spawn(move || {
            let mut rows = Vec::new();
            for (index, fixture) in fixtures.iter().enumerate() {
                if index % threads == worker {
                    rows.push((index, evaluate(&data, fixture, samples)));
                }
            }
            rows
        }));
    }
    let mut rows: Vec<(usize, Value)> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
    rows.sort_by_key(|r| r.0);
    let rows: Vec<Value> = rows.into_iter().map(|r| r.1).collect();
    let done: Vec<&Value> = rows.iter().filter(|r| r.get("skipped").is_none()).collect();
    let count = |f: &dyn Fn(&Value) -> bool| done.iter().filter(|r| f(r)).count();
    let summary = json!({
        "fights": rows.len(),
        "simulated": done.len(),
        "samplesPerFight": samples,
        "outcomeImpossible": count(&|r| !r["outcomePossible"].as_bool().unwrap()),
        "stateImpossible": count(&|r| r["outcomePossible"].as_bool().unwrap() && !r["statePossible"].as_bool().unwrap()),
        "deterministic": count(&|r| r["deterministic"].as_bool().unwrap()),
        "meanProbabilityOfRealOutcome": done.iter().map(|r| r["pReal"].as_f64().unwrap()).sum::<f64>() / done.len().max(1) as f64,
        "meanProbabilityOfRealState": done.iter().map(|r| r["pState"].as_f64().unwrap()).sum::<f64>() / done.len().max(1) as f64,
    });
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
    if !out_path.is_empty() {
        std::fs::write(&out_path, serde_json::to_string(&json!({"summary": summary, "rows": rows})).unwrap())
            .expect("write output");
    }
}
