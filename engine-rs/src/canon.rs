//! The canonical state dump of `tools/gen_env_golden.py` (`canonical`), built
//! from the Rust state so the checker can compare the two field by field.
//!
//! `toggled` holds Python object ids in the reference; here it holds the
//! shop positions of the toggled items that are still in the shop followed
//! by the count of the ones that are not. The checker compares the length
//! only, and the extended block `x` (when the trace has one) compares the
//! positions.

use crate::env::{Game, Obj, Seat, SlotType, NS};
use serde_json::{json, Value};

fn pet_row(game: &Game, seat: &Seat, p: usize) -> Value {
    let pet = &seat.pets[p];
    let d = &game.t.d;
    let copied = pet
        .over
        .as_ref()
        .and_then(|a| a.copied_from)
        .map(|k| Value::String(d.pets[k].name.clone()))
        .unwrap_or(Value::Null);
    json!([
        d.pets[pet.kind].name,
        pet.attack,
        pet.health,
        pet.tmp_attack,
        pet.tmp_health,
        pet.level,
        pet.experience,
        pet.status.name(),
        pet.ability_counter,
        pet.sell_value,
        pet.friends_hurt_counter,
        pet.hurt,
        copied
    ])
}

pub fn seat_state(game: &Game, side: usize, extended: bool) -> Value {
    let seat = &game.seats[side];
    let team: Vec<Value> = (0..NS)
        .map(|i| match seat.team[i] {
            Some(p) => pet_row(game, seat, p),
            None => Value::Null,
        })
        .collect();
    let mut shop: Vec<Value> = Vec::new();
    let mut shop_x: Vec<Value> = Vec::new();
    let mut toggled_shop: Vec<Value> = Vec::new();
    for (index, slot) in seat.slots.iter().enumerate() {
        match &slot.obj {
            Obj::Empty => {
                shop.push(json!([slot.ty.name(), Value::Null, slot.frozen, slot.cost]));
                shop_x.push(json!([slot.reward_group]));
            }
            Obj::Food(f) => {
                shop.push(json!([slot.ty.name(), game.t.e.foods[f.kind].name, slot.frozen, slot.cost]));
                shop_x.push(json!([slot.reward_group, f.attack, f.health, f.cost]));
                if seat.toggled.contains(&f.uid) {
                    toggled_shop.push(json!(index));
                }
            }
            Obj::Pet(p) => {
                let pet = &seat.pets[*p];
                shop.push(json!([
                    slot.ty.name(),
                    game.t.d.pets[pet.kind].name,
                    slot.frozen,
                    slot.cost,
                    pet.attack,
                    pet.health,
                    pet.level
                ]));
                shop_x.push(json!([slot.reward_group, pet.status.name(), pet.sell_value]));
                if seat.toggled.contains(&pet.uid) {
                    toggled_shop.push(json!(index));
                }
            }
        }
        debug_assert!(matches!(slot.ty, SlotType::Pet | SlotType::Food | SlotType::Levelup));
    }
    let mut out = json!({
        "gold": seat.gold,
        "turn": seat.turn,
        "lives": seat.lives,
        "wins": seat.wins,
        "ended": seat.turn_ended,
        "last": seat.last_battle_outcome,
        "team": team,
        "shop": shop,
        "shopAttack": seat.shop_attack,
        "shopHealth": seat.shop_health,
        "actions": seat.action_count,
        "reorders": seat.reorder_count,
        "extraReorder": false,
        "toggled": seat.toggled,
        "seenOrders": seat.seen.len(),
    });
    if extended {
        let (pos, first, last) = seat.rs.probe();
        out["x"] = json!({
            "toggledShop": toggled_shop,
            "shop": shop_x,
            "rs": [pos, first, last],
            "cursors": [seat.pet_cursor, seat.food_cursor],
            "lf": seat.lf_winner,
        });
    }
    out
}

/// `canonical(env.core)`.
pub fn canonical(game: &Game, extended: bool) -> Value {
    json!({
        "turn": game.turn,
        "seats": [seat_state(game, 0, extended), seat_state(game, 1, extended)],
    })
}

/// First difference between the reference dump and ours, as a path and both values.
pub fn first_difference(reference: &Value, ours: &Value, path: &str) -> Option<String> {
    match (reference, ours) {
        (Value::Object(a), Value::Object(b)) => {
            for (key, va) in a {
                let here = format!("{path}.{key}");
                match b.get(key) {
                    None => return Some(format!("{here}: missing on the Rust side (reference {va})")),
                    Some(vb) => {
                        if key == "toggled" {
                            let la = va.as_array().map(|x| x.len());
                            let lb = vb.as_array().map(|x| x.len());
                            if la != lb {
                                return Some(format!("{here}: {la:?} toggled items in the reference, {lb:?} here"));
                            }
                            continue;
                        }
                        if let Some(d) = first_difference(va, vb, &here) {
                            return Some(d);
                        }
                    }
                }
            }
            None
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("{path}: length {} in the reference, {} here ({reference} vs {ours})", a.len(), b.len()));
            }
            for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
                if let Some(d) = first_difference(va, vb, &format!("{path}[{i}]")) {
                    return Some(d);
                }
            }
            None
        }
        _ => {
            let same = match (reference.as_f64(), ours.as_f64()) {
                (Some(x), Some(y)) => x == y,
                _ => reference == ours,
            };
            if same {
                None
            } else {
                Some(format!("{path}: reference {reference}, here {ours}"))
            }
        }
    }
}
