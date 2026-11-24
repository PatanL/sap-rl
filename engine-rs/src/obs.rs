//! `rl/entity_observation.py`: one seat's information set as fixed-size
//! arrays. Values are computed in f64 in the reference's operation order and
//! cast to f32 on assignment, so the bytes match NumPy's.
//!
//! Layout of one observation, in the reference's key order:
//!
//! | key | dtype | shape |
//! |---|---|---|
//! | item_ids, kind_ids, status_ids, positions | i64 | [21] |
//! | numeric | f32 | [21, 16] |
//! | entity_mask | i8 | [21] |
//! | type_mask | i8 | [9] |
//! | source_mask | i8 | [9, 11] |
//! | target_mask | i8 | [9, 11, 6] |
//! | opponent_history_item_ids, _status_ids, _positions | i64 | [2, 5] |
//! | opponent_history_numeric | f32 | [2, 5, 8] |
//! | opponent_history_mask | i8 | [2, 5] |
//! | opponent_history_board_mask | i8 | [2] |

use crate::env::{decode_action, Board, Game, Obj, Seat, SlotType, MAX_REORDERS_PER_TURN, NS, NUM_ACTIONS};
use crate::envdata::MAX_SHOP_SLOTS;

pub const NUM_ENTITIES: usize = 1 + NS + MAX_SHOP_SLOTS + NS;
pub const NUM_NUMERIC: usize = 16;
pub const NUM_TYPES: usize = 9;
pub const NUM_SOURCES: usize = 11;
pub const NUM_TARGETS: usize = 6;
pub const HISTORY_BOARDS: usize = 2;
pub const HISTORY_NUMERIC: usize = 8;

/// (name, element size in bytes, element count, NumPy dtype, shape).
pub const OBS_KEYS: [(&str, usize, usize, &str, &[usize]); 15] = [
    ("item_ids", 8, NUM_ENTITIES, "int64", &[NUM_ENTITIES]),
    ("kind_ids", 8, NUM_ENTITIES, "int64", &[NUM_ENTITIES]),
    ("status_ids", 8, NUM_ENTITIES, "int64", &[NUM_ENTITIES]),
    ("positions", 8, NUM_ENTITIES, "int64", &[NUM_ENTITIES]),
    ("numeric", 4, NUM_ENTITIES * NUM_NUMERIC, "float32", &[NUM_ENTITIES, NUM_NUMERIC]),
    ("entity_mask", 1, NUM_ENTITIES, "int8", &[NUM_ENTITIES]),
    ("type_mask", 1, NUM_TYPES, "int8", &[NUM_TYPES]),
    ("source_mask", 1, NUM_TYPES * NUM_SOURCES, "int8", &[NUM_TYPES, NUM_SOURCES]),
    ("target_mask", 1, NUM_TYPES * NUM_SOURCES * NUM_TARGETS, "int8", &[NUM_TYPES, NUM_SOURCES, NUM_TARGETS]),
    ("opponent_history_item_ids", 8, HISTORY_BOARDS * NS, "int64", &[HISTORY_BOARDS, NS]),
    ("opponent_history_status_ids", 8, HISTORY_BOARDS * NS, "int64", &[HISTORY_BOARDS, NS]),
    ("opponent_history_positions", 8, HISTORY_BOARDS * NS, "int64", &[HISTORY_BOARDS, NS]),
    (
        "opponent_history_numeric",
        4,
        HISTORY_BOARDS * NS * HISTORY_NUMERIC,
        "float32",
        &[HISTORY_BOARDS, NS, HISTORY_NUMERIC],
    ),
    ("opponent_history_mask", 1, HISTORY_BOARDS * NS, "int8", &[HISTORY_BOARDS, NS]),
    ("opponent_history_board_mask", 1, HISTORY_BOARDS, "int8", &[HISTORY_BOARDS]),
];

/// Bytes of one observation with the arrays concatenated in key order.
pub const OBS_BYTES: usize = 4 * 8 * NUM_ENTITIES
    + 4 * NUM_ENTITIES * NUM_NUMERIC
    + NUM_ENTITIES
    + NUM_TYPES
    + NUM_TYPES * NUM_SOURCES
    + NUM_TYPES * NUM_SOURCES * NUM_TARGETS
    + 3 * 8 * HISTORY_BOARDS * NS
    + 4 * HISTORY_BOARDS * NS * HISTORY_NUMERIC
    + HISTORY_BOARDS * NS
    + HISTORY_BOARDS;

/// One observation, one field per reference array. (The struct itself has
/// alignment padding before the history arrays; `to_bytes` and the batch
/// writer copy field by field.)
#[repr(C)]
#[derive(Clone)]
pub struct Obs {
    pub item_ids: [i64; NUM_ENTITIES],
    pub kind_ids: [i64; NUM_ENTITIES],
    pub status_ids: [i64; NUM_ENTITIES],
    pub positions: [i64; NUM_ENTITIES],
    pub numeric: [[f32; NUM_NUMERIC]; NUM_ENTITIES],
    pub entity_mask: [i8; NUM_ENTITIES],
    pub type_mask: [i8; NUM_TYPES],
    pub source_mask: [[i8; NUM_SOURCES]; NUM_TYPES],
    pub target_mask: [[[i8; NUM_TARGETS]; NUM_SOURCES]; NUM_TYPES],
    pub history_item_ids: [[i64; NS]; HISTORY_BOARDS],
    pub history_status_ids: [[i64; NS]; HISTORY_BOARDS],
    pub history_positions: [[i64; NS]; HISTORY_BOARDS],
    pub history_numeric: [[[f32; HISTORY_NUMERIC]; NS]; HISTORY_BOARDS],
    pub history_mask: [[i8; NS]; HISTORY_BOARDS],
    pub history_board_mask: [i8; HISTORY_BOARDS],
}

impl Obs {
    pub fn zeroed() -> Obs {
        // All-zero bytes are valid for every field.
        unsafe { std::mem::zeroed() }
    }

    /// The reference's byte string: every array, in key order, native byte order.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(OBS_BYTES);
        fn put<T: Copy>(out: &mut Vec<u8>, v: &[T]) {
            let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
            out.extend_from_slice(bytes);
        }
        put(&mut out, &self.item_ids);
        put(&mut out, &self.kind_ids);
        put(&mut out, &self.status_ids);
        put(&mut out, &self.positions);
        put(&mut out, &self.numeric);
        put(&mut out, &self.entity_mask);
        put(&mut out, &self.type_mask);
        put(&mut out, &self.source_mask);
        put(&mut out, &self.target_mask);
        put(&mut out, &self.history_item_ids);
        put(&mut out, &self.history_status_ids);
        put(&mut out, &self.history_positions);
        put(&mut out, &self.history_numeric);
        put(&mut out, &self.history_mask);
        put(&mut out, &self.history_board_mask);
        out
    }
}

/// `FlatActionCodec.decode` for every flat id, built once.
fn action_triples() -> &'static [[u8; 3]; NUM_ACTIONS] {
    static TABLE: std::sync::OnceLock<[[u8; 3]; NUM_ACTIONS]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u8; 3]; NUM_ACTIONS];
        for (flat, row) in table.iter_mut().enumerate() {
            let [ty, source, target] = decode_action(flat);
            *row = [ty as u8, source as u8, target as u8];
        }
        table
    })
}

#[inline]
fn f(x: f64) -> f32 {
    x as f32
}

#[inline]
fn b(x: bool) -> f32 {
    if x {
        1.0
    } else {
        0.0
    }
}

/// `EntityObservationEncoder._encode_pet` for a live pet of the seat (team or shop).
fn encode_own_pet(seat: &Seat, game: &Game, p: usize, token: usize, friendly: bool, shop: bool, o: &mut Obs) {
    let pet = &seat.pets[p];
    let fd = &game.t.d.pets[pet.kind];
    o.item_ids[token] = game.t.e.pet_item_id[pet.kind];
    o.status_ids[token] = game.t.e.status_id[pet.status as usize];
    let n = &mut o.numeric[token];
    n[0] = f(seat.atk(p) as f64 / 50.0);
    n[1] = f(seat.hp(p) as f64 / 50.0);
    n[2] = f(pet.level as f64 / 3.0);
    let total = match pet.level {
        1 => pet.experience,
        2 => 2 + pet.experience,
        _ => 5 + pet.experience,
    };
    n[3] = f(total as f64 / 5.0);
    n[4] = match fd.tier {
        Some(t) => f(t as f64 / 6.0),
        None => 0.0,
    };
    n[7] = 1.0;
    n[8] = b(friendly);
    n[9] = b(shop);
    n[10] = 0.0;
    if friendly {
        n[11] = f(pet.ability_counter.min(5) as f64 / 5.0);
        n[12] = f(pet.friends_hurt_counter.min(4) as f64 / 4.0);
    }
    n[15] = f(pet.sell_value.min(10) as f64 / 10.0);
}

/// `EntityObservationEncoder.encode(core, side)`. Computes the seat's legal
/// actions first (which registers the current team order, as the reference's
/// mask construction does).
pub fn encode(game: &mut Game, side: usize, o: &mut Obs) {
    *o = Obs::zeroed();
    // _action_masks
    let legal = game.seats[side].avail_actions();
    let triples = action_triples();
    for (w, word) in legal.0.iter().enumerate() {
        let mut bits = *word;
        while bits != 0 {
            let flat = w * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let [ty, source, target] = triples[flat];
            o.type_mask[ty as usize] = 1;
            o.source_mask[ty as usize][source as usize] = 1;
            o.target_mask[ty as usize][source as usize][target as usize] = 1;
        }
    }
    let game: &Game = game;
    let seat = &game.seats[side];
    let opponent = &game.seats[1 - side];
    o.entity_mask = [1; NUM_ENTITIES];
    o.kind_ids[0] = 1; // GLOBAL
    let limit = MAX_REORDERS_PER_TURN;
    {
        let n = &mut o.numeric[0];
        n[0] = f(seat.reorder_count.min(limit) as f64 / limit as f64);
        n[1] = seat.last_battle_outcome.unwrap_or(0) as f32;
        n[2] = b(seat.last_battle_outcome.is_some());
        n[3] = f(seat.shop_attack.min(50) as f64 / 50.0);
        n[4] = f(seat.shop_health.min(50) as f64 / 50.0);
        n[5] = f(seat.action_count.min(50) as f64 / 50.0);
        n[6] = f(seat.toggled.len().min(MAX_SHOP_SLOTS) as f64 / MAX_SHOP_SLOTS as f64);
        n[7] = f(seat.seen.len().min(limit as usize + 1) as f64 / (limit + 1) as f64);
        n[8] = 1.0;
        n[9] = f(opponent.lives as f64 / 6.0);
        n[10] = f(opponent.wins as f64 / 10.0);
        n[11] = f(seat.gold.min(30) as f64 / 30.0);
        n[12] = f(seat.turn.min(25) as f64 / 25.0);
        n[13] = f(seat.lives as f64 / 6.0);
        n[14] = f(seat.wins as f64 / 10.0);
        n[15] = b(seat.turn_ended);
    }

    for index in 0..NS {
        let token = 1 + index;
        o.positions[token] = index as i64;
        if let Some(p) = seat.team[index] {
            o.kind_ids[token] = 2; // TEAM_PET
            encode_own_pet(seat, game, p, token, true, false, o);
        }
    }

    // Discover pairs: rank of the reward group, place within it, sibling position.
    let mut groups: Vec<u32> = seat.slots.iter().filter_map(|s| s.reward_group).collect();
    groups.sort_unstable();
    groups.dedup();
    for index in 0..MAX_SHOP_SLOTS {
        let token = 1 + NS + index;
        o.positions[token] = index as i64;
        if index >= seat.slots.len() {
            continue;
        }
        let slot = &seat.slots[index];
        match (&slot.obj, slot.ty) {
            (Obj::Pet(p), SlotType::Pet | SlotType::Levelup) => {
                o.kind_ids[token] = 3; // SHOP_PET
                encode_own_pet(seat, game, *p, token, false, true, o);
                let toggled = seat.toggled.contains(&seat.pets[*p].uid);
                let n = &mut o.numeric[token];
                n[5] = f(slot.cost as f64 / 3.0);
                n[6] = b(slot.frozen);
                n[8] = b(toggled);
                if slot.ty == SlotType::Levelup {
                    let group = slot.reward_group;
                    let positions: Vec<usize> = match group {
                        Some(_) => (0..seat.slots.len()).filter(|&i| seat.slots[i].reward_group == group).collect(),
                        None => vec![index],
                    };
                    let rank = match group {
                        Some(g) => groups.iter().position(|x| *x == g).map(|r| r + 1).unwrap_or(0),
                        None => 0,
                    };
                    n[11] = 1.0;
                    n[12] = f(rank as f64 / MAX_SHOP_SLOTS as f64);
                    n[13] = positions.iter().position(|&i| i == index).unwrap_or(0) as f32;
                    n[14] = match positions.iter().find(|&&i| i != index) {
                        Some(&sibling) => f(sibling as f64 / MAX_SHOP_SLOTS as f64),
                        None => 0.0,
                    };
                }
            }
            (Obj::Food(food), SlotType::Food) => {
                let fd = &game.t.e.foods[food.kind];
                o.kind_ids[token] = 4; // SHOP_FOOD
                o.item_ids[token] = fd.item_id;
                let n = &mut o.numeric[token];
                n[0] = f(food.attack as f64 / 50.0);
                n[1] = f(food.health as f64 / 50.0);
                n[4] = f(fd.obs_tier as f64 / 6.0);
                n[5] = f(slot.cost as f64 / 3.0);
                n[6] = b(slot.frozen);
                n[7] = 1.0;
                n[8] = b(seat.toggled.contains(&food.uid));
                n[9] = 1.0;
            }
            _ => {}
        }
    }

    // The other seat's last submitted board.
    let previous: &Option<Board> = &game.previous[1 - side];
    let older: &Option<Board> = &game.previous_previous[1 - side];
    for index in 0..NS {
        let token = 1 + NS + MAX_SHOP_SLOTS + index;
        o.positions[token] = index as i64;
        let pet = match previous.as_ref().and_then(|board| board[index]) {
            Some(p) => p,
            None => continue,
        };
        o.kind_ids[token] = 5; // OPPONENT_PET
        o.item_ids[token] = game.t.e.pet_item_id[pet.kind];
        o.status_ids[token] = game.t.e.status_id[pet.status as usize];
        let n = &mut o.numeric[token];
        n[0] = f(pet.attack as f64 / 50.0);
        n[1] = f(pet.health as f64 / 50.0);
        n[2] = f(pet.level as f64 / 3.0);
        n[3] = f(pet.total_experience as f64 / 5.0);
        n[4] = match game.t.d.pets[pet.kind].tier {
            Some(t) => f(t as f64 / 6.0),
            None => 0.0,
        };
        n[7] = 1.0;
        n[10] = 1.0;
        // Only the public base level is exposed for opponent pets.
        n[15] = f(pet.level.min(10) as f64 / 10.0);
    }

    // _encode_opponent_history
    for (board_index, board) in [older, previous].into_iter().enumerate() {
        for position in 0..NS {
            o.history_positions[board_index][position] = position as i64;
        }
        let board = match board {
            Some(b) => b,
            None => continue,
        };
        o.history_board_mask[board_index] = 1;
        let recency: f32 = if board_index == 0 { 0.5 } else { 1.0 };
        for position in 0..NS {
            o.history_numeric[board_index][position][7] = recency;
            let pet = match board[position] {
                Some(p) => p,
                None => continue,
            };
            o.history_mask[board_index][position] = 1;
            o.history_item_ids[board_index][position] = game.t.e.pet_item_id[pet.kind];
            o.history_status_ids[board_index][position] = game.t.e.status_id[pet.status as usize];
            let n = &mut o.history_numeric[board_index][position];
            n[0] = f(pet.attack as f64 / 50.0);
            n[1] = f(pet.health as f64 / 50.0);
            n[2] = f(pet.level as f64 / 3.0);
            n[3] = f(pet.total_experience as f64 / 5.0);
            n[4] = match game.t.d.pets[pet.kind].tier {
                Some(t) => f(t as f64 / 6.0),
                None => 0.0,
            };
            n[5] = b(pet.tmp_attack);
            n[6] = b(pet.tmp_health);
        }
    }
}
