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
