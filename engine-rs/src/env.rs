//! The training environment around the battle engine: shop rules, the
//! player's actions and their triggers, the shop-phase hurt/faint chain,
//! action legality, the end-of-turn freeze plan and the two-seat match flow.
//!
//! A transliteration of the path the collector uses in the Python reference
//! (`ParallelSelfPlayEnv().reset(seed, paired_tape=True)` and
//! `joint_action.step_decisions`): `sapai/shop.py`, `player.py`, `pets.py`,
//! `effects.py`, the shop use of `battle.battle_phase_hurt_and_faint`,
//! `sapai_gym1/SuperAutoPetsEnv.py`, `rl/selfplay_env.py`, `rl/end_freeze.py`
//! and `rl/joint_action.py`. Reference function names are kept in the doc
//! comments. Every random draw is made at the same place, from the same
//! stream and in the same shape as the reference.
//!
//! Streams (see PROGRESS.md for the full call-site list):
//! - shop offers and level-up choices: `tape::uniform`, domains
//!   `player/{seat}/shop/pet` and `player/{seat}/shop/food`;
//! - everything else in the shop phase: the seat's `Mt` (`shop.rs`, seeded
//!   from the tape domain `player/{seat}/ability-seed`), which
//!   `Player._connect_objects` hands to every pet and shop item;
//! - a pet created after the last `_connect_objects` draws from a private
//!   copy until the next one (`Rs::Private`, `Rs::TeamSeed`);
//! - battle seeds: tape domain `battle/turn/{turn}`.

use crate::data::*;
use crate::engine::{battle_env, EngineError, EnvPet};
use crate::envdata::{FoodData, Tables, TurnRule, MAX_SHOP_SLOTS};
use crate::rng::Mt;
use crate::tape;
use std::collections::HashMap;
use std::sync::Arc;

pub type R<T> = Result<T, EngineError>;

fn err<T>(m: impl Into<String>) -> R<T> {
    Err(EngineError(m.into()))
}

pub const NS: usize = 5;
const UNRESOLVED: i32 = i32::MIN / 4;
pub const NUM_ACTIONS: usize = 197;
pub const SHOP_ACTION_CAP: i32 = 50;
pub const MAX_REORDERS_PER_TURN: i32 = 10;
pub const MAX_TURN: i32 = 25;
pub const MAX_LIVES: i32 = 6;

// FlatActionCodec
pub const A_END_TURN: usize = 0;
pub const A_BUY_PET: usize = 1;
pub const A_BUY_FOOD: usize = 61;
pub const A_COMBINE: usize = 121;
pub const A_SELL: usize = 146;
pub const A_ROLL: usize = 151;
pub const A_FREEZE: usize = 152;
pub const A_UNFREEZE: usize = 162;
pub const A_MOVE: usize = 172;

// ActionType
pub const T_END_TURN: i64 = 0;
pub const T_ROLL: i64 = 1;
pub const T_BUY_PET: i64 = 2;
pub const T_BUY_FOOD: i64 = 3;
pub const T_SELL: i64 = 4;
pub const T_COMBINE: i64 = 5;
pub const T_FREEZE: i64 = 6;
pub const T_UNFREEZE: i64 = 7;
pub const T_MOVE: i64 = 8;
pub const NO_SOURCE: i64 = 10;
pub const NO_TARGET: i64 = 5;

/// A set of flat action ids.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ActionSet(pub [u64; 4]);

impl ActionSet {
    #[inline]
    pub fn add(&mut self, a: usize) {
        self.0[a >> 6] |= 1u64 << (a & 63);
    }
    #[inline]
    pub fn has(&self, a: usize) -> bool {
        a < NUM_ACTIONS && self.0[a >> 6] & (1u64 << (a & 63)) != 0
    }
    pub fn to_vec(&self) -> Vec<usize> {
        (0..NUM_ACTIONS).filter(|&a| self.has(a)).collect()
    }
}

/// `FlatActionCodec.encode` after the `JointAction.__post_init__` round-trip
/// check: only canonical triples are accepted.
pub fn encode_action(action: [i64; 3]) -> R<usize> {
    let [ty, source, target] = action;
    let shop = |v: i64| (0..MAX_SHOP_SLOTS as i64).contains(&v);
    let team = |v: i64| (0..NS as i64).contains(&v);
    let flat = match ty {
        T_END_TURN if source == NO_SOURCE && target == NO_TARGET => A_END_TURN,
        T_ROLL if source == NO_SOURCE && target == NO_TARGET => A_ROLL,
        T_BUY_PET if shop(source) && team(target) => A_BUY_PET + (source * 6 + target) as usize,
        T_BUY_FOOD if shop(source) && (0..=NO_TARGET).contains(&target) => {
            A_BUY_FOOD + (source * 6 + target) as usize
        }
        T_COMBINE if team(source) && team(target) => A_COMBINE + (source * 5 + target) as usize,
        T_SELL if team(source) && target == NO_TARGET => A_SELL + source as usize,
        T_FREEZE if shop(source) && target == NO_TARGET => A_FREEZE + source as usize,
        T_UNFREEZE if shop(source) && target == NO_TARGET => A_UNFREEZE + source as usize,
        T_MOVE if team(source) && team(target) => A_MOVE + (source * 5 + target) as usize,
        _ => return err(format!("Noncanonical primitive action ({ty}, {source}, {target})")),
    };
    Ok(flat)
}

/// `FlatActionCodec.decode(flat).as_tuple()`.
pub fn decode_action(flat: usize) -> [i64; 3] {
    let f = flat as i64;
    if flat == A_END_TURN {
        [T_END_TURN, NO_SOURCE, NO_TARGET]
    } else if flat == A_ROLL {
        [T_ROLL, NO_SOURCE, NO_TARGET]
    } else if flat < A_BUY_FOOD {
        let o = f - A_BUY_PET as i64;
        [T_BUY_PET, o / 6, o % 6]
    } else if flat < A_COMBINE {
        let o = f - A_BUY_FOOD as i64;
        [T_BUY_FOOD, o / 6, o % 6]
    } else if flat < A_SELL {
        let o = f - A_COMBINE as i64;
        [T_COMBINE, o / 5, o % 5]
    } else if flat < A_ROLL {
        [T_SELL, f - A_SELL as i64, NO_TARGET]
    } else if flat < A_UNFREEZE {
        [T_FREEZE, f - A_FREEZE as i64, NO_TARGET]
    } else if flat < A_MOVE {
        [T_UNFREEZE, f - A_UNFREEZE as i64, NO_TARGET]
    } else {
        let o = f - A_MOVE as i64;
        [T_MOVE, o / 5, o % 5]
    }
}

/// Which generator an object's `rs` attribute refers to.
#[derive(Clone)]
pub enum Rs {
    /// The seat's shop stream (after `Player._connect_objects`).
    Shared,
    /// `LazyRandomState(shop.seed_state)`: a level-up reward pet until the
    /// next `_connect_objects`; a copy of the shop stream taken at creation.
    Private(Box<Mt>),
    /// `LazyRandomState(team.seed_state)`: a pet summoned on the shop team,
    /// until the `_connect_objects` that ends the faint chain. Not built
    /// until something draws from it.
    TeamSeed,
    /// `MockRandomState`: a draw would come from NumPy's global generator.
    Mock,
}

#[derive(Clone)]
pub struct Pet {
    /// Stands in for `id(pet)`: never reused within a match.
    pub uid: u32,
    pub kind: usize,
    /// `_attack` / `_health`.
    pub attack: i32,
    pub health: i32,
    /// `_until_end_of_battle_*_buff`.
    pub tmp_attack: i32,
    pub tmp_health: i32,
    pub hurt: i32,
    pub status: Status,
    pub level: i32,
    pub experience: i32,
    pub ability_counter: i64,
    pub friends_hurt_counter: i32,
    pub sell_value: i32,
    /// `override_ability_dict` when `override_ability` is set.
    pub over: Option<Arc<Ability>>,
    pub rs: Rs,
    /// 0: the seat's team; 1: the stand-in opponent team of a shop faint chain.
    pub team: usize,
}

#[derive(Clone)]
pub struct Food {
    pub uid: u32,
    pub kind: usize,
    pub attack: i32,
    pub health: i32,
    pub base_attack: i32,
    pub base_health: i32,
    pub cost: i32,
    pub eaten: bool,
    pub rs: Rs,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotType {
    Pet,
    Food,
    Levelup,
}

impl SlotType {
    pub fn name(self) -> &'static str {
        match self {
            SlotType::Pet => "pet",
            SlotType::Food => "food",
            SlotType::Levelup => "levelup",
        }
    }
}

#[derive(Clone)]
pub enum Obj {
    /// A `ShopSlot("pet")` / `ShopSlot("food")` placeholder before its roll.
    Empty,
    /// Index into `Seat::pets`.
    Pet(usize),
    Food(Food),
}

/// What `set_shop_item` places: a pet kind (index into `Data::pets`) or a
/// food (index into `EnvData::foods`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShopItem {
    Pet(usize),
    Food(usize),
}

#[derive(Clone)]
pub struct ShopSlot {
    pub ty: SlotType,
    pub frozen: bool,
    pub cost: i32,
    pub reward_group: Option<u32>,
    pub obj: Obj,
}

/// `apet_idx[1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Idx {
    Int(usize),
    /// The `[team_idx, pet_idx]` list a removed pet passes through `faint_trigger`.
    List(usize, usize),
    /// `[0, None]`: a food looking for its targets.
    Absent,
}

/// The triggering entity argument.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Te {
    None,
    Pet(usize),
    Team,
}

struct EffResult {
    targets: Vec<usize>,
    /// Python's `possible == [[]]`.
    possible_empty_nested: bool,
}

impl EffResult {
    fn of(targets: Vec<usize>) -> Self {
        EffResult { targets, possible_empty_nested: false }
    }
    fn none() -> Self {
        EffResult { targets: Vec::new(), possible_empty_nested: false }
    }
}

struct Record {
    pet: usize,
    team_idx: usize,
    pet_idx: usize,
    native_post: bool,
    has_status: bool,
    listeners: Vec<usize>,
    post_observers: Vec<usize>,
}

type Layout = [Option<usize>; NS];

/// A pet's current ability: a table entry or a Parrot's copy.
enum AbRef {
    Static(&'static Ability),
    Owned(Arc<Ability>),
}

impl std::ops::Deref for AbRef {
    type Target = Ability;
    #[inline]
    fn deref(&self) -> &Ability {
        match self {
            AbRef::Static(a) => a,
            AbRef::Owned(a) => a,
        }
    }
}

/// How often each rule was exercised (acceptance item 3).
#[derive(Default, Clone)]
pub struct Coverage {
    pub counts: HashMap<String, u64>,
}

impl Coverage {
    pub fn hit(&mut self, key: String) {
        *self.counts.entry(key).or_insert(0) += 1;
    }
    pub fn merge(&mut self, other: &Coverage) {
        for (k, v) in &other.counts {
            *self.counts.entry(k.clone()).or_insert(0) += v;
        }
    }
}

macro_rules! cov {
    ($self:expr, $($arg:tt)*) => {
        if let Some(c) = $self.cov.as_mut() {
            c.hit(format!($($arg)*));
        }
    };
}

/// One submitted board as the other seat sees it later
/// (`Team.from_state(team.state)`), reduced to what the encoder reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoardPet {
    pub kind: usize,
    /// `pet.attack` / `pet.health` (temporary buffs included, capped at 50).
    pub attack: i32,
    pub health: i32,
    pub level: i32,
    pub total_experience: i32,
    pub status: Status,
    pub tmp_attack: bool,
    pub tmp_health: bool,
}

pub type Board = [Option<BoardPet>; NS];

/// Fields of one pet row in the board layout shared by `set_team`, the battle
/// event export and `eval_vs`: kind id, attack, health, total experience,
/// perk id, consumed triggers, copied species. Ids and the Parrot column are
/// those of the battle entry point (`ffi::sap_battle`, `python/sap_engine_rs.py`):
/// kind = index into the pet table (`sap_kind`), perk = index into
/// `BOARD_STATUSES` (`sap_status`), copied species -2 (not stated: a Parrot
/// copies the friend ahead, the default rule), -1 (a Parrot that copied
/// nothing) or the kind it copied. Row `i` is team slot `i`, front first; an
/// empty slot has kind -1 (the battle entry point passes a count instead).
pub const BOARD_FIELDS: usize = 7;
pub type BoardRows = [[i32; BOARD_FIELDS]; NS];
pub const EMPTY_ROW: [i32; BOARD_FIELDS] = [-1, 0, 0, 0, 0, 0, -2];

/// Perk ids of the board layout, in `Status` order.
pub const BOARD_STATUSES: [Status; 13] = [
    Status::None,
    Status::Bread,
    Status::Cake,
    Status::BoneAttack,
    Status::CoconutShield,
    Status::ExtraLife,
    Status::GarlicArmor,
    Status::HoneyBee,
    Status::MelonArmor,
    Status::PoisonAttack,
    Status::SplashAttack,
    Status::SteakAttack,
    Status::Weak,
];

/// One resolved battle: what `selfplay_env` handed to `Battle(...)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BattleEvent {
    /// `core.turn` when the battle was fought (one-based).
    pub turn: i32,
    /// `_next_battle_seed()`.
    pub seed: u32,
    /// Both teams after both seats' end-of-turn effects, before combat;
    /// stats with the temporary shop buffs folded in, as `Battle.__init__` does.
    pub teams: [BoardRows; 2],
    /// What the seven columns cannot carry and the battle still reads:
    /// per slot `[friends_hurt_counter, _hurt]` (Wolverine's shop count).
    pub carried: [[[i32; 2]; NS]; 2],
}

/// One seat: `Player`, its `Team` and `Shop`, and the per-seat bookkeeping
/// of `SuperAutoPetsEnv`.
#[derive(Clone)]
pub struct Seat {
    t: &'static Tables,
    pub seat: usize,
    pub world_seed: u64,
    // Player
    pub gold: i32,
    pub turn: i32,
    pub lives: i32,
    pub wins: i32,
    pub last_battle_outcome: Option<i32>,
    pub lf_winner: Option<bool>,
    pub turn_ended: bool,
    // Team
    pub pets: Vec<Pet>,
    pub team: [Option<usize>; NS],
    /// `opponent = Team()` of `Player._resolve_shop_hurt_and_faint`.
    oteam: [Option<usize>; NS],
    /// Seed of `team.seed_state`.
    team_seed: u32,
    /// The team was replaced by `set_team` (`Team(pets)` has `seed_state=None`):
    /// a pet summoned onto it holds a `MockRandomState`.
    team_seed_none: bool,
    // Shop
    pub slots: Vec<ShopSlot>,
    pub shop_turn: i32,
    pub shop_attack: i32,
    pub shop_health: i32,
    pslots: usize,
    fslots: usize,
    /// Turn whose rules `update_shop_rules` installed (1..=11).
    rule_turn: usize,
    next_reward_group: u32,
    /// `shop.rs`.
    pub rs: Mt,
    pub pet_cursor: u64,
    pub food_cursor: u64,
    /// Tape domains `player/{seat}/shop/pet` and `player/{seat}/shop/food`.
    pet_tape: tape::Stream,
    food_tape: tape::Stream,
    // SuperAutoPetsEnv
    pub action_count: i32,
    pub reorder_count: i32,
    /// `_freeze_toggled_*`.
    pub toggled: Vec<u32>,
    /// `_seen_team_orders_*`: each order packed by `pack_order`.
    pub seen: Vec<u128>,
    /// The seat's legal actions, valid until the next change of the seat
    /// (the reference caches them behind a state fingerprint; recomputing
    /// is idempotent, so the cache changes nothing but the cost).
    legal: Option<ActionSet>,
    next_uid: u32,
    pub cov: Option<Box<Coverage>>,
}

impl Seat {
    fn new(t: &'static Tables, seat: usize, world_seed: u64, team_seed: u32, coverage: bool) -> R<Seat> {
        // Shop.enable_paired_tape
        let ability_seed =
            (tape::uniform(world_seed, &format!("player/{seat}/ability-seed"), 0) * 4294967296.0) as u64;
        let mut s = Seat {
            t,
            seat,
            world_seed,
            gold: 10,
            turn: 1,
            lives: 6,
            wins: 0,
            last_battle_outcome: None,
            lf_winner: None,
            turn_ended: false,
            pets: Vec::with_capacity(64),
            team: [None; NS],
            oteam: [None; NS],
            team_seed,
            team_seed_none: false,
            slots: Vec::with_capacity(MAX_SHOP_SLOTS + 2),
            shop_turn: 1,
            shop_attack: 0,
            shop_health: 0,
            pslots: 0,
            fslots: 0,
            rule_turn: 1,
            next_reward_group: 1,
            rs: Mt::new(ability_seed as u32),
            pet_cursor: 0,
            food_cursor: 0,
            pet_tape: tape::Stream::new(world_seed, &format!("player/{seat}/shop/pet")),
            food_tape: tape::Stream::new(world_seed, &format!("player/{seat}/shop/food")),
            action_count: 0,
            reorder_count: 0,
            toggled: Vec::new(),
            seen: Vec::new(),
            legal: None,
            next_uid: 1,
            cov: if coverage { Some(Box::default()) } else { None },
        };
        // shop.update_shop_rules(roll_only_empty=True); player._connect_objects()
        s.update_shop_rules(1, true)?;
        s.connect_objects();
        Ok(s)
    }

    // ------------------------------------------------------------------
    // Pet helpers (sapai/pets.py)
    // ------------------------------------------------------------------

    /// A fresh identity (stands in for `id(obj)`).
    fn take_uid(&mut self) -> u32 {
        let uid = self.next_uid;
        assert!(uid < 1 << 25, "identity counter overflow");
        self.next_uid += 1;
        uid
    }

    fn new_pet(&mut self, kind: usize, team: usize, rs: Rs) -> usize {
        let fd = &self.t.d.pets[kind];
        let uid = self.take_uid();
        self.pets.push(Pet {
            uid,
            kind,
            attack: fd.base_attack.map(|v| v as i32).unwrap_or(UNRESOLVED),
            health: fd.base_health.map(|v| v as i32).unwrap_or(UNRESOLVED),
            tmp_attack: 0,
            tmp_health: 0,
            hurt: 0,
            status: fd.status.unwrap_or(Status::None),
            level: 1,
            experience: 0,
            ability_counter: 0,
            friends_hurt_counter: 0,
            sell_value: 1,
            over: None,
            rs,
            team,
        });
        self.pets.len() - 1
    }

    fn new_food(&mut self, kind: usize, rs: Rs) -> Food {
        let fd = &self.t.e.foods[kind];
        let uid = self.take_uid();
        Food {
            uid,
            kind,
            attack: fd.attack,
            health: fd.health,
            base_attack: fd.attack,
            base_health: fd.health,
            cost: fd.cost,
            eaten: false,
            rs,
        }
    }

    /// `Pet.attack`.
    #[inline]
    pub fn atk(&self, p: usize) -> i32 {
        (self.pets[p].attack + self.pets[p].tmp_attack).min(50)
    }

    /// `Pet.health`.
    #[inline]
    pub fn hp(&self, p: usize) -> i32 {
        (self.pets[p].health + self.pets[p].tmp_health).min(50)
    }

    /// `Pet.ability`.
    fn ability(&self, p: usize) -> AbRef {
        let pet = &self.pets[p];
        if let Some(a) = &pet.over {
            return AbRef::Owned(a.clone());
        }
        let t: &'static Tables = self.t;
        let level = pet.level.clamp(1, 3) as usize;
        match &t.d.pets[pet.kind].abilities[level - 1] {
            Some(a) => AbRef::Static(a),
            None => AbRef::Static(&t.d.empty_ability),
        }
    }

    /// `Pet.acts_as("pet-shark", "pet-fly")` / `("pet-fly")`.
    fn acts_as(&self, p: usize, a: usize, b: Option<usize>) -> bool {
        let kind = self.pets[p].kind;
        if kind == a || Some(kind) == b {
            return true;
        }
        match self.ability(p).copied_from {
            Some(c) => c == a || Some(c) == b,
            None => false,
        }
    }

    /// `Pet.hurt`.
    fn hurt(&mut self, p: usize, value: i32) {
        self.pets[p].health -= value;
        if value > 0 {
            self.pets[p].hurt += 1;
        }
    }

    /// `Pet.total_experience`.
    pub fn total_experience(pet: &Pet) -> R<i32> {
        match pet.level {
            1 => Ok(pet.experience),
            2 => Ok(2 + pet.experience),
            3 => Ok(5 + pet.experience),
            l => err(format!("Invalid pet level {l}")),
        }
    }

    /// `Pet.set_total_experience`.
    fn set_total_experience(pet: &mut Pet, total: i32) -> R<()> {
        let total = total.min(5);
        if total < 0 {
            return err("Total experience cannot be negative");
        }
        if total < 2 {
            pet.level = 1;
            pet.experience = total;
        } else if total < 5 {
            pet.level = 2;
            pet.experience = total - 2;
        } else {
            pet.level = 3;
            pet.experience = 0;
        }
        Ok(())
    }

    /// `Pet.gain_experience(amount=1)`.
    fn gain_experience(&mut self, p: usize) -> R<i32> {
        if self.pets[p].level >= 3 {
            return Ok(0);
        }
        let old_level = self.pets[p].level;
        let total = Self::total_experience(&self.pets[p])?;
        let pet = &mut self.pets[p];
        pet.attack = (pet.attack + 1).min(50);
        pet.health = (pet.health + 1).min(50);
        Self::set_total_experience(pet, total + 1)?;
        let level_ups = pet.level - old_level;
        pet.sell_value += level_ups;
        Ok(level_ups)
    }

    /// The generator behind `apet.rs` (`None`: a rolled food, whose stream
    /// is the shop's after `_connect_objects`).
    fn stream(&mut self, apet: Option<usize>) -> R<&mut Mt> {
        let p = match apet {
            None => return Ok(&mut self.rs),
            Some(p) => p,
        };
        if matches!(self.pets[p].rs, Rs::TeamSeed) {
            let seeded = Mt::new(self.team_seed);
            self.pets[p].rs = Rs::Private(Box::new(seeded));
            cov!(self, "stream:team-seed-materialised");
        }
        if matches!(self.pets[p].rs, Rs::Private(_)) {
            cov!(self, "stream:private-draw:{}", self.t.d.pets[self.pets[p].kind].name);
        }
        match &mut self.pets[p].rs {
            Rs::Shared => Ok(&mut self.rs),
            Rs::Private(b) => Ok(&mut **b),
            Rs::TeamSeed => unreachable!(),
            Rs::Mock => err(
                "Unsupported: the reference would draw from MockRandomState (NumPy's global generator) here",
            ),
        }
    }

    // ------------------------------------------------------------------
    // Team helpers (sapai/teams.py)
    // ------------------------------------------------------------------

    #[inline]
    fn tslots(&self, t: usize) -> &[Option<usize>; NS] {
        if t == 0 {
            &self.team
        } else {
            &self.oteam
        }
    }

    #[inline]
    fn tslots_mut(&mut self, t: usize) -> &mut [Option<usize>; NS] {
        if t == 0 {
            &mut self.team
        } else {
            &mut self.oteam
        }
    }

    /// `len(team)`.
    pub fn team_len(&self, t: usize) -> usize {
        self.tslots(t).iter().filter(|s| s.is_some()).count()
    }

    fn team_move(&mut self, t: usize, sidx: usize, tidx: usize) -> R<()> {
        let slots = self.tslots_mut(t);
        if slots[tidx].is_some() {
            return err("Attempted move to a populated position");
        }
        slots[tidx] = slots[sidx];
        slots[sidx] = None;
        Ok(())
    }

    /// `Team.move_forward(start_idx=0, end_idx=10)`.
    fn move_forward(&mut self, t: usize, start_idx: usize, end_idx: usize) -> R<()> {
        loop {
            let slots = *self.tslots(t);
            let empty = match (0..NS).find(|&i| slots[i].is_none()) {
                Some(e) => e,
                None => return Ok(()),
            };
            let mut found = None;
            for (idx, slot) in slots.iter().enumerate() {
                if slot.is_none() {
                    continue;
                }
                if idx < start_idx || idx >= end_idx {
                    continue;
                }
                if empty < idx {
                    found = Some(idx);
                    break;
                }
            }
            match found {
                Some(idx) => self.team_move(t, idx, empty)?,
                None => return Ok(()),
            }
        }
    }

    /// `Team.move_backward()`.
    fn move_backward(&mut self, t: usize) -> R<()> {
        loop {
            let slots = *self.tslots(t);
            let empty = match (0..NS).rev().find(|&i| slots[i].is_none()) {
                Some(e) => e,
                None => return Ok(()),
            };
            let mut found = None;
            for idx in (0..NS).rev() {
                if slots[idx].is_none() {
                    continue;
                }
                if empty > idx {
                    found = Some(idx);
                    break;
                }
            }
            match found {
                Some(idx) => self.team_move(t, idx, empty)?,
                None => return Ok(()),
            }
        }
    }

    fn get_idx(&self, t: usize, p: usize) -> Option<usize> {
        (0..NS).find(|&i| self.tslots(t)[i] == Some(p))
    }

    fn check_friend(&self, t: usize, p: usize) -> bool {
        self.get_idx(t, p).is_some()
    }

    fn team_remove(&mut self, t: usize, p: usize) -> R<()> {
        match self.get_idx(t, p) {
            Some(i) => {
                self.tslots_mut(t)[i] = None;
                Ok(())
            }
            None => err("Remove pet not found"),
        }
    }

    /// `Team.get_ahead(idx, n)`: nearest first.
    fn get_ahead(&self, t: usize, pet_idx: usize, n: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for idx in (0..pet_idx.min(NS)).rev() {
            if let Some(p) = self.tslots(t)[idx] {
                out.push(p);
                if out.len() >= n {
                    break;
                }
            }
        }
        out
    }

    fn own_idx(&self, p: usize) -> R<usize> {
        match self.get_idx(self.pets[p].team, p) {
            Some(i) => Ok(i),
            None => err("get_idx: pet not found on its team"),
        }
    }

    // ------------------------------------------------------------------
    // Shop (sapai/shop.py)
    // ------------------------------------------------------------------

    fn rule(&self) -> &'static TurnRule {
        let t: &'static Tables = self.t;
        &t.e.turns[self.rule_turn]
    }

    /// `Shop.pets`.
    fn shop_pets(&self) -> Vec<usize> {
        self.slots
            .iter()
            .filter_map(|s| match (s.ty, &s.obj) {
                (SlotType::Pet | SlotType::Levelup, Obj::Pet(p)) => Some(*p),
                _ => None,
            })
            .collect()
    }

    /// `Shop.update_shop_rules(turn, roll_only_empty)`.
    fn update_shop_rules(&mut self, turn: i32, roll_only_empty: bool) -> R<()> {
        if turn <= 0 {
            return err("Input turn must be greater than 0");
        }
        self.rule_turn = turn.min(11) as usize;
        let rule = self.rule();
        self.pslots = rule.pslots;
        self.fslots = rule.fslots;
        self.check_rules()?;
        self.roll_shop(roll_only_empty)
    }

    /// `Shop.check_rules`.
    fn check_rules(&mut self) -> R<()> {
        let old = std::mem::take(&mut self.slots);
        let mut levelup_slots: Vec<ShopSlot> = Vec::new();
        let mut pet_slots: Vec<ShopSlot> = Vec::new();
        let mut food_slots: Vec<ShopSlot> = Vec::new();
        let mut rest: Vec<ShopSlot> = Vec::with_capacity(old.len());
        for slot in old {
            match (slot.ty, slot.frozen) {
                (SlotType::Levelup, true) => levelup_slots.push(slot),
                (SlotType::Pet, true) => pet_slots.push(slot),
                (SlotType::Food, true) => food_slots.push(slot),
                _ => rest.push(slot),
            }
        }
        let frozen_count = levelup_slots.len() + pet_slots.len() + food_slots.len();
        if frozen_count > MAX_SHOP_SLOTS {
            return err("Invalid shop state: frozen entries already exceed the hard shop capacity");
        }
        if !levelup_slots.is_empty() {
            cov!(self, "shop:frozen-levelup-kept");
        }
        let mut ordinary_pet_capacity = self.pslots.saturating_sub(levelup_slots.len());
        if pet_slots.len() > ordinary_pet_capacity {
            ordinary_pet_capacity = pet_slots.len();
        }
        ordinary_pet_capacity = ordinary_pet_capacity.min(MAX_SHOP_SLOTS - levelup_slots.len() - food_slots.len());
        let food_capacity = self
            .fslots
            .max(food_slots.len())
            .min(MAX_SHOP_SLOTS - levelup_slots.len() - ordinary_pet_capacity);
        if food_capacity < self.fslots || ordinary_pet_capacity + levelup_slots.len() < self.pslots {
            cov!(self, "shop:capacity-squeezed-by-frozen");
        }
        for slot in rest {
            if slot.ty == SlotType::Pet && !slot.frozen && pet_slots.len() < ordinary_pet_capacity {
                pet_slots.push(slot);
            } else if slot.ty == SlotType::Food && !slot.frozen && food_slots.len() < food_capacity {
                food_slots.push(slot);
            }
        }
        while pet_slots.len() < ordinary_pet_capacity {
            pet_slots.push(ShopSlot { ty: SlotType::Pet, frozen: false, cost: 3, reward_group: None, obj: Obj::Empty });
        }
        while food_slots.len() < food_capacity {
            food_slots.push(ShopSlot {
                ty: SlotType::Food,
                frozen: false,
                cost: 3,
                reward_group: None,
                obj: Obj::Empty,
            });
        }
        // Frozen foods compact right within their category.
        let (frozen_food, loose_food): (Vec<ShopSlot>, Vec<ShopSlot>) = food_slots.into_iter().partition(|s| s.frozen);
        let mut rebuilt = levelup_slots;
        rebuilt.extend(pet_slots);
        rebuilt.extend(loose_food);
        rebuilt.extend(frozen_food);
        self.slots = rebuilt;
        Ok(())
    }

    /// `Shop._roll_slot`.
    fn roll_slot(&mut self, index: usize) -> R<()> {
        if self.slots[index].frozen {
            return err("Frozen slots must not be passed to _roll_slot");
        }
        if self.slots[index].ty == SlotType::Levelup {
            self.slots[index].ty = SlotType::Pet;
            self.slots[index].reward_group = None;
        }
        let rule = self.rule();
        match self.slots[index].ty {
            SlotType::Pet => {
                let u = self.pet_tape.uniform(self.pet_cursor);
                self.pet_cursor += 1;
                let choice = rule.avail_pets[tape::pick_equal(u, rule.avail_pets.len(), rule.pp, rule.pp_sum)];
                // Created with a private stream; every caller of Shop.roll
                // runs Player._connect_objects before anything can draw.
                let p = self.new_pet(choice, 0, Rs::Shared);
                self.slots[index].obj = Obj::Pet(p);
            }
            SlotType::Food => {
                let u = self.food_tape.uniform(self.food_cursor);
                self.food_cursor += 1;
                let choice = rule.avail_foods[tape::pick_equal(u, rule.avail_foods.len(), rule.fp, rule.fp_sum)];
                let food = self.new_food(choice, Rs::Shared);
                self.slots[index].cost = food.cost;
                self.slots[index].obj = Obj::Food(food);
            }
            SlotType::Levelup => unreachable!(),
        }
        Ok(())
    }

    /// `Shop.roll(roll_only_empty)` (no team argument on this path).
    fn roll_shop(&mut self, roll_only_empty: bool) -> R<()> {
        self.check_rules()?;
        for index in 0..self.slots.len() {
            let slot = &self.slots[index];
            let empty = matches!(slot.obj, Obj::Empty);
            let eligible = !slot.frozen && (!roll_only_empty || empty);
            if !eligible {
                continue;
            }
            self.roll_slot(index)?;
            match self.slots[index].ty {
                SlotType::Pet => {
                    if let Obj::Pet(p) = self.slots[index].obj {
                        let (sa, sh) = (self.shop_attack, self.shop_health);
                        let pet = &mut self.pets[p];
                        pet.attack = (pet.attack + sa).min(50);
                        pet.health = (pet.health + sh).min(50);
                    }
                }
                SlotType::Food => {
                    if let Obj::Food(f) = &self.slots[index].obj {
                        let cost = f.cost;
                        self.slots[index].cost = cost;
                    }
                }
                SlotType::Levelup => {}
            }
        }

        // Frozen pets compact left; the others are ordered by tier, highest first (stable).
        let pet_indices: Vec<usize> = (0..self.slots.len())
            .filter(|&i| matches!(self.slots[i].ty, SlotType::Pet | SlotType::Levelup))
            .collect();
        if !pet_indices.is_empty() {
            let mut frozen: Vec<ShopSlot> = Vec::new();
            let mut loose: Vec<(i64, ShopSlot)> = Vec::new();
            for &i in &pet_indices {
                let slot = self.slots[i].clone();
                if slot.frozen {
                    frozen.push(slot);
                } else {
                    let tier = match slot.obj {
                        Obj::Pet(p) => match self.t.d.pets[self.pets[p].kind].tier {
                            Some(t) => t,
                            None => return err("shop pet without an integer tier"),
                        },
                        _ => return err("pet shop slot without a pet"),
                    };
                    loose.push((tier, slot));
                }
            }
            loose.sort_by(|a, b| b.0.cmp(&a.0));
            let mut compacted = frozen.into_iter().chain(loose.into_iter().map(|x| x.1));
            for &i in &pet_indices {
                self.slots[i] = compacted.next().unwrap();
            }
        }

        // Freshly rolled foods: higher tier first (stable); frozen foods keep their positions.
        let food_indices: Vec<usize> =
            (0..self.slots.len()).filter(|&i| self.slots[i].ty == SlotType::Food && !self.slots[i].frozen).collect();
        if food_indices.len() > 1 {
            let mut loose: Vec<(i64, ShopSlot)> = Vec::new();
            for &i in &food_indices {
                let slot = self.slots[i].clone();
                let tier = match &slot.obj {
                    Obj::Food(f) => self.t.e.foods[f.kind].sort_tier,
                    _ => 0,
                };
                loose.push((tier, slot));
            }
            loose.sort_by(|a, b| b.0.cmp(&a.0));
            for (&i, (_, slot)) in food_indices.iter().zip(loose.into_iter()) {
                self.slots[i] = slot;
            }
        }
        Ok(())
    }

    /// `Shop.buy(obj)`: remove the slot, with its linked discover sibling.
    fn shop_buy(&mut self, idx: usize) {
        match self.slots[idx].reward_group {
            None => {
                self.slots.remove(idx);
            }
            Some(group) => {
                self.slots.retain(|s| s.reward_group != Some(group));
            }
        }
    }

    /// `Shop.levelup(count)`.
    fn shop_levelup(&mut self, count: i32) -> R<()> {
        if count < 0 {
            return err("Level-up reward count cannot be negative");
        }
        // Slots generated by this call sit at the front, in creation order.
        let mut newly_generated = 0usize;
        for _ in 0..count {
            let overflow = (self.slots.len() + 2).saturating_sub(MAX_SHOP_SLOTS);
            // `newly_generated_slots` are exactly the first `newly_generated` slots.
            let removable: Vec<usize> = (0..self.slots.len())
                .rev()
                .filter(|&i| !self.slots[i].frozen && i >= newly_generated)
                .collect();
            if removable.len() < overflow {
                cov!(self, "shop:levelup-discarded-no-room");
                continue;
            }
            if overflow > 0 {
                cov!(self, "shop:levelup-overflow-removes:{}", overflow);
            }
            let remove: Vec<usize> = removable[..overflow].to_vec();
            let mut index = 0usize;
            self.slots.retain(|_| {
                let keep = !remove.contains(&index);
                index += 1;
                keep
            });
            let levelup_tier = self.t.e.turns[self.shop_turn.clamp(1, 11) as usize].levelup_tier;
            if self.shop_turn <= 0 {
                return err("Input turn must be greater than 0");
            }
            let t: &'static Tables = self.t;
            let candidates = &t.e.pet_tier_std[levelup_tier as usize];
            if candidates.len() < 2 {
                return err(format!("Tier {levelup_tier} needs two distinct discover choices"));
            }
            // RandomTape.tier_up: two draws from the seat's pet-offer domain.
            let u0 = self.pet_tape.uniform(self.pet_cursor);
            let u1 = self.pet_tape.uniform(self.pet_cursor + 1);
            self.pet_cursor += 2;
            let (c0, c1) = tape::pick_two_distinct(u0, u1, candidates.len());
            let reward_group = self.next_reward_group;
            self.next_reward_group += 1;
            let mut reward_slots = Vec::with_capacity(2);
            for choice in [candidates[c0], candidates[c1]] {
                // Pet(choice, seed_state=shop.seed_state) right after
                // _advance_seed_state: a private copy of the shop stream.
                let private = Rs::Private(Box::new(self.rs.clone()));
                let p = self.new_pet(choice, 0, private);
                let (sa, sh) = (self.shop_attack, self.shop_health);
                let pet = &mut self.pets[p];
                pet.attack = (pet.attack + sa).min(50);
                pet.health = (pet.health + sh).min(50);
                reward_slots.push(ShopSlot {
                    ty: SlotType::Levelup,
                    frozen: false,
                    cost: 3,
                    reward_group: Some(reward_group),
                    obj: Obj::Pet(p),
                });
            }
            cov!(self, "shop:levelup-reward:tier{}", levelup_tier);
            // expanded_slots = reward_slots + surviving_slots
            let mut expanded = reward_slots;
            expanded.append(&mut self.slots);
            self.slots = expanded;
            newly_generated += 2;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Player (sapai/player.py)
    // ------------------------------------------------------------------

    /// `Player._connect_objects`: every team pet and shop item draws from the shop stream.
    fn connect_objects(&mut self) {
        for i in 0..NS {
            if let Some(p) = self.team[i] {
                self.pets[p].rs = Rs::Shared;
            }
        }
        for i in 0..self.slots.len() {
            match &mut self.slots[i].obj {
                Obj::Pet(p) => {
                    let p = *p;
                    self.pets[p].rs = Rs::Shared;
                }
                Obj::Food(f) => f.rs = Rs::Shared,
                Obj::Empty => {}
            }
        }
    }

    /// `Player._shop_trigger_order`: highest attack first, ties shuffled by the shop stream.
    fn shop_trigger_order(&mut self, pets: &[usize]) -> Vec<usize> {
        if pets.len() <= 1 {
            return pets.to_vec();
        }
        let mut attacks: Vec<i32> = pets.iter().map(|&p| self.atk(p)).collect();
        let all = attacks.clone();
        attacks.sort_unstable_by(|a, b| b.cmp(a));
        attacks.dedup();
        let mut ordered = Vec::with_capacity(pets.len());
        for attack in attacks {
            let tied: Vec<usize> = (0..pets.len()).filter(|&i| all[i] == attack).map(|i| pets[i]).collect();
            if tied.len() > 1 {
                cov!(self, "draw:shop-trigger-order-tie:{}", tied.len());
                let indices = self.rs.permutation(tied.len());
                for index in indices {
                    ordered.push(tied[index]);
                }
            } else {
                ordered.push(tied[0]);
            }
        }
        ordered
    }

    /// `Player._notify_friend_gained_perk`.
    fn notify_friend_gained_perk(&mut self, gained: usize) -> R<()> {
        let eligible: Vec<usize> = self
            .team
            .iter()
            .filter_map(|s| *s)
            .filter(|&p| matches!(self.ability(p).trigger, Trigger::GainsPerk | Trigger::EatsShopFoodOrGainsPerk))
            .collect();
        if eligible.is_empty() {
            return Ok(());
        }
        for observer in self.shop_trigger_order(&eligible) {
            self.gains_perk_trigger(observer, gained)?;
        }
        Ok(())
    }

    /// `Player._resolve_shop_hurt_and_faint`.
    fn resolve_shop_hurt_and_faint(&mut self) -> R<()> {
        self.oteam = [None; NS];
        // ShopEventContext.__init__: one priority computation before the phase.
        self.update_pet_priority();
        self.battle_phase_hurt_and_faint()?;
        if self.oteam.iter().any(|s| s.is_some()) {
            cov!(self, "shop:dirty-rat-on-stand-in-team");
        }
        self.oteam = [None; NS];
        // Summoned shop pets were created after the last ownership/RNG bind.
        self.connect_objects();
        Ok(())
    }

    /// `Player.start_turn(winner)`.
    fn start_turn(&mut self, winner: Option<bool>) -> R<()> {
        self.touch();
        self.turn_ended = false;
        self.turn += 1;
        self.gold = 10;
        self.lf_winner = winner;
        self.shop_turn = self.turn;
        self.update_shop_rules(self.turn, false)?;
        self.compact_arena();
        self.connect_objects();
        let pets: Vec<usize> = self.team.iter().filter_map(|s| *s).collect();
        for &p in &pets {
            self.prepare_start_turn(p);
        }
        let eligible: Vec<usize> =
            pets.iter().copied().filter(|&p| self.ability(p).shop_trig == ShopTrig::StartOfTurn).collect();
        for p in self.shop_trigger_order(&eligible) {
            self.sot_trigger(p)?;
        }
        Ok(())
    }

    /// Drop pets that no slot refers to any more (speed and memory only;
    /// identities are `uid`s, not arena positions).
    fn compact_arena(&mut self) {
        let mut map: Vec<usize> = vec![usize::MAX; self.pets.len()];
        let mut kept: Vec<Pet> = Vec::with_capacity(32);
        for i in 0..NS {
            if let Some(p) = self.team[i] {
                if map[p] == usize::MAX {
                    map[p] = kept.len();
                    kept.push(self.pets[p].clone());
                }
                self.team[i] = Some(map[p]);
            }
        }
        for i in 0..self.slots.len() {
            if let Obj::Pet(p) = self.slots[i].obj {
                if map[p] == usize::MAX {
                    map[p] = kept.len();
                    kept.push(self.pets[p].clone());
                }
                self.slots[i].obj = Obj::Pet(map[p]);
            }
        }
        self.pets = kept;
    }

    /// `Pet.prepare_start_turn`.
    fn prepare_start_turn(&mut self, p: usize) {
        let parrot = self.t.d.parrot;
        let pet = &mut self.pets[p];
        if pet.kind == parrot {
            pet.over = None;
        }
        pet.tmp_attack = 0;
        pet.tmp_health = 0;
        pet.ability_counter = 0;
        pet.friends_hurt_counter = 0;
    }

    /// `Player.buy_pet(shop_idx, team_idx)`.
    fn buy_pet(&mut self, shop_idx: usize, team_idx: usize) -> R<()> {
        if self.team_len(0) == NS {
            return err("Attempted to buy Pet on full team");
        }
        let p = match self.slots.get(shop_idx).map(|s| &s.obj) {
            Some(Obj::Pet(p)) => *p,
            _ => return err("Attempted to buy_pet using a non-pet object"),
        };
        if self.team[team_idx].is_some() {
            return err(format!("Attempted to buy Pet into occupied slot {team_idx}"));
        }
        let cost = self.slots[shop_idx].cost;
        if cost > self.gold {
            return err(format!("Attempted to buy Pet of cost {cost} with only {} gold", self.gold));
        }
        cov!(
            self,
            "buy-pet:{}:{}",
            self.t.d.pets[self.pets[p].kind].name,
            self.slots[shop_idx].ty.name()
        );
        self.gold -= cost;
        self.team[team_idx] = Some(p);
        self.shop_buy(shop_idx);
        if self.pets[p].kind == self.t.e.scorpion {
            self.pets[p].status = Status::PoisonAttack;
        }
        if self.pets[p].status.is_perk() {
            self.notify_friend_gained_perk(p)?;
        }
        for i in 0..NS {
            if let Some(q) = self.team[i] {
                self.buy_friend_trigger(q, p)?;
            }
        }
        for i in 0..NS {
            if let Some(q) = self.team[i] {
                self.friend_summoned_trigger(q, p)?;
            }
        }
        Ok(())
    }

    /// `Player.combine_pet_stats(keep, merged, trigger_levelup_abilities)`.
    /// Returns (discover rewards, previous level).
    fn combine_pet_stats(&mut self, keep: usize, merged: usize, trigger_levelup: bool) -> R<i32> {
        let keep_total = Self::total_experience(&self.pets[keep])?;
        let merged_total = Self::total_experience(&self.pets[merged])?;
        let keep_level = self.pets[keep].level;
        let merged_level = self.pets[merged].level;
        let combined_total = keep_total + merged_total + 1;
        let experience_gained = combined_total - keep_total.max(merged_total);
        let previous_level = keep_level.max(merged_level);
        let keep_sell_bonus = (self.pets[keep].sell_value - self.pets[keep].level).max(0);
        let merged_sell_bonus = (self.pets[merged].sell_value - self.pets[merged].level).max(0);
        let (ka, kh, kta, kth, ks) = {
            let k = &self.pets[keep];
            (k.attack, k.health, k.tmp_attack, k.tmp_health, k.status)
        };
        let (ma, mh, mta, mth, ms) = {
            let m = &self.pets[merged];
            (m.attack, m.health, m.tmp_attack, m.tmp_health, m.status)
        };
        let c_attack = (ka.max(ma) + experience_gained).min(50);
        let c_health = (kh.max(mh) + experience_gained).min(50);
        let c_tmp_attack = kta.max(mta).min((50 - c_attack).max(0));
        let c_tmp_health = kth.max(mth).min((50 - c_health).max(0));
        // get_combined_status: the higher tier wins, the kept pet on a tie.
        let tier = |s: Status| -> i32 {
            match s {
                Status::Weak | Status::PoisonAttack | Status::None => 0,
                Status::HoneyBee => 1,
                Status::BoneAttack => 2,
                Status::GarlicArmor | Status::Cake => 3,
                Status::SplashAttack | Status::Bread => 4,
                Status::CoconutShield | Status::MelonArmor | Status::SteakAttack | Status::ExtraLife => 5,
            }
        };
        let cstatus = if tier(ms) > tier(ks) { ms } else { ks };
        {
            let k = &mut self.pets[keep];
            k.attack = c_attack;
            k.health = c_health;
            k.tmp_attack = c_tmp_attack;
            k.tmp_health = c_tmp_health;
            k.status = cstatus;
            Self::set_total_experience(k, combined_total)?;
            k.sell_value = k.level + keep_sell_bonus.max(merged_sell_bonus);
        }
        let newly_crossed = (self.pets[keep].level - previous_level).max(0);
        if newly_crossed > 0 && trigger_levelup {
            self.trigger_levelup_abilities(keep, previous_level)?;
        }
        if self.pets[keep].level == 3 && keep_level == 2 && merged_level == 2 {
            cov!(self, "combine:two-level-twos-no-reward");
            return Ok(0);
        }
        Ok(newly_crossed)
    }

    /// `Pet.trigger_levelup_abilities(previous_level)`.
    fn trigger_levelup_abilities(&mut self, p: usize, previous_level: i32) -> R<()> {
        let final_level = self.pets[p].level;
        for ability_level in previous_level..final_level {
            self.pets[p].level = ability_level;
            self.levelup_trigger(p)?;
            self.pets[p].ability_counter = 0;
            self.pets[p].friends_hurt_counter = 0;
        }
        self.pets[p].level = final_level;
        Ok(())
    }

    /// `Player.buy_combine(shop_idx, team_idx)`.
    fn buy_combine(&mut self, shop_idx: usize, team_idx: usize) -> R<()> {
        let shop_pet = match self.slots.get(shop_idx).map(|s| &s.obj) {
            Some(Obj::Pet(p)) => *p,
            _ => return err("Attempted buy_combined with a non-pet shop item"),
        };
        let team_pet = match self.team[team_idx] {
            Some(p) => p,
            None => return err("Attempted buy_combined with an empty team slot"),
        };
        if self.pets[team_pet].kind != self.pets[shop_pet].kind {
            return err("Attempted combine for pets of different species");
        }
        if self.pets[team_pet].level >= 3 {
            return err("Cannot combine into a level 3 pet");
        }
        let cost = self.slots[shop_idx].cost;
        if cost > self.gold {
            return err(format!("Attempted to buy Pet of cost {cost} with only {} gold", self.gold));
        }
        cov!(
            self,
            "buy-combine:{}:{}",
            self.t.d.pets[self.pets[shop_pet].kind].name,
            self.slots[shop_idx].ty.name()
        );
        self.gold -= cost;
        self.shop_buy(shop_idx);
        let rewards = self.combine_pet_stats(team_pet, shop_pet, true)?;
        if rewards > 0 {
            self.shop_levelup(rewards)?;
        }
        for i in 0..NS {
            if let Some(q) = self.team[i] {
                self.buy_friend_trigger(q, team_pet)?;
            }
        }
        Ok(())
    }

    /// `Player.combine(pet1, pet2)`: `pet1` is kept.
    fn combine(&mut self, idx1: usize, idx2: usize) -> R<()> {
        let (pet1, pet2) = match (self.team[idx1], self.team[idx2]) {
            (Some(a), Some(b)) => (a, b),
            _ => return err("Attempted combine for Pet not on team"),
        };
        if self.pets[pet1].kind != self.pets[pet2].kind {
            return err("Attempted combine for pets of different species");
        }
        if self.pets[pet1].level >= 3 || self.pets[pet2].level >= 3 {
            return err("Cannot combine a level 3 pet");
        }
        cov!(self, "combine:{}", self.t.d.pets[self.pets[pet1].kind].name);
        let previous_level = self.pets[pet1].level.max(self.pets[pet2].level);
        let rewards = self.combine_pet_stats(pet1, pet2, false)?;
        self.team[idx2] = None;
        if self.pets[pet1].level > previous_level {
            self.trigger_levelup_abilities(pet1, previous_level)?;
        }
        if rewards > 0 {
            self.shop_levelup(rewards)?;
        }
        Ok(())
    }

    /// `Pet.eat(food)`: returns the number of levels gained.
    fn eat(&mut self, p: usize, food: &mut Food) -> R<i32> {
        let fd: &FoodData = &self.t.e.foods[food.kind];
        let pet = &mut self.pets[p];
        if fd.until_end_of_battle {
            pet.tmp_attack = (pet.tmp_attack + food.attack).min((50 - pet.attack).max(0));
            pet.tmp_health = (pet.tmp_health + food.health).min((50 - pet.health).max(0));
        } else {
            pet.attack = (pet.attack + food.attack).min(50);
            pet.health = (pet.health + food.health).min(50);
        }
        if fd.status != Status::None {
            pet.status = fd.status;
        }
        if food.kind == self.t.e.chocolate {
            return self.gain_experience(p);
        } else if food.kind == self.t.e.pill {
            self.pets[p].health = -1000;
        } else if food.kind == self.t.e.canned && !food.eaten {
            self.shop_attack += food.attack;
            self.shop_health += food.health;
            food.eaten = true;
        }
        Ok(0)
    }

    /// `Player.buy_food(shop_idx, team_idx)`; `team_idx` is `None` for the
    /// foods that pick their own targets.
    fn buy_food(&mut self, shop_idx: usize, team_idx: Option<usize>) -> R<()> {
        let mut food = match self.slots.get(shop_idx) {
            Some(ShopSlot { ty: SlotType::Food, obj: Obj::Food(f), .. }) => f.clone(),
            _ => return err("Shop slot not food"),
        };
        let fd: &'static FoodData = {
            let t: &'static Tables = self.t;
            &t.e.foods[food.kind]
        };
        let targets: Vec<usize> = match team_idx {
            None => {
                // get_target(food, [0, None], [self.team])
                if food.kind == self.t.e.canned {
                    self.shop_pets()
                } else if fd.random_friend {
                    if !matches!(food.rs, Rs::Shared) {
                        return err("Unsupported: a food outside the shop stream picks random targets");
                    }
                    self.random_friend(None, Idx::Absent, fd.target_n)?
                } else {
                    return err(format!("{} needs a purchase target", fd.name));
                }
            }
            Some(i) => {
                let p = match self.team[i] {
                    Some(p) => p,
                    None => return err("Attempted to buy food for Pet not on team"),
                };
                if food.kind == self.t.e.chocolate && self.pets[p].level >= 3 {
                    return err(format!("Cannot feed {} to level {} pet", fd.name, self.pets[p].level));
                }
                vec![p]
            }
        };
        let cost = self.slots[shop_idx].cost;
        if cost > self.gold {
            return err(format!("Attempted to buy Pet of cost {cost} with only {} gold", self.gold));
        }
        cov!(self, "buy-food:{}", fd.name);

        // Before feeding, check for Cat.
        for i in 0..NS {
            if let Some(q) = self.team[i] {
                if self.pets[q].kind == self.t.e.cat {
                    self.cat_trigger(q, &mut food)?;
                }
            }
        }

        self.gold -= cost;
        self.shop_buy(shop_idx);
        if food.kind == self.t.e.canned && !food.eaten {
            self.shop_attack += food.attack;
            self.shop_health += food.health;
            food.eaten = true;
        }
        for &p in &targets {
            let previous_level = self.pets[p].level;
            let levelups = self.eat(p, &mut food)?;
            if levelups > 0 {
                self.trigger_levelup_abilities(p, previous_level)?;
                self.shop_levelup(levelups)?;
            }
            // Only pets on the team count (Canned Food also feeds shop pets).
            if !self.check_friend(0, p) {
                continue;
            }
            for i in 0..NS {
                if let Some(q) = self.team[i] {
                    self.eats_shop_food_trigger(q, p)?;
                }
            }
        }
        // buy_food_trigger: no Turtle Pack pet has the BuyFood trigger.
        for i in 0..NS {
            if let Some(q) = self.team[i] {
                if self.ability(q).shop_trig == ShopTrig::BuyFood {
                    return err("Unsupported: BuyFood trigger");
                }
            }
        }
        self.resolve_shop_hurt_and_faint()
    }

    /// `Player.sell(team_idx)`.
    fn sell(&mut self, team_idx: usize) -> R<()> {
        let pet = match self.team[team_idx] {
            Some(p) => p,
            None => return err("Attempted to sell an empty slot"),
        };
        cov!(self, "sell:{}", self.t.d.pets[self.pets[pet].kind].name);
        let listeners: Vec<usize> = self.team.iter().filter_map(|s| *s).collect();
        self.team[team_idx] = None;
        for listener in listeners {
            self.sell_trigger(listener, pet, team_idx)?;
        }
        self.gold += self.pets[pet].sell_value;
        Ok(())
    }

    /// `Player.roll()`.
    fn roll(&mut self) -> R<()> {
        if self.gold < 1 {
            return err("Attempt to roll without gold");
        }
        self.roll_shop(false)?;
        self.connect_objects();
        self.gold -= 1;
        Ok(())
    }

    /// `Player.swap(source_idx, target_idx)`: move one slot, shifting the ones between.
    fn swap(&mut self, source: usize, target: usize) -> R<()> {
        if source == target {
            return err("Source and target positions must differ");
        }
        if self.team[source].is_none() {
            return err(format!("Attempted to move empty slot {source}"));
        }
        let mut order: Vec<Option<usize>> = self.team.to_vec();
        let moved = order.remove(source);
        order.insert(target, moved);
        self.team.copy_from_slice(&order);
        Ok(())
    }

    /// `Player.end_turn()`.
    fn end_turn(&mut self) -> R<()> {
        let eligible: Vec<usize> = self
            .team
            .iter()
            .filter_map(|s| *s)
            .filter(|&p| {
                self.ability(p).shop_trig.is_end_of_turn()
                    || matches!(self.pets[p].status, Status::Cake | Status::Bread)
            })
            .collect();
        for p in self.shop_trigger_order(&eligible) {
            self.eot_trigger(p)?;
        }
        self.turn_ended = true;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Shop-phase triggers (sapai/pets.py)
    // ------------------------------------------------------------------

    fn note_ability(&mut self, p: usize, ab: &Ability, what: &str) {
        if self.cov.is_some() {
            let name = self.t.d.pets[self.pets[p].kind].name.clone();
            let level = self.pets[p].level;
            cov!(self, "ability:{}:{}:{}:L{}", name, what, ab.effect.kind_name, level);
        }
    }

    /// The `maxTriggers` gate. Returns false when spent.
    fn take_trigger(&mut self, p: usize, ab: &Ability) -> bool {
        if let Some(max) = ab.max_triggers {
            if self.pets[p].ability_counter >= max {
                cov!(self, "max-triggers-spent:{}", self.t.d.pets[self.pets[p].kind].name);
                return false;
            }
            self.pets[p].ability_counter += 1;
        }
        true
    }

    /// `Pet.sot_trigger(reset_state=False)`.
    fn sot_trigger(&mut self, p: usize) -> R<()> {
        let ab = self.ability(p);
        if ab.shop_trig != ShopTrig::StartOfTurn {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.note_ability(p, &ab, "StartOfTurn");
        self.run_eff(&ab.effect, p, Idx::Int(pet_idx), false, Te::None, None)?;
        Ok(())
    }

    /// `Pet.cat_trigger(food)` and its `FoodMultiplier` effect.
    fn cat_trigger(&mut self, p: usize, food: &mut Food) -> R<()> {
        if food.base_attack == 0 && food.base_health == 0 {
            return Ok(());
        }
        if self.pets[p].ability_counter >= 2 {
            cov!(self, "max-triggers-spent:pet-cat");
            return Ok(());
        }
        self.pets[p].ability_counter += 1;
        let ab = self.ability(p);
        if ab.effect.shop_kind != ShopEffect::FoodMultiplier {
            return err(format!("Unsupported: Cat with effect {}", ab.effect.kind_name));
        }
        self.own_idx(p)?;
        let mult = match ab.effect.amount {
            Amount::Int(v) => v as i32 - 1,
            _ => return err("FoodMultiplier amount not understood"),
        };
        self.note_ability(p, &ab, "PurchaseFood");
        food.attack += food.base_attack * mult;
        food.health += food.base_health * mult;
        Ok(())
    }

    /// `Pet.sell_trigger(trigger, trigger_idx)`.
    fn sell_trigger(&mut self, p: usize, trigger: usize, trigger_idx: usize) -> R<()> {
        let ab = self.ability(p);
        if ab.shop_trig != ShopTrig::Sell {
            return Ok(());
        }
        if ab.triggered_by == TrigBy::SelfPet && trigger != p {
            return Ok(());
        }
        if ab.triggered_by == TrigBy::EachFriend && trigger == p {
            return Ok(());
        }
        let pet_idx = if p == trigger { trigger_idx } else { self.own_idx(p)? };
        self.note_ability(p, &ab, "Sell");
        self.run_eff(&ab.effect, p, Idx::Int(pet_idx), false, Te::Pet(trigger), None)?;
        Ok(())
    }

    /// `Pet.eats_shop_food_trigger(trigger)`.
    fn eats_shop_food_trigger(&mut self, p: usize, trigger: usize) -> R<()> {
        let ab = self.ability(p);
        if !(ab.shop_trig == ShopTrig::EatsShopFood || ab.trigger == Trigger::EatsShopFoodOrGainsPerk) {
            return Ok(());
        }
        if ab.triggered_by == TrigBy::SelfPet && trigger != p {
            return Ok(());
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.note_ability(p, &ab, "EatsShopFood");
        self.run_eff(&ab.effect, p, Idx::Int(pet_idx), false, Te::Pet(trigger), None)?;
        Ok(())
    }

    /// `Pet.gains_perk_trigger(trigger)`.
    fn gains_perk_trigger(&mut self, p: usize, trigger: usize) -> R<()> {
        let ab = self.ability(p);
        if !matches!(ab.trigger, Trigger::GainsPerk | Trigger::EatsShopFoodOrGainsPerk) {
            return Ok(());
        }
        let t = self.pets[p].team;
        if !self.check_friend(t, trigger) {
            return Ok(());
        }
        if ab.triggered_by == TrigBy::SelfPet && trigger != p {
            return Ok(());
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.note_ability(p, &ab, "GainsPerk");
        self.run_eff(&ab.effect, p, Idx::Int(pet_idx), false, Te::Pet(trigger), None)?;
        Ok(())
    }

    /// `Pet.buy_friend_trigger(trigger)`.
    fn buy_friend_trigger(&mut self, p: usize, trigger: usize) -> R<()> {
        let ab = self.ability(p);
        match ab.shop_trig {
            ShopTrig::Buy => match ab.triggered_by {
                TrigBy::SelfPet | TrigBy::Player => {
                    if trigger != p {
                        return Ok(());
                    }
                }
                TrigBy::EachFriend => {
                    if trigger == p {
                        return Ok(());
                    }
                }
                TrigBy::FriendAhead | TrigBy::Other => return err("Ability unrecognized for a Buy trigger"),
            },
            ShopTrig::BuyTier1Animal => {
                if !self.t.e.pet_tier1[self.pets[trigger].kind] {
                    return Ok(());
                }
            }
            ShopTrig::BuyAfterLoss => {
                if self.lf_winner != Some(false) {
                    return Ok(());
                }
            }
            _ => return Ok(()),
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.note_ability(p, &ab, "Buy");
        self.run_eff(&ab.effect, p, Idx::Int(pet_idx), false, Te::Pet(trigger), None)?;
        Ok(())
    }

    /// `Pet.friend_summoned_trigger(trigger)`.
    fn friend_summoned_trigger(&mut self, p: usize, trigger: usize) -> R<()> {
        let ab = self.ability(p);
        if ab.trigger != Trigger::Summoned {
            return Ok(());
        }
        if trigger == p {
            return Ok(());
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.note_ability(p, &ab, "Summoned");
        self.tiger_func(false, p, Idx::Int(pet_idx), false, Te::Pet(trigger), None)?;
        Ok(())
    }

    /// `Pet.levelup_trigger(self)`.
    fn levelup_trigger(&mut self, p: usize) -> R<()> {
        let ab = self.ability(p);
        if ab.shop_trig != ShopTrig::LevelUp {
            return Ok(());
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.note_ability(p, &ab, "LevelUp");
        self.run_eff(&ab.effect, p, Idx::Int(pet_idx), false, Te::Pet(p), None)?;
        Ok(())
    }

    /// `Pet.eot_trigger()`.
    fn eot_trigger(&mut self, p: usize) -> R<()> {
        // Shop perks are independent from the pet's native ability.
        match self.pets[p].status {
            Status::Cake => {
                self.pets[p].sell_value += 1;
                cov!(self, "perk:cake-end-of-turn");
            }
            Status::Bread => {
                let pet = &mut self.pets[p];
                pet.tmp_health = (pet.tmp_health + 7).min((50 - pet.health).max(0));
                cov!(self, "perk:bread-end-of-turn");
            }
            _ => {}
        }
        let ab = self.ability(p);
        match ab.shop_trig {
            ShopTrig::EndOfTurn => {}
            ShopTrig::EndOfTurnWith3PlusGold => {
                if self.gold < 3 {
                    return Ok(());
                }
            }
            ShopTrig::EndOfTurnWithLvl3Friend => {
                let has = self.team.iter().filter_map(|s| *s).any(|q| q != p && self.pets[q].level == 3);
                if !has {
                    return Ok(());
                }
            }
            ShopTrig::EndOfTurnWith4OrLessAnimals => {
                if self.team_len(0) > 4 {
                    return Ok(());
                }
            }
            ShopTrig::EndOfTurnAfterLoss => {
                if self.lf_winner != Some(false) {
                    return Ok(());
                }
            }
            ShopTrig::EndOfTurnOther => return err(format!("Unrecognized trigger {}", ab.trigger_name)),
            _ => return Ok(()),
        }
        // Only the first eligible Bison on a team works.
        if self.pets[p].kind == self.t.e.bison {
            let living: Vec<usize> = self.team.iter().filter_map(|s| *s).collect();
            let first = living.iter().copied().find(|&c| {
                self.pets[c].kind == self.t.e.bison && living.iter().any(|&f| f != c && self.pets[f].level == 3)
            });
            if first != Some(p) {
                cov!(self, "bison:not-first-eligible");
                return Ok(());
            }
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.note_ability(p, &ab, "EndOfTurn");
        self.run_eff(&ab.effect, p, Idx::Int(pet_idx), false, Te::None, None)?;
        Ok(())
    }

    /// `Pet.hurt_trigger(opponent_team)`.
    fn hurt_trigger(&mut self, p: usize) -> R<()> {
        if self.pets[p].hurt == 0 {
            return err("Called hurt trigger on pet that was not hurt");
        }
        self.pets[p].hurt -= 1;
        let ab = self.ability(p);
        if ab.trigger != Trigger::Hurt {
            return Ok(());
        }
        if ab.triggered_by != TrigBy::SelfPet {
            return err("Only Self trigger available for hurt_trigger");
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.note_ability(p, &ab, "Hurt");
        self.tiger_func(false, p, Idx::Int(pet_idx), true, Te::Team, None)?;
        Ok(())
    }

    /// `Pet.friend_hurt_trigger(trigger, opponent_team, pending)`.
    fn friend_hurt_trigger(&mut self, p: usize, trigger: usize, pending: &mut Vec<(usize, usize)>) -> R<()> {
        let ab = self.ability(p);
        if ab.trigger != Trigger::FourFriendsHurt {
            return Ok(());
        }
        let t = self.pets[p].team;
        if trigger == p || !self.check_friend(t, p) {
            return Ok(());
        }
        self.pets[p].friends_hurt_counter += 1;
        cov!(self, "wolverine:friend-hurt-counted-in-shop");
        if self.pets[p].friends_hurt_counter < 4 {
            return Ok(());
        }
        self.pets[p].friends_hurt_counter = 0;
        pending.push((p, trigger));
        Ok(())
    }

    /// `Pet.faint_trigger(trigger, te_idx, oteam, remove_before_effect)`.
    fn faint_trigger(
        &mut self,
        p: usize,
        trigger: usize,
        te_idx: (usize, usize),
        has_oteam: bool,
        remove_before_effect: bool,
    ) -> R<(bool, EffResult)> {
        let ab = self.ability(p);
        if ab.trigger != Trigger::Faint {
            return Ok((false, EffResult::none()));
        }
        let shark = self.t.d.shark;
        let fly = self.t.d.fly;
        if self.acts_as(p, shark, Some(fly)) && self.hp(p) <= 0 && ab.triggered_by == TrigBy::EachFriend {
            return Ok((false, EffResult::none()));
        }
        let t = self.pets[p].team;
        match ab.triggered_by {
            TrigBy::SelfPet => {
                if trigger != p {
                    return Ok((false, EffResult::none()));
                }
            }
            TrigBy::FriendAhead => {
                let idx = self.own_idx(p)?;
                let ahead = self.get_ahead(t, idx, 1);
                if ahead.is_empty() || ahead[0] != trigger {
                    return Ok((false, EffResult::none()));
                }
            }
            TrigBy::EachFriend => {
                if trigger == p || self.pets[trigger].team != t {
                    return Ok((false, EffResult::none()));
                }
            }
            _ => {}
        }
        let mut te_idx = te_idx;
        if self.acts_as(p, fly, None) {
            if let Some(i) = self.get_idx(t, trigger) {
                te_idx.1 = i;
            }
            if self.pets[trigger].kind == self.t.d.zombie_fly {
                return Ok((false, EffResult::none()));
            }
        }
        if !self.take_trigger(p, &ab) {
            return Ok((false, EffResult::none()));
        }
        let pet_idx = match self.get_idx(t, p) {
            Some(i) => Idx::Int(i),
            None => Idx::List(te_idx.0, te_idx.1),
        };
        let is_summon = matches!(
            ab.effect.kind,
            EffectKind::RespawnPet
                | EffectKind::SummonPet
                | EffectKind::SummonRandomPet
                | EffectKind::SummonStoredPets
        );
        self.note_ability(p, &ab, "Faint");
        if is_summon {
            let res = self.tiger_func(remove_before_effect, p, pet_idx, has_oteam, Te::Pet(trigger), Some(te_idx))?;
            if res.possible_empty_nested {
                if ab.max_triggers.is_some() {
                    self.pets[p].ability_counter -= 1;
                }
                cov!(self, "shop-summon:no-room");
                return Ok((false, res));
            }
            Ok((true, res))
        } else {
            let res = self.tiger_func(remove_before_effect, p, pet_idx, has_oteam, Te::Pet(trigger), None)?;
            Ok((true, res))
        }
    }

    /// `tiger_func` outside a battle: the effect runs once (`team.battle` is
    /// false, so a Tiger behind never repeats it).
    fn tiger_func(
        &mut self,
        te_fainted: bool,
        apet: usize,
        apet_idx: Idx,
        has_oteam: bool,
        te: Te,
        te_idx: Option<(usize, usize)>,
    ) -> R<EffResult> {
        let ab = self.ability(apet);
        let direct = ab.effect.pet == Some(self.t.d.bee)
            || matches!(ab.effect.kind, EffectKind::RespawnPet | EffectKind::SummonStoredPets);
        if !direct && te_fainted {
            if let Te::Pet(tp) = te {
                let t = self.pets[apet].team;
                if self.check_friend(t, tp) {
                    self.team_remove(t, tp)?;
                }
            }
        }
        self.run_eff(&ab.effect, apet, apet_idx, has_oteam, te, te_idx)
    }

    // ------------------------------------------------------------------
    // The shop faint chain (sapai/battle.py with ShopEventContext)
    // ------------------------------------------------------------------

    /// `Battle.update_pet_priority(team, Team(), shop.rs)`.
    fn update_pet_priority(&mut self) -> Vec<(usize, usize)> {
        let mut entries: Vec<(i32, usize, usize)> = Vec::with_capacity(10);
        for t in 0..2 {
            for i in 0..NS {
                if let Some(p) = self.tslots(t)[i] {
                    entries.push((self.atk(p), t, i));
                }
            }
        }
        let mut attacks: Vec<i32> = entries.iter().map(|e| e.0).collect();
        attacks.sort_unstable_by(|a, b| b.cmp(a));
        attacks.dedup();
        let mut ordered = Vec::with_capacity(entries.len());
        for attack in attacks {
            let tied: Vec<(usize, usize)> = entries.iter().filter(|e| e.0 == attack).map(|e| (e.1, e.2)).collect();
            if tied.len() > 1 {
                cov!(self, "draw:shop-priority-tie:{}", tied.len());
                let order = self.rs.permutation(tied.len());
                for index in order {
                    ordered.push(tied[index]);
                }
            } else {
                ordered.push(tied[0]);
            }
        }
        ordered
    }

    fn is_post_removal_faint(&self, p: usize) -> bool {
        let ab = self.ability(p);
        ab.trigger == Trigger::Faint
            && matches!(
                ab.effect.kind,
                EffectKind::SummonPet
                    | EffectKind::SummonRandomPet
                    | EffectKind::SummonStoredPets
                    | EffectKind::RespawnPet
            )
    }

    /// `battle_phase_hurt_and_faint(context, "shop", ...)`.
    fn battle_phase_hurt_and_faint(&mut self) -> R<()> {
        let mut records: Vec<Record> = Vec::new();
        let mut recorded: Vec<usize> = Vec::new();
        let mut pending_counters: Vec<(usize, usize)> = Vec::new();
        let mut guard = 0usize;
        loop {
            guard += 1;
            if guard > 100_000 {
                return err("hurt/faint resolution did not terminate");
            }
            let priority = self.update_pet_priority();
            let mut hurt_event = None;
            for &(team_idx, pet_idx) in &priority {
                if let Some(p) = self.tslots(team_idx)[pet_idx] {
                    if self.pets[p].hurt > 0 {
                        hurt_event = Some((team_idx, p));
                        break;
                    }
                }
            }
            if let Some((team_idx, p)) = hurt_event {
                cov!(self, "shop:hurt-event");
                self.hurt_trigger(p)?;
                // check_friend_hurt_triggers
                let snapshot = *self.tslots(team_idx);
                for observer in snapshot.iter().flatten() {
                    self.friend_hurt_trigger(*observer, p, &mut pending_counters)?;
                }
                continue;
            }
            let mut new_dead = None;
            for &(team_idx, pet_idx) in &priority {
                if let Some(p) = self.tslots(team_idx)[pet_idx] {
                    if self.hp(p) <= 0 && !recorded.contains(&p) {
                        new_dead = Some((team_idx, pet_idx, p));
                        break;
                    }
                }
            }
            let (team_idx, pet_idx, p) = match new_dead {
                Some(x) => x,
                None => {
                    if !pending_counters.is_empty() {
                        let (observer, trigger) = pending_counters.remove(0);
                        let t = self.pets[observer].team;
                        if self.check_friend(t, observer) {
                            let idx = self.own_idx(observer)?;
                            let ab = self.ability(observer);
                            self.note_ability(observer, &ab, "FourFriendsHurt");
                            self.tiger_func(false, observer, Idx::Int(idx), true, Te::Pet(trigger), None)?;
                        }
                        continue;
                    }
                    break;
                }
            };
            cov!(self, "shop:faint:{}", self.t.d.pets[self.pets[p].kind].name);
            recorded.push(p);
            let listeners: Vec<usize> =
                self.tslots(team_idx).iter().filter_map(|s| *s).filter(|&q| q != p).collect();
            let record = Record {
                pet: p,
                team_idx,
                pet_idx,
                native_post: self.is_post_removal_faint(p),
                has_status: self.pets[p].status != Status::None,
                listeners,
                post_observers: Vec::new(),
            };
            let native_post = record.native_post;
            records.push(record);
            if !native_post {
                self.faint_trigger(p, p, (team_idx, pet_idx), true, false)?;
            }
        }
        if records.len() > 1 {
            cov!(self, "shop:simultaneous-faints:{}", records.len());
        }

        // FriendAheadFainted / FriendFaints observe before removal.
        for r in 0..records.len() {
            let fainted = records[r].pet;
            let team_idx = records[r].team_idx;
            let pet_idx = records[r].pet_idx;
            let listeners = records[r].listeners.clone();
            for observer in listeners {
                if self.acts_as(observer, self.t.d.shark, Some(self.t.d.fly)) {
                    continue;
                }
                let ot = self.pets[observer].team;
                if !self.check_friend(ot, observer) {
                    continue;
                }
                self.faint_trigger(observer, fainted, (team_idx, pet_idx), true, false)?;
            }
        }

        // Synchronize deaths.
        for record in &records {
            if self.check_friend(record.team_idx, record.pet) {
                self.team_remove(record.team_idx, record.pet)?;
            }
        }
        let mut pending: Vec<Record> = Vec::with_capacity(records.len());
        for mut record in records {
            let team_idx = record.team_idx;
            record.post_observers = self
                .tslots(team_idx)
                .iter()
                .filter_map(|s| *s)
                .filter(|&q| self.acts_as(q, self.t.d.shark, Some(self.t.d.fly)))
                .collect();
            pending.push(record);
        }
        // phase "shop" is not deferred.
        self.flush_post_removal_events(pending)?;
        // battle_obj.pet_priority = update_pet_priority(...): the draw happens, the list is not used.
        self.update_pet_priority();
        Ok(())
    }

    /// `_slot_insert`.
    fn slot_insert(layout: &mut Layout, slot_idx: usize, pet: usize) -> bool {
        let size = NS;
        let target = slot_idx.min(size - 1);
        if layout[target].is_none() {
            layout[target] = Some(pet);
            return true;
        }
        for empty in (0..target).rev() {
            if layout[empty].is_none() {
                for i in empty..target {
                    layout[i] = layout[i + 1];
                }
                layout[target] = Some(pet);
                return true;
            }
        }
        for empty in (target + 1)..size {
            if layout[empty].is_none() {
                for i in ((target + 1)..=empty).rev() {
                    layout[i] = layout[i - 1];
                }
                layout[target] = Some(pet);
                return true;
            }
        }
        false
    }

    /// `_resync_layouts` (+ `_write_layout`).
    fn resync_layouts(&mut self, layouts: &mut [Layout; 2], team_idx: usize, slot_idx: usize, created: &[usize]) {
        for index in 0..2 {
            let present: Vec<usize> = self.tslots(index).iter().filter_map(|s| *s).collect();
            for position in 0..NS {
                if let Some(p) = layouts[index][position] {
                    if !present.contains(&p) {
                        layouts[index][position] = None;
                    }
                }
            }
            let fresh: Vec<usize> =
                present.iter().copied().filter(|p| !layouts[index].iter().any(|s| *s == Some(*p))).collect();
            if index != team_idx {
                if !fresh.is_empty() {
                    layouts[index] = *self.tslots(index);
                }
                continue;
            }
            let mut ordered: Vec<usize> = created.iter().copied().filter(|p| fresh.contains(p)).collect();
            for p in &fresh {
                if !ordered.contains(p) {
                    ordered.push(*p);
                }
            }
            let mut placed_all = true;
            for p in ordered {
                placed_all = Self::slot_insert(&mut layouts[index], slot_idx, p) && placed_all;
            }
            if placed_all {
                *self.tslots_mut(index) = layouts[index];
            } else {
                layouts[index] = *self.tslots(index);
            }
        }
    }

    /// `_flush_post_removal_events`.
    fn flush_post_removal_events(&mut self, pending: Vec<Record>) -> R<()> {
        if pending.is_empty() {
            return Ok(());
        }
        // Stable sort, highest attack first.
        let attacks: Vec<i32> = pending.iter().map(|r| self.atk(r.pet)).collect();
        let mut order: Vec<usize> = (0..pending.len()).collect();
        order.sort_by(|&a, &b| attacks[b].cmp(&attacks[a]));
        let mut taken: Vec<Option<Record>> = pending.into_iter().map(Some).collect();
        let pending: Vec<Record> = order.into_iter().map(|i| taken[i].take().unwrap()).collect();
        let mut layouts: [Layout; 2] = [self.team, self.oteam];

        for record in &pending {
            let pet = record.pet;
            let team_idx = record.team_idx;
            let pet_idx = record.pet_idx;
            if record.native_post {
                let (activated, res) = self.faint_trigger(pet, pet, (team_idx, pet_idx), true, true)?;
                let created = res.targets.clone();
                self.resync_layouts(&mut layouts, team_idx, pet_idx, &created);
                self.check_summon_triggers(pet, team_idx, activated, &res.targets)?;
            }
            if record.has_status {
                self.check_status_triggers(pet, team_idx, pet_idx)?;
                self.resync_layouts(&mut layouts, team_idx, pet_idx, &[]);
            }
        }

        for record in &pending {
            let fainted = record.pet;
            let team_idx = record.team_idx;
            let pet_idx = record.pet_idx;
            for &observer in &record.post_observers {
                if !self.check_friend(team_idx, observer) || self.hp(observer) <= 0 {
                    continue;
                }
                let (activated, res) = self.faint_trigger(observer, fainted, (team_idx, pet_idx), true, true)?;
                let created = res.targets.clone();
                self.resync_layouts(&mut layouts, team_idx, pet_idx, &created);
                self.check_summon_triggers(observer, team_idx, activated, &res.targets)?;
            }
        }
        Ok(())
    }

    /// `battle.check_summon_triggers` with `target_team.battle == False`.
    fn check_summon_triggers(&mut self, p: usize, team_idx: usize, activated: bool, targets: &[usize]) -> R<()> {
        if !activated {
            return Ok(());
        }
        let ab = self.ability(p);
        if !matches!(
            ab.effect.kind,
            EffectKind::RespawnPet
                | EffectKind::SummonPet
                | EffectKind::SummonRandomPet
                | EffectKind::SummonStoredPets
        ) {
            return Ok(());
        }
        let target_team = match ab.effect.team {
            Some(TeamSel::Friendly) | None => team_idx,
            Some(_) => 1 - team_idx,
        };
        if target_team != team_idx {
            return Ok(());
        }
        for &te in targets {
            cov!(self, "shop-summon:{}", self.t.d.pets[self.pets[te].kind].name);
            // target_team.player is the seat for its own team, None for the stand-in team.
            if target_team == 0
                && self.pets[te].kind == self.t.e.scorpion
                && self.pets[te].status == Status::PoisonAttack
            {
                cov!(self, "shop-summon:scorpion-perk-notify");
                self.notify_friend_gained_perk(te)?;
            }
            for observer_idx in 0..NS {
                if let Some(observer) = self.tslots(target_team)[observer_idx] {
                    self.friend_summoned_trigger(observer, te)?;
                }
            }
        }
        Ok(())
    }

    /// `battle.check_status_triggers` (Honey, Mushroom).
    fn check_status_triggers(&mut self, p: usize, team_idx: usize, pet_idx: usize) -> R<()> {
        let t: &'static Tables = self.t;
        let ability = match self.pets[p].status {
            Status::HoneyBee => t.d.honey_ability.clone(),
            Status::ExtraLife => t.d.extra_life_ability.clone(),
            _ => return Ok(()),
        };
        cov!(self, "shop:faint-with-{}", self.pets[p].status.name());
        self.pets[p].over = Some(ability);
        let (activated, res) = self.faint_trigger(p, p, (team_idx, pet_idx), false, true)?;
        self.check_summon_triggers(p, team_idx, activated, &res.targets)
    }

    // ------------------------------------------------------------------
    // Effects (sapai/effects.py)
    // ------------------------------------------------------------------

    fn run_eff(
        &mut self,
        eff: &Effect,
        apet: usize,
        apet_idx: Idx,
        has_oteam: bool,
        te: Te,
        te_idx: Option<(usize, usize)>,
    ) -> R<EffResult> {
        match eff.kind {
            EffectKind::ModifyStats => self.modify_stats(eff, apet, apet_idx, has_oteam, te),
            EffectKind::DealDamage => self.deal_damage(eff, apet, apet_idx, has_oteam, te),
            EffectKind::SummonPet => self.summon_pet(eff, apet, has_oteam, te_idx),
            EffectKind::SummonRandomPet => self.summon_random_pet(eff, apet, te_idx),
            EffectKind::RespawnPet => self.respawn_pet(eff, apet, te_idx),
            EffectKind::ApplyStatus => self.apply_status(eff, apet, apet_idx, has_oteam, te),
            EffectKind::AllOf => {
                let mut targets = Vec::new();
                for sub in &eff.effects {
                    let r = self.run_eff(sub, apet, apet_idx, has_oteam, te, None)?;
                    targets.extend(r.targets);
                }
                Ok(EffResult::of(targets))
            }
            EffectKind::TransferAbility => self.transfer_ability(eff, apet, apet_idx, has_oteam, te),
            EffectKind::NoneKind => Ok(EffResult::none()),
            EffectKind::Other => match eff.shop_kind {
                ShopEffect::GainGold => {
                    match eff.amount {
                        Amount::Int(v) => self.gold += v as i32,
                        _ => return err("GainGold amount not understood"),
                    }
                    Ok(EffResult::none())
                }
                ShopEffect::StockFood => self.stock_food(eff),
                ShopEffect::RefillShops => self.refill_shops(eff, apet),
                ShopEffect::DiscountFood => {
                    let amount = match eff.amount {
                        Amount::Int(v) => v as i32,
                        _ => return err("DiscountFood amount not understood"),
                    };
                    for slot in self.slots.iter_mut() {
                        if slot.ty == SlotType::Food {
                            slot.cost = (slot.cost - amount).max(0);
                        }
                    }
                    Ok(EffResult::none())
                }
                ShopEffect::FoodMultiplier | ShopEffect::None => {
                    err(format!("Unsupported: effect {} in the shop phase", eff.kind_name))
                }
            },
            // Battle-only effects (their triggers never fire in the shop).
            EffectKind::SummonStoredPets
            | EffectKind::TransferStats
            | EffectKind::ReduceHealth
            | EffectKind::Swallow
            | EffectKind::GainHealthPercentHighestFriend => {
                err(format!("Unsupported: effect {} in the shop phase", eff.kind_name))
            }
        }
    }

    /// `RandomFriend` of `get_target`; `apet == None` is a food.
    fn random_friend(&mut self, apet: Option<usize>, apet_idx: Idx, n: i64) -> R<Vec<usize>> {
        let fidx: Vec<usize> =
            (0..NS).filter(|&i| self.team[i].map(|p| self.hp(p) > 0).unwrap_or(false)).collect();
        if fidx.is_empty() {
            return Ok(Vec::new());
        }
        let pool: Vec<usize> = fidx.iter().copied().filter(|&i| apet_idx != Idx::Int(i)).collect();
        let n = n.min(pool.len() as i64);
        if n <= 0 {
            return Ok(Vec::new());
        }
        let combos = combinations(&pool, n as usize);
        if combos.is_empty() {
            return Ok(Vec::new());
        }
        cov!(self, "draw:random-friend:{}-of-{}", n, pool.len());
        let c = self.stream(apet)?.choice_noreplace(combos.len());
        Ok(combos[c].iter().map(|&i| self.team[i].unwrap()).collect())
    }

    /// `get_target`.
    fn get_target(
        &mut self,
        apet: usize,
        apet_idx: Idx,
        eff: &Effect,
        has_oteam: bool,
        te: Te,
        get_from: bool,
    ) -> R<Vec<usize>> {
        let target = if get_from {
            match &eff.from {
                Some(t) => t,
                None => return err("from not found in effect"),
            }
        } else if let Some(t) = &eff.target {
            t
        } else if let Some(t) = &eff.to {
            t
        } else {
            return err("Target not found");
        };
        let mut n = target.n.unwrap_or(1);
        let ft = self.pets[apet].team;
        let ot = 1 - ft;
        let fidx: Vec<usize> =
            (0..NS).filter(|&i| self.tslots(ft)[i].map(|p| self.hp(p) > 0).unwrap_or(false)).collect();
        let oidx: Vec<usize> = if has_oteam {
            (0..NS).filter(|&i| self.tslots(ot)[i].map(|p| self.hp(p) > 0).unwrap_or(false)).collect()
        } else {
            Vec::new()
        };
        let fpet = |s: &Self, i: usize| s.tslots(ft)[i].unwrap();
        let opet = |s: &Self, i: usize| s.tslots(ot)[i].unwrap();
        let int_idx = |idx: Idx| -> R<usize> {
            match idx {
                Idx::Int(i) => Ok(i),
                _ => err("target needs the activating pet's slot, but it has left the team"),
            }
        };
        if self.cov.is_some() {
            cov!(self, "target:{}", target.kind_name);
        }
        match target.kind {
            TargetKind::AdjacentAnimals => {
                let mut all: Vec<Option<usize>> = Vec::with_capacity(10);
                if has_oteam {
                    for i in (0..NS).rev() {
                        all.push(self.tslots(ot)[i]);
                    }
                }
                let base = all.len();
                for i in 0..NS {
                    all.push(self.tslots(ft)[i]);
                }
                let me = base + int_idx(apet_idx)?;
                let alive = |s: &Self, c: Option<usize>| c.map(|p| s.hp(p) > 0).unwrap_or(false);
                let mut out = Vec::new();
                for c in (0..me).rev() {
                    if alive(self, all[c]) {
                        out.push(all[c].unwrap());
                        break;
                    }
                }
                for c in (me + 1)..all.len() {
                    if alive(self, all[c]) {
                        out.push(all[c].unwrap());
                        break;
                    }
                }
                Ok(out)
            }
            TargetKind::All => {
                let mut out: Vec<usize> = fidx.iter().map(|&i| fpet(self, i)).collect();
                out.extend(oidx.iter().map(|&i| opet(self, i)));
                Ok(out)
            }
            TargetKind::AllOtherFriends => Ok(fidx.iter().map(|&i| fpet(self, i)).filter(|&p| p != apet).collect()),
            TargetKind::EachEnemy => Ok(oidx.iter().map(|&i| opet(self, i)).collect()),
            TargetKind::EachFriend => {
                Ok(fidx.iter().filter(|&&i| apet_idx != Idx::Int(i)).map(|&i| fpet(self, i)).collect())
            }
            // apet.shop is the seat's shop for every pet a shop effect can activate.
            TargetKind::EachShopAnimal => Ok(self.shop_pets()),
            TargetKind::FirstEnemy => Ok(oidx.first().map(|&i| vec![opet(self, i)]).unwrap_or_default()),
            TargetKind::LastEnemy => Ok(oidx.last().map(|&i| vec![opet(self, i)]).unwrap_or_default()),
            TargetKind::FriendAhead => {
                let me = int_idx(apet_idx)?;
                let mut out = Vec::new();
                for &i in fidx.iter().rev() {
                    if i < me {
                        out.push(fpet(self, i));
                        if out.len() as i64 >= n {
                            break;
                        }
                    }
                }
                Ok(out)
            }
            TargetKind::FriendBehind => {
                let me = int_idx(apet_idx)?;
                let mut out = Vec::new();
                for &i in fidx.iter() {
                    if i > me {
                        let p = fpet(self, i);
                        if let Some(ex) = target.exclude_status {
                            if self.pets[p].status == ex {
                                continue;
                            }
                        }
                        out.push(p);
                        if out.len() as i64 >= n {
                            break;
                        }
                    }
                }
                Ok(out)
            }
            TargetKind::HighestHealthEnemy | TargetKind::LowestHealthEnemy => {
                if oidx.is_empty() {
                    return Ok(Vec::new());
                }
                let healths: Vec<i32> = oidx.iter().map(|&i| self.hp(opet(self, i))).collect();
                let pick = if target.kind == TargetKind::HighestHealthEnemy {
                    *healths.iter().max().unwrap()
                } else {
                    *healths.iter().min().unwrap()
                };
                let range: Vec<usize> = (0..healths.len()).filter(|&k| healths[k] == pick).collect();
                let choice = range[self.stream(Some(apet))?.choice_noreplace(range.len())];
                Ok(vec![opet(self, oidx[choice])])
            }
            TargetKind::RandomEnemy => {
                if oidx.is_empty() {
                    return Ok(Vec::new());
                }
                if (oidx.len() as i64) < n {
                    n = oidx.len() as i64;
                }
                let combos = combinations(&oidx, n.max(0) as usize);
                if combos.is_empty() {
                    return Ok(Vec::new());
                }
                let c = self.stream(Some(apet))?.choice_noreplace(combos.len());
                Ok(combos[c].iter().map(|&i| opet(self, i)).collect())
            }
            TargetKind::RandomFriend => {
                if ft != 0 {
                    return err("Unsupported: RandomFriend for a pet on the stand-in team");
                }
                self.random_friend(Some(apet), apet_idx, n)
            }
            TargetKind::FrontMostFriend => Ok(fidx.first().map(|&i| vec![fpet(self, i)]).unwrap_or_default()),
            TargetKind::SelfPet => Ok(vec![apet]),
            TargetKind::StrongestFriend => err("Unsupported: StrongestFriend in the shop phase"),
            TargetKind::TriggeringEntity => match te {
                Te::Pet(p) => Ok(vec![p]),
                Te::None => Ok(Vec::new()),
                Te::Team => err("TriggeringEntity is a team"),
            },
            TargetKind::NoneKind => Ok(Vec::new()),
            TargetKind::Other => {
                if !target.level23 {
                    return err(format!("Target {} implementation not found", target.kind_name));
                }
                // Level2And3Friends (Penguin).
                let me = int_idx(apet_idx).ok();
                let mut ret: Vec<usize> = Vec::new();
                for &i in &fidx {
                    let level = if Some(i) == me { 0 } else { self.pets[fpet(self, i)].level };
                    if level > 1 {
                        ret.push(fpet(self, i));
                    }
                }
                if fidx.is_empty() {
                    return Ok(Vec::new());
                }
                let n = match target.n {
                    Some(n) => n.max(0) as usize,
                    None => ret.len(),
                };
                if ret.len() > n {
                    cov!(self, "draw:level2and3-friends:{}-of-{}", n, ret.len());
                    let len = ret.len();
                    let order = self.stream(Some(apet))?.permutation(len);
                    ret = order[..n].iter().map(|&k| ret[k]).collect();
                }
                Ok(ret)
            }
        }
    }

    /// `ModifyStats`.
    fn modify_stats(&mut self, eff: &Effect, apet: usize, apet_idx: Idx, has_oteam: bool, te: Te) -> R<EffResult> {
        let target = self.get_target(apet, apet_idx, eff, has_oteam, te, false)?;
        let mut attack_amount = eff.attack_amount.unwrap_or(0) as i32;
        let health_amount = eff.health_amount.unwrap_or(0) as i32;
        match eff.amount {
            Amount::AttackPercent(p) => {
                attack_amount = (((self.atk(apet) as i64 * p) as f64) * 0.01) as i32;
            }
            Amount::AttackDamagePercent(_) | Amount::Other => return err("ModifyStats amount dict not understood"),
            _ => {}
        }
        let including_future = match &eff.target {
            Some(t) => t.including_future,
            None => return err("ModifyStats without a target entry"),
        };
        for &tp in &target {
            let pet = &mut self.pets[tp];
            if pet.health <= 0 {
                continue;
            }
            if eff.until_end_of_battle {
                pet.tmp_attack = (pet.tmp_attack + attack_amount).min((50 - pet.attack).max(0));
                pet.tmp_health = (pet.tmp_health + health_amount).min((50 - pet.health).max(0));
            } else {
                pet.attack += attack_amount;
                if health_amount < 0 {
                    if pet.health > 0 {
                        pet.health = (pet.health + health_amount).max(1);
                    }
                } else {
                    pet.health += health_amount;
                }
            }
            if including_future {
                self.shop_attack += attack_amount;
                self.shop_health += health_amount;
            }
            let pet = &mut self.pets[tp];
            pet.attack = pet.attack.min(50);
            pet.health = pet.health.min(50);
        }
        Ok(EffResult::of(target))
    }

    /// `DealDamage`.
    fn deal_damage(&mut self, eff: &Effect, apet: usize, apet_idx: Idx, has_oteam: bool, te: Te) -> R<EffResult> {
        let target = self.get_target(apet, apet_idx, eff, has_oteam, te, false)?;
        let base = match eff.amount {
            Amount::Int(v) => v as i32,
            Amount::AttackDamagePercent(p) => (((self.atk(apet) as i64 * p) as f64) * 0.01) as i32,
            _ => return err("DealDamage amount not understood"),
        };
        for &tp in &target {
            let mut amount = base;
            if eff.double_against_tier.is_some() && eff.double_against_tier == self.t.d.pets[self.pets[tp].kind].tier {
                amount *= 2;
            }
            match self.pets[tp].status {
                Status::MelonArmor => {
                    amount = (amount - 20).max(0);
                    self.pets[tp].status = Status::None;
                }
                Status::GarlicArmor => {
                    if amount > 2 {
                        amount = (amount - 2).max(2);
                    }
                }
                Status::CoconutShield => {
                    amount = 0;
                    self.pets[tp].status = Status::None;
                }
                Status::Weak => amount += 3,
                _ => {}
            }
            self.hurt(tp, amount);
        }
        Ok(EffResult::of(target))
    }

    /// `ApplyStatus` outside a battle: a perk notifies `Player._notify_friend_gained_perk`.
    fn apply_status(&mut self, eff: &Effect, apet: usize, apet_idx: Idx, has_oteam: bool, te: Te) -> R<EffResult> {
        let target = self.get_target(apet, apet_idx, eff, has_oteam, te, false)?;
        let status = match eff.status {
            Some(s) => s,
            None => return err("ApplyStatus without status"),
        };
        let spec = eff.to.as_ref().or(eff.target.as_ref());
        let exclude = spec.and_then(|s| s.exclude_status);
        let mut applied = Vec::new();
        for &tp in &target {
            if let Some(ex) = exclude {
                if self.pets[tp].status == ex {
                    continue;
                }
            }
            self.pets[tp].status = status;
            applied.push(tp);
            // target_pet.player is None only for pets of the stand-in team.
            if status.is_perk() && self.pets[tp].team == 0 {
                cov!(self, "perk-gained-from-ability:{}", status.name());
                self.notify_friend_gained_perk(tp)?;
            }
        }
        Ok(EffResult::of(applied))
    }

    /// `TransferAbility` (Parrot, end of turn).
    fn transfer_ability(&mut self, eff: &Effect, apet: usize, apet_idx: Idx, has_oteam: bool, te: Te) -> R<EffResult> {
        let target = self.get_target(apet, apet_idx, eff, has_oteam, te, true)?;
        if let Some(&source) = target.first() {
            let t: &'static Tables = self.t;
            let mut origin = self.pets[source].kind;
            if let Some(over) = &self.pets[source].over {
                if let Some(c) = over.copied_from {
                    origin = c;
                    cov!(self, "parrot:copies-a-parrot's-copy");
                }
            }
            let level = self.pets[apet].level.clamp(1, 3) as usize;
            // native.get(f"level{apet.level}Ability", source.ability)
            let mut copied: Ability = match &t.d.pets[origin].abilities[level - 1] {
                Some(a) => (**a).clone(),
                None => (*self.ability(source)).clone(),
            };
            copied.copied_from = Some(origin);
            cov!(self, "parrot:copied:{}", t.d.pets[origin].name);
            self.pets[apet].over = Some(Arc::new(copied));
        }
        Ok(EffResult::of(target))
    }

    /// `StockFood` (Worm, Pigeon).
    fn stock_food(&mut self, eff: &Effect) -> R<EffResult> {
        let requested = eff.n.unwrap_or(1).max(0) as usize;
        let free_slots = MAX_SHOP_SLOTS.saturating_sub(self.slots.len());
        let removable: Vec<usize> = (0..self.slots.len()).rev().filter(|&i| !self.slots[i].frozen).collect();
        let generated = requested.min(free_slots + removable.len());
        let overflow = (self.slots.len() + generated).saturating_sub(MAX_SHOP_SLOTS);
        if overflow > 0 {
            cov!(self, "shop:stock-overflow-removes:{}", overflow);
            let remove: Vec<usize> = removable[..overflow].to_vec();
            let mut index = 0usize;
            self.slots.retain(|_| {
                let keep = !remove.contains(&index);
                index += 1;
                keep
            });
        }
        if generated < requested {
            cov!(self, "shop:stock-short:{}-of-{}", generated, requested);
        }
        let kind = match eff.food_name.as_ref().and_then(|n| self.t.e.food_by_name.get(n)) {
            Some(k) => *k,
            None => return err("StockFood without a known food"),
        };
        for _ in 0..generated {
            // Food(name, shop=shop): no seed state, so a MockRandomState until the next _connect_objects.
            let mut food = self.new_food(kind, Rs::Mock);
            if let Some(v) = eff.attack_amount {
                food.attack = v as i32;
                food.base_attack = food.attack;
            }
            if let Some(v) = eff.health_amount {
                food.health = v as i32;
                food.base_health = food.health;
            }
            if let Some(v) = eff.cost {
                food.cost = v as i32;
            }
            cov!(self, "shop:stocked:{}", self.t.e.foods[kind].name);
            self.slots.push(ShopSlot {
                ty: SlotType::Food,
                frozen: false,
                cost: food.cost,
                reward_group: None,
                obj: Obj::Food(food),
            });
        }
        Ok(EffResult::none())
    }

    /// `RefillShops` (Cow).
    fn refill_shops(&mut self, eff: &Effect, apet: usize) -> R<EffResult> {
        let level = self.pets[apet].level;
        let kind = match self.t.e.food_by_name.get(eff.food_name.as_deref().unwrap_or("food-milk")) {
            Some(k) => *k,
            None => return err("RefillShops without a known food"),
        };
        let count = eff.n.unwrap_or(2).max(0) as usize;
        let mut surviving: Vec<ShopSlot> =
            std::mem::take(&mut self.slots).into_iter().filter(|s| s.ty != SlotType::Food).collect();
        let missing = (surviving.len() + count).saturating_sub(MAX_SHOP_SLOTS);
        let removable: Vec<usize> = (0..surviving.len()).rev().filter(|&i| !surviving[i].frozen).collect();
        let remove: Vec<usize> = removable.iter().copied().take(missing).collect();
        if !remove.is_empty() {
            cov!(self, "shop:cow-overflow-removes:{}", remove.len());
            let mut index = 0usize;
            surviving.retain(|_| {
                let keep = !remove.contains(&index);
                index += 1;
                keep
            });
        }
        let room = MAX_SHOP_SLOTS.saturating_sub(surviving.len());
        cov!(self, "shop:cow-milks-stocked:{}-of-{}", count.min(room), count);
        for _ in 0..count.min(room) {
            let mut food = self.new_food(kind, Rs::Mock);
            food.attack = food.base_attack * level;
            food.health = food.base_health * level;
            food.cost = 0;
            surviving.push(ShopSlot { ty: SlotType::Food, frozen: false, cost: 0, reward_group: None, obj: Obj::Food(food) });
        }
        self.slots = surviving;
        Ok(EffResult::none())
    }

    /// The team shuffle shared by the friendly summons.
    fn open_summon_gap(&mut self, ft: usize, te_slot: usize) -> R<()> {
        let nahead = self.get_ahead(ft, te_slot, 5).len();
        let npets = self.team_len(ft);
        self.move_backward(ft)?;
        let end_idx = (NS - npets) + nahead;
        self.move_forward(ft, 0, end_idx)
    }

    fn max_empty(&self, t: usize) -> Option<usize> {
        (0..NS).rev().find(|&i| self.tslots(t)[i].is_none())
    }

    /// The generator of a pet placed with `team[idx] = name`.
    fn summoned_rs(&self, team: usize) -> Rs {
        if team == 0 && !self.team_seed_none {
            Rs::TeamSeed
        } else {
            Rs::Mock
        }
    }

    /// `SummonPet`.
    fn summon_pet(&mut self, eff: &Effect, apet: usize, has_oteam: bool, te_idx: Option<(usize, usize)>) -> R<EffResult> {
        let te_idx = match te_idx {
            Some(t) => t,
            None => return err("Indices of triggering entity must be provided as te_idx"),
        };
        let spet_kind = match eff.pet {
            Some(k) => k,
            None => return err("SummonPet without a known pet"),
        };
        let ft = self.pets[apet].team;
        let friendly = eff.team == Some(TeamSel::Friendly);
        let target_team = match eff.team {
            Some(TeamSel::Friendly) => {
                self.open_summon_gap(ft, te_idx.1)?;
                ft
            }
            Some(TeamSel::Enemy) => {
                if !has_oteam {
                    return Ok(EffResult::none());
                }
                self.move_backward(1 - ft)?;
                1 - ft
            }
            _ => return err("SummonPet team not understood"),
        };
        let mut count = eff.n.unwrap_or(1);
        if eff.n.is_none() {
            if spet_kind == self.t.d.ram {
                count = 2;
            } else if spet_kind == self.t.d.chick {
                count = self.pets[apet].level as i64;
            }
        }
        let mut target = Vec::new();
        for _ in 0..count {
            let slot = match self.max_empty(target_team) {
                Some(s) => s,
                None => {
                    let empty = target.is_empty();
                    return Ok(EffResult { targets: target, possible_empty_nested: empty });
                }
            };
            let rs = self.summoned_rs(target_team);
            let sp = self.new_pet(spet_kind, target_team, rs);
            self.tslots_mut(target_team)[slot] = Some(sp);
            if let Some(v) = eff.with_attack {
                self.pets[sp].attack = v as i32;
            }
            if let Some(v) = eff.with_health {
                self.pets[sp].health = v as i32;
            }
            if let Some(v) = eff.with_level {
                self.pets[sp].level = v as i32;
            }
            // A summoned token sells for its summoner's level; a Honey Bee for 1.
            if friendly && spet_kind != self.t.d.bee {
                let apet_level = self.pets[apet].level;
                let s = &mut self.pets[sp];
                s.sell_value = s.sell_value.max(s.level).max(apet_level);
            }
            if spet_kind == self.t.d.chick {
                self.pets[sp].attack = (self.atk(apet) as f64 * 0.5) as i32;
            }
            if self.pets[sp].attack == UNRESOLVED || self.pets[sp].health == UNRESOLVED {
                return err("SummonPet produced an unresolved stat");
            }
            target.push(sp);
        }
        self.move_forward(target_team, 0, 10)?;
        let empty = target.is_empty();
        Ok(EffResult { targets: target, possible_empty_nested: empty })
    }

    /// `SummonRandomPet` (Spider).
    fn summon_random_pet(&mut self, eff: &Effect, apet: usize, te_idx: Option<(usize, usize)>) -> R<EffResult> {
        let ft = self.pets[apet].team;
        let tier = match eff.tier {
            Some(t) => t,
            None => return err("SummonRandomPet without tier"),
        };
        let t: &'static Tables = self.t;
        let possible = match t.d.tier_std.get(&tier) {
            Some(p) if !p.is_empty() => p,
            _ => return err("no pets for SummonRandomPet tier"),
        };
        // The draw comes before the room check.
        let chosen = possible[self.stream(Some(apet))?.choice_replace(possible.len())];
        let te_idx = match te_idx {
            Some(t) => t,
            None => return err("SummonRandomPet without te_idx"),
        };
        self.open_summon_gap(ft, te_idx.1)?;
        let slot = match self.max_empty(ft) {
            Some(s) => s,
            None => return Ok(EffResult::none()),
        };
        let rs = self.summoned_rs(ft);
        let sp = self.new_pet(chosen, ft, rs);
        self.tslots_mut(ft)[slot] = Some(sp);
        let sattack = match eff.base_attack {
            Some(v) => v as i32,
            None => self.pets[sp].attack,
        };
        let shealth = match eff.base_health {
            Some(v) => v as i32,
            None => self.pets[sp].health,
        };
        if let Some(l) = eff.level {
            let s = &mut self.pets[sp];
            s.level = l as i32;
            s.sell_value = s.sell_value.max(s.level);
        }
        self.pets[sp].attack = sattack;
        self.pets[sp].health = shealth;
        self.move_forward(ft, 0, 10)?;
        Ok(EffResult::of(vec![sp]))
    }

    /// `RespawnPet` (Mushroom).
    fn respawn_pet(&mut self, eff: &Effect, apet: usize, te_idx: Option<(usize, usize)>) -> R<EffResult> {
        let te_idx = match te_idx {
            Some(t) => t,
            None => return err("Indices of triggering entity must be provided as te_idx"),
        };
        let ft = self.pets[apet].team;
        self.open_summon_gap(ft, te_idx.1)?;
        let slot = match self.max_empty(ft) {
            Some(s) => s,
            None => return Ok(EffResult { targets: Vec::new(), possible_empty_nested: true }),
        };
        let kind = self.pets[apet].kind;
        let rs = self.summoned_rs(ft);
        let sp = self.new_pet(kind, ft, rs);
        self.tslots_mut(ft)[slot] = Some(sp);
        if let Some(v) = eff.base_attack {
            self.pets[sp].attack = v as i32;
        }
        if let Some(v) = eff.base_health {
            self.pets[sp].health = v as i32;
        }
        self.pets[sp].level = self.pets[apet].level;
        self.move_forward(ft, 0, 10)?;
        Ok(EffResult::of(vec![sp]))
    }

    // ------------------------------------------------------------------
    // SuperAutoPetsEnv: legality and action resolution
    // ------------------------------------------------------------------

    /// `_team_order_key` as the five identities (0 for an empty slot).
    fn team_order(&self) -> [u32; NS] {
        let mut key = [0u32; NS];
        for i in 0..NS {
            if let Some(p) = self.team[i] {
                key[i] = self.pets[p].uid;
            }
        }
        key
    }

    /// `_team_order_key`.
    fn team_order_key(&self) -> u128 {
        pack_order(&self.team_order())
    }

    /// Forget the cached legal actions; called by everything that changes the seat.
    #[inline]
    fn touch(&mut self) {
        self.legal = None;
    }

    fn obj_uid(&self, slot: &ShopSlot) -> u32 {
        match &slot.obj {
            Obj::Pet(p) => self.pets[*p].uid,
            Obj::Food(f) => f.uid,
            Obj::Empty => 0,
        }
    }

    /// `SuperAutoPetsEnv._avail_actions(agent_idx)`, keys only. Registers the
    /// current team order as seen, as `_avail_move` does.
    pub fn avail_actions(&mut self) -> ActionSet {
        if let Some(set) = self.legal {
            return set;
        }
        let set = self.compute_avail_actions();
        self.legal = Some(set);
        set
    }

    fn compute_avail_actions(&mut self) -> ActionSet {
        let mut set = ActionSet::default();
        if self.action_count >= SHOP_ACTION_CAP {
            // Safety valve: END_TURN is forced legal after the hard action cap.
            set.add(A_END_TURN);
            return set;
        }
        // _avail_end_turn
        if self.gold <= 0 {
            set.add(A_END_TURN);
        }
        // _avail_buy_pets
        for (shop_idx, slot) in self.slots.iter().enumerate().take(MAX_SHOP_SLOTS) {
            if !matches!(slot.ty, SlotType::Pet | SlotType::Levelup) || slot.cost > self.gold {
                continue;
            }
            let shop_kind = match slot.obj {
                Obj::Pet(p) => self.pets[p].kind,
                _ => continue,
            };
            for team_idx in 0..NS {
                match self.team[team_idx] {
                    None => set.add(A_BUY_PET + shop_idx * 6 + team_idx),
                    Some(q) => {
                        if self.pets[q].kind == shop_kind && self.pets[q].level < 3 {
                            set.add(A_BUY_PET + shop_idx * 6 + team_idx);
                        }
                    }
                }
            }
        }
        // _avail_buy_foods
        let team_count = self.team_len(0);
        if team_count > 0 {
            for (shop_idx, slot) in self.slots.iter().enumerate().take(MAX_SHOP_SLOTS) {
                if slot.ty != SlotType::Food || slot.cost > self.gold {
                    continue;
                }
                let kind = match &slot.obj {
                    Obj::Food(f) => f.kind,
                    _ => continue,
                };
                if self.t.e.foods[kind].no_target {
                    set.add(A_BUY_FOOD + shop_idx * 6 + NS);
                } else {
                    for team_idx in 0..NS {
                        if let Some(q) = self.team[team_idx] {
                            // Pet.can_eat
                            if !(kind == self.t.e.chocolate && self.pets[q].level >= 3) {
                                set.add(A_BUY_FOOD + shop_idx * 6 + team_idx);
                            }
                        }
                    }
                }
            }
        }
        // _avail_team_combine
        if team_count > 1 {
            for source in 0..NS {
                let sp = match self.team[source] {
                    Some(p) => p,
                    None => continue,
                };
                for target in 0..NS {
                    if source == target {
                        continue;
                    }
                    let tp = match self.team[target] {
                        Some(p) => p,
                        None => continue,
                    };
                    if self.pets[sp].kind != self.pets[tp].kind {
                        continue;
                    }
                    if self.pets[sp].level >= 3 || self.pets[tp].level >= 3 {
                        continue;
                    }
                    set.add(A_COMBINE + source * 5 + target);
                }
            }
        }
        // _avail_sell
        for team_idx in 0..NS {
            if self.team[team_idx].is_some() {
                set.add(A_SELL + team_idx);
            }
        }
        // _avail_roll
        if self.gold >= 1 {
            set.add(A_ROLL);
        }
        // _avail_freeze / _avail_unfreeze
        for (shop_idx, slot) in self.slots.iter().enumerate().take(MAX_SHOP_SLOTS) {
            if matches!(slot.obj, Obj::Empty) {
                continue;
            }
            if self.toggled.contains(&self.obj_uid(slot)) {
                continue;
            }
            if slot.frozen {
                set.add(A_UNFREEZE + shop_idx);
            } else {
                set.add(A_FREEZE + shop_idx);
            }
        }
        // _avail_move
        if self.reorder_count < MAX_REORDERS_PER_TURN {
            let current = self.team_order();
            let packed = pack_order(&current);
            if !self.seen.contains(&packed) {
                self.seen.push(packed);
            }
            for source in 0..NS {
                if self.team[source].is_none() {
                    continue;
                }
                for target in 0..NS {
                    if source == target {
                        continue;
                    }
                    let mut order = current;
                    let moved = order[source];
                    if source < target {
                        for i in source..target {
                            order[i] = order[i + 1];
                        }
                    } else {
                        for i in ((target + 1)..=source).rev() {
                            order[i] = order[i - 1];
                        }
                    }
                    order[target] = moved;
                    if self.seen.contains(&pack_order(&order)) {
                        continue;
                    }
                    set.add(A_MOVE + source * 5 + target);
                }
            }
        }
        set
    }

    /// `SuperAutoPetsEnv.resolve_action(agent_idx, action)`.
    pub fn resolve_action(&mut self, flat: usize) -> R<()> {
        if !self.avail_actions().has(flat) {
            return err(format!("Agent {} attempted invalid action {flat}.", self.seat));
        }
        let [ty, source, target] = decode_action(flat);
        let (source, target) = (source as usize, target as usize);
        self.touch();
        let mut freeze_key = None;
        let mut is_roll = false;
        let mut is_swap = false;
        match ty {
            T_END_TURN => self.end_turn()?,
            T_ROLL => {
                self.roll()?;
                is_roll = true;
            }
            T_BUY_PET => {
                if self.team[target].is_none() {
                    self.buy_pet(source, target)?;
                } else {
                    self.buy_combine(source, target)?;
                }
            }
            T_BUY_FOOD => {
                if target == NS {
                    self.buy_food(source, None)?;
                } else {
                    self.buy_food(source, Some(target))?;
                }
            }
            // The destination pet is retained: player.combine(target_idx, source_idx).
            T_COMBINE => self.combine(target, source)?,
            T_SELL => self.sell(source)?,
            T_FREEZE => {
                freeze_key = Some(self.obj_uid(&self.slots[source]));
                self.slots[source].frozen = true;
            }
            T_UNFREEZE => {
                freeze_key = Some(self.obj_uid(&self.slots[source]));
                self.slots[source].frozen = false;
            }
            T_MOVE => {
                self.swap(source, target)?;
                is_swap = true;
            }
            _ => return err("unknown action type"),
        }
        if let Some(key) = freeze_key {
            if !self.toggled.contains(&key) {
                self.toggled.push(key);
            }
        } else if is_roll {
            self.toggled.clear();
        }
        self.action_count += 1;
        if self.action_count == SHOP_ACTION_CAP {
            cov!(self, "env:action-cap-reached");
        }
        if is_swap {
            self.reorder_count += 1;
            if self.reorder_count == MAX_REORDERS_PER_TURN {
                cov!(self, "env:reorder-limit-reached");
            }
            let key = self.team_order_key();
            if !self.seen.contains(&key) {
                self.seen.push(key);
            }
        }
        Ok(())
    }

    /// `end_freeze.expand_end_freeze(core, side, mask_id)`: the validated
    /// primitive plan (flat ids), nothing executed.
    pub fn expand_end_freeze(&mut self, mask_id: u32) -> R<Vec<usize>> {
        if mask_id >= 1 << MAX_SHOP_SLOTS {
            return err("Invalid freeze subset ID");
        }
        if self.turn_ended {
            return err("Player already ended this shop");
        }
        let indices: Vec<usize> = (0..MAX_SHOP_SLOTS).filter(|i| mask_id & (1 << i) != 0).collect();
        if indices.len() as i32 > (SHOP_ACTION_CAP - self.action_count).max(0) {
            return err("Freeze plan exceeds remaining action budget");
        }
        if !indices.is_empty() && self.gold != 0 {
            return err("End-of-turn freeze plans require zero gold");
        }
        if indices.iter().any(|&i| i >= self.slots.len() || matches!(self.slots[i].obj, Obj::Empty)) {
            return err("Freeze plan references an empty shop position");
        }
        let legal = self.avail_actions();
        if !legal.has(A_END_TURN) {
            return err("END_TURN is not currently legal");
        }
        let mut plan = Vec::with_capacity(indices.len() + 1);
        for index in indices {
            let action = if self.slots[index].frozen { A_UNFREEZE + index } else { A_FREEZE + index };
            if !legal.has(action) {
                return err("Freeze plan contains an illegal toggle");
            }
            plan.push(action);
        }
        plan.push(A_END_TURN);
        Ok(plan)
    }

    /// The submitted board as the other seat will see it.
    fn board(&self) -> R<Board> {
        let mut out: Board = [None; NS];
        for i in 0..NS {
            if let Some(p) = self.team[i] {
                let pet = &self.pets[p];
                out[i] = Some(BoardPet {
                    kind: pet.kind,
                    attack: self.atk(p),
                    health: self.hp(p),
                    level: pet.level,
                    total_experience: Self::total_experience(pet)?,
                    status: pet.status,
                    tmp_attack: pet.tmp_attack > 0,
                    tmp_health: pet.tmp_health > 0,
                });
            }
        }
        Ok(out)
    }

    /// The team as `Battle.__init__` copies it.
    fn battle_team(&self) -> [Option<EnvPet>; NS] {
        let mut out: [Option<EnvPet>; NS] = [None, None, None, None, None];
        for i in 0..NS {
            if let Some(p) = self.team[i] {
                let pet = &self.pets[p];
                out[i] = Some(EnvPet {
                    kind: pet.kind,
                    attack: pet.attack,
                    health: pet.health,
                    tmp_attack: pet.tmp_attack,
                    tmp_health: pet.tmp_health,
                    level: pet.level,
                    status: pet.status,
                    ability_counter: pet.ability_counter,
                    friends_hurt_counter: pet.friends_hurt_counter,
                    hurt: pet.hurt,
                    override_ability: pet.over.clone(),
                });
            }
        }
        out
    }
}

// ----------------------------------------------------------------------
// Search and ghost primitives: board rows in and out, gold, rekey.
// ----------------------------------------------------------------------

struct RowSpec {
    kind: usize,
    attack: i32,
    health: i32,
    experience: i32,
    status: Status,
    consumed: i32,
    /// -2 default rule, -1 copied nothing, otherwise the copied kind.
    copies: i32,
}

/// Validate one board; nothing is changed by a refused board.
fn parse_rows(t: &'static Tables, rows: &BoardRows) -> R<[Option<RowSpec>; NS]> {
    let mut out: [Option<RowSpec>; NS] = [None, None, None, None, None];
    for (i, row) in rows.iter().enumerate() {
        if row[0] == -1 {
            continue;
        }
        let kinds = t.d.pets.len() as i32;
        if !(0..kinds).contains(&row[0]) {
            return err(format!("board row {i}: no pet kind {}", row[0]));
        }
        if !(0..=50).contains(&row[1]) || !(1..=50).contains(&row[2]) {
            return err(format!("board row {i}: attack {} / health {} outside 0..50 / 1..50", row[1], row[2]));
        }
        if !(0..=5).contains(&row[3]) {
            return err(format!("board row {i}: total experience {} outside 0..5", row[3]));
        }
        if !(0..BOARD_STATUSES.len() as i32).contains(&row[4]) {
            return err(format!("board row {i}: no perk id {}", row[4]));
        }
        if !(0..=1000).contains(&row[5]) {
            return err(format!("board row {i}: consumed triggers {} outside 0..1000", row[5]));
        }
        if !(-2..kinds).contains(&row[6]) {
            return err(format!("board row {i}: no copied kind {}", row[6]));
        }
        out[i] = Some(RowSpec {
            kind: row[0] as usize,
            attack: row[1],
            health: row[2],
            experience: row[3],
            status: BOARD_STATUSES[row[4] as usize],
            consumed: row[5],
            copies: row[6],
        });
    }
    Ok(out)
}

/// Whether `set_team` / `eval_vs` would accept the board (they refuse nothing else).
pub fn check_board(rows: &BoardRows) -> R<()> {
    parse_rows(Tables::global(), rows).map(|_| ())
}

impl Seat {
    /// The team in board rows, as the battle receives it (`Battle.__init__`
    /// folds the temporary buffs; a Parrot states what it holds).
    pub fn export_team(&self) -> R<BoardRows> {
        let mut rows = [EMPTY_ROW; NS];
        for i in 0..NS {
            if let Some(p) = self.team[i] {
                let pet = &self.pets[p];
                let copies = if pet.kind == self.t.d.parrot {
                    pet.over.as_ref().and_then(|a| a.copied_from).map(|k| k as i32).unwrap_or(-1)
                } else {
                    -2
                };
                rows[i] = [
                    pet.kind as i32,
                    self.atk(p),
                    self.hp(p),
                    Self::total_experience(pet)?,
                    pet.status as i32,
                    pet.ability_counter as i32,
                    copies,
                ];
            }
        }
        Ok(rows)
    }

    /// Per slot `[friends_hurt_counter, _hurt]`: carried into the battle, not in the rows.
    pub fn export_carried(&self) -> [[i32; 2]; NS] {
        let mut out = [[0i32; 2]; NS];
        for i in 0..NS {
            if let Some(p) = self.team[i] {
                out[i] = [self.pets[p].friends_hurt_counter, self.pets[p].hurt];
            }
        }
        out
    }

    /// `player.team = make_team(specs); player._connect_objects()` of
    /// `runs/train-ghost-20261123/ghost_worker.py`: fresh pets (`Pet(id)`
    /// defaults for everything a row does not state: sell value 1, no
    /// temporary buffs, counters zero), the Parrot rule applied front to back
    /// (no draw, no trigger spent), then every team pet and shop item bound
    /// to the shop stream. Row `i` goes to slot `i` (`make_team` itself only
    /// produces teams without gaps). Nothing else of the seat changes: gold,
    /// shop, counters, the seen-order set and `turn_ended` stay.
    pub fn set_team(&mut self, rows: &BoardRows) -> R<()> {
        let t: &'static Tables = self.t;
        let parsed = parse_rows(t, rows)?;
        self.touch();
        self.team = [None; NS];
        for (i, spec) in parsed.iter().enumerate() {
            if let Some(spec) = spec {
                // Pet(id): a MockRandomState until _connect_objects below.
                let p = self.new_pet(spec.kind, 0, Rs::Mock);
                let pet = &mut self.pets[p];
                pet.attack = spec.attack;
                pet.health = spec.health;
                Self::set_total_experience(pet, spec.experience)?;
                pet.status = spec.status;
                pet.ability_counter = spec.consumed as i64;
                self.team[i] = Some(p);
            }
        }
        for (i, spec) in parsed.iter().enumerate() {
            let (p, spec) = match (self.team[i], spec) {
                (Some(p), Some(spec)) if spec.kind == t.d.parrot => (p, spec),
                _ => continue,
            };
            let level = self.pets[p].level.clamp(1, 3) as usize;
            match spec.copies {
                // TransferAbility(slot.pet, [0, index], [team])
                -2 => {
                    let native = match &t.d.pets[t.d.parrot].abilities[level - 1] {
                        Some(a) => a,
                        None => return err("the Parrot has no ability to copy with"),
                    };
                    self.transfer_ability(&native.effect, p, Idx::Int(i), false, Te::None)?;
                }
                -1 => {}
                // data["pets"][source].get(f"level{level}Ability") or Pet(source).ability
                source => {
                    let source = source as usize;
                    let mut copied: Ability = match &t.d.pets[source].abilities[level - 1] {
                        Some(a) => (**a).clone(),
                        None => match &t.d.pets[source].abilities[0] {
                            Some(a) => (**a).clone(),
                            None => (*t.d.empty_ability).clone(),
                        },
                    };
                    copied.copied_from = Some(source);
                    self.pets[p].over = Some(Arc::new(copied));
                }
            }
        }
        self.team_seed_none = true;
        self.connect_objects();
        Ok(())
    }

    /// `player.gold = gold` (the ghost worker sets 0 so that END is legal).
    pub fn set_gold(&mut self, gold: i32) -> R<()> {
        if !(0..=1000).contains(&gold) {
            return err(format!("gold {gold} outside 0..1000"));
        }
        self.touch();
        self.gold = gold;
        Ok(())
    }

    /// Put a chosen pet or food into shop slot `slot`, as if `Shop.roll` had
    /// just rolled it there and `Player.roll` had bound it (`Shop._roll_slot`
    /// without the draw, the stat line of `Shop.roll`, and the four
    /// assignments `Player._connect_objects` makes for a shop item, for the
    /// new item only).
    ///
    /// A pet goes into a `pet` or `levelup` slot: a fresh `Pet(name)` (base
    /// stats, level 1, no experience, the species' own status, sell value 1,
    /// counters zero) plus the shop's Canned Food bonus capped at 50; the
    /// slot becomes an ordinary pet slot (`slot_type = "pet"`,
    /// `reward_group = None`, as `_roll_slot` does to a reward slot; a reward
    /// sibling keeps its group and stays where it is), cost 3. A food goes
    /// into a `food` slot: a fresh `Food(name)` at its table cost (a Squirrel
    /// discount or a stocked item's cost of 0 is replaced). The item draws
    /// from the seat's shop stream. `frozen` becomes the slot's frozen flag.
    /// The slot keeps its position: no tier ordering is applied.
    ///
    /// Only what a roll can offer on some turn is accepted (the 60 pets and
    /// 18 foods of the Turtle roll pools, whatever the current turn); tokens,
    /// Milk and Bread Crumbs are refused, as are a slot index outside the
    /// shop list and a pet for a food slot or a food for a pet slot. Nothing
    /// is drawn and nothing else of the seat changes: gold, the other slots,
    /// the action and reorder counters, the freeze-toggled set (the new item
    /// is not in it, the replaced one stays in it if it was), the seen-order
    /// set, the offer cursors and the shop stream.
    pub fn set_shop_item(&mut self, slot: usize, item: ShopItem, frozen: bool) -> R<()> {
        let t: &'static Tables = self.t;
        let pool = &t.e.turns[11];
        if slot >= self.slots.len() {
            return err(format!("shop slot {slot} does not exist (the shop has {} slots)", self.slots.len()));
        }
        let ty = self.slots[slot].ty;
        match item {
            ShopItem::Pet(kind) => {
                if kind >= t.d.pets.len() {
                    return err(format!("no pet kind {kind}"));
                }
                if !pool.avail_pets.contains(&kind) {
                    return err(format!("{} is not a pet the Turtle pack shop can roll", t.d.pets[kind].name));
                }
                if !matches!(ty, SlotType::Pet | SlotType::Levelup) {
                    return err(format!("shop slot {slot} is a {} slot: it cannot take a pet", ty.name()));
                }
                self.touch();
                let p = self.new_pet(kind, 0, Rs::Shared);
                let (sa, sh) = (self.shop_attack, self.shop_health);
                let pet = &mut self.pets[p];
                pet.attack = (pet.attack + sa).min(50);
                pet.health = (pet.health + sh).min(50);
                let s = &mut self.slots[slot];
                s.ty = SlotType::Pet;
                s.reward_group = None;
                s.cost = 3;
                s.frozen = frozen;
                s.obj = Obj::Pet(p);
            }
            ShopItem::Food(kind) => {
                if kind >= t.e.foods.len() {
                    return err(format!("no food {kind}"));
                }
                if !pool.avail_foods.contains(&kind) {
                    return err(format!("{} is not a food the Turtle pack shop can roll", t.e.foods[kind].name));
                }
                if ty != SlotType::Food {
                    return err(format!("shop slot {slot} is a {} slot: it cannot take a food", ty.name()));
                }
                self.touch();
                let food = self.new_food(kind, Rs::Shared);
                let s = &mut self.slots[slot];
                s.cost = food.cost;
                s.frozen = frozen;
                s.obj = Obj::Food(food);
            }
        }
        Ok(())
    }

    /// The seat's share of `ParallelSelfPlayEnv.redeterminize(seed)` on the
    /// paired-tape path: `shop.enable_paired_tape(seed, seat)` (a new tape
    /// with every cursor at zero, the shop stream reseeded from the new
    /// tape's `ability-seed` domain) and `_connect_objects` (pets holding a
    /// private copy of the old stream are bound to the new one).
    fn rekey(&mut self, seed: u64) {
        let seat = self.seat;
        let ability_seed = (tape::uniform(seed, &format!("player/{seat}/ability-seed"), 0) * 4294967296.0) as u64;
        self.rs = Mt::new(ability_seed as u32);
        self.pet_tape = tape::Stream::new(seed, &format!("player/{seat}/shop/pet"));
        self.food_tape = tape::Stream::new(seed, &format!("player/{seat}/shop/food"));
        self.pet_cursor = 0;
        self.food_cursor = 0;
        self.world_seed = seed;
        self.connect_objects();
    }
}

impl Game {
    /// See `Seat::set_team`.
    pub fn set_team(&mut self, side: usize, rows: &BoardRows) -> R<()> {
        if side > 1 {
            return err("no such seat");
        }
        if self.poisoned {
            return err("An earlier step failed inside the engine; reset the game");
        }
        self.seats[side].set_team(rows)
    }

    /// See `Seat::set_gold`.
    pub fn set_gold(&mut self, side: usize, gold: i32) -> R<()> {
        if side > 1 {
            return err("no such seat");
        }
        if self.poisoned {
            return err("An earlier step failed inside the engine; reset the game");
        }
        self.seats[side].set_gold(gold)
    }

    /// See `Seat::set_shop_item`.
    pub fn set_shop_item(&mut self, side: usize, slot: usize, item: ShopItem, frozen: bool) -> R<()> {
        if side > 1 {
            return err("no such seat");
        }
        if self.poisoned {
            return err("An earlier step failed inside the engine; reset the game");
        }
        self.seats[side].set_shop_item(slot, item, frozen)
    }

    /// `ParallelSelfPlayEnv.redeterminize(seed)` on the paired-tape path:
    /// every later random draw of the match comes from the tape of `seed`.
    /// Replaced: both seats' pet and food offer domains (cursors restart at
    /// zero; level-up rewards draw from the pet domain), both shop streams
    /// (trigger-order ties, random targets, priority ties in the shop) and
    /// the battle-seed domain. Not replaced: everything already drawn, i.e.
    /// the current shops (frozen or not), teams, counters and histories, and
    /// the per-team seed that no reachable draw uses.
    pub fn rekey(&mut self, seed: u64) -> R<()> {
        if seed >= 1u64 << 63 {
            return err("rekey requires a seed in [0, 2**63)");
        }
        if self.poisoned {
            return err("An earlier step failed inside the engine; reset the game");
        }
        for seat in self.seats.iter_mut() {
            seat.rekey(seed);
        }
        self.tape_seed = seed;
        Ok(())
    }

    /// The team `side` would hand to the battle if it ended its turn now
    /// without freezing anything: its end-of-turn effects run on a private
    /// copy of the seat (same draws from the same stream position as the real
    /// END would make); a seat that has already ended is taken as it is.
    pub fn team_if_ended(&self, side: usize) -> R<[Option<EnvPet>; NS]> {
        let seat = &self.seats[side];
        if seat.turn_ended {
            return Ok(seat.battle_team());
        }
        let mut copy = seat.clone();
        copy.cov = None;
        copy.end_turn()?;
        Ok(copy.battle_team())
    }

    /// The battle team a board row block stands for (`set_team` on a private
    /// copy of a seat, so both read a board the same way).
    pub fn board_team(&self, rows: &BoardRows) -> R<[Option<EnvPet>; NS]> {
        let mut scratch = self.seats[0].clone();
        scratch.cov = None;
        scratch.set_team(rows)?;
        Ok(scratch.battle_team())
    }

    /// Battle seed `sample` against board `board` of an `eval_vs` call:
    /// `int(RandomTape(seed).uniform(f"eval/{board}") * (2**32 - 1))` at cursor `sample`.
    pub fn eval_seed(seed: u64, board: usize, sample: u64) -> u32 {
        (tape::uniform(seed, &format!("eval/{board}"), sample) * 4294967295.0) as u32
    }

    /// Fight `own` (the team of `side`) `samples` times against `other`;
    /// returns `[wins, draws, losses]` of `side`. Seat zero is the first
    /// team of the battle, as in the match.
    pub fn eval_board(
        &self,
        side: usize,
        own: &[Option<EnvPet>; NS],
        other: &[Option<EnvPet>; NS],
        board: usize,
        samples: u32,
        seed: u64,
    ) -> R<[i32; 3]> {
        let mut stream = tape::Stream::new(seed, &format!("eval/{board}"));
        let mut counts = [0i32; 3];
        for sample in 0..samples as u64 {
            let battle_seed = (stream.uniform(sample) * 4294967295.0) as u32;
            let outcome = if side == 0 {
                battle_env(&self.t.d, [own, other], battle_seed)?
            } else {
                battle_env(&self.t.d, [other, own], battle_seed)?
            };
            let column = match (outcome, side) {
                (2, _) => 1,
                (0, 0) | (1, 1) => 0,
                _ => 2,
            };
            counts[column] += 1;
        }
        Ok(counts)
    }

    /// `eval_vs` for one game: counts `[wins, draws, losses]` of `side` per board.
    /// The game is not changed.
    pub fn eval_vs(&self, side: usize, boards: &[BoardRows], samples: u32, seed: u64) -> R<Vec<[i32; 3]>> {
        if side > 1 {
            return err("no such seat");
        }
        if seed >= 1u64 << 63 {
            return err("eval_vs requires a seed in [0, 2**63)");
        }
        if self.poisoned {
            return err("An earlier step failed inside the engine; reset the game");
        }
        let own = self.team_if_ended(side)?;
        let mut out = Vec::with_capacity(boards.len());
        for (m, rows) in boards.iter().enumerate() {
            let other = self.board_team(rows)?;
            out.push(self.eval_board(side, &own, &other, m, samples, seed)?);
        }
        Ok(out)
    }
}

/// Five identities in one integer (25 bits each; `new_pet` / `new_food`
/// never hand out an identity that does not fit).
#[inline]
fn pack_order(order: &[u32; NS]) -> u128 {
    let mut key = 0u128;
    for uid in order {
        key = (key << 25) | *uid as u128;
    }
    key
}

/// `itertools.combinations(items, n)` in lexicographic order.
fn combinations(items: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let len = items.len();
    if n > len {
        return out;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    loop {
        out.push(idx.iter().map(|&i| items[i]).collect());
        let mut found = None;
        for k in (0..n).rev() {
            if idx[k] != k + len - n {
                found = Some(k);
                break;
            }
        }
        match found {
            None => return out,
            Some(k) => {
                idx[k] += 1;
                for j in (k + 1)..n {
                    idx[j] = idx[j - 1] + 1;
                }
            }
        }
    }
}

/// What one `step_decisions` call returns besides the observations.
#[derive(Clone, Debug, PartialEq)]
pub struct StepOut {
    /// `EndFreezeStep.primitive_actions`: the flat ids executed per seat.
    pub plans: [Vec<usize>; 2],
    pub rewards: [f64; 2],
    pub battle_resolved: bool,
    pub battle_result: Option<u8>,
    pub terminated: bool,
}

/// `ParallelSelfPlayEnv` after `reset(seed, paired_tape=True)`.
#[derive(Clone)]
pub struct Game {
    pub t: &'static Tables,
    pub seats: [Seat; 2],
    /// `core.turn`.
    pub turn: i32,
    pub seed: u64,
    /// `previous_agent_state` / `previous_opponent_state` by seat, and the ones before.
    pub previous: [Option<Board>; 2],
    pub previous_previous: [Option<Board>; 2],
    pub terminated: bool,
    /// Every battle seed drawn (`_paired_battle_seeds`).
    pub battle_seeds: Vec<(i32, u32)>,
    pub battles: u64,
    /// Set when a step failed after it had started to change the state (an
    /// internal error, never illegal input): the game must be reset.
    pub poisoned: bool,
    /// Seed of `_battle_tape` (the match seed until `rekey`).
    pub tape_seed: u64,
    /// The battle resolved by the last accepted step, if it resolved one.
    pub last_battle: Option<BattleEvent>,
}

impl Game {
    /// `ParallelSelfPlayEnv().reset(seed=seed, paired_tape=True)`.
    pub fn new(seed: u64, coverage: bool) -> R<Game> {
        if seed >= 1u64 << 63 {
            return err("Paired world requires explicit seed and standard single-battle outcome");
        }
        let t = Tables::global();
        // core.reset(seed % 2**32): four child seeds; the second and fourth seed the teams.
        let mut rs_random = Mt::new((seed % (1u64 << 32)) as u32);
        let mut child = [0u32; 4];
        for c in child.iter_mut() {
            *c = rs_random.interval(4294967294) as u32;
        }
        let seats = [Seat::new(t, 0, seed, child[1], coverage)?, Seat::new(t, 1, seed, child[3], coverage)?];
        let mut game = Game {
            t,
            seats,
            turn: 1,
            seed,
            previous: [None, None],
            previous_previous: [None, None],
            terminated: false,
            battle_seeds: Vec::new(),
            battles: 0,
            poisoned: false,
            tape_seed: seed,
            last_battle: None,
        };
        for seat in game.seats.iter_mut() {
            let key = seat.team_order_key();
            seat.seen = vec![key];
        }
        Ok(game)
    }

    /// `ParallelSelfPlayEnv.active_players()`.
    pub fn active(&self) -> [bool; 2] {
        [!self.seats[0].turn_ended, !self.seats[1].turn_ended]
    }

    /// `joint_action.step_decisions(env, decisions)`: one `[type, source, target]`
    /// and one END freeze mask per seat (`None` for a seat that has ended).
    /// Both seats are validated before anything changes.
    pub fn step(&mut self, actions: [Option<[i64; 3]>; 2], masks: [u32; 2]) -> R<StepOut> {
        if self.terminated {
            return err("The match is over; reset the game");
        }
        if self.poisoned {
            return err("An earlier step failed inside the engine; reset the game");
        }
        let plans = self.validate(actions, masks)?;
        let out = self.execute(plans);
        if out.is_err() {
            self.poisoned = true;
        }
        out
    }

    /// `step_with_end_freeze`, first half: both seats' plans, nothing executed.
    /// Legality only registers the current team orders, which the last
    /// observation already did, so a rejected step leaves the game as it was.
    pub fn validate(&mut self, actions: [Option<[i64; 3]>; 2], masks: [u32; 2]) -> R<[Vec<usize>; 2]> {
        let active = self.active();
        let mut plans: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        for side in 0..2 {
            let mask_id = masks[side];
            if mask_id >= 1 << MAX_SHOP_SLOTS {
                return err("Invalid freeze subset ID");
            }
            let action = actions[side];
            if !active[side] {
                if action.is_some() || mask_id != 0 {
                    return err("Ended players cannot submit actions or freeze plans");
                }
                continue;
            }
            let action = match action {
                Some(a) => a,
                None => return err("Active players must submit an action"),
            };
            // JointAction.__post_init__
            let flat = encode_action(action)?;
            if mask_id != 0 && action[0] != T_END_TURN {
                return err("Freeze subset requires END_TURN");
            }
            // env._coerce_and_validate
            if !self.seats[side].avail_actions().has(flat) {
                return err(format!("Illegal action {:?} for player {side}", action));
            }
            plans[side] = if action[0] == T_END_TURN {
                self.seats[side].expand_end_freeze(mask_id)?
            } else {
                vec![flat]
            };
        }
        Ok(plans)
    }

    /// `step_with_end_freeze`, second half, and `ParallelSelfPlayEnv.step`.
    fn execute(&mut self, plans: [Vec<usize>; 2]) -> R<StepOut> {
        self.last_battle = None;
        // Toggles first, seat by seat.
        for side in 0..2 {
            let n = plans[side].len();
            for k in 0..n.saturating_sub(1) {
                let a = plans[side][k];
                self.seats[side].resolve_action(a)?;
            }
        }
        // ParallelSelfPlayEnv.step with the final actions.
        for side in 0..2 {
            if let Some(&last) = plans[side].last() {
                if self.seats[side].turn_ended {
                    return err(format!("Player {side} already ended the turn"));
                }
                self.seats[side].resolve_action(last)?;
            }
        }
        if !(self.seats[0].turn_ended && self.seats[1].turn_ended) {
            return Ok(StepOut {
                plans,
                rewards: [0.0, 0.0],
                battle_resolved: false,
                battle_result: None,
                terminated: false,
            });
        }

        self.previous_previous = self.previous;
        self.previous = [Some(self.seats[0].board()?), Some(self.seats[1].board()?)];
        // _next_battle_seed
        let u = tape::uniform(self.tape_seed, &format!("battle/turn/{}", self.turn), 0);
        let battle_seed = (u * 4294967295.0) as u32;
        self.battle_seeds.push((self.turn, battle_seed));
        let t0 = self.seats[0].battle_team();
        let t1 = self.seats[1].battle_team();
        self.last_battle = Some(BattleEvent {
            turn: self.turn,
            seed: battle_seed,
            teams: [self.seats[0].export_team()?, self.seats[1].export_team()?],
            carried: [self.seats[0].export_carried(), self.seats[1].export_carried()],
        });
        let battle_result = battle_env(&self.t.d, [&t0, &t1], battle_seed)?;
        self.battles += 1;
        let lives_before = [self.seats[0].lives, self.seats[1].lives];
        self.player_fight_outcome(battle_result);
        let life_deltas = [self.seats[0].lives - lives_before[0], self.seats[1].lives - lives_before[1]];
        self.turn += 1;
        let terminated = self.seats[0].lives <= 0 || self.seats[1].lives <= 0 || self.turn >= MAX_TURN;

        let life_swing = life_deltas[0] - life_deltas[1];
        let battle_component = 0.1 * life_swing as f64;
        let mut rewards = [battle_component, -battle_component];
        if self.seats[1].lives <= 0 {
            rewards[0] += 2.0;
            rewards[1] -= 2.0;
        }
        if self.seats[0].lives <= 0 {
            rewards[0] -= 2.0;
            rewards[1] += 2.0;
        }
        if terminated && self.seats[0].lives > 0 && self.seats[1].lives > 0 {
            let player = (self.seats[0].lives, self.seats[0].wins);
            let opponent = (self.seats[1].lives, self.seats[1].wins);
            if player > opponent {
                rewards[0] += 2.0;
                rewards[1] -= 2.0;
            } else if opponent > player {
                rewards[0] -= 2.0;
                rewards[1] += 2.0;
            }
        }
        if !terminated {
            for seat in self.seats.iter_mut() {
                let winner = seat.lf_winner;
                seat.start_turn(winner)?;
            }
            for seat in self.seats.iter_mut() {
                seat.touch();
                seat.reorder_count = 0;
                seat.action_count = 0;
                seat.toggled.clear();
                let key = seat.team_order_key();
                seat.seen.clear();
                seat.seen.push(key);
            }
        }
        self.terminated = terminated;
        Ok(StepOut { plans, rewards, battle_resolved: true, battle_result: Some(battle_result), terminated })
    }

    /// `SuperAutoPetsEnv._player_fight_outcome`.
    fn player_fight_outcome(&mut self, outcome: u8) {
        let [a, o] = &mut self.seats;
        match outcome {
            0 => {
                a.lf_winner = Some(true);
                o.lf_winner = Some(false);
                a.last_battle_outcome = Some(1);
                o.last_battle_outcome = Some(-1);
                a.wins += 1;
                o.lives = (o.lives - 1).max(0);
            }
            1 => {
                o.lf_winner = Some(true);
                a.lf_winner = Some(false);
                a.last_battle_outcome = Some(-1);
                o.last_battle_outcome = Some(1);
                o.wins += 1;
                a.lives = (a.lives - 1).max(0);
            }
            _ => {
                a.lf_winner = Some(true);
                o.lf_winner = Some(true);
                a.last_battle_outcome = Some(0);
                o.last_battle_outcome = Some(0);
            }
        }
        // One heart back for both players after the turn-two battle.
        if self.turn == 2 {
            a.lives = (a.lives + 1).min(MAX_LIVES);
            o.lives = (o.lives + 1).min(MAX_LIVES);
        }
    }

    /// `distributed._completed_match_result(core)`: 0 / 1 / 2 from seat zero's side.
    pub fn match_result(&self) -> u8 {
        let (a, o) = (&self.seats[0], &self.seats[1]);
        if o.lives <= 0 && 0 < a.lives {
            return 0;
        }
        if a.lives <= 0 && 0 < o.lives {
            return 1;
        }
        let player = (a.lives, a.wins);
        let opponent = (o.lives, o.wins);
        if player > opponent {
            0
        } else if opponent > player {
            1
        } else {
            2
        }
    }
}
