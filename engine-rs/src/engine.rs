//! Battle engine. A transliteration of the Python reference
//! (`sapai/battle.py`, `effects.py`, `pets.py`, `teams.py`, `status.py` at
//! engine-v31), kept function-for-function where rules may change again.
//! Every random draw is made in the same place and shape as the reference so
//! that a fight with the same seed reproduces the reference's trace exactly.

use crate::data::*;
use crate::rng::Mt;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

type R<T> = Result<T, EngineError>;

fn err<T>(m: impl Into<String>) -> R<T> {
    Err(EngineError(m.into()))
}

/// Placeholder for the "?" base stats of summoned tokens until the summon sets them.
const UNRESOLVED: i32 = i32::MIN / 4;
const NSLOTS: usize = 5;
const MAX_ROUNDS: usize = 5000;

/// Input pet, front of the team first (same fields as the golden rows).
#[derive(Clone, Debug)]
pub struct PetSpec {
    pub id: String,
    pub attack: i32,
    pub health: i32,
    pub experience: i32,
    pub status: String,
    pub triggers_consumed: i32,
    /// For a Parrot only. `None`: derive the copy from the friend ahead (old
    /// behaviour, used by fuzz boards). `Some(None)`: the replay recorded the
    /// Parrot's own ability, so it copied nothing. `Some(Some(id))`: the
    /// replay recorded this pet's ability.
    pub parrot_copies: Option<Option<String>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotView {
    /// Index into `Data::pets`.
    pub kind: usize,
    pub attack: i32,
    pub health: i32,
    pub status: Status,
    pub level: u8,
}

/// `[player, opponent]`, five slots each, front first.
pub type Boards = [[Option<SlotView>; NSLOTS]; 2];

#[derive(Clone, Debug)]
pub struct Outcome {
    /// 0 player win, 1 opponent win, 2 draw.
    pub outcome: u8,
    pub final_boards: Boards,
    /// Boards after construction, after start of battle, after every attack round.
    pub trace: Vec<Boards>,
    /// The next `random_sample()` after the fight (checks the draw count).
    pub probe: f64,
    pub rounds: usize,
}

#[derive(Clone)]
struct Stored {
    pet: usize,
    with_attack: i32,
    with_health: i32,
    with_level: i32,
}

#[derive(Clone)]
struct Pet {
    kind: usize,
    /// `_attack` / `_health`.
    attack: i32,
    health: i32,
    /// `_until_end_of_battle_*_buff`.
    tmp_attack: i32,
    tmp_health: i32,
    hurt: i32,
    status: Status,
    level: i32,
    ability_counter: i64,
    friends_hurt_counter: i32,
    override_ability: Option<Arc<Ability>>,
    swallowed: Vec<Stored>,
    /// The owning team (kept after removal, like `pet.team`).
    team: usize,
}

/// `apet_idx[1]`: a slot index, or the `[team_idx, pet_idx]` list a removed
/// pet passes through `faint_trigger`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Idx {
    Int(usize),
    List(usize, usize),
}

/// The triggering entity argument.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Te {
    #[allow(dead_code)]
    None,
    Pet(usize),
    Team,
}

struct EffResult {
    /// Targets, flattened in order.
    targets: Vec<usize>,
    /// True when Python's `possible == [[]]` (a summon that produced nothing).
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
    tiger: Option<usize>,
    support_chain: Vec<usize>,
    native_post: bool,
    has_status: bool,
    listeners: Vec<usize>,
    post_observers: Vec<usize>,
}

struct KoEvent {
    pet: usize,
    team_idx: usize,
    tiger: Option<usize>,
    attack: f64,
    tie: f64,
}

type Layout = [Option<usize>; NSLOTS];

pub struct Fight<'a> {
    d: &'a Data,
    pets: Vec<Pet>,
    /// `teams[t].slots`: pet index or None for an empty slot ("pet-none").
    slots: [[Option<usize>; NSLOTS]; 2],
    rng: Mt,
    pet_priority: Vec<(usize, usize)>,
    pending: Vec<Record>,
    /// Wolverine activations waiting for the hurt and faint chain to finish.
    pending_counters: Vec<(usize, usize)>,
    attack_happened: bool,
    ko_list: Vec<KoEvent>,
}

impl<'a> Fight<'a> {
    // ------------------------------------------------------------------
    // Construction (tools/gen_golden.py make_team + Battle.__init__)
    // ------------------------------------------------------------------

    pub fn new(d: &'a Data, player: &[PetSpec], opponent: &[PetSpec], seed: u32) -> R<Self> {
        let mut f = Fight {
            d,
            pets: Vec::with_capacity(24),
            slots: [[None; NSLOTS]; 2],
            rng: Mt::new(seed),
            pet_priority: Vec::new(),
            pending: Vec::new(),
            pending_counters: Vec::new(),
            attack_happened: false,
            ko_list: Vec::new(),
        };
        for (team, specs) in [(0usize, player), (1usize, opponent)] {
            if specs.len() > NSLOTS {
                return err("more than five pets on a team");
            }
            for (index, spec) in specs.iter().enumerate() {
                let kind = match d.by_name.get(&spec.id) {
                    Some(k) => *k,
                    None => return err(format!("Pet {} not found", spec.id)),
                };
                let p = f.new_pet(kind, team);
                let status = match Status::parse(&spec.status) {
                    Some(s) => s,
                    None => return err(format!("unknown status {}", spec.status)),
                };
                let pet = &mut f.pets[p];
                pet.attack = spec.attack;
                pet.health = spec.health;
                // set_total_experience
                let total = spec.experience.min(5);
                if total < 0 {
                    return err("Total experience cannot be negative");
                }
                pet.level = if total < 2 { 1 } else if total < 5 { 2 } else { 3 };
                pet.status = status;
                pet.ability_counter = spec.triggers_consumed as i64;
                f.slots[team][index] = Some(p);
            }
            // make_team: a submitted board's Parrot has already copied the friend ahead.
            for index in 0..NSLOTS {
                if let Some(p) = f.slots[team][index] {
                    if f.pets[p].kind == d.parrot {
                        match specs.get(index).and_then(|s| s.parrot_copies.clone()) {
                            None => f.transfer_ability_setup(p, index)?,
                            Some(None) => {}
                            Some(Some(name)) => {
                                let kind = match d.by_name.get(&name) {
                                    Some(k) => *k,
                                    None => return err("unknown parrotCopies pet"),
                                };
                                let level = f.pets[p].level as usize;
                                // Same fallback as TransferAbility: a pet with no
                                // ability at this level (Scorpion) yields the empty one.
                                let base = match &d.pets[kind].abilities[level - 1] {
                                    Some(a) => a.clone(),
                                    None => d.empty_ability.clone(),
                                };
                                let mut copied = (*base).clone();
                                copied.copied_from = Some(kind);
                                f.pets[p].override_ability = Some(Arc::new(copied));
                            }
                        }
                    }
                }
            }
        }
        // Battle.__init__: fold temporary shop buffs into the battle copy.
        for pet in f.pets.iter_mut() {
            pet.attack = (pet.attack + pet.tmp_attack).min(50);
            pet.health = (pet.health + pet.tmp_health).min(50);
            pet.tmp_attack = 0;
            pet.tmp_health = 0;
        }
        f.pet_priority = f.update_pet_priority();
        Ok(f)
    }

    fn new_pet(&mut self, kind: usize, team: usize) -> usize {
        let fd = &self.d.pets[kind];
        self.pets.push(Pet {
            kind,
            attack: fd.base_attack.map(|v| v as i32).unwrap_or(UNRESOLVED),
            health: fd.base_health.map(|v| v as i32).unwrap_or(UNRESOLVED),
            tmp_attack: 0,
            tmp_health: 0,
            hurt: 0,
            status: fd.status.unwrap_or(Status::None),
            level: 1,
            ability_counter: 0,
            friends_hurt_counter: 0,
            override_ability: None,
            swallowed: Vec::new(),
            team,
        });
        self.pets.len() - 1
    }

    /// effects.TransferAbility as called from make_team (Parrot copies the friend ahead).
    fn transfer_ability_setup(&mut self, apet: usize, index: usize) -> R<()> {
        let ab = self.ability(apet);
        let from = match &ab.effect.from {
            Some(t) => t.clone(),
            None => return err("from not found in effect"),
        };
        if from.kind != TargetKind::FriendAhead {
            return err("Parrot source target is not FriendAhead");
        }
        let team = self.pets[apet].team;
        let n = from.n.unwrap_or(1);
        if n < 1 {
            return Ok(());
        }
        let mut source = None;
        for idx in (0..index).rev() {
            if let Some(p) = self.slots[team][idx] {
                if self.hp(p) > 0 {
                    source = Some(p);
                    break;
                }
            }
        }
        if let Some(src) = source {
            let level = self.pets[apet].level as usize;
            let mut kind = self.pets[src].kind;
            // v31: a Parrot behind a Parrot copies what the front Parrot
            // copied (real replays: Sheep, Parrot, Parrot gives five Ram
            // summons), not the Parrot's own end-of-turn ability.
            if let Some(over) = &self.pets[src].override_ability {
                if let Some(origin) = over.copied_from {
                    kind = origin;
                }
            }
            // source.fd.get(f"level{apet.level}Ability", source.ability)
            let base = match &self.d.pets[kind].abilities[level - 1] {
                Some(a) => a.clone(),
                None => self.ability(src),
            };
            let mut copied = (*base).clone();
            copied.copied_from = Some(kind);
            self.pets[apet].override_ability = Some(Arc::new(copied));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Pet helpers (sapai/pets.py)
    // ------------------------------------------------------------------

    #[inline]
    fn atk(&self, p: usize) -> i32 {
        (self.pets[p].attack + self.pets[p].tmp_attack).min(50)
    }

    #[inline]
    fn hp(&self, p: usize) -> i32 {
        (self.pets[p].health + self.pets[p].tmp_health).min(50)
    }

    /// `Pet.ability`.
    fn ability(&self, p: usize) -> Arc<Ability> {
        let pet = &self.pets[p];
        if let Some(a) = &pet.override_ability {
            return a.clone();
        }
        let level = pet.level.clamp(1, 3) as usize;
        match &self.d.pets[pet.kind].abilities[level - 1] {
            Some(a) => a.clone(),
            None => self.d.empty_ability.clone(),
        }
    }

    /// `Pet.acts_as`.
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

    /// `Pet.get_damage` (status.apply_damage_dict).
    fn get_damage(&self, p: usize, value: i32) -> i32 {
        match self.pets[p].status {
            Status::CoconutShield => 0,
            Status::GarlicArmor => {
                if value <= 0 {
                    0
                } else if value <= 2 {
                    value
                } else {
                    (value - 2).max(2)
                }
            }
            Status::MelonArmor => (value - 20).max(0),
            Status::Weak => {
                if value > 0 {
                    value + 3
                } else {
                    0
                }
            }
            _ => value.max(0),
        }
    }

    /// `Pet.attack_damage` (status.apply_attack_dict).
    fn attack_damage(&self, p: usize) -> i32 {
        let value = self.atk(p);
        match self.pets[p].status {
            Status::BoneAttack => {
                if value > 0 {
                    value + 3
                } else {
                    0
                }
            }
            Status::SteakAttack => {
                if value > 0 {
                    value + 20
                } else {
                    0
                }
            }
            _ => value.max(0),
        }
    }

    /// `Pet.hurt`.
    fn hurt(&mut self, p: usize, value: i32) {
        self.pets[p].health -= value;
        if value > 0 {
            self.pets[p].hurt += 1;
        }
    }

    // ------------------------------------------------------------------
    // Team helpers (sapai/teams.py)
    // ------------------------------------------------------------------

    fn team_len(&self, t: usize) -> usize {
        self.slots[t].iter().filter(|s| s.is_some()).count()
    }

    fn team_move(&mut self, t: usize, sidx: usize, tidx: usize) -> R<()> {
        if self.slots[t][tidx].is_some() {
            return err("Attempted move to a populated position");
        }
        self.slots[t][tidx] = self.slots[t][sidx];
        self.slots[t][sidx] = None;
        Ok(())
    }

    /// `Team.move_forward(start_idx=0, end_idx=10)`.
    fn move_forward(&mut self, t: usize, start_idx: usize, end_idx: usize) -> R<()> {
        loop {
            let empty = match (0..NSLOTS).find(|&i| self.slots[t][i].is_none()) {
                Some(e) => e,
                None => return Ok(()),
            };
            let mut found = None;
            for idx in 0..NSLOTS {
                if self.slots[t][idx].is_none() {
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
            let empty = match (0..NSLOTS).rev().find(|&i| self.slots[t][i].is_none()) {
                Some(e) => e,
                None => return Ok(()),
            };
            let mut found = None;
            for idx in (0..NSLOTS).rev() {
                if self.slots[t][idx].is_none() {
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
        (0..NSLOTS).find(|&i| self.slots[t][i] == Some(p))
    }

    fn check_friend(&self, t: usize, p: usize) -> bool {
        self.get_idx(t, p).is_some()
    }

    fn team_remove(&mut self, t: usize, p: usize) -> R<()> {
        match self.get_idx(t, p) {
            Some(i) => {
                self.slots[t][i] = None;
                Ok(())
            }
            None => err("Remove pet not found"),
        }
    }

    /// `Team.get_ahead(idx, n)`: nearest first.
    fn get_ahead(&self, t: usize, pet_idx: usize, n: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for idx in (0..pet_idx.min(NSLOTS)).rev() {
            if let Some(p) = self.slots[t][idx] {
                out.push(p);
                if out.len() >= n {
                    break;
                }
            }
        }
        out
    }

    /// `Team.get_behind(pet, n=1)`: the first occupied slot behind.
    fn get_behind(&self, t: usize, p: usize) -> Option<usize> {
        let pet_idx = self.get_idx(t, p).unwrap_or(0);
        ((pet_idx + 1)..NSLOTS).find_map(|i| self.slots[t][i])
    }

    // ------------------------------------------------------------------
    // Battle (sapai/battle.py)
    // ------------------------------------------------------------------

    /// `Battle.update_pet_priority`.
    fn update_pet_priority(&mut self) -> Vec<(usize, usize)> {
        let mut entries: Vec<(i32, usize, usize)> = Vec::with_capacity(10);
        for t in 0..2 {
            for i in 0..NSLOTS {
                if let Some(p) = self.slots[t][i] {
                    entries.push((self.atk(p), t, i));
                }
            }
        }
        let mut attacks: Vec<i32> = entries.iter().map(|e| e.0).collect();
        attacks.sort_unstable_by(|a, b| b.cmp(a));
        attacks.dedup();
        let mut ordered = Vec::with_capacity(entries.len());
        for attack in attacks {
            let tied: Vec<(usize, usize)> =
                entries.iter().filter(|e| e.0 == attack).map(|e| (e.1, e.2)).collect();
            if tied.len() > 1 {
                let order = self.rng.permutation(tied.len());
                for index in order {
                    ordered.push(tied[index]);
                }
            } else {
                ordered.push(tied[0]);
            }
        }
        ordered
    }

    /// `Battle.start`.
    pub fn start(&mut self) -> R<()> {
        self.pending.clear();
        // phase_move_start
        self.move_forward(0, 0, 10)?;
        self.move_forward(1, 0, 10)?;
        self.pet_priority = self.update_pet_priority();
        // phase_start
        self.battle_phase_start()?;
        // phase_hurt_and_faint
        self.battle_phase_hurt_and_faint(false)?;
        // phase_move_end
        self.move_forward(0, 0, 10)?;
        self.move_forward(1, 0, 10)?;
        self.pet_priority = self.update_pet_priority();
        Ok(())
    }

    /// `Battle.attack`: returns whether another attack should follow.
    pub fn attack(&mut self) -> R<bool> {
        self.pending.clear();
        if self.team_len(0) == 0 || self.team_len(1) == 0 {
            return Ok(false);
        }
        self.attack_happened = false;
        self.ko_list.clear();
        // phase_move_start
        self.move_forward(0, 0, 10)?;
        self.move_forward(1, 0, 10)?;
        self.battle_phase_attack_before()?;
        self.battle_phase_hurt_and_faint(false)?; // phase_hurt_and_faint_ab
        self.battle_phase_attack()?;
        self.battle_phase_attack_after()?;
        self.battle_phase_hurt_and_faint(true)?; // phase_hurt_and_faint_aa
        self.battle_phase_knockout()?; // runs phase_hurt_and_faint_k itself
        // phase_move_end
        self.move_forward(0, 0, 10)?;
        self.move_forward(1, 0, 10)?;
        Ok(self.check_battle_result() < 0)
    }

    /// `Battle.check_battle_result`.
    pub fn check_battle_result(&self) -> i32 {
        let alive = |t: usize| self.slots[t].iter().any(|s| s.map(|p| self.hp(p) > 0).unwrap_or(false));
        match (alive(0), alive(1)) {
            (true, true) => -1,
            (true, false) => 0,
            (false, true) => 1,
            (false, false) => 2,
        }
    }

    /// The loop of tools/gen_golden.py (same as `Battle.battle`).
    pub fn priority_refresh(&mut self) {
        self.pet_priority = self.update_pet_priority();
    }

    pub fn boards(&self) -> Boards {
        let mut out: Boards = [[None; NSLOTS]; 2];
        for t in 0..2 {
            for i in 0..NSLOTS {
                if let Some(p) = self.slots[t][i] {
                    out[t][i] = Some(SlotView {
                        kind: self.pets[p].kind,
                        attack: self.atk(p),
                        health: self.hp(p),
                        status: self.pets[p].status,
                        level: self.pets[p].level as u8,
                    });
                }
            }
        }
        out
    }

    pub fn random_sample(&mut self) -> f64 {
        self.rng.random_sample()
    }

    /// `battle_phase_start` (+ `check_self_summoned_triggers`).
    fn battle_phase_start(&mut self) -> R<()> {
        let pp = self.pet_priority.clone();
        for &(team_idx, pet_idx) in &pp {
            let p = match self.slots[team_idx][pet_idx] {
                Some(p) => p,
                None => continue, // "pet-none": no StartOfBattle ability
            };
            self.sob_trigger(p)?;
        }
        // check_self_summoned_triggers: only Butterfly (Summoned / Self).
        for &(team_idx, pet_idx) in &pp {
            let p = match self.slots[team_idx][pet_idx] {
                Some(p) => p,
                None => continue,
            };
            if self.hp(p) <= 0 {
                continue;
            }
            let ab = self.ability(p);
            if ab.trigger != Trigger::Summoned || ab.triggered_by != TrigBy::SelfPet {
                continue;
            }
            // func(p, [0, pet_idx], teams, te=p)
            self.run_eff(&ab.effect, p, Idx::Int(pet_idx), true, Te::Pet(p), None)?;
        }
        Ok(())
    }

    /// `check_friend_hurt_triggers`.
    fn check_friend_hurt_triggers(&mut self, hurt_pet: usize, team_idx: usize) -> R<()> {
        let snapshot = self.slots[team_idx];
        for slot in snapshot.iter() {
            if let Some(observer) = *slot {
                self.friend_hurt_trigger(observer, hurt_pet)?;
            }
        }
        Ok(())
    }

    /// `_is_post_removal_faint`.
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

    /// `_capture_tiger_support`.
    fn capture_tiger_support(&self, p: usize) -> Option<usize> {
        let t = self.pets[p].team;
        if !self.check_friend(t, p) {
            return None;
        }
        let support = self.get_behind(t, p)?;
        if self.pets[support].kind == self.d.tiger {
            Some(support)
        } else {
            None
        }
    }

    /// `battle_phase_hurt_and_faint`. `defer_post` is true for the
    /// `_aa` and `_k` phases, whose post-removal events wait for knockouts.
    fn battle_phase_hurt_and_faint(&mut self, defer_post: bool) -> R<()> {
        let mut records: Vec<Record> = Vec::new();
        let mut recorded: Vec<usize> = Vec::new();
        let mut guard = 0usize;
        loop {
            guard += 1;
            if guard > 100_000 {
                return err("hurt/faint resolution did not terminate");
            }
            let priority = self.update_pet_priority();
            let mut hurt_event = None;
            for &(team_idx, pet_idx) in &priority {
                if let Some(p) = self.slots[team_idx][pet_idx] {
                    if self.pets[p].hurt > 0 {
                        hurt_event = Some((team_idx, p));
                        break;
                    }
                }
            }
            if let Some((team_idx, p)) = hurt_event {
                self.hurt_trigger(p)?;
                self.check_friend_hurt_triggers(p, team_idx)?;
                continue;
            }
            let mut new_dead = None;
            for &(team_idx, pet_idx) in &priority {
                if let Some(p) = self.slots[team_idx][pet_idx] {
                    if self.hp(p) <= 0 && !recorded.contains(&p) {
                        new_dead = Some((team_idx, pet_idx, p));
                        break;
                    }
                }
            }
            let (team_idx, pet_idx, p) = match new_dead {
                Some(x) => x,
                None => {
                    // Counter abilities (Wolverine) wait until no hurt or
                    // faint ability is left, and run before removal.
                    if self.run_pending_counter()? {
                        continue;
                    }
                    break;
                }
            };
            recorded.push(p);
            let listeners: Vec<usize> =
                self.slots[team_idx].iter().filter_map(|s| *s).filter(|&q| q != p).collect();
            let support_chain: Vec<usize> =
                ((pet_idx + 1)..NSLOTS).filter_map(|i| self.slots[team_idx][i]).collect();
            let record = Record {
                pet: p,
                team_idx,
                pet_idx,
                tiger: self.capture_tiger_support(p),
                support_chain,
                native_post: self.is_post_removal_faint(p),
                has_status: self.pets[p].status != Status::None,
                listeners,
                post_observers: Vec::new(),
            };
            let native_post = record.native_post;
            let tiger = record.tiger;
            records.push(record);
            if !native_post {
                self.faint_trigger(p, p, (team_idx, pet_idx), true, false, tiger)?;
            }
        }

        // Post-removal summon Faints skip simultaneously dead intervening pets
        // when looking for a Tiger.
        let dead: Vec<usize> = records.iter().map(|r| r.pet).collect();
        for record in records.iter_mut() {
            if !record.native_post || record.tiger.is_some() {
                continue;
            }
            for &support in &record.support_chain {
                if self.pets[support].kind == self.d.tiger {
                    record.tiger = Some(support);
                    break;
                }
                if dead.contains(&support) {
                    continue;
                }
                break;
            }
        }

        // FriendAheadFainted / FriendFaints observe before removal.
        for r in 0..records.len() {
            let fainted = records[r].pet;
            let team_idx = records[r].team_idx;
            let pet_idx = records[r].pet_idx;
            let listeners = records[r].listeners.clone();
            for observer in listeners {
                if self.acts_as(observer, self.d.shark, Some(self.d.fly)) {
                    continue;
                }
                let ot = self.pets[observer].team;
                if !self.check_friend(ot, observer) {
                    continue;
                }
                self.faint_trigger(observer, fainted, (team_idx, pet_idx), true, false, None)?;
            }
        }

        // Synchronize deaths.
        for record in &records {
            if self.check_friend(record.team_idx, record.pet) {
                self.team_remove(record.team_idx, record.pet)?;
            }
        }

        for mut record in records {
            let team_idx = record.team_idx;
            record.post_observers = self.slots[team_idx]
                .iter()
                .filter_map(|s| *s)
                .filter(|&q| self.acts_as(q, self.d.shark, Some(self.d.fly)))
                .collect();
            self.pending.push(record);
        }

        if !defer_post {
            self.flush_post_removal_events()?;
        }
        self.pet_priority = self.update_pet_priority();
        Ok(())
    }

    /// `_slot_insert`.
    fn slot_insert(layout: &mut Layout, slot_idx: usize, pet: usize) -> bool {
        let size = NSLOTS;
        let target = slot_idx.min(size - 1);
        if layout[target].is_none() {
            layout[target] = Some(pet);
            return true;
        }
        for empty in (0..target).rev() {
            if layout[empty].is_none() {
                // layout[empty:target] = layout[empty + 1 : target + 1]
                for i in empty..target {
                    layout[i] = layout[i + 1];
                }
                layout[target] = Some(pet);
                return true;
            }
        }
        for empty in (target + 1)..size {
            if layout[empty].is_none() {
                // layout[target + 1 : empty + 1] = layout[target:empty]
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
            let present: Vec<usize> = self.slots[index].iter().filter_map(|s| *s).collect();
            for position in 0..NSLOTS {
                if let Some(p) = layouts[index][position] {
                    if !present.contains(&p) {
                        layouts[index][position] = None;
                    }
                }
            }
            let fresh: Vec<usize> = present
                .iter()
                .copied()
                .filter(|p| !layouts[index].iter().any(|s| *s == Some(*p)))
                .collect();
            if index != team_idx {
                if !fresh.is_empty() {
                    layouts[index] = self.slots[index];
                }
                continue;
            }
            let mut ordered: Vec<usize> = created.iter().copied().filter(|p| fresh.contains(p)).collect();
            // `created` may list a pet twice only if the engine returned it
            // twice; Python's list comprehension would keep both entries.
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
                // _write_layout
                self.slots[index] = layouts[index];
            } else {
                layouts[index] = self.slots[index];
            }
        }
    }

    /// `_flush_post_removal_events`.
    fn flush_post_removal_events(&mut self) -> R<()> {
        let mut pending = std::mem::take(&mut self.pending);
        if pending.is_empty() {
            return Ok(());
        }
        // Stable sort, highest attack first.
        let attacks: Vec<i32> = pending.iter().map(|r| self.atk(r.pet)).collect();
        let mut order: Vec<usize> = (0..pending.len()).collect();
        order.sort_by(|&a, &b| attacks[b].cmp(&attacks[a]));
        let mut sorted: Vec<Record> = Vec::with_capacity(pending.len());
        let mut taken: Vec<Option<Record>> = pending.drain(..).map(Some).collect();
        for i in order {
            sorted.push(taken[i].take().unwrap());
        }
        let pending = sorted;
        let mut layouts: [Layout; 2] = [self.slots[0], self.slots[1]];

        // Priority 25: native post-removal Faint, then faint equipment.
        for record in &pending {
            let pet = record.pet;
            let team_idx = record.team_idx;
            let pet_idx = record.pet_idx;
            if record.native_post {
                let (activated, res) =
                    self.faint_trigger(pet, pet, (team_idx, pet_idx), true, true, record.tiger)?;
                let created = res.targets.clone();
                self.resync_layouts(&mut layouts, team_idx, pet_idx, &created);
                self.check_summon_triggers(pet, team_idx, activated, &res.targets)?;
            }
            if record.has_status {
                self.check_status_triggers(pet, team_idx, pet_idx)?;
                self.resync_layouts(&mut layouts, team_idx, pet_idx, &[]);
            }
        }

        // Priority 26: Shark / Fly observers.
        for record in &pending {
            let fainted = record.pet;
            let team_idx = record.team_idx;
            let pet_idx = record.pet_idx;
            for &observer in &record.post_observers {
                if !self.check_friend(team_idx, observer) || self.hp(observer) <= 0 {
                    continue;
                }
                let (activated, res) =
                    self.faint_trigger(observer, fainted, (team_idx, pet_idx), true, true, None)?;
                let created = res.targets.clone();
                self.resync_layouts(&mut layouts, team_idx, pet_idx, &created);
                self.check_summon_triggers(observer, team_idx, activated, &res.targets)?;
            }
        }
        Ok(())
    }

    /// `check_summon_triggers`.
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
        // v31: a pet given to the other team (Rat's Dirty Rat) does not set
        // off that team's friend-summoned abilities. Real client, five
        // fights: Horse, Dog and Turkey leave a gifted Dirty Rat at 1/1.
        if target_team != team_idx {
            return Ok(());
        }
        for &te in targets {
            for observer_idx in 0..NSLOTS {
                if let Some(observer) = self.slots[target_team][observer_idx] {
                    self.friend_summoned_trigger(observer, te)?;
                }
            }
        }
        Ok(())
    }

    /// `check_status_triggers`.
    fn check_status_triggers(&mut self, p: usize, team_idx: usize, pet_idx: usize) -> R<()> {
        let ability = match self.pets[p].status {
            Status::HoneyBee => self.d.honey_ability.clone(),
            Status::ExtraLife => self.d.extra_life_ability.clone(),
            _ => return Ok(()),
        };
        self.pets[p].override_ability = Some(ability);
        let (activated, res) = self.faint_trigger(p, p, (team_idx, pet_idx), false, true, None)?;
        self.check_summon_triggers(p, team_idx, activated, &res.targets)
    }

    /// `battle_phase_attack_before`.
    fn battle_phase_attack_before(&mut self) -> R<()> {
        if !self.both_fronts_alive() {
            return Ok(());
        }
        let pp = self.pet_priority.clone();
        for &(team_idx, pet_idx) in &pp {
            if pet_idx != 0 {
                continue;
            }
            if let Some(p) = self.slots[team_idx][pet_idx] {
                self.before_attack_trigger(p)?;
            }
        }
        Ok(())
    }

    /// `get_attack_idx`: both physical front slots hold a living pet.
    fn both_fronts_alive(&self) -> bool {
        (0..2).all(|t| self.slots[t][0].map(|p| self.hp(p) > 0).unwrap_or(false))
    }

    /// The "next" index of `get_attack_idx`: first living pet other than slot 0.
    fn next_idx(&self, t: usize) -> Option<usize> {
        (1..NSLOTS).find(|&i| self.slots[t][i].map(|p| self.hp(p) > 0).unwrap_or(false))
    }

    /// `_make_knockout_event`.
    fn make_knockout_event(&mut self, pet: usize, team_idx: usize) -> KoEvent {
        let tiger = self.capture_tiger_support(pet);
        let attack = self.atk(pet) as f64;
        let tie = self.rng.random_sample();
        KoEvent { pet, team_idx, tiger, attack, tie }
    }

    /// `get_splash_damage`.
    fn get_splash_damage(&mut self, target: usize) -> i32 {
        let damage = self.get_damage(target, 5);
        if matches!(self.pets[target].status, Status::CoconutShield | Status::MelonArmor) {
            self.pets[target].status = Status::None;
        }
        damage
    }

    /// `battle_phase_attack`.
    fn battle_phase_attack(&mut self) -> R<()> {
        if !self.both_fronts_alive() {
            return Ok(());
        }
        let nidx = [self.next_idx(0), self.next_idx(1)];
        let p0 = self.slots[0][0].unwrap();
        let p1 = self.slots[1][0].unwrap();

        // get_attack
        let p0a = self.get_damage(p1, self.attack_damage(p0));
        let p1a = self.get_damage(p0, self.attack_damage(p1));
        for p in [p0, p1] {
            if matches!(
                self.pets[p].status,
                Status::CoconutShield | Status::MelonArmor | Status::SteakAttack
            ) {
                self.pets[p].status = Status::None;
            }
        }

        self.hurt(p0, p1a);
        self.hurt(p1, p0a);
        if self.pets[p1].status == Status::PoisonAttack && p1a > 0 {
            self.pets[p0].health = -self.pets[p0].tmp_health;
        }
        if self.pets[p0].status == Status::PoisonAttack && p0a > 0 {
            self.pets[p1].health = -self.pets[p1].tmp_health;
        }
        self.attack_happened = true;

        let mut knockout_list = Vec::new();
        if self.hp(p0) <= 0 {
            let e = self.make_knockout_event(p1, 1);
            knockout_list.push(e);
        }
        if self.hp(p1) <= 0 {
            let e = self.make_knockout_event(p0, 0);
            knockout_list.push(e);
        }

        // Chili
        if self.pets[p0].status == Status::SplashAttack {
            if let Some(i) = nidx[1] {
                if let Some(pn1) = self.slots[1][i] {
                    let dmg = self.get_splash_damage(pn1);
                    self.hurt(pn1, dmg);
                    if self.hp(pn1) <= 0 {
                        let e = self.make_knockout_event(p0, 0);
                        knockout_list.push(e);
                    }
                }
            }
        }
        if self.pets[p1].status == Status::SplashAttack {
            if let Some(i) = nidx[0] {
                if let Some(pn0) = self.slots[0][i] {
                    let dmg = self.get_splash_damage(pn0);
                    self.hurt(pn0, dmg);
                    if self.hp(pn0) <= 0 {
                        let e = self.make_knockout_event(p1, 1);
                        knockout_list.push(e);
                    }
                }
            }
        }
        self.ko_list = knockout_list;
        Ok(())
    }

    /// `battle_phase_attack_after`.
    fn battle_phase_attack_after(&mut self) -> R<()> {
        if !self.attack_happened {
            return Ok(());
        }
        let pp = self.pet_priority.clone();
        for &(team_idx, pet_idx) in &pp {
            // Both attackers stand in physical slot 0.
            if pet_idx > 1 {
                continue;
            }
            let p = match self.slots[team_idx][pet_idx] {
                Some(p) => p,
                None => continue,
            };
            let ab = self.ability(p);
            if ab.trigger != Trigger::AfterAttack {
                continue;
            }
            if pet_idx == 0 && ab.triggered_by != TrigBy::SelfPet {
                continue;
            }
            if pet_idx == 1 && ab.triggered_by != TrigBy::FriendAhead {
                continue;
            }
            self.after_attack_trigger(p)?;
        }
        Ok(())
    }

    /// `battle_phase_knockout`.
    fn battle_phase_knockout(&mut self) -> R<()> {
        if !self.attack_happened {
            return self.flush_post_removal_events();
        }
        let mut queue = std::mem::take(&mut self.ko_list);
        let mut guard = 0usize;
        while !queue.is_empty() {
            guard += 1;
            if guard > 10_000 {
                return err("knockout chain did not terminate");
            }
            // sort(key=(attack, tie), reverse=True), stable.
            queue.sort_by(|a, b| {
                b.attack
                    .partial_cmp(&a.attack)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.tie.partial_cmp(&a.tie).unwrap_or(std::cmp::Ordering::Equal))
            });
            let event = queue.remove(0);
            let apet = event.pet;
            let team_idx = event.team_idx;
            let (activated, res) = self.knockout_trigger(apet, event.tiger)?;
            if !activated {
                continue;
            }
            let opponent = 1 - team_idx;
            let mut killed: Vec<usize> = Vec::new();
            for &target in &res.targets {
                if self.pets[target].team == opponent && self.hp(target) <= 0 && !killed.contains(&target) {
                    killed.push(target);
                }
            }
            self.battle_phase_hurt_and_faint(true)?; // phase_hurt_and_faint_k
            for _ in 0..killed.len() {
                let mut chained = self.make_knockout_event(apet, team_idx);
                if chained.tiger.is_none() {
                    if let Some(inherited) = event.tiger {
                        let it = self.pets[inherited].team;
                        if self.hp(inherited) > 0 && self.check_friend(it, inherited) {
                            chained.tiger = Some(inherited);
                        }
                    }
                }
                queue.push(chained);
            }
        }
        self.flush_post_removal_events()
    }

    // ------------------------------------------------------------------
    // Triggers (sapai/pets.py)
    // ------------------------------------------------------------------

    /// The `maxTriggers` gate shared by the triggers. Returns false when spent.
    fn take_trigger(&mut self, p: usize, ab: &Ability) -> bool {
        if let Some(max) = ab.max_triggers {
            if self.pets[p].ability_counter >= max {
                return false;
            }
            self.pets[p].ability_counter += 1;
        }
        true
    }

    fn own_idx(&self, p: usize) -> R<usize> {
        match self.get_idx(self.pets[p].team, p) {
            Some(i) => Ok(i),
            None => err("get_idx: pet not found on its team"),
        }
    }

    /// `Pet.sob_trigger`.
    fn sob_trigger(&mut self, p: usize) -> R<()> {
        let ab = self.ability(p);
        if ab.trigger != Trigger::StartOfBattle {
            return Ok(());
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.tiger_func(false, p, Idx::Int(pet_idx), true, Te::Team, None, None)?;
        Ok(())
    }

    /// `Pet.before_attack_trigger`.
    fn before_attack_trigger(&mut self, p: usize) -> R<()> {
        let ab = self.ability(p);
        if ab.trigger != Trigger::BeforeAttack {
            return Ok(());
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.tiger_func(false, p, Idx::Int(pet_idx), true, Te::Team, None, None)?;
        Ok(())
    }

    /// `Pet.after_attack_trigger`.
    fn after_attack_trigger(&mut self, p: usize) -> R<()> {
        let ab = self.ability(p);
        if ab.trigger != Trigger::AfterAttack {
            return Ok(());
        }
        match ab.triggered_by {
            TrigBy::FriendAhead => {
                let t = self.pets[p].team;
                let idx = self.own_idx(p)?;
                let ahead = self.get_ahead(t, idx, 1);
                if ahead.is_empty() {
                    return Ok(());
                }
                if self.get_idx(t, ahead[0]) != Some(0) {
                    return Ok(());
                }
            }
            TrigBy::SelfPet => {}
            _ => return err("Unsupported AfterAttack trigger source"),
        }
        if !self.take_trigger(p, &ab) {
            return Ok(());
        }
        let pet_idx = self.own_idx(p)?;
        self.tiger_func(false, p, Idx::Int(pet_idx), true, Te::Team, None, None)?;
        Ok(())
    }

    /// `Pet.hurt_trigger`.
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
        self.tiger_func(false, p, Idx::Int(pet_idx), true, Te::Team, None, None)?;
        Ok(())
    }

    /// `Pet.friend_hurt_trigger` (Wolverine).
    fn friend_hurt_trigger(&mut self, p: usize, trigger: usize) -> R<()> {
        let ab = self.ability(p);
        if ab.trigger != Trigger::FourFriendsHurt {
            return Ok(());
        }
        let t = self.pets[p].team;
        if trigger == p || !self.check_friend(t, p) {
            return Ok(());
        }
        self.pets[p].friends_hurt_counter += 1;
        if self.pets[p].friends_hurt_counter < 4 {
            return Ok(());
        }
        self.pets[p].friends_hurt_counter = 0;
        // v31: the activation waits until no hurt or faint ability is left
        // (see battle_phase_hurt_and_faint). Real client, fight b836f265.
        self.pending_counters.push((p, trigger));
        Ok(())
    }

    /// Run one waiting Wolverine activation. Returns false when none is left.
    fn run_pending_counter(&mut self) -> R<bool> {
        if self.pending_counters.is_empty() {
            return Ok(false);
        }
        let (p, trigger) = self.pending_counters.remove(0);
        let t = self.pets[p].team;
        if self.check_friend(t, p) {
            let pet_idx = self.own_idx(p)?;
            self.tiger_func(false, p, Idx::Int(pet_idx), true, Te::Pet(trigger), None, None)?;
        }
        Ok(true)
    }

    /// `Pet.knockout_trigger`.
    fn knockout_trigger(&mut self, p: usize, tiger_override: Option<usize>) -> R<(bool, EffResult)> {
        let ab = self.ability(p);
        if ab.trigger != Trigger::KnockOut {
            return Ok((false, EffResult::none()));
        }
        let t = self.pets[p].team;
        if !self.check_friend(t, p) {
            let spec = ab.effect.target.as_ref().or(ab.effect.to.as_ref());
            if spec.map(|s| s.kind) == Some(TargetKind::SelfPet) {
                return Ok((false, EffResult::none()));
            }
        }
        if !self.take_trigger(p, &ab) {
            return Ok((false, EffResult::none()));
        }
        let pet_idx = self.get_idx(t, p).unwrap_or(0);
        let res = self.tiger_func(false, p, Idx::Int(pet_idx), true, Te::Team, None, tiger_override)?;
        Ok((true, res))
    }

    /// `Pet.friend_summoned_trigger`.
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
        self.tiger_func(false, p, Idx::Int(pet_idx), false, Te::Pet(trigger), None, None)?;
        Ok(())
    }

    /// `Pet.gains_perk_trigger` (Rabbit, Seal).
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
        // Direct call: no Tiger repeat, friendly team only.
        self.run_effect(p, Idx::Int(pet_idx), false, Te::Pet(trigger), None)?;
        Ok(())
    }

    /// `Pet.faint_trigger`.
    fn faint_trigger(
        &mut self,
        p: usize,
        trigger: usize,
        te_idx: (usize, usize),
        has_oteam: bool,
        remove_before_effect: bool,
        tiger_override: Option<usize>,
    ) -> R<(bool, EffResult)> {
        let ab = self.ability(p);
        if ab.trigger != Trigger::Faint {
            return Ok((false, EffResult::none()));
        }
        let shark = self.d.shark;
        let fly = self.d.fly;
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
            if self.pets[trigger].kind == self.d.zombie_fly {
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
        if is_summon {
            let res = self.tiger_func(
                remove_before_effect,
                p,
                pet_idx,
                has_oteam,
                Te::Pet(trigger),
                Some(te_idx),
                tiger_override,
            )?;
            if res.possible_empty_nested {
                if ab.max_triggers.is_some() {
                    self.pets[p].ability_counter -= 1;
                }
                return Ok((false, res));
            }
            Ok((true, res))
        } else {
            let res = self.tiger_func(
                remove_before_effect,
                p,
                pet_idx,
                has_oteam,
                Te::Pet(trigger),
                None,
                tiger_override,
            )?;
            Ok((true, res))
        }
    }

    /// `tiger_func`.
    #[allow(clippy::too_many_arguments)]
    fn tiger_func(
        &mut self,
        te_fainted: bool,
        apet: usize,
        apet_idx: Idx,
        has_oteam: bool,
        te: Te,
        te_idx: Option<(usize, usize)>,
        tiger_override: Option<usize>,
    ) -> R<EffResult> {
        let ab = self.ability(apet);
        // A Bee comes from a status, not an ability: never repeated.
        if ab.effect.pet == Some(self.d.bee) {
            return self.run_effect(apet, apet_idx, has_oteam, te, te_idx);
        }
        if matches!(ab.effect.kind, EffectKind::RespawnPet | EffectKind::SummonStoredPets) {
            return self.run_effect(apet, apet_idx, has_oteam, te, te_idx);
        }
        let t = self.pets[apet].team;
        let (pet_behind, tiger_was_captured) = if let Some(tg) = tiger_override {
            (Some(tg), true)
        } else if self.check_friend(t, apet) {
            (self.get_behind(t, apet), false)
        } else {
            (None, false)
        };
        let ability_before = ab.clone();
        let override_before = self.pets[apet].override_ability.is_some();

        if te_fainted {
            if let Te::Pet(tp) = te {
                if self.check_friend(t, tp) {
                    self.team_remove(t, tp)?;
                }
            }
        }

        let first = self.run_effect(apet, apet_idx, has_oteam, te, te_idx)?;

        let pb = match pet_behind {
            Some(pb) => pb,
            None => return Ok(first),
        };
        if self.pets[pb].kind != self.d.tiger {
            return Ok(first);
        }
        if !tiger_was_captured && self.hp(pb) <= 0 && !self.check_friend(t, pb) {
            return Ok(first);
        }

        let post_override = self.pets[apet].override_ability.clone();
        let original_level = self.pets[apet].level;
        let tiger_level = self.pets[pb].level;
        self.pets[apet].level = tiger_level;
        let repeat = (|| -> R<EffResult> {
            if let Some(cf) = ability_before.copied_from {
                let lvl = tiger_level.clamp(1, 3) as usize;
                match &self.d.pets[cf].abilities[lvl - 1] {
                    Some(a) => self.pets[apet].override_ability = Some(a.clone()),
                    None => return err("copied pet has no ability at the Tiger's level"),
                }
            } else if override_before {
                self.pets[apet].override_ability = Some(ability_before.clone());
            } else {
                self.pets[apet].override_ability = None;
            }
            let shifted = te_idx.map(|(_, i)| (0usize, i + first.targets.len()));
            self.run_effect(apet, apet_idx, has_oteam, te, shifted)
        })();
        self.pets[apet].level = original_level;
        self.pets[apet].override_ability = post_override;
        let second = repeat?;
        let mut targets = first.targets;
        targets.extend(second.targets);
        Ok(EffResult { targets, possible_empty_nested: false })
    }

    // ------------------------------------------------------------------
    // Effects (sapai/effects.py)
    // ------------------------------------------------------------------

    fn run_effect(
        &mut self,
        apet: usize,
        apet_idx: Idx,
        has_oteam: bool,
        te: Te,
        te_idx: Option<(usize, usize)>,
    ) -> R<EffResult> {
        let ab = self.ability(apet);
        self.run_eff(&ab.effect, apet, apet_idx, has_oteam, te, te_idx)
    }

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
            EffectKind::SummonStoredPets => self.summon_stored_pets(apet, has_oteam, te_idx),
            EffectKind::RespawnPet => self.respawn_pet(eff, apet, te_idx),
            EffectKind::TransferStats => self.transfer_stats(eff, apet, apet_idx, has_oteam, te),
            EffectKind::ReduceHealth => self.reduce_health(eff, apet, apet_idx, has_oteam, te),
            EffectKind::ApplyStatus => self.apply_status(eff, apet, apet_idx, has_oteam, te),
            EffectKind::AllOf => {
                let mut targets = Vec::new();
                for sub in &eff.effects {
                    let r = self.run_eff(sub, apet, apet_idx, has_oteam, te, None)?;
                    targets.extend(r.targets);
                }
                Ok(EffResult::of(targets))
            }
            EffectKind::Swallow => self.swallow(eff, apet, apet_idx, has_oteam, te),
            EffectKind::GainHealthPercentHighestFriend => self.gain_health_percent(eff, apet),
            EffectKind::NoneKind => Ok(EffResult::none()),
            EffectKind::TransferAbility | EffectKind::Other => {
                err(format!("effect {} is not supported in battle", eff.kind_name))
            }
        }
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
        let fidx: Vec<usize> = (0..NSLOTS)
            .filter(|&i| self.slots[ft][i].map(|p| self.hp(p) > 0).unwrap_or(false))
            .collect();
        let oidx: Vec<usize> = if has_oteam {
            (0..NSLOTS)
                .filter(|&i| self.slots[ot][i].map(|p| self.hp(p) > 0).unwrap_or(false))
                .collect()
        } else {
            Vec::new()
        };
        let fpet = |s: &Self, i: usize| s.slots[ft][i].unwrap();
        let opet = |s: &Self, i: usize| s.slots[ot][i].unwrap();
        let int_idx = |idx: Idx| -> R<usize> {
            match idx {
                Idx::Int(i) => Ok(i),
                Idx::List(..) => err("target needs the activating pet's slot, but it has left the team"),
            }
        };
        match target.kind {
            TargetKind::AdjacentAnimals => {
                // Opponent reversed, then friendly: the opposing front pet is
                // the next pet ahead of the friendly front.
                let mut all: Vec<Option<usize>> = Vec::with_capacity(10);
                if has_oteam {
                    for i in (0..NSLOTS).rev() {
                        all.push(self.slots[ot][i]);
                    }
                }
                let base = all.len();
                for i in 0..NSLOTS {
                    all.push(self.slots[ft][i]);
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
            TargetKind::AllOtherFriends => {
                Ok(fidx.iter().map(|&i| fpet(self, i)).filter(|&p| p != apet).collect())
            }
            TargetKind::EachEnemy => Ok(oidx.iter().map(|&i| opet(self, i)).collect()),
            TargetKind::EachFriend => Ok(fidx
                .iter()
                .filter(|&&i| apet_idx != Idx::Int(i))
                .map(|&i| fpet(self, i))
                .collect()),
            TargetKind::EachShopAnimal => Ok(Vec::new()),
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
                let choice = range[self.rng.choice_noreplace(range.len())];
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
                let c = self.rng.choice_noreplace(combos.len());
                Ok(combos[c].iter().map(|&i| opet(self, i)).collect())
            }
            TargetKind::RandomFriend => {
                if fidx.is_empty() {
                    return Ok(Vec::new());
                }
                let pool: Vec<usize> = fidx.iter().copied().filter(|&i| apet_idx != Idx::Int(i)).collect();
                let n = n.min(pool.len() as i64);
                if n == 0 {
                    return Ok(Vec::new());
                }
                let combos = combinations(&pool, n.max(0) as usize);
                if combos.is_empty() {
                    return Ok(Vec::new());
                }
                let c = self.rng.choice_noreplace(combos.len());
                Ok(combos[c].iter().map(|&i| fpet(self, i)).collect())
            }
            TargetKind::FrontMostFriend => Ok(fidx.first().map(|&i| vec![fpet(self, i)]).unwrap_or_default()),
            TargetKind::SelfPet => Ok(vec![apet]),
            TargetKind::StrongestFriend => {
                if fidx.is_empty() {
                    return err("StrongestFriend with no living friend");
                }
                let stats: Vec<i32> =
                    fidx.iter().map(|&i| self.atk(fpet(self, i)) + self.hp(fpet(self, i))).collect();
                let best = *stats.iter().max().unwrap();
                // Reference quirk: indexes the team by position in the living list.
                let max_idx: Vec<usize> = (0..stats.len()).filter(|&k| stats[k] == best).collect();
                let choice = self.rng.choice_noreplace(max_idx.len());
                match self.slots[ft][max_idx[choice]] {
                    Some(p) => Ok(vec![p]),
                    None => err("StrongestFriend picked an empty slot"),
                }
            }
            TargetKind::TriggeringEntity => match te {
                Te::Pet(p) => Ok(vec![p]),
                Te::None => Ok(Vec::new()),
                Te::Team => err("TriggeringEntity is a team"),
            },
            TargetKind::NoneKind => Ok(Vec::new()),
            TargetKind::Other => err(format!("Target {} implementation not found", target.kind_name)),
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
            if eff.double_against_tier.is_some() && eff.double_against_tier == self.d.pets[self.pets[tp].kind].tier {
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

    /// `ReduceHealth` (Skunk).
    fn reduce_health(&mut self, eff: &Effect, apet: usize, apet_idx: Idx, has_oteam: bool, te: Te) -> R<EffResult> {
        let target = self.get_target(apet, apet_idx, eff, has_oteam, te, false)?;
        let per = match eff.percentage {
            Some(p) => p as f64 * 0.01,
            None => return err("ReduceHealth without percentage"),
        };
        for &tp in &target {
            let h = self.hp(tp);
            if h > 0 {
                let v = (h as f64 * (1.0 - per)).floor() as i32;
                self.pets[tp].health = v.max(1);
            }
        }
        Ok(EffResult::of(target))
    }

    /// `ApplyStatus`.
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
            if status.is_perk() {
                // A perk gained during battle triggers Seal and Rabbit.
                let team = self.pets[tp].team;
                let snapshot = self.slots[team];
                for slot in snapshot.iter() {
                    if let Some(obs) = *slot {
                        let oab = self.ability(obs);
                        if matches!(oab.trigger, Trigger::GainsPerk | Trigger::EatsShopFoodOrGainsPerk) {
                            self.gains_perk_trigger(obs, tp)?;
                        }
                    }
                }
            }
        }
        Ok(EffResult::of(applied))
    }

    /// `GainHealthPercentHighestFriend` (Crab).
    fn gain_health_percent(&mut self, eff: &Effect, apet: usize) -> R<EffResult> {
        let ft = self.pets[apet].team;
        let friends: Vec<usize> = self.slots[ft]
            .iter()
            .filter_map(|s| *s)
            .filter(|&p| p != apet && self.hp(p) > 0)
            .collect();
        if friends.is_empty() {
            return Ok(EffResult::none());
        }
        let highest = friends.iter().map(|&p| self.hp(p)).max().unwrap();
        let candidates: Vec<usize> = friends.into_iter().filter(|&p| self.hp(p) == highest).collect();
        let chosen = self.rng.choice_replace(candidates.len());
        let source = candidates[chosen];
        let percentage = match eff.percentage {
            Some(p) => p as f64 * 0.01,
            None => return err("Crab effect without percentage"),
        };
        let gain = (self.hp(source) as f64 * percentage).ceil() as i32;
        self.pets[apet].health = (self.pets[apet].health + gain).min(50);
        Ok(EffResult::of(vec![apet]))
    }

    /// `TransferStats` (Dodo; Butterfly).
    fn transfer_stats(&mut self, eff: &Effect, apet: usize, apet_idx: Idx, has_oteam: bool, te: Te) -> R<EffResult> {
        let target = self.get_target(apet, apet_idx, eff, has_oteam, te, false)?;
        let from_self = eff.from.as_ref().map(|f| f.kind) == Some(TargetKind::SelfPet);
        let scaled = |value: i32, pct: Option<i64>| -> i32 {
            match pct {
                Some(p) => (value as f64 * (p as f64 * 0.01)) as i32,
                None => value,
            }
        };
        for &entry in &target {
            if from_self {
                if eff.copy_attack {
                    let add = scaled(self.atk(apet), eff.percentage).max(1);
                    self.pets[entry].attack = (self.pets[entry].attack + add).min(50);
                }
                if eff.copy_health {
                    return err("TransferStats from Self cannot copy health");
                }
            } else {
                let from = self.get_target(apet, apet_idx, eff, has_oteam, te, true)?;
                if from.is_empty() {
                    continue;
                }
                let src = from[0];
                if eff.copy_attack {
                    self.pets[apet].attack = scaled(self.atk(src), eff.percentage);
                }
                if eff.copy_health {
                    self.pets[apet].health = scaled(self.hp(src), eff.percentage);
                }
            }
        }
        Ok(EffResult::of(target))
    }

    /// `Swallow` (Whale).
    fn swallow(&mut self, eff: &Effect, apet: usize, apet_idx: Idx, has_oteam: bool, te: Te) -> R<EffResult> {
        let target = self.get_target(apet, apet_idx, eff, has_oteam, te, false)?;
        if target.is_empty() {
            return Ok(EffResult::of(target));
        }
        let output_level = self.pets[apet].level;
        for &tp in &target {
            let stored = Stored {
                pet: self.pets[tp].kind,
                with_attack: self.atk(tp),
                with_health: self.hp(tp),
                with_level: output_level,
            };
            self.pets[apet].swallowed.push(stored);
            self.pets[apet].override_ability = Some(self.d.stored_ability.clone());
            self.pets[tp].health = -1;
        }
        let ft = self.pets[apet].team;
        self.move_forward(ft, 0, 10)?;
        Ok(EffResult::of(target))
    }

    /// The team shuffle shared by the friendly summons: `nahead` pets stay in
    /// front of the summon position, everything else is packed to the back.
    fn open_summon_gap(&mut self, ft: usize, te_slot: usize) -> R<()> {
        let nahead = self.get_ahead(ft, te_slot, 5).len();
        let npets = self.team_len(ft);
        self.move_backward(ft)?;
        let end_idx = (NSLOTS - npets) + nahead;
        self.move_forward(ft, 0, end_idx)
    }

    fn max_empty(&self, t: usize) -> Option<usize> {
        (0..NSLOTS).rev().find(|&i| self.slots[t][i].is_none())
    }

    /// `SummonPet`, with the effect passed explicitly (see `SummonStoredPets`).
    fn summon_pet(&mut self, eff: &Effect, apet: usize, has_oteam: bool, te_idx: Option<(usize, usize)>) -> R<EffResult> {
        self.summon_pet_fields(
            apet,
            has_oteam,
            te_idx,
            eff.pet,
            eff.team,
            eff.n,
            eff.with_attack.map(|v| v as i32),
            eff.with_health.map(|v| v as i32),
            eff.with_level.map(|v| v as i32),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn summon_pet_fields(
        &mut self,
        apet: usize,
        has_oteam: bool,
        te_idx: Option<(usize, usize)>,
        pet: Option<usize>,
        team: Option<TeamSel>,
        n: Option<i64>,
        with_attack: Option<i32>,
        with_health: Option<i32>,
        with_level: Option<i32>,
    ) -> R<EffResult> {
        let te_idx = match te_idx {
            Some(t) => t,
            None => return err("Indices of triggering entity must be provided as te_idx"),
        };
        let spet_kind = match pet {
            Some(k) => k,
            None => return err("SummonPet without a known pet"),
        };
        let ft = self.pets[apet].team;
        let target_team = match team {
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
        let mut count = n.unwrap_or(1);
        if n.is_none() {
            if spet_kind == self.d.ram {
                count = 2;
            } else if spet_kind == self.d.chick {
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
            let sp = self.new_pet(spet_kind, target_team);
            self.slots[target_team][slot] = Some(sp);
            if let Some(v) = with_attack {
                self.pets[sp].attack = v;
            }
            if let Some(v) = with_health {
                self.pets[sp].health = v;
            }
            if let Some(v) = with_level {
                self.pets[sp].level = v;
            }
            if spet_kind == self.d.chick {
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
        let possible = match self.d.tier_std.get(&tier) {
            Some(p) if !p.is_empty() => p,
            _ => return err("no pets for SummonRandomPet tier"),
        };
        let chosen = possible[self.rng.choice_replace(possible.len())];
        let te_idx = match te_idx {
            Some(t) => t,
            None => return err("SummonRandomPet without te_idx"),
        };
        self.open_summon_gap(ft, te_idx.1)?;
        let slot = match self.max_empty(ft) {
            Some(s) => s,
            None => return Ok(EffResult::none()),
        };
        let sp = self.new_pet(chosen, ft);
        self.slots[ft][slot] = Some(sp);
        let sattack = match eff.base_attack {
            Some(v) => v as i32,
            None => self.pets[sp].attack,
        };
        let shealth = match eff.base_health {
            Some(v) => v as i32,
            None => self.pets[sp].health,
        };
        if let Some(l) = eff.level {
            self.pets[sp].level = l as i32;
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
        let sp = self.new_pet(kind, ft);
        self.slots[ft][slot] = Some(sp);
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

    /// `SummonStoredPets` (Whale release).
    fn summon_stored_pets(&mut self, apet: usize, has_oteam: bool, te_idx: Option<(usize, usize)>) -> R<EffResult> {
        let ft = self.pets[apet].team;
        if self.check_friend(ft, apet) {
            self.team_remove(ft, apet)?;
        }
        let stored = std::mem::take(&mut self.pets[apet].swallowed);
        let mut targets = Vec::new();
        for s in &stored {
            let r = self.summon_pet_fields(
                apet,
                has_oteam,
                te_idx,
                Some(s.pet),
                Some(TeamSel::Friendly),
                None,
                Some(s.with_attack),
                Some(s.with_health),
                Some(s.with_level),
            )?;
            targets.extend(r.targets);
        }
        let empty = targets.is_empty();
        Ok(EffResult { targets, possible_empty_nested: empty })
    }
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

/// Run one fight exactly as `tools/gen_golden.py` steps the reference.
pub fn simulate(d: &Data, player: &[PetSpec], opponent: &[PetSpec], seed: u32, trace: bool) -> Result<Outcome, EngineError> {
    let mut fight = Fight::new(d, player, opponent, seed)?;
    let mut steps = Vec::new();
    if trace {
        steps.push(fight.boards());
    }
    fight.start()?;
    if trace {
        steps.push(fight.boards());
    }
    let mut rounds = 0usize;
    loop {
        fight.priority_refresh();
        let more = fight.attack()?;
        rounds += 1;
        if trace {
            steps.push(fight.boards());
        }
        if !more {
            break;
        }
        if rounds > MAX_ROUNDS {
            return err("battle did not terminate");
        }
    }
    let result = fight.check_battle_result();
    if result < 0 {
        return err("battle ended without a result");
    }
    let probe = fight.random_sample();
    Ok(Outcome { outcome: result as u8, final_boards: fight.boards(), trace: steps, probe, rounds })
}

/// A pet as the training environment hands it to a battle: one slot of
/// `Team.copy()` of the shop team (`Pet.copy` keeps counters and a Parrot's
/// copied ability). Temporary shop buffs are still separate here;
/// `Battle.__init__` folds them.
#[derive(Clone)]
pub struct EnvPet {
    pub kind: usize,
    pub attack: i32,
    pub health: i32,
    pub tmp_attack: i32,
    pub tmp_health: i32,
    pub level: i32,
    pub status: Status,
    pub ability_counter: i64,
    pub friends_hurt_counter: i32,
    pub hurt: i32,
    pub override_ability: Option<Arc<Ability>>,
}

/// `Battle(t0, t1, seed=seed, fast_training=True).battle()` for two shop
/// teams given slot by slot (empty slots stay where they are until the
/// battle's own first move). Same rules and draws as `simulate`.
pub fn battle_env(d: &Data, teams: [&[Option<EnvPet>; NSLOTS]; 2], seed: u32) -> Result<u8, EngineError> {
    let mut f = Fight {
        d,
        pets: Vec::with_capacity(24),
        slots: [[None; NSLOTS]; 2],
        rng: Mt::new(seed),
        pet_priority: Vec::new(),
        pending: Vec::new(),
        pending_counters: Vec::new(),
        attack_happened: false,
        ko_list: Vec::new(),
    };
    for (team, slots) in teams.iter().enumerate() {
        for (index, slot) in slots.iter().enumerate() {
            if let Some(spec) = slot {
                let p = f.new_pet(spec.kind, team);
                let pet = &mut f.pets[p];
                // Battle.__init__: fold temporary shop buffs into the battle copy.
                pet.attack = (spec.attack + spec.tmp_attack).min(50);
                pet.health = (spec.health + spec.tmp_health).min(50);
                pet.level = spec.level;
                pet.status = spec.status;
                pet.ability_counter = spec.ability_counter;
                pet.friends_hurt_counter = spec.friends_hurt_counter;
                pet.hurt = spec.hurt;
                pet.override_ability = spec.override_ability.clone();
                f.slots[team][index] = Some(p);
            }
        }
    }
    f.pet_priority = f.update_pet_priority();
    f.start()?;
    let mut rounds = 0usize;
    loop {
        f.priority_refresh();
        let more = f.attack()?;
        rounds += 1;
        if !more {
            break;
        }
        if rounds > MAX_ROUNDS {
            return err("battle did not terminate");
        }
    }
    let result = f.check_battle_result();
    if result < 0 {
        return err("battle ended without a result");
    }
    Ok(result as u8)
}
