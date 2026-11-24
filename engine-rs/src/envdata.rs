//! Tables for the shop phase and the observation encoder, exported from the
//! Python reference by `tools/export_env_data.py` into
//! `data/turtle_v31_env.json`. The battle tables (`data::Data`) come from
//! `data/turtle_v31.json`; both files are compiled into the library so the
//! batch interface needs no data directory.

use crate::data::{Data, Status};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

pub const MAX_SHOP_SLOTS: usize = 10;
pub const MAX_TEAM_SLOTS: usize = 5;

/// One row of `foodRows` (a `sapai.foods.Food` built from its name).
#[derive(Clone, Debug)]
pub struct FoodData {
    pub name: String,
    pub item_id: i64,
    pub cost: i32,
    pub attack: i32,
    pub health: i32,
    pub status: Status,
    pub until_end_of_battle: bool,
    /// `Shop.roll._food_tier`.
    pub sort_tier: i64,
    /// Tier as the observation encoder reads it (0 for tokens).
    pub obs_tier: i64,
    /// Bought without a team target (Canned Food and RandomFriend foods).
    pub no_target: bool,
    /// `effect["target"]["n"]` for the RandomFriend foods.
    pub target_n: i64,
    pub random_friend: bool,
}

/// `shop.get_shop_rules(turn)` for the Standard (Turtle) pack.
#[derive(Clone, Debug)]
pub struct TurnRule {
    pub pslots: usize,
    pub fslots: usize,
    pub tier_avail: i64,
    pub levelup_tier: i64,
    /// Indices into `Data::pets`.
    pub avail_pets: Vec<usize>,
    /// Indices into `EnvData::foods`.
    pub avail_foods: Vec<usize>,
    /// The equal weight of every pet / food and `math.fsum` of the weights.
    pub pp: f64,
    pub pp_sum: f64,
    pub fp: f64,
    pub fp_sum: f64,
}

pub struct EnvData {
    pub foods: Vec<FoodData>,
    pub food_by_name: HashMap<String, usize>,
    /// Index 1..=11.
    pub turns: Vec<TurnRule>,
    /// `pet_tier_lookup_std[tier]` (index 1..=6), in the reference's order.
    pub pet_tier_std: Vec<Vec<usize>>,
    /// Observation item id per `Data::pets` index.
    pub pet_item_id: Vec<i64>,
    /// Membership in `tiers.pet_tier_lookup[1]` per `Data::pets` index.
    pub pet_tier1: Vec<bool>,
    /// Observation status id per `Status as usize`.
    pub status_id: [i64; 13],
    pub canned: usize,
    pub chocolate: usize,
    pub pill: usize,
    pub scorpion: usize,
    pub cat: usize,
    pub bison: usize,
    pub dirty_rat: usize,
}

pub const STATUS_ORDER: [Status; 13] = [
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

fn need<'a>(v: &'a Value, key: &str) -> Result<&'a Value, String> {
    v.get(key).ok_or_else(|| format!("env table: missing {key}"))
}

impl EnvData {
    pub fn from_json(d: &Data, text: &str) -> Result<EnvData, String> {
        let root: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let item_ids = need(&root, "itemIds")?.as_object().ok_or("itemIds")?;
        let item_id = |name: &str| -> Result<i64, String> {
            item_ids.get(name).and_then(|v| v.as_i64()).ok_or_else(|| format!("no item id for {name}"))
        };
        // Foods in name order (serde_json maps are sorted by key).
        let rows = need(&root, "foodRows")?.as_object().ok_or("foodRows")?;
        let mut foods = Vec::new();
        let mut food_by_name = HashMap::new();
        for (name, row) in rows {
            let int = |k: &str| -> Result<i64, String> {
                row.get(k).and_then(|v| v.as_i64()).ok_or_else(|| format!("food {name}: {k}"))
            };
            let status = row
                .get("status")
                .and_then(|s| s.as_str())
                .and_then(Status::parse)
                .ok_or_else(|| format!("food {name}: status"))?;
            let target_kind = row.get("targetKind").and_then(|s| s.as_str()).unwrap_or("");
            food_by_name.insert(name.clone(), foods.len());
            foods.push(FoodData {
                name: name.clone(),
                item_id: item_id(name)?,
                cost: int("cost")? as i32,
                attack: int("attack")? as i32,
                health: int("health")? as i32,
                status,
                until_end_of_battle: row.get("untilEndOfBattle").and_then(|b| b.as_bool()) == Some(true),
                sort_tier: int("sortTier")?,
                obs_tier: int("obsTier")?,
                no_target: row.get("noTarget").and_then(|b| b.as_bool()) == Some(true),
                target_n: int("targetN")?,
                random_friend: target_kind == "RandomFriend",
            });
        }
        let pet = |name: &str| -> Result<usize, String> {
            d.by_name.get(name).copied().ok_or_else(|| format!("pet {name} missing from battle tables"))
        };
        let food = |name: &str| -> Result<usize, String> {
            food_by_name.get(name).copied().ok_or_else(|| format!("food {name} missing"))
        };
        let turns_obj = need(&root, "turns")?;
        let mut turns = vec![];
        for turn in 0..=11 {
            if turn == 0 {
                turns.push(TurnRule {
                    pslots: 0,
                    fslots: 0,
                    tier_avail: 0,
                    levelup_tier: 0,
                    avail_pets: vec![],
                    avail_foods: vec![],
                    pp: 0.0,
                    pp_sum: 0.0,
                    fp: 0.0,
                    fp_sum: 0.0,
                });
                continue;
            }
            let row = need(turns_obj, &turn.to_string())?;
            let int = |k: &str| -> Result<i64, String> {
                row.get(k).and_then(|v| v.as_i64()).ok_or_else(|| format!("turn {turn}: {k}"))
            };
            let floats = |k: &str| -> Result<Vec<f64>, String> {
                Ok(row
                    .get(k)
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("turn {turn}: {k}"))?
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(f64::NAN))
                    .collect())
            };
            let names = |k: &str| -> Result<Vec<String>, String> {
                Ok(row
                    .get(k)
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("turn {turn}: {k}"))?
                    .iter()
                    .map(|x| x.as_str().unwrap_or("").to_string())
                    .collect())
            };
            let pp = floats("pp")?;
            let fp = floats("fp")?;
            if pp.is_empty() || fp.is_empty() || pp.iter().any(|x| *x != pp[0]) || fp.iter().any(|x| *x != fp[0]) {
                return Err(format!("turn {turn}: the port expects equal roll weights"));
            }
            let avail_pets = names("availPets")?.iter().map(|n| pet(n)).collect::<Result<Vec<_>, _>>()?;
            let avail_foods = names("availFoods")?.iter().map(|n| food(n)).collect::<Result<Vec<_>, _>>()?;
            turns.push(TurnRule {
                pslots: int("pslots")? as usize,
                fslots: int("fslots")? as usize,
                tier_avail: int("tierAvail")?,
                levelup_tier: int("levelupTier")?,
                avail_pets,
                avail_foods,
                pp: pp[0],
                pp_sum: row.get("ppSum").and_then(|v| v.as_f64()).ok_or("ppSum")?,
                fp: fp[0],
                fp_sum: row.get("fpSum").and_then(|v| v.as_f64()).ok_or("fpSum")?,
            });
        }
        let tier_obj = need(&root, "petTierStd")?;
        let mut pet_tier_std = vec![vec![]; 7];
        for (tier, list) in pet_tier_std.iter_mut().enumerate().skip(1) {
            for name in need(tier_obj, &tier.to_string())?.as_array().ok_or("petTierStd list")? {
                list.push(pet(name.as_str().unwrap_or(""))?);
            }
        }
        let mut pet_item_id = Vec::with_capacity(d.pets.len());
        for p in &d.pets {
            // "pet-none" has no id of its own; item_to_id.get(name, 0).
            pet_item_id.push(item_ids.get(&p.name).and_then(|v| v.as_i64()).unwrap_or(0));
        }
        let mut pet_tier1 = vec![false; d.pets.len()];
        for name in need(&root, "petTier1")?.as_array().ok_or("petTier1")? {
            if let Some(k) = d.by_name.get(name.as_str().unwrap_or("")) {
                pet_tier1[*k] = true;
            }
        }
        let status_ids = need(&root, "statusIds")?;
        let mut status_id = [0i64; 13];
        for (i, s) in STATUS_ORDER.iter().enumerate() {
            debug_assert_eq!(*s as usize, i);
            status_id[i] = status_ids.get(s.name()).and_then(|v| v.as_i64()).ok_or("statusIds")?;
        }
        // The action layout and limits are compiled in; refuse tables that disagree.
        let codec = need(&root, "codec")?;
        for (key, want) in [
            ("END_TURN", 0i64),
            ("BUY_PET", 1),
            ("BUY_FOOD", 61),
            ("COMBINE", 121),
            ("SELL", 146),
            ("ROLL", 151),
            ("FREEZE", 152),
            ("UNFREEZE", 162),
            ("MOVE", 172),
            ("NUM_ACTIONS", 197),
        ] {
            if codec.get(key).and_then(|v| v.as_i64()) != Some(want) {
                return Err(format!("action codec changed: {key}"));
            }
        }
        let env = need(&root, "env")?;
        for (key, want) in
            [("MAX_TURN", 25i64), ("MAX_LIVES", 6), ("MAX_REORDERS_PER_TURN", 10), ("MAX_UNSPENT_GOLD_TO_END", 0)]
        {
            if env.get(key).and_then(|v| v.as_i64()) != Some(want) {
                return Err(format!("environment constant changed: {key}"));
            }
        }
        Ok(EnvData {
            canned: food("food-canned-food")?,
            chocolate: food("food-chocolate")?,
            pill: food("food-sleeping-pill")?,
            scorpion: pet("pet-scorpion")?,
            cat: pet("pet-cat")?,
            bison: pet("pet-bison")?,
            dirty_rat: pet("pet-dirty-rat")?,
            foods,
            food_by_name,
            turns,
            pet_tier_std,
            pet_item_id,
            pet_tier1,
            status_id,
        })
    }
}

/// Battle tables plus environment tables.
pub struct Tables {
    pub d: Data,
    pub e: EnvData,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

impl Tables {
    /// The tables compiled into the library.
    pub fn global() -> &'static Tables {
        TABLES.get_or_init(|| {
            let d = Data::from_json(
                include_str!("../data/turtle_v31.json"),
                include_str!("../data/turtle_v31_tiers.json"),
            )
            .expect("embedded battle tables");
            let e = EnvData::from_json(&d, include_str!("../data/turtle_v31_env.json")).expect("embedded env tables");
            Tables { d, e }
        })
    }
}
