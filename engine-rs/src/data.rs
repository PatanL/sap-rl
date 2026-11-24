//! Rule tables exported from the Python reference (`tools/export_data.py`).
//! Abilities stay data-driven: every pet's per-level ability dictionary is
//! parsed into `Ability`, and the engine interprets it the way
//! `sapai/effects.py` interprets the dictionary.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Status {
    None,
    Bread,
    Cake,
    BoneAttack,
    CoconutShield,
    ExtraLife,
    GarlicArmor,
    HoneyBee,
    MelonArmor,
    PoisonAttack,
    SplashAttack,
    SteakAttack,
    Weak,
}

impl Status {
    pub fn parse(s: &str) -> Option<Status> {
        Some(match s {
            "none" => Status::None,
            "status-bread" => Status::Bread,
            "status-cake" => Status::Cake,
            "status-bone-attack" => Status::BoneAttack,
            "status-coconut-shield" => Status::CoconutShield,
            "status-extra-life" => Status::ExtraLife,
            "status-garlic-armor" => Status::GarlicArmor,
            "status-honey-bee" => Status::HoneyBee,
            "status-melon-armor" => Status::MelonArmor,
            "status-poison-attack" => Status::PoisonAttack,
            "status-splash-attack" => Status::SplashAttack,
            "status-steak-attack" => Status::SteakAttack,
            "status-weak" => Status::Weak,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Status::None => "none",
            Status::Bread => "status-bread",
            Status::Cake => "status-cake",
            Status::BoneAttack => "status-bone-attack",
            Status::CoconutShield => "status-coconut-shield",
            Status::ExtraLife => "status-extra-life",
            Status::GarlicArmor => "status-garlic-armor",
            Status::HoneyBee => "status-honey-bee",
            Status::MelonArmor => "status-melon-armor",
            Status::PoisonAttack => "status-poison-attack",
            Status::SplashAttack => "status-splash-attack",
            Status::SteakAttack => "status-steak-attack",
            Status::Weak => "status-weak",
        }
    }

    /// `sapai.status.PERK_STATUSES`: everything except "none" and Weak.
    pub fn is_perk(self) -> bool {
        !matches!(self, Status::None | Status::Weak)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trigger {
    Faint,
    Summoned,
    StartOfBattle,
    AfterAttack,
    BeforeAttack,
    Hurt,
    KnockOut,
    FourFriendsHurt,
    GainsPerk,
    EatsShopFoodOrGainsPerk,
    /// Any trigger that never fires during a battle (Sell, Buy, StartOfTurn, ...)
    /// and the "none" trigger of ability-less pets.
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrigBy {
    SelfPet,
    FriendAhead,
    EachFriend,
    Player,
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EffectKind {
    ModifyStats,
    DealDamage,
    SummonPet,
    SummonRandomPet,
    SummonStoredPets,
    RespawnPet,
    TransferStats,
    TransferAbility,
    ReduceHealth,
    ApplyStatus,
    AllOf,
    Swallow,
    GainHealthPercentHighestFriend,
    NoneKind,
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TargetKind {
    AdjacentAnimals,
    All,
    AllOtherFriends,
    EachEnemy,
    EachFriend,
    EachShopAnimal,
    FirstEnemy,
    FriendAhead,
    FriendBehind,
    HighestHealthEnemy,
    LastEnemy,
    LowestHealthEnemy,
    RandomEnemy,
    RandomFriend,
    FrontMostFriend,
    SelfPet,
    StrongestFriend,
    TriggeringEntity,
    NoneKind,
    Other,
}

#[derive(Clone, Debug)]
pub struct Target {
    pub kind: TargetKind,
    pub kind_name: String,
    pub n: Option<i64>,
    pub exclude_status: Option<Status>,
    /// `target["includingFuture"] is True` (shop phase only).
    pub including_future: bool,
    /// kind == "Level2And3Friends" (Penguin, shop phase only; `TargetKind::Other` for battles).
    pub level23: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Amount {
    Absent,
    Int(i64),
    AttackDamagePercent(i64),
    AttackPercent(i64),
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TeamSel {
    Friendly,
    Enemy,
    Other,
}

#[derive(Clone, Debug)]
pub struct Effect {
    pub kind: EffectKind,
    pub kind_name: String,
    pub attack_amount: Option<i64>,
    pub health_amount: Option<i64>,
    pub amount: Amount,
    pub target: Option<Target>,
    pub to: Option<Target>,
    pub from: Option<Target>,
    pub until_end_of_battle: bool,
    /// Summoned pet, as an index into `Data::pets` (None when absent or unknown).
    pub pet: Option<usize>,
    pub with_attack: Option<i64>,
    pub with_health: Option<i64>,
    pub with_level: Option<i64>,
    pub team: Option<TeamSel>,
    pub n: Option<i64>,
    pub status: Option<Status>,
    pub effects: Vec<Effect>,
    pub tier: Option<i64>,
    pub base_attack: Option<i64>,
    pub base_health: Option<i64>,
    pub level: Option<i64>,
    pub percentage: Option<i64>,
    pub double_against_tier: Option<i64>,
    pub copy_attack: bool,
    pub copy_health: bool,
    /// Shop-phase effect kinds, which battles never run (`EffectKind::Other` there).
    pub shop_kind: ShopEffect,
    /// `effect["food"]` (StockFood, RefillShops).
    pub food_name: Option<String>,
    /// `effect["cost"]` (StockFood).
    pub cost: Option<i64>,
}

/// Effects that only exist in the shop phase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShopEffect {
    None,
    GainGold,
    StockFood,
    RefillShops,
    DiscountFood,
    FoodMultiplier,
}

/// Every trigger name the shop phase tests for (`Ability::trigger` folds
/// these into `Trigger::Other`, which is all a battle needs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShopTrig {
    None,
    Sell,
    Buy,
    BuyAfterLoss,
    BuyTier1Animal,
    BuyFood,
    StartOfTurn,
    EndOfTurn,
    EndOfTurnWith3PlusGold,
    EndOfTurnWithLvl3Friend,
    EndOfTurnWith4OrLessAnimals,
    EndOfTurnAfterLoss,
    /// Any other name starting with "EndOfTurn" (the reference raises on it).
    EndOfTurnOther,
    LevelUp,
    EatsShopFood,
    /// Not a shop-only trigger (see `Ability::trigger`).
    Battle,
}

impl ShopTrig {
    pub fn is_end_of_turn(self) -> bool {
        matches!(
            self,
            ShopTrig::EndOfTurn
                | ShopTrig::EndOfTurnWith3PlusGold
                | ShopTrig::EndOfTurnWithLvl3Friend
                | ShopTrig::EndOfTurnWith4OrLessAnimals
                | ShopTrig::EndOfTurnAfterLoss
                | ShopTrig::EndOfTurnOther
        )
    }
}

#[derive(Clone, Debug)]
pub struct Ability {
    pub trigger: Trigger,
    pub trigger_name: String,
    pub triggered_by: TrigBy,
    pub max_triggers: Option<i64>,
    pub effect: Effect,
    /// `_copiedFrom` set by `TransferAbility` (Parrot).
    pub copied_from: Option<usize>,
    /// The trigger as the shop phase reads it.
    pub shop_trig: ShopTrig,
}

#[derive(Clone, Debug)]
pub struct PetData {
    pub name: String,
    /// Integer tier; None for "Summoned"/"none".
    pub tier: Option<i64>,
    /// None when the table holds "?" or "none".
    pub base_attack: Option<i64>,
    pub base_health: Option<i64>,
    /// `fd["status"]` when the key is present.
    pub status: Option<Status>,
    /// `fd["level{n}Ability"]` when the key is present.
    pub abilities: [Option<Arc<Ability>>; 3],
}

pub struct Data {
    pub engine_sha256: String,
    pub pets: Vec<PetData>,
    pub by_name: HashMap<String, usize>,
    /// `sapai.pets.empty_ability`.
    pub empty_ability: Arc<Ability>,
    /// Whale's override after `Swallow`.
    pub stored_ability: Arc<Ability>,
    pub honey_ability: Arc<Ability>,
    pub extra_life_ability: Arc<Ability>,
    /// `pet_tier_lookup_std[tier]`, in the reference's order.
    pub tier_std: HashMap<i64, Vec<usize>>,
    pub tiger: usize,
    pub shark: usize,
    pub fly: usize,
    pub zombie_fly: usize,
    pub ram: usize,
    pub chick: usize,
    pub bee: usize,
    pub parrot: usize,
}

fn int(v: Option<&Value>) -> Option<i64> {
    v.and_then(|x| x.as_i64())
}

fn parse_target(v: Option<&Value>) -> Option<Target> {
    let obj = v?.as_object()?;
    let kind_name = obj.get("kind").and_then(|k| k.as_str()).unwrap_or("").to_string();
    let kind = match kind_name.as_str() {
        "AdjacentAnimals" => TargetKind::AdjacentAnimals,
        "All" => TargetKind::All,
        "AllOtherFriends" => TargetKind::AllOtherFriends,
        "EachEnemy" => TargetKind::EachEnemy,
        "EachFriend" => TargetKind::EachFriend,
        "EachShopAnimal" => TargetKind::EachShopAnimal,
        "FirstEnemy" => TargetKind::FirstEnemy,
        "FriendAhead" => TargetKind::FriendAhead,
        "FriendBehind" => TargetKind::FriendBehind,
        "HighestHealthEnemy" => TargetKind::HighestHealthEnemy,
        "LastEnemy" => TargetKind::LastEnemy,
        "LowestHealthEnemy" => TargetKind::LowestHealthEnemy,
        "RandomEnemy" => TargetKind::RandomEnemy,
        "RandomFriend" => TargetKind::RandomFriend,
        "FrontMostFriend" => TargetKind::FrontMostFriend,
        "Self" => TargetKind::SelfPet,
        "StrongestFriend" => TargetKind::StrongestFriend,
        "TriggeringEntity" => TargetKind::TriggeringEntity,
        "none" => TargetKind::NoneKind,
        _ => TargetKind::Other,
    };
    Some(Target {
        including_future: obj.get("includingFuture").and_then(|b| b.as_bool()) == Some(true),
        level23: kind_name == "Level2And3Friends",
        kind,
        kind_name,
        n: int(obj.get("n")),
        exclude_status: obj
            .get("excludeStatus")
            .and_then(|s| s.as_str())
            .and_then(Status::parse)
            .filter(|s| *s != Status::None),
    })
}

fn parse_effect(v: &Value, by_name: &HashMap<String, usize>) -> Effect {
    let empty = serde_json::Map::new();
    let obj = v.as_object().unwrap_or(&empty);
    let kind_name = obj.get("kind").and_then(|k| k.as_str()).unwrap_or("none").to_string();
    let kind = match kind_name.as_str() {
        "ModifyStats" => EffectKind::ModifyStats,
        "DealDamage" => EffectKind::DealDamage,
        "SummonPet" => EffectKind::SummonPet,
        "SummonRandomPet" => EffectKind::SummonRandomPet,
        "SummonStoredPets" => EffectKind::SummonStoredPets,
        "RespawnPet" => EffectKind::RespawnPet,
        "TransferStats" => EffectKind::TransferStats,
        "TransferAbility" => EffectKind::TransferAbility,
        "ReduceHealth" => EffectKind::ReduceHealth,
        "ApplyStatus" => EffectKind::ApplyStatus,
        "AllOf" => EffectKind::AllOf,
        "Swallow" => EffectKind::Swallow,
        "GainHealthPercentHighestFriend" => EffectKind::GainHealthPercentHighestFriend,
        "none" => EffectKind::NoneKind,
        _ => EffectKind::Other,
    };
    let amount = match obj.get("amount") {
        None | Some(Value::Null) => Amount::Absent,
        Some(Value::Object(m)) => {
            if let Some(p) = m.get("attackDamagePercent").and_then(|x| x.as_i64()) {
                Amount::AttackDamagePercent(p)
            } else if let Some(p) = m.get("attackPercent").and_then(|x| x.as_i64()) {
                Amount::AttackPercent(p)
            } else {
                Amount::Other
            }
        }
        Some(x) => x.as_i64().map(Amount::Int).unwrap_or(Amount::Other),
    };
    Effect {
        kind,
        attack_amount: int(obj.get("attackAmount")),
        health_amount: int(obj.get("healthAmount")),
        amount,
        target: parse_target(obj.get("target")),
        to: parse_target(obj.get("to")),
        from: parse_target(obj.get("from")),
        until_end_of_battle: obj.get("untilEndOfBattle").and_then(|b| b.as_bool()) == Some(true),
        pet: obj
            .get("pet")
            .and_then(|p| p.as_str())
            .and_then(|p| by_name.get(p).copied()),
        with_attack: int(obj.get("withAttack")),
        with_health: int(obj.get("withHealth")),
        with_level: int(obj.get("withLevel")),
        team: obj.get("team").and_then(|t| t.as_str()).map(|t| match t {
            "Friendly" => TeamSel::Friendly,
            "Enemy" => TeamSel::Enemy,
            _ => TeamSel::Other,
        }),
        n: int(obj.get("n")),
        status: obj.get("status").and_then(|s| s.as_str()).and_then(Status::parse),
        effects: obj
            .get("effects")
            .and_then(|e| e.as_array())
            .map(|a| a.iter().map(|e| parse_effect(e, by_name)).collect())
            .unwrap_or_default(),
        tier: int(obj.get("tier")),
        base_attack: int(obj.get("baseAttack")),
        base_health: int(obj.get("baseHealth")),
        level: int(obj.get("level")),
        percentage: int(obj.get("percentage")),
        double_against_tier: int(obj.get("doubleAgainstTier")),
        copy_attack: obj.get("copyAttack").and_then(|b| b.as_bool()) == Some(true),
        copy_health: obj.get("copyHealth").and_then(|b| b.as_bool()) == Some(true),
        shop_kind: match kind_name.as_str() {
            "GainGold" => ShopEffect::GainGold,
            "StockFood" => ShopEffect::StockFood,
            "RefillShops" => ShopEffect::RefillShops,
            "DiscountFood" => ShopEffect::DiscountFood,
            "FoodMultiplier" => ShopEffect::FoodMultiplier,
            _ => ShopEffect::None,
        },
        food_name: obj.get("food").and_then(|f| f.as_str()).map(|f| f.to_string()),
        cost: int(obj.get("cost")),
        kind_name,
    }
}

pub fn parse_ability(v: &Value, by_name: &HashMap<String, usize>) -> Ability {
    let trigger_name = v.get("trigger").and_then(|t| t.as_str()).unwrap_or("none").to_string();
    let trigger = match trigger_name.as_str() {
        "Faint" => Trigger::Faint,
        "Summoned" => Trigger::Summoned,
        "StartOfBattle" => Trigger::StartOfBattle,
        "AfterAttack" => Trigger::AfterAttack,
        "BeforeAttack" => Trigger::BeforeAttack,
        "Hurt" => Trigger::Hurt,
        "KnockOut" => Trigger::KnockOut,
        "FourFriendsHurt" => Trigger::FourFriendsHurt,
        "GainsPerk" => Trigger::GainsPerk,
        "EatsShopFoodOrGainsPerk" => Trigger::EatsShopFoodOrGainsPerk,
        _ => Trigger::Other,
    };
    let triggered_by = match v
        .get("triggeredBy")
        .and_then(|t| t.get("kind"))
        .and_then(|k| k.as_str())
        .unwrap_or("")
    {
        "Self" => TrigBy::SelfPet,
        "FriendAhead" => TrigBy::FriendAhead,
        "EachFriend" => TrigBy::EachFriend,
        "Player" => TrigBy::Player,
        _ => TrigBy::Other,
    };
    let shop_trig = match trigger_name.as_str() {
        "none" => ShopTrig::None,
        "Sell" => ShopTrig::Sell,
        "Buy" => ShopTrig::Buy,
        "BuyAfterLoss" => ShopTrig::BuyAfterLoss,
        "BuyTier1Animal" => ShopTrig::BuyTier1Animal,
        "BuyFood" => ShopTrig::BuyFood,
        "StartOfTurn" => ShopTrig::StartOfTurn,
        "EndOfTurn" => ShopTrig::EndOfTurn,
        "EndOfTurnWith3PlusGold" => ShopTrig::EndOfTurnWith3PlusGold,
        "EndOfTurnWithLvl3Friend" => ShopTrig::EndOfTurnWithLvl3Friend,
        "EndOfTurnWith4OrLessAnimals" => ShopTrig::EndOfTurnWith4OrLessAnimals,
        "EndOfTurnAfterLoss" => ShopTrig::EndOfTurnAfterLoss,
        "LevelUp" => ShopTrig::LevelUp,
        "EatsShopFood" => ShopTrig::EatsShopFood,
        other if other.starts_with("EndOfTurn") => ShopTrig::EndOfTurnOther,
        _ => ShopTrig::Battle,
    };
    Ability {
        shop_trig,
        trigger,
        trigger_name,
        triggered_by,
        max_triggers: int(v.get("maxTriggers")),
        effect: parse_effect(v.get("effect").unwrap_or(&Value::Null), by_name),
        copied_from: None,
    }
}

impl Data {
    /// Load `turtle_v31.json` and `turtle_v31_tiers.json` from a directory.
    pub fn load_dir(dir: &std::path::Path) -> Result<Data, String> {
        let main = std::fs::read_to_string(dir.join("turtle_v31.json")).map_err(|e| e.to_string())?;
        let tiers = std::fs::read_to_string(dir.join("turtle_v31_tiers.json")).map_err(|e| e.to_string())?;
        Data::from_json(&main, &tiers)
    }

    pub fn from_json(main: &str, tiers: &str) -> Result<Data, String> {
        let root: Value = serde_json::from_str(main).map_err(|e| e.to_string())?;
        let pets_obj = root.get("pets").and_then(|p| p.as_object()).ok_or("no pets table")?;
        // Stable order: sorted by name (serde_json's map is ordered by key).
        let names: Vec<String> = pets_obj.keys().cloned().collect();
        let by_name: HashMap<String, usize> =
            names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let mut pets = Vec::with_capacity(names.len());
        for name in &names {
            let entry = &pets_obj[name];
            let mut abilities: [Option<Arc<Ability>>; 3] = [None, None, None];
            for (i, slot) in abilities.iter_mut().enumerate() {
                let key = format!("level{}Ability", i + 1);
                if let Some(v) = entry.get(&key) {
                    if !v.is_null() {
                        *slot = Some(Arc::new(parse_ability(v, &by_name)));
                    }
                }
            }
            pets.push(PetData {
                name: name.clone(),
                tier: int(entry.get("tier")),
                base_attack: int(entry.get("baseAttack")),
                base_health: int(entry.get("baseHealth")),
                status: entry.get("status").and_then(|s| s.as_str()).and_then(Status::parse),
                abilities,
            });
        }
        let statuses = root.get("statuses").ok_or("no statuses table")?;
        let status_ability = |name: &str| -> Result<Arc<Ability>, String> {
            let v = statuses
                .get(name)
                .and_then(|s| s.get("ability"))
                .ok_or_else(|| format!("no ability for {name}"))?;
            Ok(Arc::new(parse_ability(v, &by_name)))
        };
        let empty_value: Value = serde_json::json!({
            "trigger": "none", "triggeredBy": {"kind": "none"}, "effect": {"kind": "none"}
        });
        let stored_value: Value = serde_json::json!({
            "trigger": "Faint", "triggeredBy": {"kind": "Self"}, "effect": {"kind": "SummonStoredPets"}
        });
        let tiers_root: Value = serde_json::from_str(tiers).map_err(|e| e.to_string())?;
        let mut tier_std = HashMap::new();
        if let Some(obj) = tiers_root.get("pet_tier_lookup_std").and_then(|o| o.as_object()) {
            for (tier, list) in obj {
                let tier: i64 = tier.parse().map_err(|_| "bad tier key".to_string())?;
                let mut ids = Vec::new();
                for name in list.as_array().ok_or("tier list")? {
                    let name = name.as_str().ok_or("tier entry")?;
                    ids.push(*by_name.get(name).ok_or_else(|| format!("tier pet {name} missing"))?);
                }
                tier_std.insert(tier, ids);
            }
        }
        let id = |name: &str| -> Result<usize, String> {
            by_name.get(name).copied().ok_or_else(|| format!("pet {name} missing from tables"))
        };
        Ok(Data {
            engine_sha256: root
                .get("engineSha256")
                .and_then(|s| s.as_str())
                .unwrap_or("")
                .to_string(),
            empty_ability: Arc::new(parse_ability(&empty_value, &by_name)),
            stored_ability: Arc::new(parse_ability(&stored_value, &by_name)),
            honey_ability: status_ability("status-honey-bee")?,
            extra_life_ability: status_ability("status-extra-life")?,
            tier_std,
            tiger: id("pet-tiger")?,
            shark: id("pet-shark")?,
            fly: id("pet-fly")?,
            zombie_fly: id("pet-zombie-fly")?,
            ram: id("pet-ram")?,
            chick: id("pet-chick")?,
            bee: id("pet-bee")?,
            parrot: id("pet-parrot")?,
            pets,
            by_name,
        })
    }
}
