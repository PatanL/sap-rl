#!/usr/bin/env python3
"""Dump the tables the Rust environment port needs on top of turtle_v31.json.

Usage (inside the training image, PYTHONPATH on the packaged engine):
  export_env_data.py <out.json>

Written next to data/turtle_v31.json as data/turtle_v31_env.json:
  turns        turn-1..turn-11 shop rules (get_shop_rules: pet slots, food slots, tiers, level-up tier)
  petTierStd   shop.pet_tier_lookup_std, in the reference's list order (roll pools, level-up pools)
  foodTierStd  shop.food_tier_lookup_std, same
  petTier1     tiers.pet_tier_lookup[1] (Dragon's BuyTier1Animal test)
  pets         per pet: tier as Pet.tier holds it, and every key of the table entry
  foods        per food: cost, tier, ability
  itemIds      EntityObservationEncoder.item_to_id
  statusIds    EntityObservationEncoder.status_to_id
"""
import json, math, sys
from sapai.data import data
import sapai.shop as shop
import sapai.tiers as tiers
from sapai_gym1.rl.entity_observation import EntityObservationEncoder
from sapai_gym1.rl.action_schema import FlatActionCodec as C
from sapai_gym1 import SuperAutoPetsEnv

enc = EntityObservationEncoder()
turns = {}
for turn in range(1, 12):
    pslots, fslots, tier_avail, levelup_tier, avail_pets, avail_foods, pp, fp = shop.get_shop_rules(turn)
    turns[str(turn)] = {"pslots": pslots, "fslots": fslots, "tierAvail": tier_avail, "levelupTier": levelup_tier,
                        "availPets": list(avail_pets), "availFoods": list(avail_foods),
                        "pp": [float(x) for x in pp], "fp": [float(x) for x in fp],
                        # RandomTape.sample: threshold = uniform * math.fsum(weights)
                        "ppSum": math.fsum(float(x) for x in pp), "fpSum": math.fsum(float(x) for x in fp)}
from sapai import Food
food_rows = {}
for name, entry in data["foods"].items():
    if name == "food-none":
        continue
    food = Food(name)
    effect = entry["ability"]["effect"]
    target = effect.get("target", {})
    sort_tier = 0
    try:
        sort_tier = int(entry["tier"])          # Shop.roll._food_tier
    except (TypeError, ValueError):
        pass
    obs_tier = (food.fd.get("tier", entry["tier"]) if isinstance(entry["tier"], int) else 0)  # entity_observation
    food_rows[name] = {
        "cost": food.cost, "attack": food.attack, "health": food.health, "status": food.status,
        "untilEndOfBattle": bool(food.apply_until_end_of_battle), "sortTier": sort_tier, "obsTier": obs_tier,
        # SuperAutoPetsEnv._avail_buy_foods: bought without a target when Canned Food or a RandomFriend effect
        "noTarget": bool(name == "food-canned-food" or ("target" in effect and target.get("kind") == "RandomFriend")),
        "targetKind": (effect.get("target") or effect.get("to") or {}).get("kind"),
        "targetN": target.get("n", 1) if isinstance(target, dict) else 1,
    }
out = {
    "foodRows": food_rows,
    "turns": turns,
    "petTierStd": {str(k): v for k, v in shop.pet_tier_lookup_std.items()},
    "foodTierStd": {str(k): v for k, v in shop.food_tier_lookup_std.items()},
    "petTier1": list(tiers.pet_tier_lookup[1]),
    "pets": {k: v for k, v in data["pets"].items()},
    "foods": {k: v for k, v in data["foods"].items()},
    "statuses": data["statuses"],
    "itemIds": enc.item_to_id,
    "statusIds": enc.status_to_id,
    "codec": {k: getattr(C, k) for k in ("END_TURN", "BUY_PET", "BUY_FOOD", "COMBINE", "SELL", "ROLL", "FREEZE", "UNFREEZE", "MOVE", "NUM_ACTIONS")},
    "env": {"MAX_TURN": SuperAutoPetsEnv.MAX_TURN, "MAX_LIVES": SuperAutoPetsEnv.MAX_LIVES,
            "MAX_REORDERS_PER_TURN": SuperAutoPetsEnv.MAX_REORDERS_PER_TURN,
            "MAX_UNSPENT_GOLD_TO_END": SuperAutoPetsEnv.MAX_UNSPENT_GOLD_TO_END},
}
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True, default=str)
print(len(out["pets"]), "pets", len(out["foods"]), "foods", len(out["itemIds"]), "item ids")
