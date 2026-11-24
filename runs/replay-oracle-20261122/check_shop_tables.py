#!/usr/bin/env python3
"""Compare what real shops contained with the engine's static tables.

Reads every end-of-turn Build snapshot in the downloaded replays and checks:
shop slot counts and tier by turn, which pets and foods appear at which tier,
their base stats and prices. Stdlib only.

Usage: check_shop_tables.py <engine-tables.json> <calculator dir> <raw dir> [bundle matches dir]
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

tables = json.load(open(sys.argv[1]))
data = Path(sys.argv[2]) / "src/assets/data"
PETS = {str(p["Id"]): p["Name"] for p in json.loads((data / "pets.json").read_text())}
FOOD = {str(p["Id"]): p["Name"] for p in json.loads((data / "food.json").read_text())}
eng_pets = {v["name"]: v for v in tables["pets"].values()}
rollable = {tables["pets"][k]["name"] for k in tables["rollable"]}
eng_foods = {v["name"]: v for v in tables["foods"].values()}
turns = {v["index"]: v for v in tables["turns"].values()}

def perspectives():
    for path in sorted(Path(sys.argv[3]).glob("*.json")):
        yield from json.loads(path.read_text())["perspectives"]["perspectives"]
    if len(sys.argv) > 4:
        for path in sorted(Path(sys.argv[4]).glob("*/BUNDLE.json")):
            yield from json.loads(path.read_text())["perspectives"]["perspectives"]

snapshots = 0
slots, tiers = Counter(), Counter()
pet_seen, food_seen, reward_seen = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)   # name -> Counter(tier at sighting)
pet_stats, pet_price, food_price = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
board_species = Counter()
for per in perspectives():
    for action in per["raw"]["Actions"]:
        if action["Type"] != 0 or not action.get("Build"):
            continue
        bor = json.loads(action["Build"])["Bor"]
        turn, tier = bor.get("Tur"), bor.get("Ti")
        snapshots += 1
        expect = turns.get(min(turn, max(turns)))
        slots[(turn if turn < 12 else 12, bor.get("MiSC"), bor.get("SpSC"), expect["animalShopSlots"], expect["foodShopSlots"])] += 1
        tiers[(turn if turn < 12 else 12, tier, expect["tiersAvailable"])] += 1
        for pet in (bor.get("Mins") or {}).get("Items") or []:
            if pet:
                board_species[PETS.get(str(pet.get("Enu") or 0), f"#{pet.get('Enu')}")] += 1
        for pet in bor.get("MiSh") or []:
            if not pet:
                continue
            name = PETS.get(str(pet.get("Enu") or 0), f"#{pet.get('Enu')}")
            pet_price[name][pet.get("Pri")] += 1
            if pet.get("Rwrd"):
                # Level-up reward: offered from the next tier, so it says
                # nothing about the normal pool at this tier.
                reward_seen[name][tier] += 1
                continue
            pet_seen[name][tier] += 1
            if (pet.get("Lvl") or 1) == 1 and not pet.get("Rwrd"):
                pet_stats[name][((pet.get("At") or {}).get("Perm"), (pet.get("Hp") or {}).get("Perm"))] += 1
        for item in bor.get("SpSh") or []:
            if not item:
                continue
            name = FOOD.get(str(item.get("Enu") or 0), f"#{item.get('Enu')}")
            food_seen[name][tier] += 1
            food_price[name][item.get("Pri")] += 1

print("snapshots:", snapshots)
print("\n== slots by turn (turn, real pet slots, real food slots, engine pet, engine food): count; mismatches only")
for key, n in sorted(slots.items()):
    if key[1] != key[3] or key[2] != key[4]:
        print("  ", key, n)
print("== tier by turn (turn, real tier, engine tier): mismatches only")
for key, n in sorted(tiers.items()):
    if key[1] != key[2]:
        print("  ", key, n)
print("\n== pets seen in real shops")
for name in sorted(pet_seen):
    eng = eng_pets.get(name)
    lowest = min(pet_seen[name])
    issues = []
    if eng is None or name not in rollable:
        issues.append("NOT ROLLABLE IN ENGINE")
    else:
        if eng["tier"] != lowest:
            issues.append(f"tier engine {eng['tier']} vs lowest real sighting {lowest}")
        base = pet_stats[name].most_common(1)
        if base and base[0][0] != (eng["attack"], eng["health"]):
            issues.append(f"base stats engine {eng['attack']}/{eng['health']} vs real {dict(pet_stats[name].most_common(3))}")
    if set(pet_price[name]) != {3}:
        issues.append(f"prices {dict(pet_price[name])}")
    if issues:
        print(f"   {name} (seen {sum(pet_seen[name].values())}):", "; ".join(issues))
bad_rewards = {n: dict(c) for n, c in reward_seen.items() if n in eng_pets and any(eng_pets[n]["tier"] != t + 1 for t in c if t < 6)}
print("   level-up reward pets not exactly one tier above the shop tier:", bad_rewards)
print("   engine rollable pets never seen in a real shop:", sorted(rollable - set(pet_seen)))
print("   species on real boards that the engine cannot roll:", {n: c for n, c in board_species.items() if n not in rollable})
print("\n== foods seen in real shops")
for name in sorted(food_seen):
    eng = eng_foods.get(name)
    lowest = min(food_seen[name])
    note = "NOT IN ENGINE" if eng is None else (f"tier engine {eng['tier']} vs real {lowest}" if eng["tier"] != lowest else "")
    cost = "" if eng is None or set(food_price[name]) == {eng["cost"]} else f"price engine {eng['cost']} vs real {dict(food_price[name])}"
    print(f"   {name}: seen {sum(food_seen[name].values())}, lowest tier {lowest} {note} {cost}")
print("   engine foods never seen:", sorted(set(eng_foods) - set(food_seen)))
