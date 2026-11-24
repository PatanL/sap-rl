#!/usr/bin/env python3
"""Turn envcheck --coverage output into the coverage table of the report.

Usage: coverage_report.py <coverage.json> [<coverage.json> ...]   (counts are summed)
Prints: per pet and trigger the number of shop-phase activations at levels 1/2/3,
per food the number of purchases, and every other counted shop rule. Lists
what the tables say should be reachable but was never hit.
"""
import collections, json, sys

counts = collections.Counter()
for path in sys.argv[1:]:
    for key, value in json.load(open(path)).items():
        counts[key] += value
tables = json.load(open(__file__.rsplit("/", 2)[0] + "/data/turtle_v31_env.json"))
old = json.load(open(__file__.rsplit("/", 2)[0] + "/data/turtle_v31.json"))
SHOP_TRIGGERS = {"Sell": "Sell", "Buy": "Buy", "BuyTier1Animal": "Buy", "StartOfTurn": "StartOfTurn", "EndOfTurn": "EndOfTurn",
                 "EndOfTurnWithLvl3Friend": "EndOfTurn", "EndOfTurnAfterLoss": "EndOfTurn", "LevelUp": "LevelUp",
                 "EatsShopFoodOrGainsPerk": "EatsShopFood", "PurchaseFood": "PurchaseFood", "Faint": "Faint",
                 "Summoned": "Summoned", "Hurt": "Hurt", "FourFriendsHurt": "FourFriendsHurt"}
ability = collections.defaultdict(lambda: [0, 0, 0])
for key, value in counts.items():
    if key.startswith("ability:"):
        _, pet, what, effect, level = key.split(":")
        ability[(pet, what, effect)][int(level[1]) - 1] += value
print("pet | trigger | effect | L1 | L2 | L3")
missing = []
for pet in sorted(old["rollable"]):
    entry = tables["pets"][pet]
    for level in (1, 2, 3):
        ab = entry.get(f"level{level}Ability")
        if not ab or ab["trigger"] not in SHOP_TRIGGERS:
            continue
        what = SHOP_TRIGGERS[ab["trigger"]]
        key = (pet, what, ab["effect"]["kind"])
        if ability[key][level - 1] == 0 and not (ab["trigger"] == "LevelUp" and level == 3):
            missing.append(f"{pet} {ab['trigger']} L{level}")
for key in sorted(ability):
    print(" | ".join(key), "|", " | ".join(str(v) for v in ability[key]))
print("\nfood purchases:", ", ".join(f"{k.split(':')[1]}={v}" for k, v in sorted(counts.items()) if k.startswith("buy-food:")))
never = [f for f in tables["foodRows"] if counts.get("buy-food:" + f, 0) == 0]
print("foods never bought:", never or "none")
print("\nnative shop-phase abilities never activated at a level:", missing or "none")
groups = collections.defaultdict(list)
for key, value in sorted(counts.items()):
    head = key.split(":")[0]
    if head not in ("ability", "buy-food", "buy-pet", "buy-combine", "combine", "sell", "parrot"):
        groups[head].append(f"{key.split(':', 1)[1] if ':' in key else key}={value}")
print()
for head, items in groups.items():
    print(f"{head}: " + ", ".join(items))
for head in ("buy-pet", "buy-combine", "combine", "sell"):
    rows = {k: v for k, v in counts.items() if k.startswith(head + ":")}
    species = {k.split(":")[1] for k in rows}
    lacking = sorted(set(old["rollable"]) - species)
    print(f"{head}: {sum(rows.values())} over {len(species)} species; rollable species never seen: {lacking or 'none'}")
print("level-up rewards bought:", sum(v for k, v in counts.items() if k.startswith("buy-pet:") and k.endswith(":levelup")),
      "combined onto a copy:", sum(v for k, v in counts.items() if k.startswith("buy-combine:") and k.endswith(":levelup")))
print("parrot copies:", sum(v for k, v in counts.items() if k.startswith("parrot:copied:")), "over",
      len([k for k in counts if k.startswith("parrot:copied:")]), "species;", "copy of a copy:", counts.get("parrot:copies-a-parrot's-copy", 0))
