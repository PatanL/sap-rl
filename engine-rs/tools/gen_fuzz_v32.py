#!/usr/bin/env python3
"""Random boards aimed at the v32 rules, in the fixtures format gen_golden.py reads.

Like gen_fuzz_fixtures.py, but drawn mostly from the pets the v32 rules touch (start-of-battle
pets with Dodo, Skunk at 50 health, Crab, Wolverine with Rhino and blast pets, Fly and Shark with
several friends that die together, Tiger behind summoners and behind pets with a use limit, Dog),
with low health so that many pets die in one step, with empty slots inside a board (a
{"id": "pet-none"} spec) and now and then an empty board. Stdlib only.

Usage: gen_fuzz_v32.py <data/turtle_v31.json> <out.json> [count] [seed]
"""
import json
import random
import sys

data = json.load(open(sys.argv[1]))
count = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 20261124)
rollable = data["rollable"]
statuses = ["status-bone-attack", "status-coconut-shield", "status-extra-life", "status-garlic-armor",
            "status-honey-bee", "status-melon-armor", "status-poison-attack", "status-splash-attack",
            "status-steak-attack", "status-weak"]
groups = {
    "start": ["pet-dodo", "pet-skunk", "pet-crab", "pet-whale", "pet-dolphin", "pet-crocodile",
              "pet-leopard", "pet-mosquito", "pet-armadillo", "pet-dodo", "pet-crab", "pet-skunk"],
    "wolverine": ["pet-wolverine", "pet-wolverine", "pet-rhino", "pet-hedgehog", "pet-badger",
                  "pet-blowfish", "pet-hippo", "pet-camel", "pet-gorilla", "pet-tiger", "pet-peacock"],
    "fly": ["pet-fly", "pet-fly", "pet-shark", "pet-parrot", "pet-hedgehog", "pet-badger", "pet-deer",
            "pet-sheep", "pet-cricket", "pet-spider", "pet-rooster", "pet-tiger", "pet-whale",
            "pet-mammoth", "pet-ox", "pet-turtle"],
    "tiger": ["pet-tiger", "pet-tiger", "pet-snake", "pet-hippo", "pet-ox", "pet-gorilla", "pet-fly",
              "pet-deer", "pet-cricket", "pet-sheep", "pet-rhino", "pet-hedgehog", "pet-elephant",
              "pet-rat", "pet-spider"],
    "dog": ["pet-dog", "pet-dog", "pet-horse", "pet-turkey", "pet-sheep", "pet-deer", "pet-cricket",
            "pet-rooster", "pet-fly", "pet-hedgehog", "pet-kangaroo"],
}
names = sorted(groups)


def pet(pool, small, flat):
    roll = rng.random()
    name = rng.choice(pool) if roll < 0.75 else rng.choice(rollable)
    if flat is not None and rng.random() < 0.6:
        attack = flat
    elif small:
        attack = rng.randint(1, 6)
    else:
        attack = min(50, int(rng.expovariate(1 / 9)) + 1)
    health = rng.randint(1, 6) if small else min(50, int(rng.expovariate(1 / 11)) + 1)
    if rng.random() < 0.06:
        health = 50
    return {
        "id": name,
        "attack": attack,
        "health": health,
        "experience": rng.choice([0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5]),
        "status": rng.choice(statuses) if rng.random() < 0.3 else "none",
        "triggersConsumed": rng.choice([0, 0, 0, 0, 0, 0, 1, 2, 3, 4]),
    }


def board(pool, small, flat):
    if rng.random() < 0.01:
        return []
    size = rng.randint(1, 5) if rng.random() < 0.4 else rng.randint(3, 5)
    pets = [pet(pool, small, flat) for _ in range(size)]
    if rng.random() < 0.2 and size < 5:
        # empty slots inside the board; the last slot stays a pet
        for _ in range(rng.randint(1, 5 - size)):
            pets.insert(rng.randint(0, len(pets) - 1), {"id": "pet-none"})
    return pets


fixtures = []
for index in range(count):
    small = rng.random() < 0.6
    flat = rng.randint(1, 8) if rng.random() < 0.3 else None
    pools = [groups[rng.choice(names)] for _ in range(2)]
    fixtures.append({
        "battleId": f"fuzz3-{index:05d}",
        "player": board(pools[0] + pools[1][:3], small, flat),
        "opponent": board(pools[1] + pools[0][:3], small, flat),
    })
json.dump({"fixtures": fixtures}, open(sys.argv[2], "w"))
print(len(fixtures), "fuzz fixtures",
      sum(any(p["id"] == "pet-none" for p in f["player"] + f["opponent"]) for f in fixtures), "with empty slots",
      sum(not f["player"] or not f["opponent"] for f in fixtures), "with an empty board")
