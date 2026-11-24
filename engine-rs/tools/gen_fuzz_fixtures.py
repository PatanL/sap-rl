#!/usr/bin/env python3
"""Random boards for Rust-vs-Python parity, in the fixtures format gen_golden.py reads.

Real fights rarely contain some combinations (Whale with a Tiger behind, a
Parrot copying a summoner, Honey or Mushroom on summoners, many equal attacks).
These boards are not meant to be legal shop results; they only need to be
boards the reference engine can resolve. Stdlib only.

Usage: gen_fuzz_fixtures.py <data/turtle_v31.json> <out.json> [count] [seed]
"""
import json
import random
import sys

data = json.load(open(sys.argv[1]))
count = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 20261123)
rollable = data["rollable"]
tokens = ["pet-bus", "pet-ram", "pet-zombie-cricket", "pet-zombie-fly", "pet-chick", "pet-bee",
          "pet-dirty-rat", "pet-sloth"]
statuses = ["status-bread", "status-cake", "status-bone-attack", "status-coconut-shield",
            "status-extra-life", "status-garlic-armor", "status-honey-bee", "status-melon-armor",
            "status-poison-attack", "status-splash-attack", "status-steak-attack", "status-weak"]
# Pets whose interactions are the hard part; drawn more often.
hot = ["pet-tiger", "pet-whale", "pet-parrot", "pet-fly", "pet-shark", "pet-deer", "pet-sheep",
       "pet-spider", "pet-rooster", "pet-cricket", "pet-rat", "pet-hedgehog", "pet-badger",
       "pet-blowfish", "pet-wolverine", "pet-rhino", "pet-hippo", "pet-ox", "pet-turtle",
       "pet-mammoth", "pet-elephant", "pet-snake", "pet-gorilla", "pet-seal", "pet-rabbit",
       "pet-horse", "pet-dog", "pet-turkey", "pet-crab", "pet-skunk", "pet-dodo", "pet-camel"]


def pet(small):
    roll = rng.random()
    name = rng.choice(hot) if roll < 0.55 else rng.choice(tokens) if roll > 0.97 else rng.choice(rollable)
    if small:
        attack, health = rng.randint(1, 6), rng.randint(1, 8)
    else:
        attack, health = min(50, int(rng.expovariate(1 / 9)) + 1), min(50, int(rng.expovariate(1 / 11)) + 1)
    return {
        "id": name,
        "attack": attack,
        "health": health,
        "experience": rng.choice([0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 5]),
        "status": rng.choice(statuses) if rng.random() < 0.4 else "none",
        "triggersConsumed": rng.choice([0, 0, 0, 0, 0, 1, 2, 3]),
    }


fixtures = []
for index in range(count):
    small = rng.random() < 0.5
    fixtures.append({
        "battleId": f"fuzz-{index:05d}",
        "player": [pet(small) for _ in range(rng.randint(1, 5))],
        "opponent": [pet(small) for _ in range(rng.randint(1, 5))],
    })
json.dump({"fixtures": fixtures}, open(sys.argv[2], "w"))
print(len(fixtures), "fuzz fixtures")
