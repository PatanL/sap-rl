#!/usr/bin/env python3
"""Emit golden battle traces from the Python reference engine.

For every distinct real fight in a fixtures file and a few seeds, record the
board after construction, after start of battle and after every attack round,
the outcome, and one extra random draw taken after the fight. The Rust engine
must reproduce each trace exactly, seed for seed, so a port bug shows up as a
first diverging round rather than as noise in an outcome distribution.

Usage (inside the training image):
  gen_golden.py <engine dir> <fixtures.json> <out.jsonl> [seeds per fight] [workers] [limit]
"""
import json
import multiprocessing
import sys

ENGINE, FIXTURES, OUTPUT = sys.argv[1:4]
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
LIMIT = int(sys.argv[6]) if len(sys.argv) > 6 else 0
sys.path.insert(0, ENGINE)

from sapai import Battle, Pet, Team  # noqa: E402
from sapai.effects import TransferAbility  # noqa: E402
from sapai.data import data  # noqa: E402
import copy  # noqa: E402


def make_team(specs):
    pets = []
    for spec in specs:
        pet = Pet(spec["id"])
        pet._attack = int(spec["attack"])
        pet._health = int(spec["health"])
        pet.set_total_experience(int(spec["experience"]))
        pet.status = spec["status"]
        pet.ability_counter = int(spec.get("triggersConsumed", 0))
        pets.append(pet)
    team = Team(pets)
    for index, slot in enumerate(team):
        if slot.empty or slot.pet.name != "pet-parrot":
            continue
        if "parrotCopies" in specs[index]:
            # The replay records which ability the Parrot ended the turn with.
            source = specs[index]["parrotCopies"]
            if source:
                # Same fallback as TransferAbility for a pet with no ability
                # at this level (Scorpion).
                ability = data["pets"][source].get(f"level{slot.pet.level}Ability")
                if ability is None:
                    ability = Pet(source).ability
                ability = copy.deepcopy(ability)
                ability["_copiedFrom"] = source
                slot.pet.set_ability(ability)
        else:
            TransferAbility(slot.pet, [0, index], [team])
    return team


def board(team):
    return [None if slot.empty else
            [slot.pet.name, int(slot.pet.attack), int(slot.pet.health), slot.pet.status, int(slot.pet.level)]
            for slot in team]


def spec(pet):
    out = {key: pet[key] for key in ("id", "attack", "health", "experience", "status")} | {
        "triggersConsumed": int(pet.get("triggersConsumed", 0))}
    if "parrotCopies" in pet:
        out["parrotCopies"] = pet["parrotCopies"]
    return out


def run(fixture):
    rows = []
    player = [spec(p) for p in fixture["player"]]
    opponent = [spec(p) for p in fixture["opponent"]]
    for index in range(SEEDS):
        seed = (1234567 + index * 2654435761) % (2 ** 32)
        row = {"battleId": fixture["battleId"], "seed": seed, "player": player, "opponent": opponent}
        try:
            battle = Battle(make_team(player), make_team(opponent), seed=seed, fast_training=True)
            trace = [[board(battle.t0), board(battle.t1)]]
            battle.start()
            trace.append([board(battle.t0), board(battle.t1)])
            rounds = 0
            while True:
                battle.pet_priority = battle.update_pet_priority(battle.t0, battle.t1, battle.rs)
                more = battle.attack(rounds)
                rounds += 1
                trace.append([board(battle.t0), board(battle.t1)])
                if not more:
                    break
            row.update(trace=trace, outcome=int(battle.check_battle_result()),
                       probe=float(battle.rs.random_sample()))
        except Exception as error:
            row.update(error=repr(error)[:200])
        rows.append(row)
    return rows


def main():
    fixtures = [f for f in json.load(open(FIXTURES))["fixtures"] if "mirrorOf" not in f]
    if LIMIT:
        fixtures = fixtures[:LIMIT]
    count = errors = 0
    with multiprocessing.Pool(WORKERS) as pool, open(OUTPUT, "w") as out:
        for rows in pool.imap(run, fixtures, chunksize=8):
            for row in rows:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1
                errors += "error" in row
    print(json.dumps({"fights": len(fixtures), "rows": count, "errors": errors}))


if __name__ == "__main__":
    main()
