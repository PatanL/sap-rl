#!/usr/bin/env python3
"""Print the engine's board after start-of-battle and after every attack round.

Usage (in the training image): trace_sapai.py <engine dir> <fixtures.json> <battle id prefix> [seed]
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "/dev/null", "1", "1"] + sys.argv[3:]
import run_sapai  # noqa: E402  (reuses make_team; its argv parsing is satisfied above)
from sapai import Battle  # noqa: E402

prefix = sys.argv[6]
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 1234567
fixture = next(f for f in json.load(open(sys.argv[2]))["fixtures"] if f["battleId"].startswith(prefix))


def show(team):
    return " ".join("___" if slot.empty else f"{slot.pet.name[4:]}({slot.pet.attack}/{slot.pet.health}{'*' + slot.pet.status[7:] if slot.pet.status != 'none' else ''})"
                    for slot in team)


battle = Battle(run_sapai.make_team(fixture["player"]), run_sapai.make_team(fixture["opponent"]), seed=seed)
print("init   P:", show(battle.t0), "| O:", show(battle.t1))
battle.start()
print("start  P:", show(battle.t0), "| O:", show(battle.t1))
index = 0
while True:
    # Same order as Battle.battle(): refresh ability priority, then attack.
    battle.pet_priority = battle.update_pet_priority(battle.t0, battle.t1, battle.rs)
    more = battle.attack(index)
    if not more:
        break
    index += 1
    print(f"atk {index:2d} P:", show(battle.t0), "| O:", show(battle.t1))
print("final  P:", show(battle.t0), "| O:", show(battle.t1), "->", ("player", "opponent", "draw")[battle.check_battle_result()])
print("real   :", fixture["realOutcome"], [(s["name"], s["attack"], s["health"]) for s in sorted(fixture["survivors"], key=lambda s: -s["slot"])])
