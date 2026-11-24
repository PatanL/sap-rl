#!/usr/bin/env python3
"""Run the Python sapai engine on real replay battles and compare with the game.

The game's RNG is unknown, so each fight is simulated over many seeds:
- "impossible": the real outcome never occurs in the simulations (a definite
  disagreement, given enough seeds);
- "state miss": the real outcome occurs but the real surviving board (pets and
  their attack/health) never does;
- deterministic fights (one simulated final state across all seeds) must match
  exactly.

Usage (inside the training image, engine path first on sys.path):
  run_sapai.py <engine dir> <fixtures.json> <out.json> [samples] [workers]
"""
import copy
import json
import multiprocessing
import sys
from collections import Counter

ENGINE, FIXTURES, OUTPUT = sys.argv[1:4]
SAMPLES = int(sys.argv[4]) if len(sys.argv) > 4 else 200
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
sys.path.insert(0, ENGINE)

from sapai import Battle, Pet, Team  # noqa: E402
from sapai.effects import TransferAbility  # noqa: E402
from sapai.data import data as ENGINE_DATA  # noqa: E402

OUTCOMES = ("player", "opponent", "draw")


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
                ability = ENGINE_DATA["pets"][source].get(f"level{slot.pet.level}Ability")
                if ability is None:
                    ability = Pet(source).ability
                ability = copy.deepcopy(ability)
                ability["_copiedFrom"] = source
                slot.pet.set_ability(ability)
        else:
            TransferAbility(slot.pet, [0, index], [team])
    return team


def final_state(team):
    return tuple((slot.pet.name, int(slot.pet.attack), int(slot.pet.health))
                 for slot in team if not slot.empty)


def real_state(fixture):
    # Winner's survivors only. Player occupies slots 0-4 with slot 4 at the
    # front; the opponent occupies 5-9. Compare order-free and both orders.
    pets = sorted(fixture["survivors"], key=lambda s: -s["slot"])
    return tuple((p["id"], int(p["attack"]), int(p["health"])) for p in pets)


def evaluate(fixture):
    unknown = [p["id"] for side in ("player", "opponent") for p in fixture[side]
               if p["id"] not in ENGINE_DATA["pets"]]
    row = {"battleId": fixture["battleId"], "matchId": fixture["matchId"],
           "turn": fixture["turn"], "realOutcome": fixture["realOutcome"]}
    if unknown:
        row.update(skipped="unknown pets", unknown=unknown)
        return row
    try:
        player, opponent = make_team(fixture["player"]), make_team(fixture["opponent"])
        outcomes, states = Counter(), Counter()
        for sample in range(SAMPLES):
            battle = Battle(player, opponent, seed=(1234567 + sample * 2654435761) % (2**32),
                            fast_training=True)
            outcome = OUTCOMES[battle.battle()]
            outcomes[outcome] += 1
            winner = {"player": battle.t0, "opponent": battle.t1}.get(outcome)
            states[(outcome, final_state(winner) if winner is not None else ())] += 1
    except Exception as error:  # an engine crash on a legal board is a finding
        row.update(skipped="engine error", error=repr(error)[:300])
        return row
    real = real_state(fixture)
    real_outcome = fixture["realOutcome"]
    same_outcome = [state for (outcome, state) in states if outcome == real_outcome]
    ordered = any(state == real or state == real[::-1] for state in same_outcome)
    unordered = any(sorted(state) == sorted(real) for state in same_outcome)
    row.update(
        simOutcomes=dict(outcomes),
        pReal=outcomes[real_outcome] / SAMPLES,
        distinctStates=len(states),
        deterministic=len(states) == 1,
        outcomePossible=outcomes[real_outcome] > 0,
        statePossible=unordered,
        stateOrderMatches=ordered,
        realState=real,
        topSimStates=[[outcome, list(state), count] for (outcome, state), count in states.most_common(3)],
    )
    return row


def main():
    data = json.load(open(FIXTURES))
    fixtures = [f for f in data["fixtures"] if "mirrorOf" not in f]
    with multiprocessing.Pool(WORKERS) as pool:
        rows = pool.map(evaluate, fixtures, chunksize=4)
    done = [r for r in rows if "skipped" not in r]
    summary = {
        "fights": len(rows),
        "simulated": len(done),
        "skipped": dict(Counter(r["skipped"] for r in rows if "skipped" in r)),
        "samplesPerFight": SAMPLES,
        "outcomeImpossible": sum(not r["outcomePossible"] for r in done),
        "stateImpossible": sum(r["outcomePossible"] and not r["statePossible"] for r in done),
        "deterministic": sum(r["deterministic"] for r in done),
        "deterministicOutcomeWrong": sum(r["deterministic"] and not r["outcomePossible"] for r in done),
        "deterministicStateWrong": sum(r["deterministic"] and r["outcomePossible"] and not r["statePossible"] for r in done),
        "meanProbabilityOfRealOutcome": sum(r["pReal"] for r in done) / max(1, len(done)),
    }
    json.dump({"summary": summary, "rows": rows}, open(OUTPUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
