#!/usr/bin/env python3
"""Three-way triage of fights the engine fails: real game, sapai, SAP-Calculator.

For each failing fight the calculator is run on the same board. If the
calculator reproduces the real result and sapai does not, the engine is wrong
and the calculator's code is the mechanism reference. If neither reproduces
it, suspect the converter (dropped fields) before the engine.

Also assigns each fight to dev or held-out by match id, so fixes developed on
dev can be checked for regressions on fights they were not fitted to.

Usage: triage.py <fixtures.json> <engine result.json> <calculator dir> <out.json> [samples]
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

FIXTURES, RESULT, CALCULATOR, OUTPUT = sys.argv[1:5]
SAMPLES = int(sys.argv[5]) if len(sys.argv) > 5 else 200
HERE = Path(__file__).resolve().parent
EQUIPMENT = {"Meat Bone", "Coconut", "Mushroom", "Garlic", "Honey", "Melon", "Peanut", "Chili", "Steak", "Cake"}
WINNER = {"player": "player", "opponent": "opponent", "draw": "draw"}


def split_of(match_id: str) -> str:
    return "heldout" if int(hashlib.sha256(match_id.encode()).hexdigest(), 16) % 5 < 2 else "dev"


def calc_pet(pet):
    return {"name": pet["name"], "attack": pet["attack"], "health": pet["health"],
            "exp": pet["experience"], "triggersConsumed": pet.get("triggersConsumed", 0),
            "equipment": {"name": pet["perk"]} if pet["perk"] in EQUIPMENT else None}


def calc_config(fixture):
    player = [calc_pet(p) for p in fixture["player"]]
    opponent = [calc_pet(p) for p in fixture["opponent"]]
    return {"playerPack": "Turtle", "opponentPack": "Turtle", "turn": fixture["turn"],
            "playerPets": player + [None] * (5 - len(player)),
            "opponentPets": opponent + [None] * (5 - len(opponent)),
            "seed": 20261122, "simulationCount": SAMPLES, "logsEnabled": True,
            "maxLoggedBattles": SAMPLES}


def run_calculator(configs, log=False):
    args = ["node", str(HERE / "calc_states.js"), CALCULATOR] + (["--log"] if log else [])
    done = subprocess.run(args, input=json.dumps(configs), text=True, capture_output=True, check=True)
    return json.loads(done.stdout)


def real_stats(fixture):
    pets = sorted(fixture["survivors"], key=lambda s: s["slot"])
    stats = [[int(p["attack"]), int(p["health"])] for p in pets]
    return sorted(stats)


def main():
    fixtures = {f["battleId"]: f for f in json.load(open(FIXTURES))["fixtures"]}
    rows = [r for r in json.load(open(RESULT))["rows"] if "skipped" not in r]
    failing = [r for r in rows if not r["outcomePossible"] or not r["statePossible"]]
    results = run_calculator([calc_config(fixtures[r["battleId"]]) for r in failing])
    out, kinds = [], Counter()
    for row, calc in zip(failing, results):
        fixture = fixtures[row["battleId"]]
        real = fixture["realOutcome"]
        key = {"player": "playerWins", "opponent": "opponentWins", "draw": "draws"}[real]
        calc_outcome = calc[key] > 0
        side = {"player": "player", "opponent": "opponent"}.get(real)
        want = real_stats(fixture)
        calc_state = any(
            state["winner"] == real and (side is None or sorted((state["board"] or {}).get(side, [])) == want)
            for state in calc["states"])
        sapai_outcome = row["outcomePossible"]
        if calc_state:
            kind = "engine bug (calculator matches real)"
        elif calc_outcome and not sapai_outcome:
            kind = "engine winner wrong; calculator winner right, board not"
        elif calc_outcome:
            kind = "both miss the real board"
        else:
            kind = "both miss the real winner"
        kinds[kind] += 1
        out.append({"battleId": row["battleId"], "split": split_of(fixture["matchId"]), "turn": fixture["turn"],
                    "kind": kind, "real": real, "realStats": want,
                    "sapaiOutcomes": row["simOutcomes"], "sapaiDeterministic": row["deterministic"],
                    "calcOutcomes": {k: calc[k] for k in ("playerWins", "opponentWins", "draws")},
                    "calcDeterministic": len(calc["states"]) == 1,
                    "pets": sorted({p["name"] for p in fixture["player"] + fixture["opponent"]}),
                    "size": len(fixture["player"]) + len(fixture["opponent"])})
    summary = {"failing": len(failing), "kinds": dict(kinds),
               "bySplit": dict(Counter(r["split"] for r in out)),
               "allFightsBySplit": dict(Counter(split_of(fixtures[r["battleId"]]["matchId"]) for r in rows))}
    json.dump({"summary": summary, "rows": out}, open(OUTPUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
