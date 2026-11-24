#!/usr/bin/env python3
"""Find real fights whose result reveals exactly one random choice.

For each real fight, run the reference engine over many seeds with a logging
wrapper around the battle's random generator. Keep fights where every run
makes exactly one random target choice (tie-break shuffles aside), the final
board differs for each option, and the real final board matches one option.
Each kept fight gives one data point: the game's recorded Seed and which
option the real game picked.

Usage (training image): collect.py <engine dir> <oracle dir> <fixtures.json> <out.json> [workers]
"""
import json, multiprocessing, sys
ENGINE, ORACLE, FIXTURES, OUT = sys.argv[1:5]
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
sys.path.insert(0, ENGINE); sys.path.insert(0, ORACLE)
sys.argv = ["run_sapai.py", ENGINE, FIXTURES, "/dev/null", "1", "1"]
import numpy as np
import run_sapai
from sapai import Battle


class Logging:
    """RandomState stand-in that records every draw."""
    def __init__(self, seed):
        self.inner = np.random.RandomState(seed); self.log = []
    def choice(self, a, size=None, replace=True, p=None):
        out = self.inner.choice(a, size=size, replace=replace, p=p)
        options = len(a) if hasattr(a, "__len__") else int(a)
        count = int(np.prod(size)) if size is not None else 1
        kind = "shuffle" if (not replace and count == options and options > 1) else "pick"
        picked = out.tolist() if hasattr(out, "tolist") else out
        self.log.append((kind, options, picked, [str(x) for x in a] if hasattr(a, "__len__") and options <= 6 else None))
        return out
    def random_sample(self, *args):
        self.log.append(("sample", 0, None, None)); return self.inner.random_sample(*args)
    def get_state(self): return self.inner.get_state()
    def set_state(self, state): return self.inner.set_state(state)
    def seed(self, seed): return self.inner.seed(seed)


def run(fixture):
    if "mirrorOf" in fixture:
        return None
    real = sorted(run_sapai.real_state(fixture)); real_outcome = fixture["realOutcome"]
    by_option = {}; picks_seen = None
    try:
        for index in range(48):
            seed = (1234567 + index * 2654435761) % (2 ** 32)
            battle = Battle(run_sapai.make_team(fixture["player"]), run_sapai.make_team(fixture["opponent"]), seed=seed, fast_training=True)
            proxy = Logging(seed); battle.rs = proxy
            battle._bind_battle_rng(battle.t0); battle._bind_battle_rng(battle.t1)
            battle.pet_priority = battle.update_pet_priority(battle.t0, battle.t1, battle.rs)
            outcome = run_sapai.OUTCOMES[battle.battle()]
            winner = {"player": battle.t0, "opponent": battle.t1}.get(outcome)
            state = (outcome, tuple(sorted(run_sapai.final_state(winner))) if winner is not None else ())
            picks = [e for e in proxy.log if e[0] == "pick" and e[1] > 1]
            samples = [e for e in proxy.log if e[0] == "sample"]
            if len(picks) != 1:
                return None
            _, options, picked, labels = picks[0]
            before = proxy.log.index(picks[0])
            shuffles_before = [e[1] for e in proxy.log[:before] if e[0] == "shuffle"]
            option = picked[0] if isinstance(picked, list) else picked
            sig = (options, tuple(shuffles_before), tuple(labels or ()))
            if picks_seen is None: picks_seen = sig
            elif picks_seen != sig: return None
            if option in by_option and by_option[option] != state: return None
            by_option[option] = state
    except Exception:
        return None
    if picks_seen is None or len(by_option) != picks_seen[0]:
        return None
    if len(set(by_option.values())) != len(by_option):
        return None
    matches = [o for o, s in by_option.items() if s == (real_outcome, tuple(real))]
    if len(matches) != 1:
        return {"battleId": fixture["battleId"], "unmatched": True}
    return {"battleId": fixture["battleId"], "seed": fixture["seed"], "turn": fixture["turn"], "options": picks_seen[0],
            "shufflesBefore": list(picks_seen[1]), "labels": list(picks_seen[2]), "realOption": int(matches[0]) if not isinstance(matches[0], str) else matches[0],
            "player": [p["name"] for p in fixture["player"]], "opponent": [p["name"] for p in fixture["opponent"]]}


def main():
    fixtures = json.load(open(FIXTURES))["fixtures"]
    with multiprocessing.Pool(WORKERS) as pool:
        rows = [r for r in pool.imap(run, fixtures, chunksize=16) if r]
    good = [r for r in rows if not r.get("unmatched")]
    json.dump({"points": good, "unmatched": len(rows) - len(good)}, open(OUT, "w"), indent=1)
    print(json.dumps({"dataPoints": len(good), "singleChoiceButRealUnmatched": len(rows) - len(good)}))

if __name__ == "__main__":
    main()
