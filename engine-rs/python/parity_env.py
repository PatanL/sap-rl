"""Play random legal full matches in the training environment and compare the
Rust battle with the Python battle at every fight, on the same seed.

Usage (training image): parity_env.py <lib .so> <data dir> <matches> [first match index]
"""
import random, sys, time
import sap_engine_rs
from sapai.battle import Battle as PythonBattle
import sapai_gym1.rl.selfplay_env as selfplay_env
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C

sap_engine_rs.install(sys.argv[1], sys.argv[2])
MATCHES = int(sys.argv[3]); FIRST = int(sys.argv[4]) if len(sys.argv) > 4 else 0
TALLY = {"battles": 0, "equal": 0, "different": 0, "unsupported": {}, "python_s": 0.0, "rust_s": 0.0, "examples": []}


class Checking:
    def __init__(self, t0, t1, seed=None, fast_training=False):
        self.args = (t0, t1, seed, fast_training)
    def battle(self):
        t0, t1, seed, fast = self.args
        start = time.perf_counter(); expected = PythonBattle(t0, t1, seed=seed, fast_training=fast).battle(); TALLY["python_s"] += time.perf_counter() - start
        TALLY["battles"] += 1
        try:
            start = time.perf_counter(); got = sap_engine_rs.battle(t0, t1, seed); TALLY["rust_s"] += time.perf_counter() - start
        except sap_engine_rs.Unsupported as reason:
            TALLY["unsupported"][str(reason)] = TALLY["unsupported"].get(str(reason), 0) + 1
            return expected
        if got == expected:
            TALLY["equal"] += 1
        else:
            TALLY["different"] += 1
            if len(TALLY["examples"]) < 5:
                show = lambda team: [(s.pet.name, s.pet.attack, s.pet.health, s.pet.level, s.pet.status, s.pet.ability_counter, s.pet.ability.get("_copiedFrom")) for s in team if not s.empty]
                TALLY["examples"].append({"seed": seed, "python": expected, "rust": got, "t0": show(t0), "t1": show(t1)})
        return expected


selfplay_env.Battle = Checking
env = ParallelSelfPlayEnv()
decisions = 0
for index in range(FIRST, FIRST + MATCHES):
    seed = 991100000 + index
    rng = random.Random(seed)
    env.reset(seed=seed, paired_tape=True)
    for _ in range(6000):
        actions = []
        for side, active in enumerate(env.active_players()):
            if not active:
                actions.append(None); continue
            actions.append(C.decode(rng.choice(tuple(env.core._avail_actions(side))))); decisions += 1
        if env.step(*actions).terminated:
            break
    else:
        raise RuntimeError("match did not finish")
TALLY["matches"] = MATCHES; TALLY["decisions"] = decisions
TALLY["python_ms_per_battle"] = 1000 * TALLY["python_s"] / max(1, TALLY["battles"])
TALLY["rust_ms_per_battle_including_packing"] = 1000 * TALLY["rust_s"] / max(1, TALLY["equal"] + TALLY["different"])
import json; print(json.dumps(TALLY, indent=1, default=str))
