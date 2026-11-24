"""Time the training environment under random legal play (observations included).

Usage: time_env.py <matches> [--rust-battle LIB DATA]
Legal actions are read from the observation's masks' source (one
_avail_actions call per decision), as the collector's validation does.
"""
import random, sys, time
import sapai_gym1.rl.selfplay_env as selfplay_env
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C

if "--rust-battle" in sys.argv:
    import sap_engine_rs
    at = sys.argv.index("--rust-battle")
    sap_engine_rs.install(sys.argv[at + 1], sys.argv[at + 2])
    selfplay_env.Battle = sap_engine_rs.RustBattle
MATCHES = int(sys.argv[1])
env = ParallelSelfPlayEnv()
decisions = battles = 0
start = time.perf_counter()
for index in range(MATCHES):
    seed = 771100000 + index
    rng = random.Random(seed)
    env.reset(seed=seed, paired_tape=True)
    for _ in range(6000):
        actions = []
        for side, active in enumerate(env.active_players()):
            if not active:
                actions.append(None); continue
            actions.append(C.decode(rng.choice(tuple(env.core._avail_actions(side))))); decisions += 1
        step = env.step(*actions)
        battles += int(step.battle_resolved)
        if step.terminated:
            break
seconds = time.perf_counter() - start
print(f"{decisions} decisions, {battles} battles, {seconds:.2f}s, {1000*seconds/decisions:.3f} ms/decision, {decisions/seconds:.0f} decisions/s")
