"""Profile the training environment's step with battles handed to Rust.

Random legal play, observations included (as in collection). Prints the
functions that take the most time. Usage: profile_env.py <lib> <data dir> <matches>
"""
import cProfile, pstats, random, sys, time, io
import sap_engine_rs
import sapai_gym1.rl.selfplay_env as selfplay_env
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C

sap_engine_rs.install(sys.argv[1], sys.argv[2])
selfplay_env.Battle = sap_engine_rs.RustBattle
MATCHES = int(sys.argv[3])
env = ParallelSelfPlayEnv()
COUNT = {"decisions": 0, "battles": 0}


def play():
    for index in range(MATCHES):
        seed = 771100000 + index
        rng = random.Random(seed)
        env.reset(seed=seed, paired_tape=True)
        for _ in range(6000):
            actions = []
            for side, active in enumerate(env.active_players()):
                if not active:
                    actions.append(None); continue
                actions.append(C.decode(rng.choice(tuple(env.core._avail_actions(side))))); COUNT["decisions"] += 1
            step = env.step(*actions)
            COUNT["battles"] += int(step.battle_resolved)
            if step.terminated:
                break


start = time.perf_counter(); play(); plain = time.perf_counter() - start
print(f"plain: {COUNT['decisions']} decisions, {COUNT['battles']} battles, {plain:.2f}s, {1000*plain/COUNT['decisions']:.3f} ms/decision, rust battles {sap_engine_rs.STATS['rust']}, fallbacks {sap_engine_rs.STATS['fallback']}")
COUNT.update(decisions=0, battles=0)
profile = cProfile.Profile(); profile.enable(); play(); profile.disable()
for key in ("tottime", "cumulative"):
    out = io.StringIO(); pstats.Stats(profile, stream=out).sort_stats(key).print_stats(28)
    lines = [l for l in out.getvalue().splitlines() if l.strip()]
    print(f"\n==== by {key}"); print("\n".join(l[:170] for l in lines[4:36]))
