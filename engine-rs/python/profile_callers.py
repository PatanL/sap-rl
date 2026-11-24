import cProfile, pstats, random, sys, io
import sap_engine_rs
import sapai_gym1.rl.selfplay_env as selfplay_env
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C
sap_engine_rs.install(sys.argv[1], sys.argv[2]); selfplay_env.Battle = sap_engine_rs.RustBattle
env = ParallelSelfPlayEnv()
def play(n):
    for index in range(n):
        seed = 771100000 + index; rng = random.Random(seed); env.reset(seed=seed, paired_tape=True)
        for _ in range(6000):
            actions = [C.decode(rng.choice(tuple(env.core._avail_actions(s)))) if active else None for s, active in enumerate(env.active_players())]
            if env.step(*actions).terminated: break
p = cProfile.Profile(); p.enable(); play(int(sys.argv[3])); p.disable()
out = io.StringIO(); st = pstats.Stats(p, stream=out)
for name in sys.argv[4:]:
    st.print_callers(name)
print("\n".join(l[:200] for l in out.getvalue().splitlines() if l.strip())[:6000])
