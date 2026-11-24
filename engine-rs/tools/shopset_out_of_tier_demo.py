"""Out-of-tier item on turn 2: what the reference does with it afterwards, and the port next to it."""
import json, sys, numpy as np
import gen_env_golden as G, gen_env_ghost as H, gen_env_shopset as S
from sap_env_rs import SapVecEnv, NO_DECISION
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai.data import data
G.EXT = True
env = SapVecEnv(1, library=sys.argv[1]); seed = 4242
py = ParallelSelfPlayEnv(); py.reset(seed=seed, paired_tape=True); env.reset(np.array([seed], dtype=np.int64))
keep = []
def same(tag):
    a, b = G.canonical(py.core), env.state(0)
    for seat in (0, 1):
        x, y = dict(a["seats"][seat]), dict(b["seats"][seat]); x.pop("toggled"); y.pop("toggled")
        assert x == y, (tag, seat)
def step(f0, f1, m0=0, m1=0):
    flats = [f0, f1]
    act = np.full((1, 2, 3), NO_DECISION, dtype=np.int64); act[:, :, 1] = 10; act[:, :, 2] = 5
    dec = []
    for seat, flat in enumerate(flats):
        if flat is None: dec.append(None); continue
        assert flat in G.legal_of(py)[seat], ("illegal", seat, flat)
        act[0, seat] = C.decode(flat).as_tuple(); dec.append(JointAction(C.decode(flat), None if flat else 0))
    env.step(act, np.zeros((1, 2), dtype=np.int64)); assert env.status[0] == 0, env.error(0)
    r = step_decisions(py, dec); same("step"); return r
def gold(seat, v):
    (py.core.agent, py.core.opponent)[seat].gold = v; env.set_gold(0, seat, v)
def put(seat, slot, name, frozen=False):
    S.place((py.core.agent, py.core.opponent)[seat], slot, name, frozen, keep); env.set_shop_item(0, seat, slot, name, frozen); same("place")
def shop():
    return [(row[1], "tier %d" % (data["pets"][row[1]]["tier"] if row[0] != "food" else data["foods"][row[1]]["tier"]), row[0], "frozen" if row[2] else "", "cost %d" % row[3]) + tuple(row[4:6]) for row in env.state(0)["seats"][0]["shop"]]
def me():
    s = env.state(0)["seats"][0]; return {"gold": s["gold"], "turn": s["turn"], "team": [None if r is None else (r[0], r[1], r[2], "level %d" % r[5], "sell %d" % r[9]) for r in s["team"]]}
gold(0, 0); gold(1, 0); step(0, 0)
print("turn 2 shop as rolled:", shop())
T5 = "pet-seal"
put(0, 0, T5); put(0, 1, T5, True); put(0, 2, T5)
print("three", T5, "(tier 5) placed, the second frozen:", shop())
gold(1, 0); step(1 + 0 * 6 + 0, 0)
print("bought the first:", me())
step(1 + 0 * 6 + 0, None); step(1 + 0 * 6 + 0, None)
print("the other two combined into it:", me())
print("level-up rewards on offer (turn 2):", shop())
put(0, 0, "pet-dragon", True)
print("a frozen Dragon (tier 6) placed into the first reward slot:", shop(), "reward groups", [r[0] for r in env.state(0)["seats"][0]["x"]["shop"]])
step(151, None)
print("after a roll:", shop(), me()["gold"], "gold")
step(146 + 0, None)
print("sold the level-2 Seal:", me())
gold(0, 0); step(0, None)
print("next turn:", shop(), "turn", me()["turn"])
print("reference and port agreed after every edit and step")
