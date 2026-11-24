"""Fingerprint full matches of the training environment under random legal play.

For each match seed, every step contributes the legal actions offered, the
action taken, the observations returned, the rewards and the players' full
state to a running SHA-256. Two engines that are behaviourally identical
(same rules, same random draws in the same order) give the same digests, so
this is the safety net for speed changes to the Python engine and, later, for
a Rust shop port.

Usage: env_trace.py <out.jsonl> <matches> [first index] [--rust-battle LIB DATA]
"""
import hashlib, json, random, sys, time
import numpy as np
import sapai_gym1.rl.selfplay_env as selfplay_env
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C

OUT, MATCHES = sys.argv[1], int(sys.argv[2])
FIRST = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 0
if "--rust-battle" in sys.argv:
    import sap_engine_rs
    at = sys.argv.index("--rust-battle")
    sap_engine_rs.install(sys.argv[at + 1], sys.argv[at + 2])
    selfplay_env.Battle = sap_engine_rs.RustBattle


def feed(digest, value):
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode()); digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            digest.update(str(key).encode()); feed(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"["); [feed(digest, v) for v in value]; digest.update(b"]")
    elif hasattr(value, "__dataclass_fields__"):
        for key in value.__dataclass_fields__:
            digest.update(key.encode()); feed(digest, getattr(value, key))
    else:
        digest.update(repr(value).encode())


def state_of(player):
    state = dict(player.state)
    return json.dumps(state, sort_keys=True, default=str)


env = ParallelSelfPlayEnv()
start = time.perf_counter(); total = 0
with open(OUT, "w") as out:
    for index in range(FIRST, FIRST + MATCHES):
        seed = 551100000 + index
        rng = random.Random(seed)
        digest = hashlib.sha256()
        feed(digest, env.reset(seed=seed, paired_tape=True))
        decisions = battles = 0
        for _ in range(6000):
            actions = []
            for side, active in enumerate(env.active_players()):
                if not active:
                    actions.append(None); continue
                keys = tuple(env.core._avail_actions(side))
                digest.update(repr(keys).encode())
                code = rng.choice(keys); digest.update(str(code).encode())
                actions.append(C.decode(code)); decisions += 1
            step = env.step(*actions)
            feed(digest, step)
            digest.update(state_of(env.core.agent).encode()); digest.update(state_of(env.core.opponent).encode())
            battles += int(step.battle_resolved)
            if step.terminated:
                break
        else:
            raise RuntimeError("match did not finish")
        total += decisions
        out.write(json.dumps({"seed": seed, "decisions": decisions, "battles": battles, "digest": digest.hexdigest()}) + "\n")
elapsed = time.perf_counter() - start
print(json.dumps({"matches": MATCHES, "decisions": total, "seconds": round(elapsed, 2), "ms_per_decision": round(1000 * elapsed / total, 4)}))
