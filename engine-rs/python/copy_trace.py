"""Fingerprint matches in which the environment is deep-copied, pickled and
rebuilt from saved state while it is played.

env_trace.py never copies the game. Search, rollouts and worker snapshots do,
so a speed change to how pets and shop slots hold their random generators must
also be identical across copy, pickle and Player.from_state. Each match
alternates the three every few steps and keeps playing on the copy; the digest
covers legal actions, actions, observations and both players' saved state.

Usage: copy_trace.py <out.jsonl> <matches>
"""
import copy, hashlib, json, pickle, random, sys
import numpy as np
from sapai.player import Player
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C


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


out = open(sys.argv[1], "w")
copies = {"deepcopy": 0, "pickle": 0, "from_state": 0}
for index in range(int(sys.argv[2])):
    seed = 661100000 + index
    rng = random.Random(seed)
    env = ParallelSelfPlayEnv()
    env.reset(seed=seed, paired_tape=False)
    digest = hashlib.sha256(); decisions = 0
    for step_index in range(6000):
        if step_index % 7 == 3:
            kind = ("deepcopy", "pickle", "from_state")[(step_index // 7) % 3]
            if kind == "deepcopy":
                env = copy.deepcopy(env)
            elif kind == "pickle":
                env = pickle.loads(pickle.dumps(env))
            else:
                for name in ("agent", "opponent"):
                    player = getattr(env.core, name)
                    state = player.state
                    rebuilt = Player.from_state(json.loads(json.dumps(state)))
                    feed(digest, rebuilt.state)
            copies[kind] += 1
        actions = []
        for side, active in enumerate(env.active_players()):
            if not active:
                actions.append(None); continue
            legal = tuple(env.core._avail_actions(side)); feed(digest, legal)
            choice = rng.choice(legal); feed(digest, choice); actions.append(C.decode(choice)); decisions += 1
        step = env.step(*actions)
        feed(digest, step)
        for name in ("agent", "opponent"):
            feed(digest, getattr(env.core, name).state)
        if step.terminated:
            break
    out.write(json.dumps({"seed": seed, "decisions": decisions, "digest": digest.hexdigest()}) + "\n"); out.flush()
print(json.dumps({"matches": int(sys.argv[2]), "copies": copies}))
