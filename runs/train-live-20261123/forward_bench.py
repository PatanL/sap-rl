"""How many decisions per second can the acting network serve at large batch?

Collects real observations by playing a few hundred environments at random for
a while, tiles them to each batch size, and times `model.act` (sampling, BF16
autocast, per-seat memory supplied) on the GPU. Forward only: no environment,
no transfer of observations from other processes. This is the ceiling for
games per second set by the network alone.

Usage: forward_bench.py <checkpoint> <out.json>
"""
import json, random, sys, time
import numpy as np
import torch
from sapai_gym1.rl.model import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C

ck = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=True)
model = EntityTransformerPolicy(EntityTransformerConfig(**ck["config"]))
model.load_state_dict(ck["model"], strict=True)
model = model.eval().requires_grad_(False).cuda()
params = sum(p.numel() for p in model.parameters())
rng = random.Random(3); pool = []
for index in range(96):
    env = ParallelSelfPlayEnv(); obs = env.reset(seed=5000 + index, paired_tape=True)
    for _ in range(rng.randrange(20, 400)):
        acts = [C.decode(rng.choice(tuple(env.core._avail_actions(s)))) if a else None for s, a in enumerate(env.active_players())]
        step = env.step(*acts); obs = step.observations
        if step.terminated:
            break
    pool.extend(obs)
keys = list(pool[0]); rows = []
with torch.inference_mode():
    for batch in (64, 256, 1024, 2048, 4096, 8192, 16384):
        picks = [pool[i % len(pool)] for i in range(batch)]
        inp = {k: torch.from_numpy(np.stack([o[k] for o in picks])).cuda() for k in keys}
        memory = model.initial_memory(batch, "cuda")
        def call():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.act(inp, memory=memory, deterministic=False)
            torch.cuda.synchronize(); return out
        try:
            call(); call()
            times = []
            for _ in range(8 if batch <= 4096 else 4):
                start = time.perf_counter(); call(); times.append(time.perf_counter() - start)
        except torch.OutOfMemoryError:
            rows.append({"batch": batch, "error": "out of memory"}); break
        median = sorted(times)[len(times) // 2]
        rows.append({"batch": batch, "ms": 1000 * median, "decisions_per_s": batch / median,
                     "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
        print(json.dumps(rows[-1]), flush=True)
json.dump({"checkpoint": sys.argv[1], "parameters": params, "rows": rows}, open(sys.argv[2], "w"), indent=1)
