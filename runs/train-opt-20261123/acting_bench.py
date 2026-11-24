"""Acting alone on the Rust environment: seat decisions/s for the plain and the compiled encoder at several widths,
one policy (no frozen opponents), sampled actions, no learner. Usage: acting_bench.py <out.json>"""
import json, sys, time
import numpy as np, torch
from runtime import verify_engine
from learner import prepare
from rust_adapter import RustVector
from pathlib import Path
import fast_encoder

info = verify_engine(); torch.set_num_threads(1)
model, *_ = prepare(Path('/sap/checkpoints/update-025772.pt'), None, 'parent', info, 1); model = model.cuda().eval().requires_grad_(False)
out = {}
for compiled in (False, True):
    if compiled: fast_encoder.enable(model)
    for n in (512, 2048, 4096):
        env = RustVector(n, 10); o = env.output; rng = np.random.default_rng(5); env.reset(np.arange(n), rng.integers(2 ** 62, size=n), o)
        memory = model.initial_memory(2 * n, 'cuda'); moves = np.zeros((n, 2, 3), 'i8'); freeze = np.zeros((n, 2), 'i8'); seat = 0; begin = None
        with torch.inference_mode():
            for wave in range(100000):
                if wave == 30: torch.cuda.synchronize(); begin = time.perf_counter(); seat = 0
                ids = np.flatnonzero(o['active'].ravel()); ix = torch.from_numpy(ids).cuda()
                ob = {k: torch.from_numpy(v.reshape(-1, *v.shape[2:])[ids]).cuda() for k, v in o['obs'].items()}
                with torch.autocast('cuda', dtype=torch.bfloat16): a = model.act(ob, memory=memory.index_select(0, ix))
                memory.index_copy_(0, ix, a.memory.float()); moves.reshape(-1, 3)[ids] = a.actions.cpu().numpy(); freeze.ravel()[ids] = a.end_freeze_mask.cpu().numpy()
                env.step(moves, freeze, o); seat += len(ids)
                done = np.flatnonzero(o['terminated'])
                if len(done):
                    env.reset(done, rng.integers(2 ** 62, size=len(done)), o); memory.index_fill_(0, torch.as_tensor(np.stack((done * 2, done * 2 + 1), 1).ravel(), device='cuda'), 0.)
                if begin and time.perf_counter() - begin > 25: break
        torch.cuda.synchronize(); rate = seat / (time.perf_counter() - begin); env.close()
        out[f"{'compiled' if compiled else 'plain'} {n}"] = rate; print(f"{'compiled' if compiled else 'plain'} games {n}: {rate:.0f} seat decisions/s", flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
