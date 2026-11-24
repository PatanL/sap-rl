"""Cost of one acting call at several batch sizes: plain, compiled encoder, and the whole `act` through torch.compile.
The fixed cost per call is what frozen opponents pay every wave (one small call per frozen network).
Usage: act_compile_bench.py <out.json>"""
import json, statistics, sys, time
import numpy as np, torch
from pathlib import Path
from runtime import verify_engine
from learner import prepare
from rust_adapter import RustVector
import fast_encoder

info = verify_engine(); torch.set_num_threads(1)
model, *_ = prepare(Path('/sap/checkpoints/update-025772.pt'), None, 'parent', info, 1); model = model.cuda().eval().requires_grad_(False)
n = 2048; env = RustVector(n, 8); o = env.output; rng = np.random.default_rng(5); env.reset(np.arange(n), rng.integers(2 ** 62, size=n), o)
# advance a little with random legal-ish play by the policy so the rows are not all turn-1 states
memory = model.initial_memory(2 * n, 'cuda'); moves = np.zeros((n, 2, 3), 'i8'); freeze = np.zeros((n, 2), 'i8')
with torch.inference_mode():
    for _ in range(40):
        ids = np.flatnonzero(o['active'].ravel()); ix = torch.from_numpy(ids).cuda()
        ob = {k: torch.from_numpy(v.reshape(-1, *v.shape[2:])[ids]).cuda() for k, v in o['obs'].items()}
        with torch.autocast('cuda', dtype=torch.bfloat16): a = model.act(ob, memory=memory.index_select(0, ix))
        memory.index_copy_(0, ix, a.memory.float()); moves.reshape(-1, 3)[ids] = a.actions.cpu().numpy(); freeze.ravel()[ids] = a.end_freeze_mask.cpu().numpy()
        env.step(moves, freeze, o)
ids = np.flatnonzero(o['active'].ravel())
rows = {k: torch.from_numpy(v.reshape(-1, *v.shape[2:])[ids]).cuda() for k, v in o['obs'].items()}; mem = memory.index_select(0, torch.from_numpy(ids).cuda())
print('rows available', len(ids), flush=True)
out = {}

def bench(label, call):
    res = {}
    for b in (8, 32, 128, 512, min(2048, len(ids))):
        ob = {k: v[:b] for k, v in rows.items()}; m = mem[:b]; ts = []
        with torch.inference_mode():
            for i in range(40):
                torch.cuda.synchronize(); t = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16): r = call(ob, m)
                r.actions.cpu(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        res[b] = statistics.median(ts[10:]) * 1000
    out[label] = res; print(label, {b: round(v, 2) for b, v in res.items()}, 'ms per call', flush=True)
    json.dump(out, open(sys.argv[1], 'w'), indent=1)

bench('plain', lambda ob, m: model.act(ob, memory=m))
fast_encoder.enable(model)
bench('compiled encoder', lambda ob, m: model.act(ob, memory=m))
try:
    whole = torch.compile(lambda ob, m: model.act(ob, memory=m), dynamic=True)
    bench('compiled whole act', whole)
except Exception as err:
    out['compiled whole act'] = dict(failed=repr(err)[:600]); print('whole act failed', repr(err)[:600], flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
