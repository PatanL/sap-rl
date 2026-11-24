"""Weight-update and acting cost for one network size, with the update variants side by side.

Arm: parent (14.35M) or fresh (1.35M, random weights; timing only). Same saved 8,192-decision batch for both (for the
small network the stored start-of-turn memory is cut to its width, which is fine for timing).
Update variants, each timed on 8 updates after 3 warm-ups, cap 3,072 unless --cap:
  encoder      compact layout + compiled encoder (what runs today)
  encoder+gru  the same with the recurrence as one cuDNN call
  autotune     compiled encoder in max-autotune-no-cudagraphs mode
  whole        the whole sequence evaluation (encoder, recurrence in one call, heads) compiled from a plain policy
Then a profile of the first variant (operator counts, GPU and CPU time) and acting calls at several batch sizes.
Usage: profile_arm.py parent|fresh <out.json> [cap]"""
import dataclasses, gc, json, os, statistics, sys, time
from pathlib import Path
import numpy as np, torch
os.environ['SPEED_GEOMETRY'] = 'sorted'
from fixture import get_fixture
from equivalence import settings, verify_engine
from learner import prepare, optimizer_for
import learner_fast
from learner_fast import update_fast
from learner_ops import enable_learner
import compact_sequence, fast_encoder

arm, out_path = sys.argv[1], sys.argv[2]; CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 3072
settings(True); info = verify_engine(); turns = get_fixture(8192); out = {'arm': arm, 'cap': CAP}
PARENT = Path('/sap/checkpoints/update-025772.pt')

def build():
    model, state, rates, sem, *_ = prepare(PARENT, None, arm, info, 7)
    model.cuda(); opt = optimizer_for(model, state if arm == 'parent' else None, rates, 1)
    for g in opt.param_groups: g['lr'] = 0.
    return model, opt, sem

probe, _, _ = build(); width = probe.config.memory_dim; out['parameters'] = sum(p.numel() for p in probe.parameters()); del probe
if turns[0].initial_memory.shape[-1] != width:
    turns = [dataclasses.replace(t, initial_memory=t.initial_memory[..., :width].clone()) for t in turns]
sync = torch.cuda.synchronize; clock = time.perf_counter
plain_evaluate = compact_sequence.evaluate

def timed(label, setup):
    os.environ['OPT_CUDNN_GRU'] = '0'; fast_encoder.MODE.clear(); fast_encoder._COMPILED.clear(); compact_sequence.evaluate = plain_evaluate
    model, opt, sem = build(); enable_learner(model)
    try:
        setup(model)
        ts = []
        for _ in range(11):
            sync(); t = clock(); m = update_fast(model, opt, turns, sem, 'cuda', CAP, compact=True); sync(); ts.append(clock() - t)
        out[label] = dict(seconds=statistics.median(ts[3:]), rate=m['decisions'] / statistics.median(ts[3:]), first=ts[0], microbatches=m['microbatches'])
    except Exception as err:
        out[label] = dict(failed=repr(err)[:500]); torch.cuda.empty_cache()
    print(label, json.dumps(out[label])[:400], flush=True); json.dump(out, open(out_path, 'w'), indent=1)
    return model, opt, sem

def whole(model):
    os.environ['OPT_CUDNN_GRU'] = '1'
    compact_sequence.evaluate = torch.compile(plain_evaluate, dynamic=True)

model, opt, sem = timed('encoder', lambda m: fast_encoder.enable(m))
# profile of the current variant
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    sync(); t = clock(); update_fast(model, opt, turns, sem, 'cuda', CAP, compact=True); sync(); wall = clock() - t
ev = prof.key_averages(); dev = lambda e: getattr(e, 'self_device_time_total', None) or getattr(e, 'self_cuda_time_total', 0)
launches = sum(e.count for e in ev if e.key in ('cudaLaunchKernel', 'cuLaunchKernel'))
out['profile'] = dict(wall=wall, cpu_self_s=sum(e.self_cpu_time_total for e in ev) / 1e6, kernel_launches=launches,
                      top_gpu=[[e.key[:70], round(dev(e) / 1e6, 4), e.count] for e in sorted(ev, key=dev, reverse=True)[:14]],
                      top_cpu=[[e.key[:60], round(e.self_cpu_time_total / 1e6, 4), e.count] for e in sorted(ev, key=lambda e: e.self_cpu_time_total, reverse=True)[:10]])
print('profile', json.dumps(out['profile'])[:1500], flush=True)
del model, opt; gc.collect()
timed('encoder+gru', lambda m: (fast_encoder.enable(m), os.environ.__setitem__('OPT_CUDNN_GRU', '1')))
timed('autotune', lambda m: (fast_encoder.MODE.update(mode='max-autotune-no-cudagraphs'), fast_encoder.enable(m)))
timed('whole', whole)
compact_sequence.evaluate = plain_evaluate; os.environ['OPT_CUDNN_GRU'] = '0'; fast_encoder.MODE.clear(); fast_encoder._COMPILED.clear()

# acting calls
model, _, _ = build(); model = model.eval().requires_grad_(False); fast_encoder.enable(model)
from rust_adapter import RustVector
n = 2048; env = RustVector(n, 5); o = env.output; env.reset(np.arange(n), np.random.default_rng(5).integers(2 ** 62, size=n), o)
ids = np.flatnonzero(o['active'].ravel()); rows = {k: torch.from_numpy(v.reshape(-1, *v.shape[2:])[ids]).cuda() for k, v in o['obs'].items()}; mem = model.initial_memory(len(ids), 'cuda')
act = {}
with torch.inference_mode():
    for b in (8, 64, 512, 2048, 4096):
        b = min(b, len(ids)); ob = {k: v[:b] for k, v in rows.items()}; ts = []
        for i in range(30):
            sync(); t = clock()
            with torch.autocast('cuda', dtype=torch.bfloat16): r = model.act(ob, memory=mem[:b])
            r.actions.cpu(); sync(); ts.append(clock() - t)
        act[b] = dict(ms=statistics.median(ts[8:]) * 1000, rows_per_s=b / statistics.median(ts[8:]))
out['acting'] = act; print('acting', json.dumps(act), flush=True)
json.dump(out, open(out_path, 'w'), indent=1)
