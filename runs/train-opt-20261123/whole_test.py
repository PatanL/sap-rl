"""Is the whole-update compile stable on varying batches? Fresh (1.35M) arm. Times each update on a different random
subset of the saved turns (different turn lengths and row counts) and prints any update slower than 3x the median,
which would be a recompile. Also reports the start-up (first update) time, to see whether the on-disk compile cache helps.
Usage: whole_test.py <arm> <out.json> [updates]"""
import dataclasses, json, os, random, statistics, sys, time
from pathlib import Path
import torch
os.environ['SPEED_GEOMETRY'] = 'sorted'; os.environ['OPT_CUDNN_GRU'] = '1'
from fixture import get_fixture
from equivalence import settings, verify_engine
from learner import prepare, optimizer_for
from learner_fast import update_fast
from learner_ops import enable_learner
import compact_sequence
arm, out_path = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
settings(True); info = verify_engine(); turns = get_fixture(8192)
model, state, rates, sem, *_ = prepare(Path('/sap/checkpoints/update-025772.pt'), None, arm, info, 7); model.cuda()
opt = optimizer_for(model, state if arm == 'parent' else None, rates, 1)
for g in opt.param_groups: g['lr'] = 0.
w = model.config.memory_dim
if turns[0].initial_memory.shape[-1] != w: turns = [dataclasses.replace(t, initial_memory=t.initial_memory[..., :w].clone()) for t in turns]
enable_learner(model); compact_sequence.evaluate = torch.compile(compact_sequence.evaluate, dynamic=True)
rng = random.Random(3); ts = []; sizes = []
for i in range(N):
    batch = turns if i == 0 else rng.sample(turns, rng.randint(len(turns) // 4, len(turns)))
    torch.cuda.synchronize(); t = time.perf_counter(); m = update_fast(model, opt, batch, sem, 'cuda', rng.choice((1536, 3072, 6144)) if i else 3072, compact=True); torch.cuda.synchronize()
    ts.append(time.perf_counter() - t); sizes.append(m['decisions']); print(i, m['decisions'], m['microbatches'], round(ts[-1], 2), flush=True)
med = statistics.median(ts[1:]); slow = [(i, round(t, 1)) for i, t in enumerate(ts) if i and t > 3 * med]
out = dict(arm=arm, first=ts[0], median=med, slow=slow, rate=statistics.median(s / t for s, t in zip(sizes[1:], ts[1:])))
json.dump(out, open(out_path, 'w'), indent=1); print(json.dumps(out))
