"""Which learner layouts are faithful, judged against the FP32 gradient of the reference update (not against the
reference's own BF16 rounding), and how fast each is.

truth      = reference update (learner.update, 128-row groups) in FP32
candidates = reference in BF16; learner_fast in BF16 with each geometry and cap; compact
For each: relative L2 error of the accumulated gradient against truth, cosine, loss, seconds per update (median of 3
after a warm-up), microbatches. Usage: layout_truth.py <out.json>
"""
import gc, json, os, statistics, sys, time
import torch
from fixture import get_fixture
from equivalence import execute, new, rel, settings, verify_engine
from learner_fast import update_fast
from learner_ops import enable_learner

settings(True); info = verify_engine(); turns = get_fixture(8192)
out = {}
_, truth, _ = execute(False, 'cuda', info, turns, 'fp32', 128); gc.collect()


def timing(cap, compact):
    model, opt, sem = new('cuda', info); enable_learner(model)
    for g in opt.param_groups: g['lr'] = 0.
    samples = []
    for _ in range(4):
        torch.cuda.synchronize(); t = time.perf_counter(); m = update_fast(model, opt, turns, sem, 'cuda', cap, compact=compact)
        torch.cuda.synchronize(); samples.append(time.perf_counter() - t)
    del model, opt; gc.collect(); torch.cuda.empty_cache()
    return statistics.median(samples[1:]), m


def record(name, fast, prec, cap, geometry=None, compact=False):
    if geometry: os.environ['SPEED_GEOMETRY'] = geometry
    try:
        result, grads, _ = execute(fast, 'cuda', info, turns, prec, cap, compact) if fast else execute(False, 'cuda', info, turns, prec, cap)
        row = dict(error=rel(truth, grads), loss=result['loss'])
        if fast and prec == 'bf16':
            seconds, m = timing(cap, compact)
            row.update(seconds=seconds, rate=m['decisions'] / seconds, microbatches=m['microbatches'], padded=m['padded_rows'])
    except Exception as err:  # keep the table going
        row = dict(failed=repr(err)[:300]); torch.cuda.empty_cache()
    out[name] = row; print(name, json.dumps(row), flush=True)
    json.dump(out, open(sys.argv[1], 'w'), indent=1); gc.collect()


record('reference bf16', False, 'bf16', 128)
record('reference fp32 again', False, 'fp32', 128)
for geometry in ('block-both', 'sorted', 'aligned'):
    for cap in (1536, 3072):
        record(f'{geometry} cap {cap} bf16', True, 'bf16', cap, geometry)
record('sorted cap 1536 fp32', True, 'fp32', 1536, 'sorted')
record('sorted compact cap 1536 fp32', True, 'fp32', 1536, 'sorted', True)
record('sorted compact cap 1536 bf16', True, 'bf16', 1536, 'sorted', True)
record('sorted compact cap 3072 bf16', True, 'bf16', 3072, 'sorted', True)
