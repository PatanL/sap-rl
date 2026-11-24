"""Compact + compiled learner with the cuDNN recurrence: gradient against the FP32 reference, and speed."""
import gc, json, os, statistics, sys, time
import torch
import equivalence, learner_ops, fast_encoder
from fixture import get_fixture
from equivalence import execute, rel, settings, verify_engine, new
from learner_fast import update_fast
settings(True); info = verify_engine(); turns = get_fixture(8192); out = {}
_, truth, _ = execute(False, 'cuda', info, turns, 'fp32', 128); gc.collect()
os.environ['SPEED_GEOMETRY'] = 'sorted'
plain = learner_ops.enable_learner
def both(m): plain(m); fast_encoder.enable(m)
learner_ops.enable_learner = both
for gru in ('0', '1'):
    os.environ['OPT_CUDNN_GRU'] = gru
    for prec in ('fp32', 'bf16'):
        result, grads, _ = execute(True, 'cuda', info, turns, prec, 3072, True)
        out[f'gru={gru} {prec}'] = dict(error=rel(truth, grads), loss=result['loss']); print(f'gru={gru} {prec}', json.dumps(out[f'gru={gru} {prec}']), flush=True)
    model, opt, sem = new('cuda', info); both(model)
    for g in opt.param_groups: g['lr'] = 0.
    samples = []
    for _ in range(6):
        torch.cuda.synchronize(); t = time.perf_counter(); m = update_fast(model, opt, turns, sem, 'cuda', 3072, compact=True); torch.cuda.synchronize(); samples.append(time.perf_counter() - t)
    out[f'gru={gru} seconds'] = statistics.median(samples[2:]); print(f'gru={gru} update seconds', out[f'gru={gru} seconds'], 'rate', m['decisions'] / out[f'gru={gru} seconds'], flush=True)
    del model, opt; gc.collect()
json.dump(out, open(sys.argv[1], 'w'), indent=1)
