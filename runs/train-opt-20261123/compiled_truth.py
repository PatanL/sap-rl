"""Gradient of the full update with the compiled encoder, against the FP32 reference gradient (same test as layout_truth.py)."""
import gc, json, os, sys
import torch
import equivalence
from fixture import get_fixture
from equivalence import execute, rel, settings, verify_engine
import fast_encoder

settings(True); info = verify_engine(); turns = get_fixture(8192); out = {}
_, truth, _ = execute(False, 'cuda', info, turns, 'fp32', 128); gc.collect()
plain_new = equivalence.new
def compiled_new(device, info):
    m, opt, sem = plain_new(device, info)
    class Hook:  # enable() must come after enable_learner(), which execute() calls right after new()
        pass
    return m, opt, sem
os.environ['SPEED_GEOMETRY'] = 'sorted'
import learner_ops
plain_enable = learner_ops.enable_learner
def enable_both(m):
    plain_enable(m); fast_encoder.enable(m)
learner_ops.enable_learner = enable_both
for name, prec in (('compiled sorted compact cap 3072 fp32', 'fp32'), ('compiled sorted compact cap 3072 bf16', 'bf16')):
    result, grads, _ = execute(True, 'cuda', info, turns, prec, 3072, True)
    out[name] = dict(error=rel(truth, grads), loss=result['loss']); print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
