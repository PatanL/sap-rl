"""Entity encoder alone, forward + backward on the 8,192 saved decisions: as it is against torch.compile variants."""
import json, statistics, sys, time
import torch
from fixture import get_fixture
from equivalence import new, settings, verify_engine

settings(True); info = verify_engine(); turns = get_fixture(8192)
flat = {k: torch.cat([s.observations[k] for s in turns]).cuda() for k in turns[0].observations}
rows = flat['item_ids'].shape[0]
out = {'rows': rows}
sync = torch.cuda.synchronize


def bench(policy, label, reps=6):
    samples = []; ref = None
    for i in range(reps):
        policy.zero_grad(set_to_none=True)
        sync(); t = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = policy.encode_entities(flat)
        loss = tok.float().square().mean(); loss.backward(); sync()
        samples.append(time.perf_counter() - t)
    sync(); t = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        tok = policy.encode_entities(flat)
    sync(); fwd = time.perf_counter() - t
    out[label] = dict(first=samples[0], median=statistics.median(samples[2:]), forward_only=fwd, loss=float(loss), rows_per_s=rows / statistics.median(samples[2:]))
    print(label, out[label], flush=True); json.dump(out, open(sys.argv[1], 'w'), indent=1)
    return tok.float()


policy, _, _ = new('cuda', info); policy.train()
base = bench(policy, 'as is')
plain = policy.transformer
for label, kw in (('compile default', {}), ('compile dynamic', dict(dynamic=True)), ('compile max-autotune-no-cudagraphs', dict(mode='max-autotune-no-cudagraphs'))):
    try:
        policy.transformer = torch.compile(plain, **kw)
        tok = bench(policy, label)
        out[label]['max_abs_diff_vs_as_is'] = float((tok - base).abs().max()); out[label]['rel'] = float((tok - base).norm() / base.norm())
    except Exception as err:
        out[label] = dict(failed=repr(err)[:400]); print(label, out[label], flush=True)
    policy.transformer = plain
json.dump(out, open(sys.argv[1], 'w'), indent=1)
