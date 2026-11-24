"""Where one weight update spends its time (production settings: learner_fast, block-both geometry, fused loss, BF16).

Reports, on the saved 8,192-decision batch of runs/train-speed-20261123:
  gemm      what this GPU does on a plain BF16 matmul of the feed-forward shape (the ceiling for the big layers)
  caps      seconds per update at several padded-row caps
  split     for the production cap: time in pack (CPU gather + copy to GPU), forward, backward, optimizer
  profile   top operators by GPU time for one update, and total GPU time against wall time
Usage (scoring container): learner_profile.py <out.json>
"""
import json, os, statistics, sys, time
import torch
from fixture import get_fixture
from equivalence import new, settings, verify_engine
import learner_fast
from learner_fast import update_fast, plan, pack, micro_loss
from learner_ops import enable_learner

out = {}
COMPACT = os.environ.get('OPT_COMPACT') == '1'; CAP = int(os.environ.get('OPT_CAP', '1536'))
settings(True); info = verify_engine(); turns = get_fixture(8192)
sync = torch.cuda.synchronize
clock = time.perf_counter


def gemm():
    a = torch.randn(32768, 384, device='cuda', dtype=torch.bfloat16); w = torch.randn(384, 1536, device='cuda', dtype=torch.bfloat16)
    for _ in range(5): a @ w
    sync(); t = clock()
    for _ in range(50): a @ w
    sync(); s = clock() - t
    return 50 * 2 * 32768 * 384 * 1536 / s / 1e12


out['gemm_tflops_bf16'] = gemm()
print('gemm', out['gemm_tflops_bf16'], flush=True)

model, opt, sem = new('cuda', info); enable_learner(model)
if os.environ.get('OPT_COMPILE') == '1':
    import fast_encoder; fast_encoder.enable(model)
for g in opt.param_groups: g['lr'] = 0.
out['caps'] = {}
for cap in (1536, 3072, 6144):
    try:
        samples = []
        for _ in range(4):
            sync(); t = clock(); m = update_fast(model, opt, turns, sem, 'cuda', cap, compact=COMPACT); sync(); samples.append(clock() - t)
        out['caps'][cap] = dict(seconds=samples, median=statistics.median(samples[1:]), rate=m['decisions'] / statistics.median(samples[1:]),
                                microbatches=m['microbatches'], padded=m['padded_rows'], decisions=m['decisions'], peak=torch.cuda.max_memory_reserved())
    except torch.cuda.OutOfMemoryError:
        out['caps'][cap] = dict(oom=True); torch.cuda.empty_cache()
    print(cap, out['caps'][cap], flush=True)

# Split of one update at the production cap.
groups = plan(turns, CAP, 128); n = sum(s.length for s in turns)
split = dict(pack=0., forward_loss=0., backward=0., optimizer=0., plan=0.)
for rep in range(3):
    acc = dict.fromkeys(split, 0.)
    sync(); t = clock(); groups = plan(turns, CAP, 128); acc['plan'] = clock() - t
    opt.zero_grad(set_to_none=True)
    real_pack = learner_fast.pack
    for group in groups:
        box = {}
        def timed_pack(parts, device, compact=False):
            sync(); t0 = clock(); r = real_pack(parts, device, compact); sync(); box['pack'] = clock() - t0; return r
        learner_fast.pack = timed_pack
        sync(); t = clock(); loss = micro_loss(model, group, sem, 'cuda', True, n, COMPACT); sync(); total = clock() - t
        learner_fast.pack = real_pack
        acc['pack'] += box['pack']; acc['forward_loss'] += total - box['pack']
        t = clock(); loss.backward(); sync(); acc['backward'] += clock() - t
    t = clock(); torch.nn.utils.clip_grad_norm_(model.parameters(), 40.); opt.step(); sync(); acc['optimizer'] = clock() - t
    split = acc
out['split_seconds'] = split; out['split_total'] = sum(split.values())
print('split', split, flush=True)

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    sync(); t = clock(); update_fast(model, opt, turns, sem, 'cuda', CAP, compact=COMPACT); sync(); wall = clock() - t
events = prof.key_averages()
def cuda_time(e): return getattr(e, 'self_device_time_total', None) or getattr(e, 'self_cuda_time_total', 0)
rows = sorted(events, key=cuda_time, reverse=True)[:25]
out['profile'] = dict(wall_with_profiler=wall, gpu_seconds=sum(cuda_time(e) for e in events) / 1e6,
                      cpu_self_seconds=sum(e.self_cpu_time_total for e in events) / 1e6,
                      top_gpu=[[e.key, round(cuda_time(e) / 1e6, 4), e.count] for e in rows],
                      top_cpu=[[e.key, round(e.self_cpu_time_total / 1e6, 4), e.count] for e in sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:25]])
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out['profile'])[:3000])
