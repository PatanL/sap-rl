"""Weight-update and acting speed by network size, with the compiled encoder in autotune mode (random weights; timing only).
Sizes: name -> (d_model, layers, heads, memory, opponent-history width); feed-forward width is 4 * d_model.
Usage: profile_sizes.py <out.json> [size names...]"""
import dataclasses, gc, json, os, statistics, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
os.environ['SPEED_GEOMETRY'] = 'sorted'
from fixture import get_fixture
from equivalence import settings, verify_engine
from learner import prepare, optimizer_for, read_checkpoint
from learner_fast import update_fast
from learner_ops import enable_learner
from sapai_gym1.rl.model import EntityTransformerPolicy, EntityTransformerConfig
import fast_encoder
SIZES = {'1.35M': (128, 4, 4, 192, 48), '2.6M': (176, 4, 8, 256, 64), '4.47M': (224, 5, 8, 320, 80), '8.7M': (304, 6, 8, 448, 96), '14.35M': (384, 6, 8, 512, 128)}
out_path = sys.argv[1]; names = sys.argv[2:] or list(SIZES)
settings(True); info = verify_engine(); base_turns = get_fixture(8192); out = {}
parent = read_checkpoint(Path('/sap/checkpoints/update-025772.pt')); rates = {g['name']: 0. for g in parent['optimizer']['param_groups']}
_, _, _, sem, *_ = prepare(Path('/sap/checkpoints/update-025772.pt'), None, 'parent', info, 7)
sync = torch.cuda.synchronize; clock = time.perf_counter
from rust_adapter import RustVector
n = 2048; env = RustVector(n, 5); o = env.output; env.reset(np.arange(n), np.random.default_rng(5).integers(2 ** 62, size=n), o)
ids = np.flatnonzero(o['active'].ravel()); rows = {k: torch.from_numpy(v.reshape(-1, *v.shape[2:])[ids]).cuda() for k, v in o['obs'].items()}
for name in names:
    d, layers, heads, mem, hist = SIZES[name]
    cfg = replace(EntityTransformerConfig(**parent['config']), d_model=d, num_layers=layers, nhead=heads, dim_feedforward=4 * d, memory_dim=mem, opponent_history_dim=hist)
    res = {'config': SIZES[name]}
    for mode in ('default', 'max-autotune-no-cudagraphs'):
        fast_encoder.MODE.clear(); fast_encoder._COMPILED.clear()
        if mode != 'default': fast_encoder.MODE['mode'] = mode
        torch.manual_seed(1); model = EntityTransformerPolicy(cfg).cuda(); res['parameters'] = sum(p.numel() for p in model.parameters())
        opt = optimizer_for(model, None, rates, 1); enable_learner(model); fast_encoder.enable(model)
        turns = [dataclasses.replace(t, initial_memory=(t.initial_memory[..., :mem].clone() if mem <= t.initial_memory.shape[-1] else torch.zeros(mem))) for t in base_turns]
        ts = []
        for _ in range(10):
            sync(); t = clock(); m = update_fast(model, opt, turns, sem, 'cuda', 3072, compact=True); sync(); ts.append(clock() - t)
        res[f'update {mode}'] = dict(seconds=statistics.median(ts[3:]), rate=m['decisions'] / statistics.median(ts[3:]), first=ts[0])
        if mode != 'default':
            actor = EntityTransformerPolicy(cfg).cuda().eval().requires_grad_(False); fast_encoder.enable(actor); act = {}
            with torch.inference_mode():
                for b in (64, 512, 2048, 4096):
                    b = min(b, len(ids)); ob = {k: v[:b] for k, v in rows.items()}; memory = actor.initial_memory(b, 'cuda'); tt = []
                    for i in range(24):
                        sync(); t = clock()
                        with torch.autocast('cuda', dtype=torch.bfloat16): r = actor.act(ob, memory=memory)
                        r.actions.cpu(); sync(); tt.append(clock() - t)
                    act[b] = round(b / statistics.median(tt[8:]))
            res['acting rows/s'] = act; del actor
        del model, opt; gc.collect(); torch.cuda.empty_cache()
    u = res['update max-autotune-no-cudagraphs']['rate']; a = res['acting rows/s'][max(res['acting rows/s'])]
    res['ideal learned/s (mirror only, shared GPU)'] = round(1 / (1 / u + 1.3 / a))
    out[name] = res; print(name, json.dumps(res), flush=True); json.dump(out, open(out_path, 'w'), indent=1)
