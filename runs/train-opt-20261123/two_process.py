"""Would a separate learner process help? Run with role=learner or role=actor in two containers at the same time and
compare their rates with the rates each reaches alone and with the single-process run.
learner: compact + compiled weight updates on the saved batch, back to back, for SECONDS.
actor:   compiled acting on the Rust environment at 512 games for SECONDS.
Usage: two_process.py learner|actor <seconds> <out.json>"""
import json, os, sys, time
import numpy as np, torch
role, seconds, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
if role == 'learner':
    os.environ['SPEED_GEOMETRY'] = 'sorted'
    from fixture import get_fixture
    from equivalence import new, settings, verify_engine
    from learner_fast import update_fast
    from learner_ops import enable_learner
    import fast_encoder
    settings(True); info = verify_engine(); turns = get_fixture(8192)
    model, opt, sem = new('cuda', info); enable_learner(model); fast_encoder.enable(model)
    for g in opt.param_groups: g['lr'] = 0.
    for _ in range(3): update_fast(model, opt, turns, sem, 'cuda', 3072, compact=True)
    torch.cuda.synchronize(); begin = time.perf_counter(); n = 0; stamps = []
    while time.perf_counter() - begin < seconds:
        m = update_fast(model, opt, turns, sem, 'cuda', 3072, compact=True); torch.cuda.synchronize(); n += m['decisions']; stamps.append(time.perf_counter() - begin)
    json.dump(dict(role=role, decisions_per_s=n / stamps[-1], updates=len(stamps), stamps=stamps), open(out, 'w'))
    print('learner', n / stamps[-1])
else:
    from runtime import verify_engine
    from learner import prepare
    from rust_adapter import RustVector
    from pathlib import Path
    import fast_encoder
    info = verify_engine(); torch.set_num_threads(1)
    model, *_ = prepare(Path('/sap/checkpoints/update-025772.pt'), None, 'parent', info, 1); model = model.cuda().eval().requires_grad_(False); fast_encoder.enable(model)
    n = 512; env = RustVector(n, 8); o = env.output; rng = np.random.default_rng(5); env.reset(np.arange(n), rng.integers(2 ** 62, size=n), o)
    memory = model.initial_memory(2 * n, 'cuda'); moves = np.zeros((n, 2, 3), 'i8'); freeze = np.zeros((n, 2), 'i8'); seat = 0; begin = None; marks = []
    with torch.inference_mode():
        for wave in range(10 ** 9):
            if wave == 30: torch.cuda.synchronize(); begin = time.perf_counter(); seat = 0
            ids = np.flatnonzero(o['active'].ravel()); ix = torch.from_numpy(ids).cuda()
            ob = {k: torch.from_numpy(v.reshape(-1, *v.shape[2:])[ids]).cuda() for k, v in o['obs'].items()}
            with torch.autocast('cuda', dtype=torch.bfloat16): a = model.act(ob, memory=memory.index_select(0, ix))
            memory.index_copy_(0, ix, a.memory.float()); moves.reshape(-1, 3)[ids] = a.actions.cpu().numpy(); freeze.ravel()[ids] = a.end_freeze_mask.cpu().numpy()
            env.step(moves, freeze, o); seat += len(ids)
            done = np.flatnonzero(o['terminated'])
            if len(done):
                env.reset(done, rng.integers(2 ** 62, size=len(done)), o); memory.index_fill_(0, torch.as_tensor(np.stack((done * 2, done * 2 + 1), 1).ravel(), device='cuda'), 0.)
            if begin: marks.append((time.perf_counter() - begin, seat))
            if begin and time.perf_counter() - begin > seconds: break
    json.dump(dict(role=role, seat_per_s=seat / marks[-1][0], marks=marks[::20]), open(out, 'w'))
    print('actor', seat / marks[-1][0])
