"""Paired greedy scoring on the Rust vector environment.

Same seeds, pairing, forward-batch order and record layout as Codex's evaluate.py
(runs/train-vector-20261123, mounted as /job), with the Python environment replaced by
`SapVecEnv` from engine-rs-pinned/v31g-env. With the same --batch the forward batches are the
same rows in the same order, so the records should equal evaluate.py's game for game; a larger
--batch changes BF16 rounding inside a forward and can flip a near-tied greedy choice.

Usage (inside the scoring container, see score_g.sh):
  evaluate_vec.py <candidate.pt> <opponent.pt> <out.json> --mode sanity --pairs 2048 [--batch 2048] [--threads 4]
"""
import argparse, datetime, hashlib, sys, time
from pathlib import Path
import numpy as np
import torch
from evaluate import load, world_ids, interval, atomic
from config import REV, PAYLOAD
from runtime import verify_engine

ENV_DIR = '/sap/engine-rs-pinned/v31g-env'
sys.path.insert(0, ENV_DIR)
from sap_env_rs import SapVecEnv, NO_DECISION  # noqa: E402

LIB = ENV_DIR + '/libsap_engine.so'
SAMPLED = [False]   # --sampled: both sides draw moves from their policies instead of taking the top one


@torch.inference_mode()
def play(seeds, models, device, batch, threads):
    tasks = [(i, s, side) for i, s in enumerate(seeds) for side in (0, 1)]
    records, decisions = [], 0
    seat_index = np.arange(2)[None, :]
    for start in range(0, len(tasks), batch):
        group = tasks[start:start + batch]
        n = len(group)
        env = SapVecEnv(n, threads=threads, library=LIB)
        if env.reset(np.array([seed for _, seed, _ in group], dtype=np.int64)):
            raise RuntimeError('reset refused')
        memories = [m.initial_memory(n * 2, device) for m in models]
        candidate_seat = np.array([side for _, _, side in group])
        done = np.zeros(n, dtype=bool)
        for _ in range(2500):
            if done.all():
                break
            actions = np.full((n, 2, 3), NO_DECISION, dtype=np.int64)
            actions[:, :, 1] = 10
            actions[:, :, 2] = 5
            masks = np.zeros((n, 2), dtype=np.int64)
            active = env.active.astype(bool) & ~done[:, None]
            for j in (0, 1):  # 0 = candidate, 1 = opponent
                games, seats = np.nonzero(active & ((seat_index != candidate_seat[:, None]) == bool(j)))
                if len(games) == 0:
                    continue
                inp = {name: torch.from_numpy(array[games, seats]).to(device) for name, array in env.obs.items()}
                index = torch.from_numpy(games * 2 + seats).to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                    out = models[j].act(inp, memory=memories[j].index_select(0, index), deterministic=not SAMPLED[0])
                memories[j].index_copy_(0, index, out.memory.float())
                actions[games, seats] = out.actions.cpu().numpy()
                if out.end_freeze_mask is not None:
                    masks[games, seats] = out.end_freeze_mask.cpu().numpy()
                decisions += len(games)
            if env.step(actions, masks):
                bad = int(np.nonzero(env.status < 0)[0][0])
                raise RuntimeError(f'game {bad} seed {group[bad][1]} status {env.status[bad]}: {env.error(bad)}')
            for k in np.nonzero(env.terminated.astype(bool) & ~done)[0]:
                i, seed, seat = group[k]
                winner = int(env.winner[k])
                records.append(dict(pair=i, seed=seed, seat=seat, score=.5 if winner == 2 else float(winner == seat),
                                    last_battle=int(env.state(int(k))['turn']) - 1))
                done[k] = True
        if not done.all():
            raise RuntimeError('Incomplete match; no score emitted')
        env.close()
    return sorted(records, key=lambda r: (r['pair'], r['seat'])), decisions


def main():
    p = argparse.ArgumentParser()
    for name in ('candidate', 'opponent', 'out'):
        p.add_argument(name, type=Path)
    p.add_argument('--mode', choices=['screen', 'confirm', 'panel', 'sanity'], required=True)
    p.add_argument('--pairs', type=int, required=True)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--gib', type=float, default=6.0)
    p.add_argument('--sampled', action='store_true')
    a = p.parse_args(); SAMPLED[0] = a.sampled
    if a.sampled: torch.manual_seed(20261123)
    if a.out.exists():
        raise FileExistsError(a.out)
    meta = verify_engine()
    if meta['payload_sha256'] != PAYLOAD or meta['simulator_revision'] != REV:
        raise ValueError('Wrong engine')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = 'cuda'
    _, total = torch.cuda.mem_get_info(); torch.cuda.set_per_process_memory_fraction(a.gib * 1024 ** 3 / total)
    m, c = load(a.candidate, device); n, o = load(a.opponent, device)
    started = time.monotonic()
    seeds = world_ids(a.mode, a.pairs, c['sha256'], o['sha256'])
    rows, decisions = play(seeds, [m, n], device, a.batch, a.threads)
    assert [(r['seed'], r['seat']) for r in rows] == [(s, seat) for s in seeds for seat in (0, 1)]
    pairs = np.array([r['score'] for r in rows]).reshape(-1, 2).mean(1)
    seconds = time.monotonic() - started
    with open(LIB, 'rb') as fp:
        library = hashlib.file_digest(fp, 'sha256').hexdigest()
    atomic(a.out, dict(protocol='paired-greedy-v31g-v1' if not a.sampled else 'paired-sampled-v31g-v1', environment='rust-vector', library_sha256=library, batch=a.batch,
                       mode=a.mode, candidate=c, opponent=o, engine=PAYLOAD, revision=REV,
                       score=float(pairs.mean()), ci95=interval(pairs), pairs=len(pairs), games=len(rows),
                       wdl=[sum(r['score'] == v for r in rows) for v in (1, .5, 0)],
                       seconds=seconds, decisions=decisions, decisions_per_second=decisions / seconds,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(), records=rows,
                       utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(f'score {pairs.mean():.4f} games {len(rows)} seconds {seconds:.1f} decisions/s {decisions / seconds:.0f}')


if __name__ == '__main__':
    main()
