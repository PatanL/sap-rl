"""Record the boards a checkpoint takes into battle in greedy self-play, on the Rust vector environment.

Same output as runs/train-vector-20261123/record_boards.py's `*-battles.jsonl` (one line per battle:
{"match", "turn", "teams": [seat 0 team, seat 1 team]}, pets front to back), but the teams come from the
environment's own battle event (both teams exactly as handed to the battle engine), so nothing is inferred.
Usage: record_boards_rust.py <checkpoint> <out battles.jsonl> [--matches 1000] [--seed 20261123] [--batch 1000] [--threads 5]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from evaluate import load
from rust_adapter import RustVector
from board_event import BoardCodec
from runtime import verify_engine


@torch.inference_mode()
def main(a):
    verify_engine(); torch.set_num_threads(1)
    model, meta = load(a.checkpoint, 'cuda'); codec = BoardCodec()
    seeds = [(a.seed * 1000003 + i * 7919) % 2 ** 31 for i in range(a.matches)]
    battles = decisions = 0; begin = time.perf_counter()
    with a.output.open('x') as archive:
        for start in range(0, len(seeds), a.batch):
            worlds = seeds[start:start + a.batch]; n = len(worlds); env = RustVector(n, a.threads)
            try:
                o = env.output; env.reset(np.arange(n), worlds, o)
                memory = model.initial_memory(n * 2, 'cuda')
                moves = np.zeros((n, 2, 3), 'i8'); freeze = np.zeros((n, 2), 'i8')
                while np.any(o['active']):
                    ids = np.flatnonzero(o['active'].ravel()); ix = torch.from_numpy(ids).to('cuda')
                    ob = {k: torch.from_numpy(v.reshape(-1, *v.shape[2:])[ids]).to('cuda') for k, v in o['obs'].items()}
                    with torch.autocast('cuda', dtype=torch.bfloat16): out = model.act(ob, memory=memory.index_select(0, ix), deterministic=True)
                    memory.index_copy_(0, ix, out.memory.float())
                    moves.reshape(-1, 3)[ids] = out.actions.cpu().numpy(); freeze.ravel()[ids] = out.end_freeze_mask.cpu().numpy()
                    env.step(moves, freeze, o); decisions += len(ids)
                    resolved = np.flatnonzero(o['battle_resolved'])
                    if len(resolved):
                        turns, _, teams = env.native.battle_events()
                        for g in resolved:
                            archive.write(json.dumps({'match': f'bot-{a.tag}-{worlds[g]}', 'turn': int(turns[g]),
                                                      'teams': [codec.team(teams[g, 0]), codec.team(teams[g, 1])]}) + '\n')
                        battles += len(resolved)
                assert o['terminated'].all()
            finally: env.close()
    print(json.dumps(dict(matches=a.matches, battles=battles, decisions=decisions, seconds=time.perf_counter() - begin, checkpoint=meta, batch=a.batch, seed=a.seed)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('checkpoint', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--matches', type=int, default=1000); p.add_argument('--seed', type=int, default=20261123)
    p.add_argument('--batch', type=int, default=1000); p.add_argument('--threads', type=int, default=5); p.add_argument('--tag', default='greedy')
    main(p.parse_args())
