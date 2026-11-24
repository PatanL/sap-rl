"""collector_g.py (Codex, runs/train-vector-20261123) with ranked players' boards as a third opponent type.

Nothing in collector_g.py, the learner or the engine is edited. This wrapper replaces three things at run time:
  * `collector_g.Opponents` -> GhostOpponents: per new game, with probability SAP_GHOST_SHARE one seat is a
    ghost (owner -3); otherwise with probability SAP_FROZEN_SHARE of all games one seat is a frozen network as
    before (parent, or with probability 1/2 a recent snapshot); otherwise both seats are the live network;
  * `pool.models[-3]` -> a stand-in whose `act` returns END for every row, so the coordinator needs no change
    and spends no network time on ghost seats;
  * `worker_profile.profiled_worker` -> ghost_worker.worker (same directory), which runs inside every spawned
    actor process and swaps the ranked board in before each battle.
Optional SAP_BASE_LR overrides the base learning rate exactly as collector_g_lr.py does.
The opponent-prediction heads will now also see ranked players' boards in the opponent history: that is
opponent modelling. No human move is ever a training target.
"""
import argparse, os
from pathlib import Path
from types import SimpleNamespace
import torch
import collector_g
import worker_profile
import ghost_worker
from ghost_worker import GHOST
from opponents import Opponents, LIVE, PARENT

GHOST_SHARE = float(os.environ['SAP_GHOST_SHARE'])
FROZEN_SHARE = float(os.environ['SAP_FROZEN_SHARE'])
if not (0 <= GHOST_SHARE <= 1 and 0 <= FROZEN_SHARE <= 1 and GHOST_SHARE + FROZEN_SHARE <= 1):
    raise ValueError('shares must be probabilities that sum to at most 1')


class EndOnly:
    """Stands where a frozen network would: every ghost row ends its turn, no freeze, memory untouched."""

    def act(self, obs, memory):
        n = memory.shape[0]
        actions = torch.tensor([0, 10, 5], dtype=torch.long, device=memory.device).repeat(n, 1)
        return SimpleNamespace(actions=actions, end_freeze_mask=torch.zeros(n, dtype=torch.long, device=memory.device),
                               log_prob=torch.zeros(n, device=memory.device), memory=memory)


class GhostOpponents(Opponents):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models[GHOST] = EndOnly()

    def retire(self):
        super().retire()
        self.models[GHOST] = self.models.get(GHOST) or EndOnly()

    def reset_games(self, indices):
        for g in indices:
            self.owners[2 * g:2 * g + 2] = LIVE
            key = 'mirror'
            u = self.rng.random()
            if u < GHOST_SHARE:
                seat = int(self.rng.integers(2)); self.owners[2 * g + seat] = GHOST
                key = f'ghost:seat{seat}'
            elif u < GHOST_SHARE + FROZEN_SHARE:
                frozen = PARENT
                if self.recent and self.rng.random() < .5:
                    frozen = self.recent[int(self.rng.integers(len(self.recent)))][0]
                seat = int(self.rng.integers(2)); self.owners[2 * g + seat] = frozen
                key = f'parent:seat{seat}' if frozen == PARENT else f'snapshot-{frozen}:seat{seat}'
            self.assignments[key] += 1
        self.retire()


collector_g.Opponents = GhostOpponents
worker_profile.profiled_worker = ghost_worker.worker

if os.environ.get('SAP_BASE_LR'):
    BASE = float(os.environ['SAP_BASE_LR'])
    _prepare = collector_g.prepare

    def prepare(*args, **kwargs):
        out = list(_prepare(*args, **kwargs))
        rates = dict(out[2])
        if set(rates) != {'base', 'opponent_history', 'opponent_adapter', 'end_freeze'}:
            raise ValueError(f'Unexpected rate groups {rates}')
        print(f'collector_g_ghost: base rate {rates["base"]} -> {BASE}; others {rates}', flush=True)
        rates['base'] = BASE
        out[2] = rates
        return tuple(out)

    collector_g.prepare = prepare

if __name__ == '__main__':
    # Same options as collector_g.py's own entry point (copied; that file is frozen).
    p = argparse.ArgumentParser()
    p.add_argument('--resume', type=Path); p.add_argument('--checkpoint-every', type=int, default=250)
    p.add_argument('--workers', type=int, default=8); p.add_argument('--games', type=int, default=96); p.add_argument('--target', type=int, default=8192)
    p.add_argument('--priority-inference', action='store_true'); p.add_argument('--capture-at', type=int, default=0); p.add_argument('--fingerprints', action='store_true')
    p.add_argument('--compact', action='store_true'); p.add_argument('--fast-actor', action='store_true')
    p.add_argument('--battle-backend', choices=['python', 'rust'], default='python'); p.add_argument('--audit-battles', action='store_true')
    p.add_argument('--concurrent', action='store_true'); p.add_argument('--learner', choices=['old', 'fast'], default='fast')
    p.add_argument('--cap', type=int, default=512); p.add_argument('--updates', type=int, required=True); p.add_argument('--max-lag', type=int, default=1)
    p.add_argument('--backlog', type=float, default=1.); p.add_argument('--seconds', type=float, default=810); p.add_argument('--seed', type=int, default=2577223)
    p.add_argument('--save', action='store_true'); p.add_argument('--profile', action='store_true'); p.add_argument('--out', type=Path, required=True)
    print(f'collector_g_ghost: ghost share {GHOST_SHARE}, frozen share {FROZEN_SHARE}, pool {ghost_worker.POOL_SHA256[:16]}', flush=True)
    collector_g.run(p.parse_args())
