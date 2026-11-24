"""collector_g.py with the base learning rate overridden (SAP_BASE_LR); every other rate and all code unchanged.

collector_g.py takes its rates from the source checkpoint's optimizer groups (base 2e-6, opponent_history 1e-4,
opponent_adapter 1e-5, end_freeze 5e-5) and has no option for them. This wrapper replaces only rates['base'] in what
`prepare` returns, on a fresh start and on every resume, then runs collector_g.run with the same arguments.
"""
import argparse, os
from pathlib import Path
import collector_g

BASE = float(os.environ['SAP_BASE_LR'])
_prepare = collector_g.prepare


def prepare(*args, **kwargs):
    out = list(_prepare(*args, **kwargs))
    rates = dict(out[2])
    if set(rates) != {'base', 'opponent_history', 'opponent_adapter', 'end_freeze'}:
        raise ValueError(f'Unexpected rate groups {rates}')
    print(f'collector_g_lr: base rate {rates["base"]} -> {BASE}; others {rates}', flush=True)
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
    collector_g.run(p.parse_args())
