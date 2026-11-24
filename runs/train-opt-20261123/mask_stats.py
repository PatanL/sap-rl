import torch, numpy as np
from fixture import get_fixture
turns = get_fixture(8192)
m = torch.cat([s.observations['entity_mask'] for s in turns]).bool().numpy()
valid = m.sum(1)
print('rows', len(m), 'slots', m.shape[1], 'mean valid', valid.mean().round(2), 'percentiles 10/50/90/100', np.percentile(valid, [10, 50, 90, 100]))
print('per-slot occupancy', m.mean(0).round(2).tolist())
print('histogram', np.bincount(valid, minlength=22).tolist())
last = np.array([np.flatnonzero(r).max() for r in m]); print('mean index of last valid slot', last.mean().round(2))
