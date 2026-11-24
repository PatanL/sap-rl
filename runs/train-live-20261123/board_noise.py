"""Board-index noise: re-record checkpoints with other self-play seeds (same tool the watcher uses)."""
import json, sys
sys.path.insert(0, '/home/dzliu/pzliu/sap-rl/runs/train-vector-20261123')
from board_metric import run
ROOT = '/home/dzliu/pzliu/sap-rl/'
jobs = [('checkpoints/update-025772.pt', 20261124), ('runs/train-live-20261123/g-hist-frozen50/update-027022.pt', 20261124),
        ('runs/train-live-20261123/g-hist-frozen50/update-027772.pt', 20261124), ('checkpoints/update-025772.pt', 20261125)]
out = []
for ck, seed in jobs:
    r = run(ROOT + ck, seed=seed)
    out.append(dict(checkpoint=ck, seed=seed, index=r['index'], scores=r['scores']))
    print(json.dumps(out[-1]), flush=True)
    json.dump(out, open(ROOT + 'runs/train-live-20261123/board-noise.json', 'w'), indent=1)
