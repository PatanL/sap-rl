"""More board-index recordings of 25,772 with other self-play seeds (noise estimate)."""
import json, sys
sys.path.insert(0, '/home/dzliu/pzliu/sap-rl/runs/train-vector-20261123')
from board_metric import run
ROOT = '/home/dzliu/pzliu/sap-rl/'
out = []
for seed in (20261126, 20261127, 20261128):
    r = run(ROOT + 'checkpoints/update-025772.pt', seed=seed)
    out.append(dict(checkpoint='checkpoints/update-025772.pt', seed=seed, index=r['index'], scores=r['scores']))
    print(json.dumps(out[-1]), flush=True)
    json.dump(out, open(ROOT + 'runs/train-live-20261123/board-noise2.json', 'w'), indent=1)
