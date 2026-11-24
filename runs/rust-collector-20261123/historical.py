"""Read existing experiment receipts for the training-signal analysis. No experiment is launched."""
from pathlib import Path
import json
R=Path('../')
def read(p):return json.loads(p.read_text())
control=R/'battle-equity-runtime-20260909-v1'
ev=[json.loads(line) for line in (R/'train-live-20261123/g-hist-frozen50/updates.jsonl').open()]
long=read(R/'train-live-20261123/g-hist-frozen50/result.json')
board=read(R/'train-live-20261123/board-noise.json')
old=read(R/'train-speed-20261123/results/proof/battle-parity.json')
profile=read(R/'train-speed-20261123/results/proof/result.json')
out=dict(decisions_per_live_turn=sum(x['decisions'] for x in ev)/sum(x['turns'] for x in ev),
 long_run=dict(learned=long['learned'],matches=long['matches'],updates=long['updates']),
 old_proof_battles_per_second=old['battles']/profile['elapsed'],
 old_proof_battles_per_seat_decision=old['battles']/profile['seat'],
 cost=read(control/'cost-v1/final-review-v1.json')['controllerDecision'],
 strength=read(control/'strength-v1/state.json')['evaluations'],previous_board_indices=board)
Path('results/history.json').write_text(json.dumps(out,indent=2))
