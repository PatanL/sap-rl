"""Inputs are JSON objects in the pinned sapai Team.state schema, slot 0 at front."""
import argparse,hashlib,json
from pathlib import Path
import sapai
from sapai import Battle,Team
from sapai.turtle_pack import SIMULATOR_REVISION

def public(t):
    return [None if s.empty else dict(species=s.pet.name,attack=int(s.pet.attack),health=int(s.pet.health),level=int(s.pet.level),perk=s.pet.status) for s in t]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('seed',type=int)
p.add_argument('--full-state',action='store_true')
a=p.parse_args()
assert SIMULATOR_REVISION=='turtle-v30-food-right-reorder10-v1-20260911'
assert 0<=a.seed<2**32
assert hashlib.sha256(Path(sapai.__file__).with_name('battle.py').read_bytes()).hexdigest()=='03332ca6c673d4c138b6ffed7c4272a497e9660f42ff875964c327ef98ecd923'
b=Battle(Team.from_state(json.loads(a.left.read_text())),Team.from_state(json.loads(a.right.read_text())),seed=a.seed,fast_training=True)
result=b.battle()
out=dict(seed=a.seed,engine=SIMULATOR_REVISION,module=sapai.__file__,outcome=result,
         winner=('left','right','draw')[result],left=public(b.t0),right=public(b.t1))
if a.full_state:
    out['leftState'],out['rightState']=b.t0.state,b.t1.state
print(json.dumps(out))
