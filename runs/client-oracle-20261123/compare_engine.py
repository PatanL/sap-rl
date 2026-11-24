"""Same integer seed (unsigned for NumPy), native start and attack rounds; original and candidate frame pairing."""
import json,hashlib,sys,copy
from pathlib import Path
D=Path(__file__).resolve().parent/'session2';O=D.parent.parent/'replay-oracle-20261122'
def pin():
 root=D/'engine-pin'
 if (root/'MANIFEST.json').exists():return root
 real=O/'engine-v31';files={}
 for p in sorted((real/'sapai').rglob('*.py')):
  b=p.read_bytes();rel=p.relative_to(real);q=root/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b);files[str(rel)]=hashlib.sha256(b).hexdigest()
 for rel,h in files.items():assert hashlib.sha256((real/rel).read_bytes()).hexdigest()==h
 (root/'MANIFEST.json').write_text(json.dumps(dict(files=files,hash=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()),indent=2));return root
def run(f):
 root=pin();sys.path.insert(0,str(root));sys.path.insert(1,str(O))
 import importlib.util
 old=sys.argv[:];sys.argv=['run_sapai.py',str(root),'-','-', '1','1']
 try:
  spec=importlib.util.spec_from_file_location('snap_ref',O/'run_sapai.py');oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
 finally:sys.argv=old
 from sapai import Battle
 battle=Battle(oracle.make_team(f['player']),oracle.make_team(f['opponent']),seed=f['seed']%2**32)
 def board(team):return [dict(pet=s.pet.name[4:].replace('-',' ').title(),attack=int(s.pet.attack),health=int(s.pet.health),perk=s.pet.status,slot=i) for i,s in enumerate(team) if not s.empty]
 def state(stage):return dict(stage=stage,player=board(battle.t0),opponent=board(battle.t1))
 rows=[state('init')];battle.start();rows.append(state('start'));i=0
 while any(not s.empty and s.pet.health>0 for s in battle.t0) and any(not s.empty and s.pet.health>0 for s in battle.t1):
  battle.pet_priority=battle.update_pet_priority(battle.t0,battle.t1,battle.rs)
  more=battle.attack(i);i+=1;rows.append(state(f'after-attack-{i:02d}'))
  if not more:break
 rows.append(state('final'));return dict(battle_id=f['battleId'],recorded_seed=f['seed'],engine_seed=f['seed']%2**32,steps=rows)
def main():
 rr={v['battleId']:v for v in json.load(open(O/'fixtures-340.json'))['fixtures']}
 for job in json.load(open(D/'dev-manifest.json')):
  assert int(hashlib.sha256(job['match'].encode()).hexdigest(),16)%5>=2
  dest=D/'engine-traces';dest.mkdir(exist_ok=True)
  (dest/(job['id']+'.json')).write_text(json.dumps(run(rr[job['id']]),indent=2))
if __name__=='__main__':main()
