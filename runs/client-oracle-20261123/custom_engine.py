"""Compare the staged native implementation on the submitted physical gaps."""
from pathlib import Path
import json,sys
from select_fights import D
from compare_engine import pin
sys.path.insert(0,str(pin()))
from sapai import Battle,Team,Pet
pets={int(v['Id']):v['Name'] for v in json.loads((D.parent.parent.parent/'SAP-Calculator/src/assets/data/pets.json').read_text())}
def team(spec):
 t=Team()
 for i,v in enumerate(spec['Mins']['Items']):
  if not v:continue
  p=Pet('pet-'+pets[v.get('Enu',0)].lower().replace(' ','-'));p._attack=v['At']['Perm'];p._health=v['Hp']['Perm'];p.set_total_experience(0)
  if v.get('Perk')==1:p.status='status-extra-life'
  t[4-i]=p
 return t
def main():
 jobs=json.load(open(D/'custom-manifest.json'))
 for j in jobs:
  b=json.load(open(D/'fixtures'/j['file']));bt=Battle(team(b['UserBoard']),team(b['OpponentBoard']),seed=b['Seed'])
  def board(t):return [[i,s.pet.name,int(s.pet.attack),int(s.pet.health),s.pet.status] for i,s in enumerate(t) if not s.empty]
  def state(stage):return dict(stage=stage,player=board(bt.t0),opponent=board(bt.t1))
  out=[state('init')];bt.start();out.append(state('start'))
  for i in range(100):
   if not len(bt.t0) or not len(bt.t1):break
   bt.pet_priority=bt.update_pet_priority(bt.t0,bt.t1,bt.rs);more=bt.attack(i);out.append(state(f'after-attack-{i+1:02d}'))
   if not more:break
  (D/'engine-traces'/j['file']).write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
