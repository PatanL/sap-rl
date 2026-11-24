"""Compare native END and imported snapshots with identical orientations, opponents and seeds."""
import argparse,collections as C,json,multiprocessing as mp,sys,zlib
from pathlib import Path
from receipts import load_rows
from replay import compile_turn,engine_turn,expected
from sapai import Team,Pet,Battle
from sapai.effects import TransferAbility
# Reuse Claude's converter and exact scoring helpers (module parses CLI even on import).
import importlib.util
argv=sys.argv[:]
sys.argv=['run_sapai.py','/engine','-','-', '200','6']
spec=importlib.util.spec_from_file_location('oracle_battle','/oracle/run_sapai.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle);sys.argv=argv
def signature(team):
 out=[]
 for s in team:
  if s.empty:out.append(None);continue
  p=s.pet
  out.append(dict(name=p.name,level=p.level,xp=p.total_experience,attack=int(p.attack),hp=int(p.health),rawA=int(p._attack),rawH=int(p._health),
       perk=p.status,uses=p.ability_counter,hurt=p.friends_hurt_counter,copy=p.ability.get('_copiedFrom'),
       ability=p.ability,swallowed=p.swallowed_pets))
 return out
def probe(t0,t1,fixture):
 real=sorted(oracle.real_state(fixture));outcome=fixture['realOutcome'];wins=False;ordered=False
 for i in range(200):
  b=Battle(t0,t1,seed=(1234567+i*2654435761)%2**32,fast_training=True);res=oracle.OUTCOMES[b.battle()]
  if res!=outcome:continue
  wins=True;winner={'player':b.t0,'opponent':b.t1}.get(res)
  state=oracle.final_state(winner) if winner is not None else ()
  if sorted(state)==real:return dict(outcome=True,state=True,first=i)
 return dict(outcome=wins,state=False,first=None)
def compare(team,reference,which,other,f):
 def evaluate(t):return probe(t,other,f) if which==0 else probe(other,t,f)
 snap=evaluate(reference);chain=evaluate(team)
 extra={}
 if snap['state'] and not chain['state']:
  def changed(mode):
   t=team.copy();rr=[s.pet for s in reference if not s.empty];nn=[s.pet for s in t if not s.empty]
   if 'count' in mode:
    for p,q in zip(nn,rr):p.ability_counter=q.ability_counter;p.friends_hurt_counter=q.friends_hurt_counter
   if 'temp' in mode:
    for p,q in zip(nn,rr):
     p._attack=q._attack;p._health=q._health;p._until_end_of_battle_attack_buff=0;p._until_end_of_battle_health_buff=0
   if 'parrot' in mode:
    import copy
    for p,q in zip(nn,rr):
     if p.name=='pet-parrot':p.override_ability=q.override_ability;p.override_ability_dict=copy.deepcopy(q.override_ability_dict)
   if 'slots' in mode:t.move_forward()
   return t
  for v in ('count','temp','parrot','slots','count+temp+parrot+slots'):extra[v]=evaluate(changed(v))
 return snap,chain,extra
def one(arg):
 row,witnesses=arg;results=[]
 for n,r in witnesses.items():
  tag=(row['match'],n);f=FIX[tag];side=0 if f['participationId']==row['seat'] else 1
  actualspec=(f['player'],f['opponent'])[side];other=oracle.make_team((f['opponent'],f['player'])[side])
  ref=oracle.make_team(actualspec);pick=[];k=str(n)
  steps,names,inferred,_,_=compile_turn(row,n,'near-right')
  kinds={int(i):v[0] for i,v in row['known'].items()};kinds.update({i:'pet' for i in inferred})
  out=engine_turn(row['states'][str(n-1)],row['states'][k],row['actions'][k],names,kinds,r['seed'],on_end=pick.append)
  assert out.get('state')==expected(row['states'][k]) and not out['costErrors'] and len(pick)==1
  own=pick[0].team;orig=signature(own);ss=signature(ref);aa=[x for x in orig if x];bb=[x for x in ss if x]
  for x,y in zip(aa,bb):
   assert all(x[t]==y[t] for t in ('name','attack','hp','level','xp','perk')),(tag,side,x,y)
  assert len(aa)==len(bb)
  pairs=[[x['name'],x['uses'],y['uses']] for x,y in zip(aa,bb)]
  diffs=[[j,x['name'],[key for key in x if x[key]!=y[key]]] for j,(x,y) in enumerate(zip(aa,bb)) if x!=y]
  snap,chain,extra=compare(own,ref,side,other,f)
  assert signature(own)==orig,'Battle mutated the original shop team'
  rec=dict(key=r['key'],battle=f['battleId'],side=side,split=r['split'],snap=snap,chain=chain,extra=extra,counters=pairs,diffs=diffs,gaps=[x is None for x in orig]!=[x is None for x in ss])
  if chain['state']!=snap['state']:rec.update(native=orig,imported=ss)
  results.append(rec)
 return results
if __name__=='__main__':
 FIX={(f['matchId'],f['turn']):f for f in json.load(open('/oracle/fixtures-340.json'))['fixtures'] if 'mirrorOf' not in f}
 by=C.defaultdict(dict)
 for (m,p,n),r in load_rows('v31t')[0].items():
  if r.get('reproduced'):by[m,p][n]=r
 rows=[json.loads(l) for l in open('seats.jsonl')]
 with mp.Pool(6) as pool,open('handoff/chain.jsonl','w',buffering=1) as fp:
  for batch in pool.imap_unordered(one,[(s,by[s['match'],s['seat']]) for s in rows]):
   for rec in batch:fp.write(json.dumps(rec,separators=(',',':'))+'\n')
