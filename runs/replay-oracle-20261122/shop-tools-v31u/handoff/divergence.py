import json,collections as C,importlib,sys
from chain import oracle,probe
from replay import compile_turn,engine_turn
from sapai import Battle
from receipts import load_rows
mod=importlib.import_module('sapai.battle')
def phases(t0,t1,seed):
 history=[];original=mod.battle_phase
 def wrapped(b,phase,teams,*args):
  out=original(b,phase,teams,*args)
  history.append((phase,[[[s.pet.name,int(s.pet.attack),int(s.pet.health),s.pet.status] for s in t if not s.empty] for t in teams]))
  return out
 mod.battle_phase=wrapped
 try:Battle(t0,t1,seed=seed,fast_training=True).battle()
 finally:mod.battle_phase=original
 return history
if __name__=='__main__':
 rows=load_rows('v31t')[0];fixtures={f['battleId']:f for f in json.load(open('/oracle/fixtures-340.json'))['fixtures']};seats={(s['match'],s['seat']):s for s in map(json.loads,open('seats.jsonl'))}
 counter=C.Counter();examples={};splitstat=C.Counter()
 for x in map(json.loads,open('handoff/chain.jsonl')):
  if not x['extra']:continue
  key=tuple(x['key']);m,p,n=key;r=rows[key];s=seats[m,p];f=fixtures[x['battle']];j=x['side']
  _,names,im,_,_=compile_turn(s,n,'near-right');kinds={int(i):v[0] for i,v in s['known'].items()};kinds.update({i:'pet' for i in im})
  got=[];engine_turn(s['states'][str(n-1)],s['states'][str(n)],s['actions'][str(n)],names,kinds,r['seed'],on_end=got.append)
  t=got[0].team;other=oracle.make_team((f['opponent'],f['player'])[j]);ref=oracle.make_team((f['player'],f['opponent'])[j])
  seed=(1234567+x['snap']['first']*2654435761)%2**32
  def args(a):return (a,other) if j==0 else (other,a)
  a,b=phases(*args(t),seed),phases(*args(ref),seed)
  for i,(v,w) in enumerate(zip(a,b)):
   if v!=w:
    counter[v[0]]+=1
    if v[0] not in examples:examples[v[0]]=dict(key=key,seed=seed,previous=a[i-1] if i else None,native=v,snapshot=w)
    break
  for field in ('hp','attack'):
   u=t.copy()
   for slot in u:
    if slot.empty:continue
    q=slot.pet
    if field=='hp':q._health=int(q.health);q._until_end_of_battle_health_buff=0
    else:q._attack=int(q.attack);q._until_end_of_battle_attack_buff=0
   b0=Battle(*args(u),seed=seed,fast_training=True);o=oracle.OUTCOMES[b0.battle()];team={'player':b0.t0,'opponent':b0.t1}.get(o)
   if o==f['realOutcome'] and sorted(oracle.final_state(team) if team else ())==sorted(oracle.real_state(f)):splitstat[field]+=1
 json.dump(dict(phaseCounts=counter,oneSeedRecovery=splitstat,examples=examples),open('handoff/first-divergence.json','w'),indent=2)
 print(counter,splitstat)
