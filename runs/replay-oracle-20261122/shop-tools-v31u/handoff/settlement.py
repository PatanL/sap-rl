"""Replay outcomes through the environment's normal END/END settlement and reward path."""
import json,collections as C,sys,hashlib
from pathlib import Path
from sapai_gym1.rl import selfplay_env as api
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.distributed import _completed_match_result
def de(v):return json.loads(v) if isinstance(v,str) else v or {}
class OutcomeBattle:
 value=None
 def __init__(self,*args,**kwargs):pass
 def battle(self):return self.value
def one(rows):
 pid=[r['ParticipationId'] for r in rows];m=rows[0]['MatchId'];obs={};last=[r['LastTurn'] for r in rows];dec=[r['Outcome'] for r in rows]
 for j,r in enumerate(rows):
  for a in r['Actions']:
   n=a['Turn'];typ=a['Type']
   if typ not in (0,1):continue
   rec=obs.setdefault(n,dict(hearts={},wins={},modehearts={},outcomes={}))
   if typ==0:
    b=de(a['Build'])['Bor'];bt=de(a['Battle'])
    rec['hearts'][j]=b['LiMa']-b.get('LoPs',0);rec['wins'][j]=b.get('Vic',0)
    o={1:j,2:1-j,3:2}[bt['Outcome']];rec['outcomes'][j]=o
   else:
    for q in de(a['Mode']).get('Opponents',[]):
     if q.get('ParticipationId') in pid:rec['modehearts'][pid.index(q['ParticipationId'])]=q.get('Lives',0)
 env=api.ParallelSelfPlayEnv();env.reset(seed=222);end=AutoregressiveAction(0);logs=[];terminal=None;reward_winner=None
 for n,rec in sorted(obs.items()):
  if not rec['outcomes']:continue
  os=set(rec['outcomes'].values());assert len(os)==1
  if terminal is not None:continue
  assert env.core.turn==n
  OutcomeBattle.value=os.pop()
  env.core.agent.gold=env.core.opponent.gold=0
  env.core._agent_actions_cache=env.core._opponent_actions_cache=None
  res=env.step(end,end);life=[env.core.agent.lives,env.core.opponent.lives];wins=[env.core.agent.wins,env.core.opponent.wins]
  logs.append(dict(turn=n,lives=life,wins=wins,record=rec,reward=res.rewards,terminated=res.terminated))
  if res.terminated:
   terminal=n;reward_winner=_completed_match_result(env.core)
 native_winner=None if terminal is None else reward_winner
 real_winner=dec.index(1) if dec.count(1)==1 else None
 return dict(match=m,split='heldout' if int(hashlib.sha256(m.encode()).hexdigest(),16)%5<2 else 'dev',records=logs,unique_turns=len([r for r in obs.values() if r['outcomes']]),last_battle=max([n for n,r in obs.items() if r['outcomes']],default=None),
   real_last=last,actual_final=dec,native_terminal_battle=terminal,native_terminal_clock=env.core.turn if terminal else None,
   native_winner=native_winner,actual_winner=real_winner,seat_battles=[sum(1 for r in obs.values() if j in r['outcomes']) for j in (0,1)])
if __name__=='__main__':
 api.Battle=OutcomeBattle
 files=sorted(Path('/oracle/raw').glob('*.json'))+sorted(Path('/archive').glob('*/BUNDLE.json'))
 out=[one([s['raw'] for s in json.load(f.open())['perspectives']['perspectives']]) for f in files]
 with open('handoff/settlement.jsonl','w') as fp:
  for r in out:fp.write(json.dumps(r,separators=(',',':'))+'\n')
