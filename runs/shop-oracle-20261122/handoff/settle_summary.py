import json,collections as C
from pathlib import Path
rs=list(map(json.loads,open('handoff/settlement.jsonl')))
def calc(xs):
 c=C.Counter();miss=[];badrows=[]
 for r in xs:
  c['matches']+=1;c['terminal']+=r['native_terminal_battle'] is not None;c['bothLastClock']+=all(t==r['native_terminal_clock'] for t in r['real_last']);c['winner']+=r['native_winner']==r['actual_winner'];c['lastBattle']+=r['last_battle']==r['native_terminal_battle']
  errors=0
  for v in r['records']:
   for s,k in [('hearts','lives'),('wins','wins'),('modehearts','lives')]:
    for j,value in v['record'][s].items():
     c[s]+=1;c[s+'Ok']+=value==v[k][int(j)]
     if value!=v[k][int(j)]:errors+=1;badrows.append((r['match'],v['turn'],s,j,value,v[k][int(j)]))
   c['rounds']+=1
  if (r['native_terminal_battle']!=r['last_battle'] or r['native_winner']!=r['actual_winner'] or errors or not all(t==r['native_terminal_clock'] for t in r['real_last'])):
   miss.append({k:r[k] for k in ('match','actual_final','actual_winner','native_winner','native_terminal_battle','last_battle','real_last','seat_battles')})
  else:c['allMatch']+=1
 return dict(counts=c,miss=miss,observed_mismatches=badrows)
groups={'all':rs,'winLoss':[r for r in rs if sorted(r['actual_final'])==[1,2]],'code3':[r for r in rs if 3 in r['actual_final']]}
z={k:calc(v) for k,v in groups.items()};Path('handoff/settlement-summary.json').write_text(json.dumps(z,indent=2))
for k,v in z.items():print(k,v['counts'],v['miss'][:1],v['observed_mismatches'][:3])
