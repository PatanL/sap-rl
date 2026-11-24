import sys,json,collections as C
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from identity import compile_turn
d={tuple(r['key']):r for r in map(json.loads,open('strict-v31t-1.jsonl')) if r.get('skip')}
counts=C.Counter();sizes=C.Counter();tops=[]
for r in map(json.loads,open('seats.jsonl')):
 for k,a in r['actions'].items():
  key=(r['match'],r['seat'],int(k))
  if key not in d:continue
  tr,names,imp,issues,prob=compile_turn(r,int(k),'near-right')
  buy={v['id'] for v in a if v['t'] in (6,7) and v['id'] not in names}
  ref={v['id'] for v in a if v['t']==9}|{v['target'] for v in a if v.get('target')}|{i for v in a for i,_ in v.get('orders',[])}
  extra=ref-names.keys()-buy
  sizes[len(buy),len(extra)]+=1;reason=d[key]['skip']; counts[reason]+=1
  tops.append((len(buy)+len(extra),key,r['states'][k]['tier'],[(v['t'],v.get('id'),v.get('target'),v.get('spell',{}).get('name')) for v in a]))
print(counts,sizes)
for x in sorted(tops,reverse=True)[:6]:print(x)
