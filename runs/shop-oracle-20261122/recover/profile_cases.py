import sys,json,collections as C
from pathlib import Path
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from identity import compile_turn
from raw_checks import sales_ledger
def jobs():
 d={tuple(r['key']):r for r in map(json.loads,open(P.parent/'strict-v31t-1.jsonl')) if r.get('skip')}
 for row in map(json.loads,open(P.parent/'seats.jsonl')):
  for k,acts in row['actions'].items():
   key=(row['match'],row['seat'],int(k))
   if key in d:yield row,int(k),d[key]
def inspect(row,n):
 k=str(n);acts=row['actions'][k];tr,names,imp,issues,probs=compile_turn(row,n,'near-right')
 allids=set();buys=set(); links=[]
 for a,(op,dst) in zip(acts,tr):
  t=a['t']
  if t in (6,7,9):allids.add(a['id'])
  if t==6:buys.add(a['id'])
  if a.get('target'):allids.add(a['target'])
  allids.update(i for i,x in a.get('orders',[]))
  if t==7:links.append((a['id'],a['target']))
 missing=sorted(allids-set(names));b=sorted(set(missing)&buys);e=sorted(set(missing)-buys)
 return acts,tr,names,missing,b,e,links
if __name__=='__main__':
 hist=C.Counter();g=C.Counter()
 for r,n,orig in jobs():
  acts,tr,names,miss,b,e,links=inspect(r,n)
  hist[len(b),len(e)]+=1
  if len(b)>=3:
   print(n,r['states'][str(n)]['tier'],b,[(a['t'],a.get('id'),(a.get('spell')or{}).get('name')) for a in acts if a.get('id') in b or a['t']==8], 'extra',e)
 print(hist)
