"""Use input-side offers listed immediately after a refresh; do not use terminal stats."""
from collections import Counter
import itertools,json
from pathlib import Path
P=Path(__file__).resolve().parent
# The static pool table gives exact shop-tier membership of each named standard species.
odds=json.load(open(P.parent/'engine-odds.json'))
TIER={name:next(int(t) for t in sorted(odds,key=int) if name in odds[t] and odds[t][name]>0) for name in odds['6']}
TIER['Sloth']=1
def clues(row,n):
 k=str(n);acts=row['actions'][k];prev=row['states'][str(n-1)];stdtier=row['states'][k]['tier']
 frozen={v['id'] for v in prev['shop']+prev['food'] if v['f']}
 kinds={int(i):v[0] for i,v in row['known'].items()};kinds.update({a['id']:'pet' for a in acts if a['t']==6})
 edges=[];ordinary=set()
 for j,a in enumerate(acts):
  for i,flag in a.get('freeze',[]):
   if flag:frozen.add(i)
   else:frozen.discard(i)
  if a['t'] in (4,5) and j+1<len(acts):
   listing=acts[j+1].get('freeze',[])
   ids=[i for i,flag in listing if kinds.get(i)=='pet' and i not in frozen]
   ordinary.update(ids)
   edges.extend(itertools.combinations(ids,2))
 return sorted(set(edges)),ordinary
if __name__=='__main__':
 stat=Counter();bad=[]
 for r in map(json.loads,open(P.parent/'seats.jsonl')):
  for k,after in r['states'].items():
   n=int(k)
   if n<2:continue
   edges,ids=clues(r,n);known=r['known']
   for i,j in edges:
    if str(i) not in known or str(j) not in known:continue
    x=known[str(i)][1];y=known[str(j)][1];a=TIER.get(x);b=TIER.get(y)
    if a is None or b is None:continue
    stat[r['split'],'known pairs']+=1
    if a<b or (a==b and i>j):
     stat[r['split'],'violations']+=1
     if len(bad)<5:bad.append((r['match'],n,i,x,j,y))
 print(stat,bad)
