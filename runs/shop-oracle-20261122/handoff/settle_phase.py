import json,collections as C
from pathlib import Path
P=Path(__file__).resolve().parent
res={r['match']:r for r in map(json.loads,(P/'settlement.jsonl').open())};out=C.Counter();examples=[]
for f in sorted(Path('../replay-oracle-20261122/raw').glob('*.json'))+sorted(Path('../expert-replay-ingest-20260926-v1/corpus/matches').glob('*/BUNDLE.json')):
 seats=[s['raw'] for s in json.load(open(f))['perspectives']['perspectives']];r=res[seats[0]['MatchId']];prev=[6,6];by={};pre={};post={}
 for j,s in enumerate(seats):
  for a in s['Actions']:
   if a['Type']!=0:continue
   n=a['Turn'];b=json.loads(a['Battle']);build=json.loads(a['Build'])['Bor'];ub=b['UserBoard']
   pre[n,j]=ub['LiMa']-ub.get('LoPs',0);post[n,j]=build['LiMa']-build.get('LoPs',0)
 for v in r['records']:
  n=v['turn']
  for j in (0,1):
   if (n,j) not in pre:continue
   out['pre observations']+=1
   if pre[n,j]!=prev[j]:out['pre miss '+str(n)]+=1
   if post[n,j]!=v['lives'][j]:out['post miss '+str(n)]+=1
   if (n+1,j) in pre and pre[n+1,j]!=post[n,j]:
    out['between-round change '+str(n)]+=1
  if (n,0) in pre and (n,1) in pre:
   delta=[post[n,j]-pre[n,j] for j in (0,1)]
   rw=.1*(delta[0]-delta[1]);bonus=(2 if r['native_winner']==0 else -2 if r['native_winner']==1 else 0) if v['terminated'] else 0
   out['paired real life-swing rewards']+=1
   if abs(rw+bonus-v['reward'][0])>1e-5:
    out['immediate reward mismatch '+str(n)]+=1
    if len(examples)<4:examples.append([r['match'],n,prev,[pre[n,j] for j in (0,1)],[post[n,j] for j in (0,1)],v['lives'],rw+bonus,v['reward'][0]])
  prev=v['lives']
json.dump(dict(counts=out,examples=examples),open(P/'settlement-phase.json','w'),indent=2)
print(dict(out),examples[:2])
