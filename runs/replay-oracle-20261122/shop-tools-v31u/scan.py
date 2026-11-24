import json,collections as C
from pathlib import Path
src=Path('../replay-oracle-20261122/raw')
count=C.Counter();examples={}; bkeys=C.Counter(); reqkeys=C.Counter();orders=C.Counter(); other=C.Counter(); unique=set()
for path in sorted(src.glob('*.json')):
 for per in json.loads(path.read_text())['perspectives']['perspectives']:
  r=per['raw']
  pkey=r['ParticipationId']
  if pkey in unique:count['dupe']+=1;continue
  unique.add(pkey)
  for i,a in enumerate(r['Actions']):
   typ=a['Type'];req=json.loads(a.get('Request') or '{}');resp=json.loads(a.get('Response') or '{}');d=req.get('Data',{});count[typ]+=1
   reqkeys[(typ,','.join(sorted(set(req)-{'Version','Data'})))]+=1
   for key in d:
    if key not in ('BuildId','BoardHash'):other[key]+=1
   if typ not in examples:examples[typ]={'keys':list(a),'req':req,'resp':resp}
   if a.get('Build') and typ==0:
    build=json.loads(a['Build']);bor=build['Bor'];bkeys.update(bor.keys())
    for k in bor:
     if k not in ('Mins','MiSh','SpSh') and k not in examples and bor[k]:examples[k]=str(bor[k])[:240]
    for k in ('Mins','MiSh','SpSh'):
     for item in (bor[k].get('Items',[]) if k=='Mins' else bor[k]) or []:
      if not item:continue
      for j in item:
       if j not in ('Id','Cosm','Abil','Enu','At','Hp','Own','Loc','Lvl','Pri','Poi','Exp','Perk','Fro','WFro','Rwrd'):
        if (k,j) not in examples: examples[k,j]=str(item)[:800]
print('count',count, 'seats',len(unique),'reqkeys',reqkeys,'data fields',other)
for k,v in examples.items():print(k,v)
