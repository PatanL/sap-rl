"""Report raw Spell evidence without assigning names to the two absent enums."""
from pathlib import Path
import json,sys,hashlib
from collections import Counter
P=Path(__file__).resolve().parent; sys.path.insert(0,str(P))
from profile_cases import jobs,inspect as describe
lookup={(r['match'],r['seat'],n):(r,n) for r,n,o in jobs() if o['skip'].startswith('unmapped')}
base=P.parent.parent
out=[]
for path in sorted((base/'replay-oracle-20261122/raw').glob('*.json'))+sorted((base/'expert-replay-ingest-20260926-v1/corpus/matches').glob('*/BUNDLE.json')):
 for perspective in json.load(path.open())['perspectives']['perspectives']:
  raw=perspective['raw'];by={}
  for a in raw['Actions']:
   t=a['Turn'];k=(raw['MatchId'],raw['ParticipationId'],t)
   if k not in lookup:continue
   if a['Type']!=8:continue
   event=json.loads(a['Response'])['Event']['Event'];spell=event['Spell']
   by.setdefault(k,[]).append(dict(quoted_id=spell.get('Enu',0),uid=spell['Id']['Uni'],target=event.get('Target',{}).get('Uni'),price=spell.get('Pri',0),
    price_key_present='Pri' in spell,power=spell.get('Pow'),stock=spell.get('Sto'),buff=spell.get('Buff'),fields=sorted(spell)))
  for k,events in by.items():
   row,n=lookup[k];acts,tr,names,*_=describe(row,n);before=row['states'][str(n-1)];after=row['states'][str(n)]
   source=next(e['target'] for e in events if e['quoted_id']==23);uids=[e['target'] for e in events if e['quoted_id'] in (130,131)]
   def item(b,i):return next((v for v in b['board'] if v and v['id']==i),None)
   out.append(dict(key=k,split=row['split'],events=events,chocolate_target=source,known_target=names.get(source),before_cow=item(before,source),after_cow=item(after,source),
     before_recipient=item(before,uids[0]),after_recipient=item(after,uids[0]),other_pets=[v['name'] for v in after['board'] if v]))
(P/'food-receipts.json').write_text(json.dumps(out,indent=2))
print('turns',len(out),'enums',Counter(e['quoted_id'] for r in out for e in r['events'] if e['quoted_id'] in (130,131)))
