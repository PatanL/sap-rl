"""Additional fitted-seller counts: never use fitted names to assign failures to a species."""
import json,csv,gzip,collections as C
from pathlib import Path
P=Path(__file__).resolve().parent
sells={(r['match'],r['seat'],int(n)):[a['id'] for a in acts if a['t']==9] for r in map(json.loads,open(P.parent/'seats.jsonl')) for n,acts in r['actions'].items()}
prior={tuple(r['key']):r for r in map(json.loads,open(P/'input-certificates.jsonl'))}
stats=C.defaultdict(C.Counter)
for r in map(json.loads,gzip.open(P/'assignments.jsonl.gz','rt')):
 k=tuple(r['key']); sold=sells[k]
 if len(sold)<2 or r['unmapped_food']:continue
 ids=r['buy']+r['endogenous'];idx=[j for j,i in enumerate(ids) if i in sold]
 if not idx:continue
 mn=None;mx=set()
 for label,_,_ in r['witnesses']:
    names={label[j] for j in idx};mx.update(names);mn=names if mn is None else mn&names
 q=prior[k]
 possible=set(x for i,v in zip(q['buy_ids'],q['buy_marginals']) if i in sold for x in v)
 for sp in ('all',r['split']):
    for nm in possible:stats[sp,nm]['raw_admissible_missing_bought_seller']+=1
    for nm in mx:stats[sp,nm]['at_least_one_witness_bought_or_summoned_seller']+=1
    for nm in mn or ():stats[sp,nm]['every_witness_has_seller_somewhere']+=1
with open(P/'fitted-sellers.csv','w') as f:
 fields=['split','species','raw_admissible_missing_bought_seller','at_least_one_witness_bought_or_summoned_seller','every_witness_has_seller_somewhere']
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for (s,n),c in sorted(stats.items()):w.writerow(dict(split=s,species=n,**{k:c[k] for k in fields[2:]}))
