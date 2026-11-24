import json,collections as C
from pathlib import Path
P=Path(__file__).resolve().parent
rs=list(map(json.loads,open(P/'rechecked.jsonl')))
assert len(rs)==710
s={}
for div in ('all','dev','heldout'):
    rows=[r for r in rs if div=='all' or r['split']==div]
    s[div]=dict(turns=len(rows),mapped=sum(not r['missing_food'] for r in rows),hit=sum(r['passes'] for r in rows),
                fixed=sum(not r['old_matches'] and r['passes'] for r in rows),broken=sum(bool(r['old_matches']) and not r['passes'] for r in rows),
                preserved=sum(r['preserved_assignments'] for r in rows),lost=sum(r['lost_assignments'] for r in rows))
pills={}
for ds in ('pill','pill-fresh'):
    allrows=list(map(json.loads,open(P/(ds+'.jsonl'))));t={}
    for target in ('Deer','Sheep','Spider','Mushroom'):
        rows=[r for r in allrows if 'compatible'in r and (r['source']==target if target!='Mushroom' else r.get('perk')=='Mushroom')]
        t[target]=dict(usable=len(rows),compatible=sum(r['compatible'] for r in rows),fixed_new_slots=sum(bool(r['new_ids']) and len(r['new_slot_sets'])==1 for r in rows),
                       direct_end=sum(r['direct'] for r in rows),source_named=sum(r['source_named_in_record'] for r in rows))
    pills[ds]=dict(total=len(allrows),groups=t)
(P/'summary.json').write_text(json.dumps(dict(inference=s,pill=pills),indent=2))
print(s,pills)
