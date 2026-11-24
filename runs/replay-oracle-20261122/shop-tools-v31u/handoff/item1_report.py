import json,collections as C
from pathlib import Path
p=Path('handoff')
def full(root):
 rows={}
 for cap in (1,64,1024):
  for r in map(json.loads,open(root/f'full-{cap}.jsonl')):rows[tuple(r['key'])]=r
 return rows
new=full(p);old=full(Path('.'))
states={}
for name,fn in [('requested fields',lambda r:'core joint' in r['ever']),('plus remaining offer contents',lambda r:'inventory content joint' in r['ever']),('plus pet order',lambda r:'pet order joint' in r['ever']),('plus food order',lambda r:'food order joint' in r['ever']),('plus all content and both orders',lambda r:r['reproduced'])]:
 counts=C.Counter((fn(old[k]),fn(r)) for k,r in new.items() if not r.get('skip'));states[name]=[sum(v for (x,y),v in counts.items() if x),sum(v for (x,y),v in counts.items() if y),counts[False,True],counts[True,False]]
(p/'full-comparison.json').write_text(json.dumps(states,indent=2))
table='\n'.join('| '+k+' | '+' | '.join(str(n) for n in v)+' |' for k,v in states.items())
text=f"""
# Shop verification, battle handoff and match settlement

The frozen Python source for this audit is [pinned-handoff](pinned-handoff), digest §{json.load(open('engine-pin-handoff.json'))['sha256']}§. Checks used the archived 340 matches on at most six CPU cores. Findings below refer to this copy. [Per-file hashes](engine-pin-handoff.json)

## 1. Shop changes

**Food order is still wrong on the same 956 pairs.** The engine matched 1,511 of 2,467 co-generated ordinary food pairs. Both the original shop seeds and the new v31t witness seeds give those same counts. The sorter reads §slot.obj.fd["tier"]§. A §Food§ stores its **ability dictionary** in §fd§, and the tier lives in §data["foods"][slot.obj.name]["tier"]§. The fallback returns zero, so stable sorting preserves draw order. Across 6,730 common completed terminal witnesses, **zero** pet-order or food-order comparison flags changed. [Same-pair rerun](handoff/order.json) · [Food construction](pinned-handoff/sapai/foods.py#L49) · [Tier lookup](pinned-handoff/sapai/shop.py#L329)

Best joint reproduction at a common 1,024 shop seeds, with 6,735 attempted turns in each version:

| Endpoint | Original audit | Current pin | Newly matched | Newly missed |
|---|---:|---:|---:|---:|
{table}

These order rows also require the requested board, frozen stock and wallet. An increase there can come from a corrected board while the slot order stays the same. [Paired comparison](handoff/full-comparison.json)

**Spider sale is fixed.** On the real turn where a level-two Spider was pilled, all ten possible summons sold for two gold. With the one-gold Dolphin sale, the player could make the recorded three-gold purchase and complete the turn in all 256 seeds. The changed line is in §SummonRandomPet§. [Sale receipt](handoff/spider.json) · [Implementation](pinned-handoff/sapai/effects.py#L1294)

| Other summon probes | Native level | Sale through Player.sell |
|---|---:|---:|
| Level-two Deer pilled, 10/6 Bus with Chili | 1 | 1 |
| Level-three Deer pilled, 15/9 Bus | 1 | 1 |
| Level-two Sheep pilled, two 4/4 Rams | 1 each | 1 each |
| Whale releases a Fish through SummonStoredPets, Whale level two / three | 2 / 3 | **1 / 1** |

Deer scales the Bus’s stats and its descriptor omits §withLevel§. The Bus sale agrees with its native level; the corpus did not provide a confirmed pilled level-two Deer for real-game comparison. The Whale probe invoked native swallow/release and then sold the released pet. Stored releases ordinarily happen in battle, so this is a forced reachability test of the object’s sale value. §SummonStoredPets§ delegates to §SummonPet§, which sets §level§ from §withLevel§ without updating §sell_value§. [All summon probes](handoff/summons.json) · [Deer descriptor](pinned-handoff/sapai/turtle_pack.py#L186) · [Stored release](pinned-handoff/sapai/effects.py#L1420)
"""
with open('REPORT-2.md','a') as f:f.write(text.replace('§',chr(96)).strip()+'\n')
print(states)
