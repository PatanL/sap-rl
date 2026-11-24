import json,collections as C
from pathlib import Path
def summarize():
 rows=[json.loads(l) for l in open('handoff/chain.jsonl')];assert len(rows)==6673
 archive={r['battleId']:r for r in json.load(open('../replay-oracle-20261122/v31i-result-340.json'))['rows']}
 assert all(r['snap']['state']==archive[r['battle']]['statePossible'] and r['snap']['outcome']==archive[r['battle']]['outcomePossible'] for r in rows)
 by=C.defaultdict(list)
 for r in rows:by[r['battle']].append(r)
 unique=[min(b,key=lambda r:r['side']) for b in by.values()]
 def stat(rs):
  bad=[r for r in rs if r['snap']['state'] and not r['chain']['state']]
  return dict(n=len(rs),snapshot_fail=sum(not r['snap']['state'] for r in rs),chain_fail=sum(not r['chain']['state'] for r in rs),
    snap_outcome_fail=sum(not r['snap']['outcome'] for r in rs),chain_outcome_fail=sum(not r['chain']['outcome'] for r in rs),
    chained_only_fail=len(bad),snap_only_fail=sum(r['chain']['state'] and not r['snap']['state'] for r in rs),
    rescues={k:sum(r['extra'][k]['state'] for r in bad) for k in ('count','temp','parrot','slots','count+temp+parrot+slots')},
    mismatch_turns=sum(any(a!=b for nm,a,b in r['counters']) for r in rs))
 counts=C.Counter((nm,a,b) for r in rows for nm,a,b in r['counters'] if a!=b)
 out=dict(seats=stat(rows),unique=stat(unique),dev=stat([r for r in unique if r['split']=='dev']),heldout=stat([r for r in unique if r['split']=='heldout']),
          by_side=C.Counter(x['side'] for x in unique),both=len(rows)-len(unique),counter_pairs=sum(len(r['counters']) for r in rows),
          either_nonzero=sum(a!=0 or b!=0 for r in rows for _,a,b in r['counters']),counter_mismatches=[[list(k),v] for k,v in counts.items()])
 Path('handoff/chain-summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':summarize()
