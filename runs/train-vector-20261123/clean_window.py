"""Compare the final proof update clock with the overlapping one-core CPU import check removed."""
from datetime import datetime
import json,subprocess
from pathlib import Path
def born(p):
    return datetime.fromisoformat(subprocess.check_output(['stat','-c','%w',str(p)],text=True).strip()).timestamp()
r=Path('results');u=r/'final-proof/updates.jsonl'
ev=list(map(json.loads,u.open()));origin=born(u)-ev[0]['elapsed']
start=born(r/'transfer.log')-origin-1;end=(r/'transfer.json').stat().st_mtime-origin+1
used=[];drop=[]
for x,y in zip(ev,ev[1:]):
    dt=y['elapsed']-x['elapsed']
    (drop if x['elapsed']<end and y['elapsed']>start else used).append((dt,y['decisions']))
print(json.dumps(dict(startup_excluded=1,probe_intersecting_updates=len(drop),probe_intersecting_seconds=sum(v for v,n in drop),retained_updates=len(used),clean_learned_s=sum(n for v,n in used)/sum(v for v,n in used)),indent=2))
