"""Session 3: locate the original Battle bytes of every currently failing fight (rust-v31g, 5,000 seeds)
and write byte-exact fixtures s3f-<id8>.json plus the capture manifest `s3real`."""
import json,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent;D=HERE/'session2';R=HERE.parent/'replay-oracle-20261122'
def h(s):return hashlib.sha256(s.encode()).hexdigest()
bad={}
for n in ['340','fresh','fresh2','fresh3','fresh4']:
    for r in json.load(open(R/f'rust-v31g-5000-{n}.json'))['rows']:
        if r.get('statePossible') is False or r.get('outcomePossible') is False:bad[r['battleId']]=dict(set=n,**{k:r[k] for k in ('deterministic','outcomePossible','statePossible','turn','matchId','realOutcome','distinctStates','pReal')})
src=[]
for sub in ('raw','raw-fresh','raw-fresh2','raw-fresh3','raw-fresh4','raw-players'):src+=sorted((R/sub).glob('*.json'))
src+=sorted((R/'expert-replay-ingest-20260926-v1/corpus/matches').glob('*/BUNDLE.json'))
found={}
for path in src:
    try:doc=json.loads(path.read_text())
    except Exception:continue
    for pers in doc['perspectives']['perspectives']:
        for a in pers['raw']['Actions']:
            if not a.get('Battle'):continue
            s=a['Battle']
            # cheap pre-filter on the id text before parsing
            i=s.find('"Id":"')
            bid=s[i+6:i+42]
            if bid in bad and bid not in found:
                b=json.loads(s);assert b['Id']==bid
                found[bid]=(s,str(path.relative_to(R)))
missing=[b for b in bad if b not in found];print('failing',len(bad),'found',len(found),'missing',missing)
prev={p.name[4:] for p in (D/'captures').glob('dev-*')}
jobs=[]
for bid,(s,origin) in found.items():
    name='s3f-'+bid[:8]+'.json';dst=D/'fixtures'/name
    if dst.exists():assert dst.read_text()==s
    else:dst.write_bytes(s.encode())
    b=json.loads(s);m=bad[bid]
    jobs.append(dict(file=name,id=bid,seed=b['Seed'],sha256=h(s),origin=origin,previously_traced=bid[:8] in prev,**m))
jobs.sort(key=lambda j:(j['previously_traced'],not j['deterministic'],j['outcomePossible'],j['id']))
(D/'s3real-manifest.json').write_text(json.dumps(jobs,indent=1))
print(len(jobs),'jobs;',sum(j['deterministic'] for j in jobs),'deterministic;',sum(j['previously_traced'] for j in jobs),'re-captures')
