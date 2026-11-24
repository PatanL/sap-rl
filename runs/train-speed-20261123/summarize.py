"""Normalize old/new receipts and report warm- and steady-window throughput."""
import json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent/'results'
def summary(name):
    d=P/name;new=(d/'result.json').exists();r=json.load(open(d/('result.json' if new else 'RESULT.json')))
    ev=list(map(json.loads,open(d/'updates.jsonl')))
    key='elapsed' if new else 'elapsed_s';n=len(ev)
    total=r['learned'] if new else r['trained_rows'];t=r['elapsed'] if new else r['elapsed_s']
    eli=r['delivered'];stale=r['stale'] if new else r['discarded_stale'];seat=r['seat'] if new else r['seat_decisions']
    warm=min(4,max(1,n//4));base=ev[warm-1];last=ev[-1]
    kept=sum(e['decisions'] for e in ev[warm:]);duration=last[key]-base[key];wstale=last['stale']-base['stale']
    return dict(name=name,updates=n,learned=total,sec=t,learned_s=total/t,accepted_s=eli/t,seat_s=seat/t,stale_fraction=stale/(stale+total),
                warm_updates=warm,steady_s=kept/duration,steady_stale=wstale/(kept+wstale),matches=r.get('matches',r.get('complete_matches_collected')),
                peak=r.get('peak_bytes',r.get('peak_gpu_reserved')),gpu=r.get('gpu_util_mean'))
def ledger(path):
    e=list(map(json.loads,path.open()));n=sum(v['decisions'] for v in e)
    return dict(updates=len(e),learned_rows=n,learned_s=n/e[-1]['elapsed_s'],stale_fraction=e[-1]['stale']/(e[-1]['stale']+n))
if __name__=='__main__':
    out=[]
    for p in sorted(P.iterdir()):
        if (p/'result.json').exists() or (p/'RESULT.json').exists():out.append(summary(p.name))
    (P/'throughput.json').write_text(json.dumps(out,indent=2))
    for r in out:print(r['name'],r['updates'],round(r['learned_s']),round(r['steady_s']),round(100*r['stale_fraction'],1))
