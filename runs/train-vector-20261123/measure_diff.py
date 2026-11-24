"""Pronounce equivalence only where there is an actual receipt for it."""
import json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def receipt(dir):
    r=read(dir/'result.json');ev=list(map(json.loads,(dir/'updates.jsonl').open()));base=ev[3];last=ev[-1]
    n=sum(x['decisions'] for x in ev[4:]);dropped=last['stale']-base['stale'];sec=last['elapsed']-base['elapsed']
    return dict(overall_learned_s=r['learned_s'],steady_learned_s=n/sec,stale=r['stale_share'],
                steady_stale=dropped/(dropped+n),delivered_s=r['collected_s'],seat_s=r['seat_s'],games_s=r['matches']/r['elapsed'],gpu=r['gpu_util_mean'])
def profile(dir):
    r=read(dir/'result.json');t=r['elapsed'];v=r['main_seconds']
    spans=dict(engine=['environment','battle','shop_api'],encoding=['encoding'],inference=['inference_service'],
               learner_wait=['await_learner'],ipc_wait=['ipc_wait','actor_wait'],control=['other','drain','publish'])
    out={k:100*sum(v.get(x,0) for x in vs)/t for k,vs in spans.items()}
    assert abs(sum(out.values())-100)<.05
    out.update(learner_active=100*r['learner_seconds']/t,overlap=100*r['overlap_with_learner_seconds'].get('inference_service',0)/t,gpu=r['gpu_util_mean'])
    worker=r['worker_seconds'];out['worker_wait']=100*worker['actor_wait']/sum(worker.values())
    return out
if __name__=='__main__':
    old=P.parent/'train-speed-20261123/results';new=P/'results'
    r={k:{'short':receipt(d/short),'profile':profile(d/prof),'proof':receipt(d/'proof')} for k,d,short,prof in [('c',old,'larger-cap','profile-final'),('g',new,'matched','profile')]}
    (new/'engine-comparison.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
