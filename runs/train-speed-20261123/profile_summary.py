"""Three wall clocks: coordinator critical accounting, parallel CPU worker capacity, and overlap."""
import json,statistics
from pathlib import Path
def main():
    p=Path('results')
    r=json.load(open(p/('profile-final' if (p/'profile-final/result.json').exists() else 'profile')/'result.json'));t=r['elapsed'];main=r['main_seconds'];over=r['overlap_with_learner_seconds'];worker=r['worker_seconds']
    tags={'engine':{'shop_api','environment','battle'}, 'public observation encoding':{'encoding'}, 'inference service':{'inference_service'},
          'waiting on learner publication':{'await_learner'},'IPC / other worker waiting':{'ipc_wait','actor_wait'},'copy/publish/control':set()}
    dims={}
    for name,keys in tags.items():
        if name=='copy/publish/control':keys=set(main)-set.union(*[vs for vs in tags.values()])
        dims[name]=100*sum(main.get(k,0) for k in keys)/t
    assert abs(sum(dims.values())-100)<.05
    summed=sum(worker.values());cpu={k:100*v/summed for k,v in worker.items()}
    ab=json.load(open(p/'boundary-bench.json'));seconds={k:statistics.median(v['s'] for v in xs) for k,xs in ab.items()}
    spans={'rust_battle_real_saving':seconds['python']-seconds['rust'],'ideal_battle_saving':seconds['python']-seconds['cached'],
           'encoding_with_cached_battles':seconds['cached']-seconds['cached-noobs'],'remaining_shop_gym_vs_oracle':seconds['cached-noobs']-seconds['replay']}
    out=dict(main_exclusive_percent=dims,main_detail_percent={k:v/t*100 for k,v in main.items()},worker_lane_percent=cpu,
             gpu_utilization=r['gpu_util_mean'],learner_active_pct=r['learner_seconds']/t*100,
             simultaneous_learner_and_inference_pct=100*over.get('inference_service',0)/t,fully_overlapped_waves=r['fully_overlapped_inference_waves'],
             fully_overlapped_seats=r['seats_fully_served_during_updates'],ablation_seconds=seconds,measured_differences=spans)
    (p/'profile-summary.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
