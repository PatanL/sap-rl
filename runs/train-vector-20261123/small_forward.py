"""Sampled BF16 full actor, real tiled g observations, same interface for all networks."""
import json,sys,time,statistics
from pathlib import Path
sys.path.append('/sap/runs/fast-collect-20261122')
from common import model_for,ParallelSelfPlayEnv
import numpy as np,torch
from sapai_gym1.rl.action_schema import FlatActionCodec as C
def main():
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
    rng=np.random.default_rng(3);pool=[]
    for i in range(48):
        e=ParallelSelfPlayEnv();obs=e.reset(seed=5000+i,paired_tape=True)
        for _ in range(int(rng.integers(20,310))):
            acts=[C.decode(int(rng.choice(tuple(e.core._avail_actions(s))))) if yes else None for s,yes in enumerate(e.active_players())]
            st=e.step(*acts);obs=st.observations
            if st.terminated:break
        pool.extend(obs)
    out={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for name in ('small','mid','medium','parent'):
            model=model_for(name).cuda().eval();out[name]=dict(parameters=sum(p.numel() for p in model.parameters()),random=name!='parent',rows=[])
            for batch in (1024,2048,4096,8192,16384):
                ix=np.arange(batch)%len(pool);ob={k:torch.from_numpy(np.stack([x[k] for x in pool])[ix]).cuda() for k in pool[0]};memory=model.initial_memory(batch,'cuda')
                try:
                    ts=[]
                    for j in range(8):
                        torch.cuda.synchronize();t=time.perf_counter();model.act(ob,memory=memory);torch.cuda.synchronize();ts.append(time.perf_counter()-t)
                    out[name]['rows'].append(dict(batch=batch,rate=batch/statistics.median(ts[2:]),seconds=ts))
                except torch.cuda.OutOfMemoryError:out[name]['rows'].append(dict(batch=batch,oom=True))
                del ob,memory;torch.cuda.empty_cache()
            del model;torch.cuda.empty_cache();Path('results/forward.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
