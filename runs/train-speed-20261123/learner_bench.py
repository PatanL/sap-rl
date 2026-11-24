import argparse,time,json,sys,gc,statistics,os
from pathlib import Path
import torch
from fixture import get_fixture
from equivalence import new,settings,verify_engine
from learner import update
from learner_fast import update_fast
def main():
    p=argparse.ArgumentParser();p.add_argument('--compact',action='store_true');p.add_argument('--rows',default=8192,type=int);p.add_argument('--sizes',default='ref,128,256,512,768,1024');p.add_argument('--out',default='results/learner-bench.json')
    a=p.parse_args();settings(True);info=verify_engine();seq=get_fixture(a.rows);out={}
    for sz in a.sizes.split(','):
        model,opt,sem=new('cuda',info)
        if sz!='ref' and os.environ.get('SPEED_FUSED_LOSS')=='1':
            from learner_ops import enable_learner
            enable_learner(model)
        for group in opt.param_groups:group['lr']=0.
        samples=[]
        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(4):
                torch.cuda.synchronize();t=time.perf_counter()
                if sz=='ref':m=update(model,opt,seq,sem,'cuda',128)
                else:m=update_fast(model,opt,seq,sem,'cuda',int(sz),compact=a.compact)
                torch.cuda.synchronize();samples.append(time.perf_counter()-t)
            out[sz]=dict(seconds=samples,decisions=m['decisions'],microbatches=m['microbatches'],rate=m['decisions']/statistics.median(samples[1:]),peak=torch.cuda.max_memory_reserved())
        except torch.cuda.OutOfMemoryError as err:
            out[sz]=dict(oom=True)
        del model,opt;gc.collect();torch.cuda.empty_cache();Path(a.out).write_text(json.dumps(out,indent=2));print(sz,out[sz],flush=True)
if __name__=='__main__':main()
