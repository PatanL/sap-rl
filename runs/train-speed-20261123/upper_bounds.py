"""Try larger coalesced head/loss batches at stable encoder shapes, on both saved fixtures."""
import json,os,gc,statistics,time
from pathlib import Path
from equivalence import *
from learner_fast import update_fast
from learner_ops import enable_learner
def main():
    settings(True);info=verify_engine();refs={};records=[]
    for cap in (1536,1024,768):
        ok=True;rec=dict(cap=cap,checks={})
        for fi,prec in [('results/g48/batch.pt','bf16'),('results/batch.pt','bf16'),('results/g48/batch.pt','fp32'),('results/batch.pt','fp32')]:
            os.environ['SPEED_BATCH']=fi;seq=get_fixture(8192)
            if (fi,prec) not in refs:refs[fi,prec]=execute(False,'cuda',info,seq,prec,128)
            l,g,d=refs[fi,prec]
            try:m,g2,d2=execute(True,'cuda',info,seq,prec,cap)
            except torch.cuda.OutOfMemoryError:rec['checks'][fi+' '+prec]={'oom':True};ok=False;break
            gd=rel(g,g2);pd=rel(d,d2);del g2,d2
            errs=dict(loss_abs=abs(m['loss']-l['loss']),gradient_relative_l2=gd['relative_l2'],delta_relative_l2=pd['relative_l2'],weight_max_abs=pd['max_abs'])
            rec['checks'][fi+' '+prec]=errs;ok=all(errs[k]<=v for k,v in TOLS[prec].items())
            if not ok:break
        rec['passed']=ok
        if ok:
            os.environ['SPEED_BATCH']='results/g48/batch.pt';seq=get_fixture(8192)
            m,opt,sem=new('cuda',info);enable_learner(m)
            for g in opt.param_groups:g['lr']=0.
            ts=[]
            for _ in range(4):
                torch.cuda.synchronize();t=time.perf_counter();x=update_fast(m,opt,seq,sem,'cuda',cap);torch.cuda.synchronize();ts.append(time.perf_counter()-t)
            rec['rate']=x['decisions']/statistics.median(ts[1:]);del m,opt
        records.append(rec);Path('results/upper-bounds.json').write_text(json.dumps(records,indent=2));print(cap,ok,rec.get('rate'),flush=True)
        if ok:break
if __name__=='__main__':main()
