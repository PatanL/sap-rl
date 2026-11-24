import os,json,gc,statistics,time
from pathlib import Path
from equivalence import *
from learner_fast import update_fast
def main():
    settings(True);info=verify_engine();results=[];refs={}
    for layout,cap in [('block-head-history',768),('block-history',256),('block-history',384),('block-both',512)]:
        os.environ['SPEED_GEOMETRY']=layout;rec=dict(layout=layout,cap=cap,checks={});ok=True
        for fi,prec in [('results/g48/batch.pt','bf16'),('results/batch.pt','bf16'),('results/g48/batch.pt','fp32'),('results/batch.pt','fp32')]:
            os.environ['SPEED_BATCH']=fi;seq=get_fixture(8192)
            if (fi,prec) not in refs:refs[fi,prec]=execute(False,'cuda',info,seq,prec,128)
            lhs,g,d=refs[fi,prec];rhs,g2,d2=execute(True,'cuda',info,seq,prec,cap)
            gg=rel(g,g2);dd=rel(d,d2);del g2,d2;gc.collect()
            errs=dict(loss_abs=abs(rhs['loss']-lhs['loss']),gradient_relative_l2=gg['relative_l2'],delta_relative_l2=dd['relative_l2'],weight_max_abs=dd['max_abs'])
            rec['checks'][fi+' '+prec]=errs;ok=all(errs[k]<=v for k,v in TOLS[prec].items())
            if not ok:break
        if ok:
            from learner_ops import enable_learner
            m,opt,sem=new('cuda',info);enable_learner(m)
            for group in opt.param_groups:group['lr']=0.
            ts=[]
            for _ in range(4):
                torch.cuda.synchronize();t=time.perf_counter();v=update_fast(m,opt,seq,sem,'cuda',cap);torch.cuda.synchronize();ts.append(time.perf_counter()-t)
            rec['rate']=v['decisions']/statistics.median(ts[1:]);del m,opt
        results.append(rec);Path('results/tiles.json').write_text(json.dumps(results,indent=2));print(layout,cap,ok,rec.get('rate'),flush=True)
if __name__=='__main__':main()
