"""Profile and independently qualify additional conservative packings. All trials use the same two saved batches."""
import time,statistics,json,gc,os
from equivalence import *
from pathlib import Path
from learner_fast import update_fast
def main():
    settings(True);info=verify_engine();out=[];originals={}
    # Isolate the short/long boundary implicated by the independent capture.
    for geom,cap,compact in [('short',128,False),('short',256,False),('short',512,False),('short',1024,False),
                              ('guard16',512,False),('aligned',256,False),('aligned',512,False),('sorted',256,True),('aligned',512,True)]:
        os.environ['SPEED_GEOMETRY']=geom;rec=dict(layout=geom,cap=cap,compact=compact,checks={})
        for fi in ['results/g48/batch.pt','results/batch.pt']:
            os.environ['SPEED_BATCH']=fi;seq=get_fixture(8192)
            if fi not in originals:originals[fi]=execute(False,'cuda',info,seq,'bf16',128)
            l,g,d=originals[fi];m,gg,dd=execute(True,'cuda',info,seq,'bf16',cap,compact)
            gd=rel(g,gg);pd=rel(d,dd);del gg,dd
            errs=dict(loss_abs=abs(m['loss']-l['loss']),gradient_relative_l2=gd['relative_l2'],delta_relative_l2=pd['relative_l2'],weight_max_abs=pd['max_abs'])
            passed=all(errs[k]<=v for k,v in TOLS['bf16'].items());rec['checks'][fi]=dict(passed=passed,errors=errs)
            if not passed:break
        if len(rec['checks'])==2 and passed:
            model,opt,sem=new('cuda',info)
            for group in opt.param_groups:group['lr']=0.
            times=[]
            for _ in range(4):
                torch.cuda.synchronize();t=time.perf_counter();m=update_fast(model,opt,seq,sem,'cuda',cap,compact=compact);torch.cuda.synchronize();times.append(time.perf_counter()-t)
            rec.update(rate=m['decisions']/statistics.median(times[1:]),seconds=times,microbatches=m['microbatches'])
            del model,opt;gc.collect();torch.cuda.empty_cache()
        out.append(rec);Path('results/supplement.json').write_text(json.dumps(out,indent=2))
        print(geom,cap,compact,passed,rec.get('rate'),flush=True)
if __name__=='__main__':main()
