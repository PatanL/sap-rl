import json,os,gc,statistics,time
from pathlib import Path
from equivalence import *
from learner_fast import update_fast
def main():
    settings(True);info=verify_engine();out=[];refs={}
    for layout in ('block-boot','block-head','block-history','block-entity'):
        os.environ['SPEED_GEOMETRY']=layout;result=dict(layout=layout,cap=768,checks={})
        for fi in ('results/g48/batch.pt','results/batch.pt'):
            os.environ['SPEED_BATCH']=fi;seq=get_fixture(8192)
            if fi not in refs:refs[fi]=execute(False,'cuda',info,seq,'bf16',128)
            l,g,d=refs[fi]
            m,y,z=execute(True,'cuda',info,seq,'bf16',768)
            e=dict(loss_abs=abs(m['loss']-l['loss']),gradient_relative_l2=rel(g,y)['relative_l2'],
                   delta_relative_l2=rel(d,z)['relative_l2'],weight_max_abs=rel(d,z)['max_abs'])
            passed=all(e[k]<=v for k,v in TOLS['bf16'].items());result['checks'][fi]=e;del y,z;gc.collect()
            if not passed:break
        if passed:
            model,opt,sem=new('cuda',info)
            for group in opt.param_groups:group['lr']=0.
            secs=[]
            for i in range(4):
                torch.cuda.synchronize();t=time.perf_counter();m=update_fast(model,opt,seq,sem,'cuda',768);torch.cuda.synchronize();secs.append(time.perf_counter()-t)
            result['rate']=m['decisions']/statistics.median(secs[1:]);del model,opt
        out.append(result);Path('results/shape-trial.json').write_text(json.dumps(out,indent=2));print(layout,passed,result.get('rate'),flush=True)
if __name__=='__main__':main()
