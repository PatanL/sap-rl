"""Independent reproduction of the selected full-batch gradient/Adam comparison at delivery."""
import json,os,gc
from pathlib import Path
from equivalence import *
def main():
    settings(True);info=verify_engine();c=json.load(open('results/final-choice.json'))
    os.environ['SPEED_GEOMETRY']=c['geometry'];os.environ['SPEED_FUSED_LOSS']=str(int(c['fused']))
    result=[];ok=True
    for fi in ['results/batch.pt','results/g48/batch.pt']:
        os.environ['SPEED_BATCH']=fi;seq=get_fixture(8192)
        for prec in ('fp32','bf16'):
            lhs,g,d=execute(False,'cuda',info,seq,prec,128)
            rhs,g2,d2=execute(True,'cuda',info,seq,prec,c['qualifier'],'--compact' in c['args'])
            y=rel(g,g2);z=rel(d,d2);err=dict(loss_abs=abs(rhs['loss']-lhs['loss']),gradient_relative_l2=y['relative_l2'],delta_relative_l2=z['relative_l2'],weight_max_abs=z['max_abs'])
            passed=all(err[k]<=v for k,v in TOLS[prec].items());ok=ok and passed
            result.append(dict(layout=c['geometry'],cap=c['qualifier'],fused=c['fused'],batch=fi,prec=prec,rows=lhs['decisions'],reference=lhs,fast=rhs,cosine=y['cosine'],gradient_max_abs=y['max_abs'],difference=err,passed=passed))
            del g,d,g2,d2;gc.collect();torch.cuda.empty_cache()
    Path('results/final-exactness.json').write_text(json.dumps(result,indent=2))
    if not ok:raise RuntimeError('Final numerical verification failed')
if __name__=='__main__':main()
