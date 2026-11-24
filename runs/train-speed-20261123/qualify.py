"""Rank physical layouts by isolated speed, then accept the first passing the fixed accuracy bounds."""
import os,sys,gc,json
from pathlib import Path
from equivalence import settings,new,verify_engine,get_fixture,execute,rel,TOLS,torch
def main():
    settings(True);seq=get_fixture(8192);info=verify_engine()
    choices=[]
    for file,layout,compact in [('learner-bench','aligned',False),('compact-sweep','sorted',True),('aligned-compact-sweep','aligned',True)]:
        for size,stats in json.load(open('results/'+file+'.json')).items():
            if size!='ref' and not stats.get('oom') and stats['peak']<7*2**30:choices.append((stats['rate'],layout,compact,int(size)))
    choices.sort(reverse=True);choices.append((0.,'short',False,512))
    original={}
    for prec in ['fp32','bf16']:original[prec]=execute(False,'cuda',info,seq,prec,128)
    trials=[]
    for rate,layout,compact,cap in choices:
        os.environ['SPEED_GEOMETRY']=layout;record=dict(layout=layout,compact=compact,cap=cap,rate=rate,comparisons={})
        for prec in ['bf16','fp32']:
            lhs,g,d=original[prec]
            try:rhs,g2,d2=execute(True,'cuda',info,seq,prec,cap,compact)
            except torch.cuda.OutOfMemoryError:
                record['comparisons'][prec]=dict(passed=False,oom=True);gc.collect();torch.cuda.empty_cache();break
            gd=rel(g,g2);pd=rel(d,d2);del g2,d2
            errors=dict(loss_abs=abs(rhs['loss']-lhs['loss']),gradient_relative_l2=gd['relative_l2'],delta_relative_l2=pd['relative_l2'],weight_max_abs=pd['max_abs'])
            ok=all(errors[k]<=tol for k,tol in TOLS[prec].items())
            record['comparisons'][prec]=dict(passed=ok,difference=errors,gradient=gd,parameter_delta=pd,reference=lhs,fast=rhs)
            if not ok:break
        trials.append(record);Path('results/qualification.json').write_text(json.dumps(trials,indent=2))
        print(layout,compact,cap,{p:r['passed'] for p,r in record['comparisons'].items()},flush=True)
        if len(record['comparisons'])==2 and all(r['passed'] for r in record['comparisons'].values()):
            Path('results/qualified.json').write_text(json.dumps(record,indent=2));return
    raise RuntimeError('No configuration met the predeclared tolerances')
if __name__=='__main__':main()
