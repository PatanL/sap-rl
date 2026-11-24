"""Independent check on the BF16 behavior trajectories transported by the live-style GPU collector."""
import json,os,gc
from pathlib import Path
import torch
from equivalence import execute,settings,verify_engine,get_fixture,rel,TOLS
P=Path('results')
def main():
    settings(True);info=verify_engine();q=json.load(open(P/'qualified.json'))
    opts=[(q['layout'],q['compact'],q['cap']),('short',False,512),('legacy',False,128)]
    trials=[];ref={}
    for geom,compact,cap in opts:
        checks={}
        for batch in ('results/g48/batch.pt','results/batch.pt'):
            os.environ['SPEED_BATCH']=batch;seq=get_fixture(8192)
            for prec in ('bf16','fp32'):
                if (batch,prec) not in ref:ref[batch,prec]=execute(False,'cuda',info,seq,prec,128)
                l,g,d=ref[batch,prec];os.environ['SPEED_GEOMETRY']=geom
                m,gg,dd=execute(True,'cuda',info,seq,prec,cap,compact)
                gd=rel(g,gg);pd=rel(d,dd);del gg,dd
                errs=dict(loss_abs=abs(m['loss']-l['loss']),gradient_relative_l2=gd['relative_l2'],delta_relative_l2=pd['relative_l2'],weight_max_abs=pd['max_abs'])
                good=all(errs[k]<=v for k,v in TOLS[prec].items())
                checks[batch+' '+prec]=dict(passed=good,errors=errs,decisions=m['decisions'],cosine=gd['cosine'],gradient_max_abs=gd['max_abs'],loss_reference=l['loss'],loss_fast=m['loss'])
                if not good:break
            if not good:break
        trials.append(dict(layout=geom,compact=compact,cap=cap,checks=checks));(P/'crossqual.json').write_text(json.dumps(trials,indent=2))
        if all(r['passed'] for r in checks.values()) and len(checks)==4:
            (P/'verified-config.json').write_text(json.dumps(trials[-1],indent=2));return
    raise RuntimeError('Failed second-fixture numerical check')
if __name__=='__main__':main()
