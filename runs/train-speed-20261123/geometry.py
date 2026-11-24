import os,json,gc,time
from pathlib import Path
from equivalence import *
def main():
    settings(True);info=verify_engine();seq=get_fixture(8192);prec='bf16'
    base,g1,d1=execute(False,'cuda',info,seq,prec,128)
    results=[]
    for geom,cap,compact in [('legacy',128,False),('aligned',512,False),('short',512,False),('medium',512,False),('long',512,False),('sorted',512,True),('short',512,True),('aligned',512,True)]:
        os.environ['SPEED_GEOMETRY']=geom
        t=time.perf_counter();rhs,g2,d2=execute(True,'cuda',info,seq,prec,cap,compact)
        grad=rel(g1,g2);delta=rel(d1,d2)
        diff=dict(loss_abs=abs(rhs['loss']-base['loss']),gradient_relative_l2=grad['relative_l2'],delta_relative_l2=delta['relative_l2'],weight_max_abs=delta['max_abs'])
        passed=all(v<=TOLS[prec][k] for k,v in diff.items())
        results.append(dict(layout=geom,cap=cap,compact=compact,passed=passed,difference=diff,microbatches=rhs['microbatches']))
        Path('results/geometries.json').write_text(json.dumps(results,indent=2))
        print(geom,compact,passed,diff,flush=True);del g2,d2;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
