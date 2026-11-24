"""Replay every prior concrete witness; re-search the prior negative turn cohort on v31c."""
import sys,os,gzip,json,multiprocessing as mp
from pathlib import Path
S=Path('/sap/runs/shop-oracle-20261122');sys.path[:0]=[str(S/'recover'),str(S)]
os.environ['SAP_ENGINE_ROOT']='/engine/sapai'
from search import run_case
from infer import install,trial,prep,jobs
from evidence import ordered
from order_evidence import clues
def case(arg):
    row,n,o,old=arg;p=prep(row,n);p['clues']=clues(row,n)
    keep=0;first=None;lost=0;calls=0
    for label,seed,sp in old['witnesses']:
        ok,why,out,ctx=trial(p,label[:len(p['b'])],seed,sp);calls+=1
        if ok and [ctx.assigned[i] for i in p['e']]==label[len(p['b']):]:
            keep+=1
            if first is None:first=[label,seed,sp]
        else:lost+=1
    found=keep>0;cap=0;previous=None
    if not found:
        pill=any(a['t']==8 and a['spell']['name']=='Sleeping Pill' for a in p['acts'])
        stages=(1,64,1024,8192) if pill and any(a['t']==9 for a in p['acts']) else (1,64)
        for cap in stages:
            result=run_case((row,n,o,cap,previous));previous=result;calls+=result['engineCalls']
            for label,groups,si,sp in result['witnesses']:
                import itertools
                for v in itertools.product(*groups):
                    if not ordered(p,{**p['names'],**dict(zip(p['b']+p['e'],v))}):continue
                    ok,_,_,ctx=trial(p,v[:len(p['b'])],si,sp);calls+=1
                    if ok and tuple(ctx.assigned[i] for i in p['e'])==v[len(p['b']):]:
                        first=[v,si,sp];found=True;break
                if found:break
            if found:break
    return dict(key=o['key'],split=row['split'],old_matches=old['n'],preserved_assignments=keep,lost_assignments=lost,passes=found,
                missing_food=old['unmapped_food'],witness=first,retry_cap=cap,calls=calls)
if __name__=='__main__':
    prior={tuple(x['key']):x for x in map(json.loads,gzip.open(S/'recover/assignments.jsonl.gz','rt'))}
    os.environ['BIRTH_IDS']='1'
    with open('shop/rechecked.jsonl','w',buffering=1) as fp,mp.Pool(6,initializer=install) as pool:
        for v in pool.imap_unordered(case,[(r,n,o,prior[tuple(o['key'])]) for r,n,o in jobs()]):fp.write(json.dumps(v)+'\n')
