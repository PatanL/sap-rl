"""Expand successful assignments to actual species, enforce witnessed ordering, replay each with a concrete native witness."""
import collections as C,itertools as IT,json,math,multiprocessing as mp,gzip,sys
from pathlib import Path
from infer import *
from evidence import ordered
FILES=['pass1-low','pass1-high','concrete','pass32-low','pass8-high','pass256-small','pass1024-small','births','pass2048-zero','pass8192-zero','postorder','rare']
def results():
 d=C.defaultdict(list)
 for f in FILES:
    path=P/(f+'.jsonl')
    if not path.exists():continue
    for l in open(path):
        try:r=json.loads(l)
        except json.JSONDecodeError:continue
        d[tuple(r['key'])].append(r)
 return d
def one(arg):
    row,n,orig,passes=arg;p=prep(row,n);from order_evidence import clues
    p['clues']=clues(row,n);p['clauses']=stock_clauses(p)
    witnesses=C.defaultdict(list);index=p['b']+p['e'];prior=set();misorder=set();bad=[]
    sumcalls=0
    for rec in passes:
        sumcalls+=rec['engineCalls']
        for label,sets,si,sp in rec['witnesses']:
            for v in IT.product(*sets):
                prior.add(v);d={**p['names'],**dict(zip(index,v))}
                if not ordered(p,d):
                    misorder.add(v);continue
                if (si,sp) not in witnesses[v]:witnesses[v].append((si,sp))
    ok=[];checked=0
    for v,options in sorted(witnesses.items()):
        for si,sp in options:
            hit,why,_,ctx=trial(p,v[:len(p['b'])],si,sp)
            checked+=1
            if hit and tuple(ctx.assigned[i] for i in p['e'])==v[len(p['b']):]:
                ok.append([v,si,sp]);break
        else:bad.append(v)
    unique_by_uid={i:sorted({v[j] for v,si,sp in ok}) for j,i in enumerate(index)}
    return dict(key=orig['key'],split=row['split'],skip=orig['skip'],buy=p['b'],endogenous=p['e'],unmapped_food=passes[0]['unknownFood'],n=len(ok),
                before_order=len(prior),excluded_by_order=len(misorder),failed_individual_recheck=len(bad),bad_examples=bad[:6],
                marginals=unique_by_uid,seedCap=max(r['seedCap'] for r in passes),native_trials=sumcalls,individual_rechecks=checked,witnesses=ok)
def main():
 d=results();js=[(r,n,o,d[tuple(o['key'])]) for r,n,o in jobs()]
 with gzip.open(P/'assignments.jsonl.gz','wt',compresslevel=3) as fp,open(P/'verified.jsonl','w',buffering=1) as sm,mp.Pool(6,initializer=install) as pool:
    for v in pool.imap_unordered(one,js):
        fp.write(json.dumps(v,separators=(',',':'))+'\n')
        del v['witnesses'];sm.write(json.dumps(v,separators=(',',':'))+'\n')
if __name__=='__main__':main()
