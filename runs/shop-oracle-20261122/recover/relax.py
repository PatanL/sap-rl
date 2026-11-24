"""Negative control: remove the Pig-wallet and named-stock shortcuts for nonmatching mapped turns."""
import itertools,json,multiprocessing as mp,sys,math
from search import *
from order_evidence import clues
from evidence import ordered
def one(arg):
 row,n,o=arg;p=prep(row,n);p['clues']=clues(row,n)
 if o['skip'].startswith('unmapped'):return dict(key=o['key'],food=True)
 p['wallet']=None;p['clauses']=[];calls=0;combos=0
 for g in tuples(p):
  v=tuple(x[0] for x in g);nm={**p['names'],**dict(zip(p['b'],v))}
  if not ordered(p,nm):continue
  combos+=1
  cap=32 if math.prod(map(len,p['groups']))<=150 else 2
  for si in range(cap):
   stack=[dict(spider=(),ids=())];varied=False
   while stack:
    path=stack.pop();hit,why,out,ctx=trial(p,v,si,path);calls+=1;varied|=ctx.varied
    if len(path['spider'])<ctx.spider_calls:
     k=len(path['spider']);stack.extend(dict(path,spider=path['spider']+(j,)) for j in range(ctx.spider_sizes[k]));continue
    if len(path['ids'])<len(ctx.tag_sizes):
     k=len(path['ids']);stack.extend(dict(path,ids=path['ids']+(j,)) for j in range(ctx.tag_sizes[k]));continue
    if hit:return dict(key=o['key'],witness=[v,ctx.assigned,si,path],calls=calls)
   if not varied:break
 return dict(key=o['key'],witness=None,groupTuples=combos,calls=calls)
if __name__=='__main__':
 keys={tuple(x['key']) for x in map(json.loads,open(P/'verified.jsonl')) if x['n']==0}
 with mp.Pool(6,initializer=install) as pool,open(P/'relaxation.jsonl','w',buffering=1) as fp:
  for x in pool.imap_unordered(one,[(r,n,o) for r,n,o in jobs() if tuple(o['key']) in keys]):fp.write(json.dumps(x)+'\n')
