"""Find a concrete competing identity that completes the recorded actions and wallet, but differs at the endpoint."""
import itertools as IT,json,multiprocessing as mp
from search import *
from evidence import ordered
from order_evidence import clues
def one(arg):
 row,n,o,truth=arg;p=prep(row,n);p['clues']=clues(row,n);p['clauses']=stock_clauses(p)
 ids=p['b']+p['e'];label=tuple(truth['marginals'][str(i)][0] for i in ids)
 for sets in tuples(p):
  for values in IT.product(*sets):
   if not p['e'] and values==label:continue
   nm={**p['names'],**dict(zip(p['b'],values))}
   if not ordered(p,nm):continue
   for si in range(4):
    stack=[{'spider':(),'ids':()}];varied=False
    while stack:
     path=stack.pop();hit,why,sim,ctx=trial(p,values,si,path);varied|=ctx.varied
     sp=path['spider'];ix=path['ids']
     if len(sp)<ctx.spider_calls:
      stack.extend(dict(path,spider=sp+(j,)) for j in range(ctx.spider_sizes[len(sp)]));continue
     if len(ix)<len(ctx.tag_sizes):
      stack.extend(dict(path,ids=ix+(j,)) for j in range(ctx.tag_sizes[len(ix)]));continue
     if not sim or 'state' not in sim or sim['costErrors'] or sim['genIssues'] or set(ctx.assigned)!=set(p['e']):continue
     if any(sim['state'][k]!=p['want'][k] for k in ('gold','gain','spent')):continue
     alt=values+tuple(ctx.assigned[i] for i in p['e'])
     if alt!=label:
      return dict(key=o['key'],fit=list(label),action_wallet_competitor=alt,failed_fields=why,seedIndex=si,choices=path)
    if not varied:break
 return dict(key=o['key'],fit=list(label),action_wallet_competitor=None)
if __name__=='__main__':
 prior={tuple(x['key']):x for x in map(json.loads,open(P/'input-certificates.jsonl'))}
 rr={tuple(x['key']):x for x in map(json.loads,open(P/'verified.jsonl')) if x['n']==1 and not prior[tuple(x['key'])]['certified']}
 with mp.Pool(6,initializer=install) as pool,open(P/'contrast.jsonl','w',buffering=1) as fp:
  for v in pool.imap_unordered(one,[(r,n,o,rr[tuple(o['key'])]) for r,n,o in jobs() if tuple(o['key']) in rr]):fp.write(json.dumps(v,separators=(',',':'))+'\n')
