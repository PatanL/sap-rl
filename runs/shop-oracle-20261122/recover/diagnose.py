"""Describe residual mismatches; this only examines the archived development/held-out corpus, never fresh turns."""
import json
from search import *
from order_evidence import clues
from evidence import ordered
def one(arg):
 row,n,o=arg;p=prep(row,n);p['clues']=clues(row,n);p['clauses']=stock_clauses(p)
 foods=[a for a in p['acts'] if a['t']==8 and a['spell']['name'].startswith('#')]
 if foods:
    # Name used only to read the native inventory at that UID; all 24 aliases are tested in the main matrix.
    p['acts']=[dict(a,spell={**a['spell'],'name':'Milk'}) if a in foods else a for a in p['acts']]
    for a in foods:p['names'][a['id']]='Milk'
 best=None;whycount=C.Counter()
 for groups in tuples(p):
    vals=tuple(x[0] for x in groups);nm={**p['names'],**dict(zip(p['b'],vals))}
    if not ordered(p,nm):continue
    for si in range(12):
        stack=[()]
        while stack:
            sp=stack.pop();hit,why,out,ctx=trial(p,vals,si,sp)
            if len(sp)<ctx.spider_calls:
                stack.extend(sp+(j,) for j in range(ctx.spider_sizes[len(sp)]));continue
            whycount[why]+=1
            if out is None:continue
            got=out.get('state',out.get('partial'));want=p['want'];b1=got['board'];b2=want['board']
            compare=sum(sum(a!=b for a,b in zip(x,y)) if x and y else 8*int(x!=y) for x,y in zip(b1,b2))
            absdiff=sum(sum(abs(x[j]-y[j]) for j in range(3,7)) for x,y in zip(b1,b2) if x and y and x[0]==y[0])
            score=(int('state' not in out),-out.get('at',0),len(why.split(',')),compare,absdiff)
            if best is None or score<best[0]:best=(score,dict(names=dict(zip(p['b'],vals)),endogenous=ctx.assigned,at=out.get('at'),error=out.get('error'),why=why,state=got,seed=si,spider=sp))
 return dict(key=o['key'],split=row['split'],skip=o['skip'],before=p['before'],expected=p['want'],actions=p['acts'],known_names={i:p['names'].get(i) for a in p['acts'] for i in (a.get('id'),a.get('target')) if i is not None},
    best=best[1] if best else None,diagnostic_counts=whycount,shape=inspect(row,n)[1])
if __name__=='__main__':
 rs={tuple(x['key']):x for x in map(json.loads,open(P/'pass2048-zero.jsonl')) if not x['matchingAssignments']}
 with mp.Pool(6,initializer=install) as pool,open(P/'negative.jsonl','w') as fp:
  for x in pool.imap_unordered(one,[(r,n,o) for r,n,o in jobs() if tuple(o['key']) in rs]):fp.write(json.dumps(x,separators=(',',':'))+'\n')
