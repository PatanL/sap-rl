import argparse,itertools as IT,json,math,multiprocessing as mp,collections as C,time,sys,os
from pathlib import Path
from infer import *
def tuples(p):
    b=p['b']; groups=p['groups']; wallet=p['wallet']
    def rec(j,chosen):
        vals={u:v[0] for u,v in zip(b,chosen)}
        for nm,ids in p['clauses']:
            if not any(i not in vals or vals[i]==nm for i in ids):return
        if wallet:
            base,coeff,total=wallet
            earned=base+sum(w for u,w in zip(b,coeff) if vals.get(u)=='Pig')
            cap=sum(w for u,w in zip(b,coeff) if u not in vals)
            if earned>total or earned+cap<total:return
        if j==len(b):
            if allowed(p,[x[0] for x in chosen]):yield tuple(chosen)
            return
        for ns in groups[j]:yield from rec(j+1,chosen+[ns])
    return rec(0,[])

def run_case(item):
    row,n,rec,tier,old=item
    p=prep(row,n);p['clauses']=stock_clauses(p)
    if os.environ.get('CONCRETE')=='1':p['groups']=[[[x] for x in group] for group in p['domain']]
    key=rec['key'];b=p['b'];e=p['e']
    unknown_food=sorted({a['spell']['name'] for a in p['acts'] if a['t']==8 and a['spell']['name'].startswith('#')})
    foodvars=['Apple','Honey','Cupcake','Meat Bone','Sleeping Pill','Garlic','Salad Bowl','Canned Food','Pear','Chili','Chocolate','Sushi','Melon','Mushroom','Pizza','Steak','Cake','Bread','Milk','Better Milk','Best Milk','Better Apple','Best Apple','Bread Crumbs'] if unknown_food else []
    p['food_slots']={a['id']:a['spell']['name'] for a in p['acts'] if a['t']==8 and a['spell']['name'].startswith('#')}
    if old:
        seen={tuple(x[0]):x[1:] for x in old['witnesses']}
    else:seen={}
    start=time.monotonic();reasons=C.Counter();roles={};runs=0;classes=0;combos=0;outstates=0
    seed_solutions=set(old.get('goodRandomSeeds',[]) if old else [])
    # Previous matching seeds are tried early on a new label. The regular schedule is the same for each label.
    limit=tier
    representative_classes=[]
    best=None; source_roles={}
    for group in tuples(p):
        classes+=1; combos+=math.prod(map(len,group))
        vals=tuple(v[0] for v in group)
        for ftuple in IT.product(foodvars,repeat=len(unknown_food)):
            aliases=dict(zip(unknown_food,ftuple))
            if aliases:
                p['acts']=[dict(a,spell={**a['spell'],'name':aliases[p['food_slots'][a['id']]]}) if a.get('id') in p['food_slots'] and a['t']==8 else a for a in p['acts']]
                for uid,enum in p['food_slots'].items():p['names'][uid]=aliases[enum]
            if not e and vals+ftuple in seen:continue
            seeds=list(sorted(seed_solutions)[:3])+list(range(limit))
            if old:seeds=[0]+[s for s in seeds if s>=old['seedCap'] or s in seed_solutions]
            for si in dict.fromkeys(seeds):
                permute=os.environ.get('BIRTH_IDS')=='1';stack=[{'spider':(), 'ids':()} if permute else ()];had_spider=False;reject=False;stochastic=False
                while stack:
                    sp=stack.pop()
                    hit,why,out,ctx=trial(p,vals,si,sp);runs+=1
                    for u,role in ctx.roles.items():source_roles.setdefault(u,set()).add(tuple(role))
                    # For unlabelled generated board items enumerate every branch of every Spider in the trace.
                    stochastic=stochastic or ctx.varied
                    spseq=sp['spider'] if permute else sp
                    if len(spseq)<ctx.spider_calls:
                        stack.extend((dict(sp,spider=spseq+(j,)) if permute else spseq+(j,)) for j in range(ctx.spider_sizes[len(spseq)]));had_spider=True;continue
                    if permute and len(sp['ids'])<len(ctx.tag_sizes):
                        stack.extend(dict(sp,ids=sp['ids']+(j,)) for j in range(ctx.tag_sizes[len(sp['ids'])]));continue
                    if hit:
                        endogenous=tuple(ctx.assigned[i] for i in e)
                        label=vals+endogenous+ftuple
                        if label not in seen:seen[label]=[list(map(list,group))+[[v] for v in endogenous+ftuple],si,sp if permute else list(sp)]
                        seed_solutions.add(si)
                    else:
                        reasons[why]+=1
                        if best is None or (out and 'state'in out and why.count(',')<best[0]):
                            best=[why.count(',') if out and 'state'in out else 100,why,vals,si,(out or {}).get('at'),(out or {}).get('error')]
                    # A wrong natural generation pool before any Pill RNG cannot be repaired with a new buff target.
                    if not hit and why in ('generation pool','duplicate discover siblings','both discover alternatives requested') and not any((a.get('spell') or {}).get('name')=='Sleeping Pill' for a in p['acts']):reject=True
                if (not e and vals+ftuple in seen) or reject or not stochastic:break
    allw=[[list(k),*v] for k,v in seen.items()]
    successes=sum(math.prod(len(g) for g in w[1]) for w in allw)
    return dict(key=key,split=row['split'],skip=rec['skip'],unknownBuys=b,unknownEndogenous=e,unknownFood=unknown_food,domain=[list(x) for x in p['domain']],
          groups=p['groups'],wallet=p['wallet'],stockClauses=p['clauses'],nativeRoles={str(i):sorted(v) for i,v in source_roles.items()}, classes=classes,combos=combos,
          seedCap=tier,engineCalls=runs,matchingAssignments=successes,witnesses=allw,goodRandomSeeds=sorted(seed_solutions),reasons=reasons.most_common(6),nearest=best,
          elapsed=round(time.monotonic()-start,3),ops=[x[0] for x in p['tr']],sells=sum(a['t']==9 for a in p['acts']))
if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--tier',type=int,default=1);pa.add_argument('--min',type=int,default=0);pa.add_argument('--max',type=int,default=9);pa.add_argument('--prev',nargs='*',default=[]);pa.add_argument('--max-witness',type=int,default=10**12);pa.add_argument('--out',required=True);pa.add_argument('--workers',type=int,default=6);a=pa.parse_args()
    prior={tuple(x['key']):x for f in a.prev for x in map(json.loads,open(f))}
    js=[]
    for r,n,o in jobs():
        _,_,_,_,b,_,_=inspect(r,n)
        k=tuple(o['key']);v=prior.get(k)
        if a.min<=len(b)<=a.max and (not a.prev or v is not None and v['matchingAssignments']<=a.max_witness):js.append((r,n,o,a.tier,v))
    with open(a.out,'w',buffering=1) as fp,mp.Pool(a.workers,initializer=install) as pool:
        for x in pool.imap_unordered(run_case,js,chunksize=1):fp.write(json.dumps(x,separators=(',',':'))+'\n')
