"""Infer only the post-Pill slots constrained by later, real UID operations."""
import sys,os,json,itertools as it,multiprocessing as mp,collections as C,zlib
from pathlib import Path
S=Path('/sap/runs/shop-oracle-20261122');sys.path.insert(0,str(S))
os.environ['SAP_ENGINE_ROOT']='/engine/sapai'
from replay import engine_turn,compile_turn
def pill(a):return a['t']==8 and a['spell']['name']=='Sleeping Pill'
def suffix_constraints(after,acts,j,names):
    # The first later BoardOrders is a set observation: its destinations are commands, not observations.
    suffix=[];exact=True
    for a in acts[j+1:]:
        if 'orders' in a:
            anchor={i for i,_ in a['orders']};exact=False;break
        if pill(a):return None
        suffix.append(a)
    else:anchor={v['id'] for v in after['board'] if v}
    real_end=tuple(v['id'] if v else None for v in after['board'])
    seen=anchor.copy();steps=[]
    for a in reversed(suffix):
        t=a['t'];i=a.get('id')
        if t in (7,9):
            if i in seen:return None
            seen.add(i);steps.append(('remove',i,None))
        elif t==6:
            if i in seen:seen.remove(i);steps.append(('buy',i,a['x']))
            else:steps.append(('merge',i,a['x']))
    if len(seen)>5:return None
    results=[]
    for start in set(it.permutations([*sorted(seen),*([None]*(5-len(seen)))])):
        possibilities={start}
        for op,i,x in reversed(steps):
            updates=set()
            for b in possibilities:
                if op=='remove':
                    if i in b:updates.add(tuple(None if v==i else v for v in b))
                elif op=='merge':
                    target=b[x]
                    if target is not None and (i not in names or target not in names or names[i]==names[target]):updates.add(b)
                elif b[x] is None:
                    y=list(b);y[x]=i;updates.add(tuple(y))
                elif names.get(b[x])!=names.get(i) or i not in names or b[x] not in names:
                    # Admit both legal shift directions to the first vacancy. Retain only invariant conclusions.
                    for direction in (-1,1):
                        hole=x
                        while 0<=hole<5 and b[hole] is not None:hole+=direction
                        if 0<=hole<5:
                            y=list(b)
                            while hole!=x:y[hole]=y[hole-direction];hole-=direction
                            y[x]=i;updates.add(tuple(y))
            possibilities=updates
        if (exact and real_end in possibilities) or (not exact and any(set(b)-{None}==anchor for b in possibilities)):results.append(start)
    return seen,results,exact,not steps and exact
def matches(r,b,old,names,only_members=False):
    if only_members:
        if {i for i in r if i in old}!={i for i,_ in b if i in old}:return False
        exp=[names[i] for i in r if i is not None and i not in old and i in names]
        got=[s for i,s in b if i is not None and i not in old]
        return len([i for i in r if i is not None])==sum(i is not None for i,_ in b) and not (C.Counter(exp)-C.Counter(got))
    for i,(k,s) in zip(r,b):
        if i is None or i in old:
            if i!=k:return False
        elif k is None or k in old or (i in names and names[i]!=s):return False
    return True
def work(row):
    result=[]
    for k,acts in row['actions'].items():
        n=int(k)
        if n<2 or k not in row['states'] or not any(pill(a) for a in acts):continue
        prev=row['states'][str(n-1)];after=row['states'][k]
        _,names,im,_,_=compile_turn(row,n,'near-right');kinds={int(i):v[0] for i,v in row['known'].items()};kinds.update({i:'pet' for i in im})
        for j,a in enumerate(acts):
            if not pill(a):continue
            key=[row['match'],row['seat'],n,j];source=names.get(a['target']);q=dict(key=key,source=source,source_named_in_record=source is not None,split=row['split'])
            constraint=suffix_constraints(after,acts,j,names)
            if constraint is None:
                result.append(dict(q,skip='second Pill before anchor or unresolved suffix'));continue
            members,layouts,exact,direct=constraint
            if not layouts:
                result.append(dict(q,skip='inconsistent suffix'));continue
            if any(b['t']==6 and b['id'] not in names for b in acts[:j]):
                result.append(dict(q,skip='unnamed earlier purchase'));continue
            if any(b['t']==8 and b['spell']['name'].startswith('#') for b in acts[:j]):
                result.append(dict(q,skip='unmapped earlier food'));continue
            seedbase=zlib.crc32('/'.join(map(str,key)).encode());mem=False;good=False;matched_input=False
            q.update(possible_layouts=len(layouts),direct=direct,end_anchor=exact)
            for trial in range(200):
                try:v=engine_turn(prev,after,acts[:j+1],names,kinds,(seedbase+trial*2654435761)%2**32,trace=True)
                except Exception:continue
                lg=v['log']
                if v.get('error') or v['genIssues'] or len(lg)!=j+1:continue
                before=lg[-2];out=lg[-1];old=set(before['uids'])-{None}
                if a['target'] not in old:continue
                target=before['state']['board'][before['uids'].index(a['target'])]
                src=source or target[0];perk=target[-1]
                # Ordinary non-summoners are counted only if they actually had Mushroom.
                q.update(source=src,perk=perk)
                if src not in ('Deer','Sheep','Spider') and perk!='Mushroom':break
                matched_input=True
                new=members-old
                slots={tuple(x for x,i in enumerate(b) if i in new) for b in layouts}
                signatures={tuple((None if i is None else ('old',i) if i in old else ('new',names.get(i))) for i in b) for b in layouts}
                native=[(i,s[0] if s else None) for i,s in zip(out['uids'],out['state']['board'])]
                projected=[[None if i is None else i if i in old else names.get(i,'?') for i in b] for b in layouts]
                q.update(new_ids=sorted(new),new_slot_sets=sorted(slots),unique_view=len(signatures)==1,native_example=native,example_real=projected[0],trial=trial+1)
                if not matches(layouts[0],native,old,names,True):continue
                mem=True
                if any(matches(b,native,old,names) for b in layouts):good=True;break
            if matched_input:q.update(member_match=mem,compatible=good)
            else:q['skip']='outside requested species/perk or prefix unresolved'
            result.append(q)
    return result
if __name__=='__main__':
    fresh='--fresh' in sys.argv
    rows=list(map(json.loads,open(S/('recover/fresh/seats.jsonl' if fresh else 'seats.jsonl'))))
    with open('shop/pill-fresh.jsonl' if fresh else 'shop/pill.jsonl','w',buffering=1) as f,mp.Pool(6) as pool:
        for batch in pool.imap_unordered(work,rows):
            for r in batch:f.write(json.dumps(r)+'\n')
