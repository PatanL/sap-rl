"""Enumerate native food-target choices and compare observer-health accounting offline."""
import copy,itertools,json,math
from collections import Counter,defaultdict
from contextlib import contextmanager
from case import details,load
from receipts import load_rows,size
from replay import Pet,Player,expected,differences
import sapai.player as player_mod

@contextmanager
def events(path):
    original_target=player_mod.get_target;original_food=Player.buy_food;original_rabbit=Pet.eats_shop_food_trigger
    stack=[];seen=[];sizes=[]
    def choose(obj,*args,**kwargs):
        chosen,possible=original_target(obj,*args,**kwargs)
        if obj.name in ('food-sushi','food-pizza','food-salad-bowl'):
            ix=len(sizes);sizes.append(len(possible))
            if possible:chosen=possible[path[ix] if ix<len(path) else 0]
        return chosen,possible
    def buy(p,slot,target=None):
        event={'food':slot.obj.name,'observerEvents':[]};seen.append(event);stack.append(event)
        try:return original_food(p,slot,target)
        finally:stack.pop()
    def rabbit(p,target=None):
        enabled=stack and p.name=='pet-rabbit'
        if enabled:old=target._health
        out=original_rabbit(p,target)
        if enabled:stack[-1]['observerEvents'].append([p.uid,p.level,target.uid,bool(p.team.check_friend(target)),target._health-old])
        return out
    player_mod.get_target=choose;Player.buy_food=buy;Pet.eats_shop_food_trigger=rabbit
    try:yield seen,sizes
    finally:player_mod.get_target=original_target;Player.buy_food=original_food;Pet.eats_shop_food_trigger=original_rabbit

def observer_deltas(seen,allow_reorder,ignore_shop,limit=4000):
    # Native target sets stay fixed. Only the assignment of the three observer bonuses is recomputed.
    # This never feeds a comparison state back into the engine.
    schedules=[({},Counter())];native=Counter()
    for event in seen:
        observers=defaultdict(list)
        for rid,level,who,board,delta in event['observerEvents']:
            if board:native[who]+=delta
            observers[rid].append((who,level,board))
        if not observers:continue
        common=next(iter(observers.values()))
        if any([v[0] for v in arr]!=[v[0] for v in common] for arr in observers.values()):return []
        targets=[v[0] for v in common]
        seqs=list(itertools.permutations(range(len(targets)))) if allow_reorder and event['food'] in ('food-pizza','food-salad-bowl','food-sushi') else [list(range(len(targets)))]
        updated=[]
        for used,delta in schedules:
            orders=itertools.product(seqs,repeat=len(observers)) if allow_reorder=='independent' else ([seq]*len(observers) for seq in seqs)
            for order in orders:
                u=used.copy();d=delta.copy()
                for (rid,arr),seq in zip(observers.items(),order):
                    for j in seq:
                        who,lvl,board=arr[j]
                        if u.get(rid,0)>=3 or (ignore_shop and not board):continue
                        if board:d[who]+=lvl
                        u[rid]=u.get(rid,0)+1
                updated.append((u,d))
        reduced={ (tuple(sorted(u.items())),tuple(sorted(d.items()))):(u,d) for u,d in updated }
        if len(reduced)>limit:return []
        schedules=list(reduced.values())
    return [(delta,native) for _,delta in schedules]
def explains(d,seen,reorder,ignore_shop):
    end=expected(d['after']);out=d['run'];ds=differences(end,out)
    if ds!=['permanent']:return False
    s=out['state'];last=out['log'][-1];byuid={i:j for j,i in enumerate(last['uids']) if i is not None}
    # Avoid treating capped health, sales, or merges involving an earlier food recipient as additive.
    if any(v and max(v[3:5])>=48 for v in s['board']+end['board']):return False
    if any(s['board'][j] and s['board'][j][:4]!=end['board'][j][:4] for j in byuid.values()):return False
    targetids={e[2] for food in seen for e in food['observerEvents'] if e[3]}
    first_food=next((j for j,a in enumerate(d['actions']) if a['t']==8),len(d['actions']))
    changed=set()
    for a,(_,dest) in zip(d['actions'][first_food+1:],d['steps'][first_food+1:]):
        if a['t'] in (6,7,9):changed.update(i for i in (a.get('id'),a.get('target'),dest) if i is not None)
    for delta,native in observer_deltas(seen,reorder,ignore_shop):
        if any(delta[i]!=native[i] for i in changed):continue
        if all(s['board'][j][4]+delta[i]-native[i]==end['board'][j][4] for i,j in byuid.items()):return True
    return False
def study(r,max_paths=2048):
    row,n=load(r['key']);seed=r['seed']
    with events(()) as (_,sizes):details(row,n,seed)
    lengths=list(sizes);work=math.prod(lengths);hist=Counter();supports=set();found={};other=0
    if work>max_paths:return dict(key=r['key'],paths=work,skipped='path count')
    for path in itertools.product(*(range(i) for i in lengths)):
        with events(path) as (seen,sz):d=details(row,n,seed)
        if sz!=lengths or 'state' not in d['run']:other+=1;continue
        state=d['run']['state'];supports.add(json.dumps(state,sort_keys=True));hist[','.join(differences(expected(d['after']),d['run']))]+=1
        for name,mode in [('permute food notifications',(True,False)),('omit shop pets from Rabbit',(False,True)),('both',(True,True)),('per-observer ordering',('independent',False))]:
            if name not in found and explains(d,seen,*mode):
                found[name]=dict(nativeTargetIndices=path,events=copy.deepcopy(seen))
    return dict(key=r['key'],nativeChoiceCounts=lengths,paths=work,distinctFoodStates=len(supports),nativeMatches=hist[''],other=other,explainedBy=found)
if __name__=='__main__':
    rr,_=load_rows();miss=[r for r in rr.values() if r.get('diff')]
    keys={tuple(r['key']) for r in sorted(miss,key=size)[:10]}
    # Generalize the fixed diagnostic to the entire Rabbit-associated failure set.
    rr=[r for r in miss if ('Rabbit' in r['pets'] and set(r['foods']).intersection({'Canned Food','Sushi','Salad Bowl','Pizza'})) or tuple(r['key']) in keys]
    import multiprocessing as mp
    with mp.Pool(6) as pool:out=list(pool.imap_unordered(study,rr))
    json.dump(out,open('food-probes.json','w'),indent=2)
    print('cases',len(out),'exact comparison signatures',Counter(k for d in out for k in d.get('explainedBy',{})))
