"""CPU-only constrained shop oracle. Opaque discarded offers are never treated as purchases."""
import collections as C,itertools as IT,json,math,sys,time,zlib,os,inspect as pyinspect
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path[:0]=[str(P.parent),str(P)]
import replay as rp
from profile_cases import jobs,inspect
from raw_checks import sales_ledger
from sapai.shop import pet_tier_lookup_std as POOLS, get_shop_rules
from sapai.rand import MockRandomState
import numpy as np
from copy import deepcopy
TIER={rp.data['pets'][x]['name']:t for t in range(1,7) for x in POOLS[t]}
TIER['Sloth']=1
STOCK_FOODS={'Milk','Better Milk','Best Milk','Bread Crumbs','Better Apple','Best Apple','#130','#131'}
BASE_SHOP=rp.OracleShop
ORIGINAL_CHOICE=MockRandomState.choice
class Invalid(BaseException):
    """This trial contradicts recorded identity or the generation pool."""
class Forced(Exception):pass
class Context:
    def __init__(self,buys,extra,known,spider=()):
        self.tag_path=spider.get('ids',[]) if isinstance(spider,dict) else []
        if isinstance(spider,dict):spider=spider['spider']
        self.tag_sizes=[];self.buys=buys;self.extra=extra;self.known=known;self.assigned={};self.roles={};self.spider=spider;self.spider_calls=0;self.spider_sizes=[];self.varied=False;self.pill=False
CTX=None
def choice(self,a,*args,**kw):
    # Sample only alternatives of Spider's native SummonRandomPet choice. Preserve normal RNG draw.
    value=ORIGINAL_CHOICE(self,a,*args,**kw)
    caller=pyinspect.currentframe().f_back.f_code.co_name
    if CTX is not None and len(a)>1 and caller not in ('levelup','_roll_slot','roll','roll_levelup','SummonRandomPet'):
        if caller!='update_pet_priority' or CTX.pill:CTX.varied=True
    if caller=='SummonRandomPet' and CTX is not None:
        i=CTX.spider_calls;CTX.spider_calls+=1;CTX.spider_sizes.append(len(a))
        if i<len(CTX.spider):return np.asarray([a[CTX.spider[i]]])
    return value
class AuditShop(BASE_SHOP):
    def draw(self,kind,reward=False):
        uid=self.unc+1;name=self.known.get(uid)
        if name and self.kinds.get(uid) and self.kinds[uid]!=kind:raise Invalid('recorded UID allocated to wrong category')
        if name and kind=='pet' and name not in TIER:raise Invalid('nonrollable token in pet roll')
        if kind=='pet' and uid in CTX.buys:
            CTX.roles[uid]=('reward' if reward else 'roll',self.levelup_tier if reward else self.tier_avail)
            if name is not None:
                tier=TIER.get(name)
                if (reward and (tier!=self.levelup_tier or name=='Sloth')) or (not reward and (not tier or tier>self.tier_avail)):
                    raise Invalid('generation pool')
        if uid in CTX.buys and kind!='pet':raise Invalid('purchase UID allocated to food')
        if uid in CTX.extra:raise Invalid('summon UID allocated to shop')
        if kind=='food' and name in STOCK_FOODS:raise Invalid('stock-only food allocated to normal roll')
        return super().draw(kind,reward)
    def tag_new(self,team):
        old=len(self.native_items)
        born=[s.pet for s in team if not s.empty and getattr(s.pet,'uid',None) is None]
        super().tag_new(team)
        if len(born)>1:
            idx=len(CTX.tag_sizes);CTX.tag_sizes.append(math.factorial(len(born)))
            if idx<len(CTX.tag_path) and CTX.tag_path[idx]:
                ids=[s.uid for s in born];chosen=next(IT.islice(IT.permutations(ids),CTX.tag_path[idx],None))
                for pet,i in zip(born,chosen):pet.uid=i
                remap=dict(zip(ids,chosen))
                self.native_items[old:]=[(remap.get(i,i),nm) for i,nm in self.native_items[old:]]
        for uid,name in self.native_items[old:]:
            if uid in CTX.buys:raise Invalid('purchased UID allocated to summon/stock')
            if uid in CTX.extra:
                CTX.assigned[uid]=name
            if uid in CTX.known and CTX.known[uid]!=name:raise Invalid('generated item contradicted by recorded identity')
    def levelup(self,count=1):
        super().levelup(count)
        groups=C.defaultdict(list)
        for s in self:
            if s.slot_type=='levelup' and s.reward_group is not None and not getattr(s.obj,'opaque',False):
                groups[s.reward_group].append((s.obj.uid,s.obj.name))
        for pair in groups.values():
            if len(pair)==2 and all(i in CTX.buys for i,_ in pair):raise Invalid('both discover alternatives requested')
            if len(pair)==2 and pair[0][1]==pair[1][1] and any(i in CTX.buys for i,_ in pair):raise Invalid('duplicate discover siblings')

def install():
    rp.OracleShop=AuditShop;MockRandomState.choice=choice

def prep(row,n):
    acts,tr,names,missing,b,e,links=inspect(row,n)
    before=row['states'][str(n-1)];end=row['states'][str(n)]
    kinds={int(i):s[0] for i,s in row['known'].items()}
    for i in set(names)-kinds.keys():kinds[i]='pet'
    for i in b:kinds[i]='pet'
    tier=end['tier'];max_tier=min(tier+1,6)
    # A reward needs a level crossing before acquisition. Before any possible merge/Chocolate only ordinary draws exist.
    groups=[];domain=[];neutral=[];wallet=None
    pos={a['id']:j for j,a in enumerate(acts) if a['t']==6}
    sale={a['id']:j for j,a in enumerate(acts) if a['t']==9}
    pill=any((a.get('spell') or {}).get('name')=='Sleeping Pill' for a in acts)
    for uid in b:
        j=pos[uid];prior=acts[:j]
        reward_possible=any(op in ('buy merge','board merge','shop merge') for op,_ in tr[:j]) or any((a.get('spell') or {}).get('name')=='Chocolate' for a in prior)
        ns=sorted(x for x,t in TIER.items() if t<=(max_tier if reward_possible else tier))
        # If source appears later, placing it cannot absorb it into a different UID already at this square.
        dst=tr[j][1]
        if not pill and dst in names and uid in sale and tr[j][0]=='buy':
            ns=[x for x in ns if x!=names[dst]]
        domain.append(ns)
        # A same-tier, non-triggering pet bought then immediately sold cannot affect the suffix by its
        # personal stats. Keep all possible collisions, stock/ability effects, and any potential
        # discover-sibling constraint separate; do not compress single-UID cases.
        ephemeral=(uid in sale and sale[uid]==j+1 and not pill and len(b)>1 and tr[j][0]=='buy' and (dst is None or dst in names)  )
        if any(uid in xy for xy in links):ephemeral=False
        neutral.append(ephemeral)
        buckets=C.defaultdict(list)
        for x in ns:
            key=(TIER[x],'neutral') if ephemeral and x not in {names.get(uid-1),names.get(uid+1)} and x not in ('Otter','Beaver','Pig','Duck','Pigeon','Cow','Scorpion') else (TIER[x],x)
            buckets[key].append(x)
        groups.append(list(buckets.values()))
    # Independent ledger: supplied prices and sale identities/levels; the only unknown income
    # term on a no-pill Turtle turn is the extra Pig sale bonus. Check its linearity.
    if not pill and not e and not any((a.get('spell') or {}).get('name','').startswith('#') for a in acts):
        neutral_names={**names,**{i:'(ordinary)' for i in b}}
        def sale_income(labels):
            # Keep the pre-turn frozen flag/stack when masking a known control. Replace only its identity.
            prior=dict(before,shop=[dict(v,name=labels[v['id']]) if v['id'] in b else v for v in before['shop']])
            return sales_ledger(prior,acts,tr,labels)[0]
        base=sale_income(neutral_names);coeff=[]
        if base is not None:
            for uid in b:
                v=sale_income({**neutral_names,uid:'Pig'});coeff.append(v-base)
            v=sale_income({**neutral_names,**dict.fromkeys(b,'Pig')})
            if v==base+sum(coeff):wallet=(base,coeff,end['gain'])
    return dict(row=row,n=n,acts=acts,tr=tr,names=names,b=b,e=e,missing=missing,links=links,before=before,end=end,want=rp.expected(end),kinds=kinds,
                groups=groups,domain=domain,neutral=neutral,wallet=wallet,seed=zlib.crc32(f"{row['match']}/{row['seat']}/{n}".encode()))

def stock_clauses(p):
    acts=p['acts'];names=p['names'];b=set(p['b'])
    clauses=set()
    for j,a in enumerate(acts):
        if a['t']!=8:continue
        name=a['spell']['name'];uid=a['id']
        if uid<=p['before']['unc'] or name not in {'Milk','Better Milk','Best Milk','Bread Crumbs'}:continue
        producer='Pigeon' if name=='Bread Crumbs' else 'Cow'
        op=9 if producer=='Pigeon' else 6
        start=0;stop=j;frozen=False
        for h,prev in enumerate(acts[:j+1]):
            for i,flag in prev.get('freeze',[]):
                if i==uid:frozen=flag
            if prev['t'] in (4,5) and not frozen:start=h+1
            referenced=[i for i,_ in prev.get('freeze',[])]+[i for i,_ in prev.get('orders',[])]+[prev.get('target') or 0,prev.get('id') or 0]
            if max(referenced)>=uid:stop=h;break
        candidates=[v['id'] for v in acts[start:stop] if v['t']==op]
        if any(names.get(i)==producer or (i not in names and i not in b) for i in candidates):continue
        unknown=tuple(sorted({i for i in candidates if i in b}))
        if unknown:clauses.add((producer,unknown))
    return sorted(clauses)
def allowed(p,vals):
    if p['wallet']:
        base,coeff,total=p['wallet']
        if base+sum(c for c,x in zip(coeff,vals) if x=='Pig')!=total:return False
    d={**p['names'],**dict(zip(p['b'],vals))}
    if any(not any(d.get(i)==name for i in ids) for name,ids in p.get('clauses',[])):return False
    return all(d.get(x)==d.get(y) for x,y in p['links'] if x in d and y in d)

def trial(p,vals,i,spider=()):
    global CTX
    known={**p['names'],**dict(zip(p['b'],vals))}
    CTX=Context(p['b'],p['e'],known,spider);CTX.pill=any((a.get('spell') or {}).get('name')=='Sleeping Pill' for a in p['acts'])
    try:
        uids=[]
        def endpoint(player):
            uids.extend(None if s.empty else getattr(s.pet,'uid',None) for s in reversed(player.team.slots))
        out=rp.engine_turn(p['before'],p['end'],p['acts'],known,p['kinds'],(p['seed']+i*2654435761)%2**32,on_end=endpoint)
    except (Invalid,rp.Unresolved) as err:
        return False,str(err),None,CTX
    ds=rp.differences(p['want'],out)
    if 'state' in out and uids!=[v['id'] if v else None for v in p['end']['board']]:ds.append('end board UIDs')
    if set(CTX.assigned)!=set(p['e']):ds.append('unidentified endogenous UID')
    if out['genIssues']:ds.append('generation kind')
    return not ds,','.join(ds) or 'match',out,CTX

if __name__=='__main__':
 install();hist=C.Counter()
 for row,n,rec in jobs():
  p=prep(row,n);hist[len(p['b'])]+=math.prod(len(x) for x in p['groups'])
 print(hist)
