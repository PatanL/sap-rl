"""Native shop-turn replay. The shim supplies offer identities and translates replay UIDs."""
import argparse, collections, hashlib, json, multiprocessing as mp, os, sys, zlib
from pathlib import Path
import numpy as np
ENGINE=Path(os.environ['SAP_ENGINE_ROOT']).resolve()
sys.path.insert(0,str(ENGINE))
import sapai
assert Path(sapai.__file__).resolve().parent==ENGINE/'sapai'
from sapai import Player,Shop,Team,Pet,Food
from sapai.shop import ShopSlot
from sapai.teams import TeamSlot
from sapai.data import data
from identity import compile_turn,insertion_hole
STATUS={'Meat Bone':'status-bone-attack','Coconut':'status-coconut-shield','Mushroom':'status-extra-life','Garlic':'status-garlic-armor',
        'Honey':'status-honey-bee','Melon':'status-melon-armor','Peanut':'status-poison-attack','Chili':'status-splash-attack',
        'Steak':'status-steak-attack','Cake':'status-cake','Bread':'status-bread',None:'none'}
PERK={v:k for k,v in STATUS.items()}
def key(s):return s.lower().replace(' ','-')
class Unresolved(Exception):pass
class Divergence(Exception):pass

def make_pet(name):
 eid='pet-'+key(name)
 if eid not in data['pets']:raise Unresolved('unsupported pet identity: '+name)
 return Pet(eid)
def make_food(name):
 scale=2 if name.startswith('Better ') else 3 if name.startswith('Best ') else 1
 base=name.split(' ',1)[1] if scale>1 else name
 eid='food-'+key(base)
 if eid not in data['foods']:raise Unresolved('unsupported food identity: '+name)
 f=Food(eid)
 for k in ('attack','health','base_attack','base_health'):setattr(f,k,getattr(f,k)*scale)
 return f
def restore_pet(d):
 p=make_pet(d['name']);p._attack=d['a'];p._health=d['h']
 p._until_end_of_battle_attack_buff=d['ta'];p._until_end_of_battle_health_buff=d['th']
 p.set_total_experience(d['xp']);assert p.level==d['lvl']
 if d['perk'] not in STATUS:raise Unresolved('unsupported perk identity: '+str(d['perk']))
 p.status=STATUS[d['perk']];p.sell_value=p.level+d['sellBonus'];p.ability_counter=d['uses'];p.uid=d['id'];return p
def label(o):return getattr(o,'uid',None)
def real_food_name(f):
 name=data['foods'][f.name]['name']
 if name in ('Milk','Apple'):
  power=f.attack
  return {2:'Better ',3:'Best '}.get(power,'')+name
 return name
def pet_stats(p):
 return [data['pets'][p.name]['name'],int(p.level),int(p.total_experience),int(p._attack),int(p._health),
         int(p._until_end_of_battle_attack_buff),int(p._until_end_of_battle_health_buff),PERK.get(p.status,p.status)]
def expected_pet(v):return [v[k] for k in ('name','lvl','xp','a','h','ta','th','perk')]
def expected_frozen(s):
 out=[[v['id'],'pet',*expected_pet(v),v['cost']] for v in s['shop'] if v['f']]
 out+=[[v['id'],'food',v['name'],v['cost']] for v in s['food'] if v['f']]
 return sorted(out)
def food_type(name):return name in ('Milk','Better Milk','Best Milk','Better Apple','Best Apple','Bread Crumbs')
class OracleShop(Shop):
    def prepare(self,before,known,kinds):
        self.unc=before['unc'];self.known=known;self.kinds=kinds;self.audit=[];self.native_items=[];self.generated=[];self.id_problems=[]
        stock=[]
        for group,items in [('pet',before['shop']),('food',before['food'])]:
            for v in items:
                if group=='pet':
                    obj=restore_pet(v);slot=ShopSlot(obj,slot_type='levelup' if v['reward'] else 'pet',reward_group=-min(v['links'] or [v['id']]) if v['reward'] else None)
                else:
                    obj=make_food(v['name']);obj.uid=v['id'];slot=ShopSlot(obj)
                slot.frozen=v['f'];slot.cost=v['cost'];stock.append(slot)
        self._nslots=len(stock);self.slots=stock
        self.shop_attack=sum(v.get('At',0) or 0 for v in before['buffs'])
        self.shop_health=sum(v.get('Hp',0) or 0 for v in before['buffs'])
    def draw(self,kind,reward=False):
        self.unc+=1;i=self.unc;name=self.known.get(i);tag=self.kinds.get(i)
        if tag and tag!=kind:
            self.id_problems.append(('generation category',i,tag,kind))
            name=None
        if name is None:
            obj=Pet() if kind=='pet' else Food()
            if kind=='pet':obj.tier=0;obj._attack=obj._health=0
            else:obj.cost=0
            obj.opaque=True
        else:
            obj=make_pet(name) if kind=='pet' else make_food(name)
        obj.uid=i;obj.shop=self;obj.rs=self.rs
        self.generated.append((i,kind,'reward' if reward else 'roll',name))
        return obj
    def _roll_slot(self,slot):
        if slot.slot_type=='levelup':
            slot.slot_type='pet';slot.reward_group=None
        slot.obj=self.draw(slot.slot_type)
        if slot.slot_type=='food':slot.cost=slot.obj.cost
    def levelup(self,count=1):
        old={id(s) for s in self.slots}
        super().levelup(count)
        new=[s for s in self.slots if id(s) not in old]
        for s in sorted(new,key=lambda s:s.reward_group):
            s.obj=self.draw('pet',reward=True)
            s.obj._attack=min(50,s.obj._attack+self.shop_attack);s.obj._health=min(50,s.obj._health+self.shop_health)
    def tag_new(self,team):
        # Only natural stock/summons get tags here. Their identity and stats come entirely from the engine.
        for s in self.slots:
            if label(s.obj) is not None:continue
            self.unc+=1;s.obj.uid=self.unc
            name=real_food_name(s.obj) if isinstance(s.obj,Food) else data['pets'][s.obj.name]['name']
            self.native_items.append((self.unc,name))
        for s in team:
            if s.empty or label(s.pet) is not None:continue
            self.unc+=1;s.pet.uid=self.unc;self.native_items.append((self.unc,data['pets'][s.pet.name]['name']))

def state_engine(p,spent):
 board=[None if s.empty else pet_stats(s.pet) for s in reversed(p.team.slots)]
 frozen=[]
 for s in p.shop:
  if not s.frozen:continue
  if getattr(s.obj,'opaque',False):frozen.append([label(s.obj),'unknown',s.cost]);continue
  if s.slot_type=='food':frozen.append([label(s.obj),'food',real_food_name(s.obj),int(s.cost)])
  else:frozen.append([label(s.obj),'pet',*pet_stats(s.obj),int(s.cost)])
 return dict(board=board,frozen=sorted(frozen),spent=int(spent),gold=int(p.gold),gain=int(p.gold-10+spent))
def expected(s):return dict(board=[expected_pet(v) if v else None for v in s['board']],frozen=expected_frozen(s),spent=s['spent'],gold=s['gold'],gain=s['gain'])
def engine_turn(before,after,acts,names,kinds,seed,trace=False):
 np.random.seed(seed)
 shop=OracleShop(slots=[],turn=before['turn'],fixed_rules=False);shop.prepare(before,names,kinds)
 team=Team()
 for v in before['board']:
  if v:team[4-v['x']]=restore_pet(v)
 p=Player(team=team,shop=shop,turn=before['turn'],gold=before['gold'],wins=before['wins'],record_actions=False)
 spent=0;log=[];cost_errors=[];which='';index=0
 def byid(i):
  for slot in team:
   if not slot.empty and label(slot.pet)==i:return slot.pet
  return None
 def offer(i):
  for s in shop:
   if label(s.obj)==i:return s
  return None
 def shop_item(i):
  s=offer(i)
  if s is None:raise Divergence(f'shop UID {i} absent')
  if getattr(s.obj,'opaque',False):raise Unresolved('unidentified purchased item')
  return s
 try:
  for index,a in enumerate(acts):
   t=a['t'];which={4:'start',5:'roll',6:'buy',7:'merge',8:'food',9:'sell',11:'end',12:'name'}.get(t,str(t))
   if 'orders' in a:
    desired={i:x for i,x in a['orders']};actual={label(s.pet):s for s in team if not s.empty}
    if set(actual)!=set(desired):raise Divergence(f'pre-action board ID set differs: {sorted(actual)} vs {sorted(desired)}')
    slots=[TeamSlot() for _ in range(5)]
    for i,x in desired.items():slots[4-x]=actual[i]
    team.slots=slots
   for i,flag in a.get('freeze',[]):
    s=offer(i)
    if s is None:
     raise Divergence(f'freeze references absent UID {i}')
    if flag and not s.frozen:p.freeze(shop.slots.index(s))
    elif not flag and s.frozen:p.unfreeze(shop.slots.index(s))
   i=a.get('id');paid=0;cost_observed=None;gold0=p.gold
   if t==4:p.start_turn(winner={1:True,2:False,3:None}.get(before['result']))
   elif t==5:p.roll();paid=1
   elif t==6:
    target_idx=4-a['x'];target=team[target_idx];src=byid(i)
    if src:
     which='move';p.swap(team.index(src),target_idx)
    else:
     stock=shop_item(i)
     if not isinstance(stock.obj,Pet):raise Divergence(f'UID {i} is not a pet')
     paid=stock.cost
     if not target.empty and target.pet.name==stock.obj.name:
      which='buy merge';p.buy_combine(stock,target)
     else:
      if not target.empty:
       board=[None if s.empty else label(s.pet) for s in reversed(team.slots)]
       hole=insertion_hole(a['x'],board,'near-right')
       if hole is None:raise Divergence('occupied buy destination on full board')
       # Shift one slot at a time toward the empty square, preserving pet identity.
       pos=4-hole
       while pos!=target_idx:
        neighbour=pos+(1 if target_idx>pos else -1);team.move(neighbour,pos);pos=neighbour
      p.buy_pet(stock,team_idx=target_idx)
   elif t==7:
    src=byid(i);dst=byid(a['target'])
    if dst is None:raise Divergence(f'merge target UID {a["target"]} absent')
    if src is not None:which='board merge';p.combine(dst,src)
    else:
     which='shop merge';stock=shop_item(i);paid=stock.cost;p.buy_combine(stock,dst)
   elif t==8:
    stock=shop_item(i)
    if not isinstance(stock.obj,Food):raise Divergence(f'UID {i} is not a food')
    # The independently logged spell makes an ID mapping error distinguishable from a stat mismatch.
    if a['spell'] and real_food_name(stock.obj)!=a['spell']['name']:
     raise Divergence(f'food identity: engine {real_food_name(stock.obj)}, real {a["spell"]["name"]}')
    paid=stock.cost;cost_observed=a['spell']['cost']
    target=byid(a['target']) if a.get('target') else None
    if a.get('target') and target is None:raise Divergence(f'food target UID {a["target"]} absent')
    p.buy_food(stock,target)
   elif t==9:
    obj=byid(i)
    if obj is None:raise Divergence(f'sale references absent board UID {i}')
    p.sell(obj)
   elif t==11:p.end_turn()
   elif t!=12:raise Unresolved(f'undecoded action {t}')
   if cost_observed is not None and paid!=cost_observed:
    cost_errors.append([index,i,int(paid),int(cost_observed)])
   spent+=paid;shop.tag_new(team)
   if trace:log.append(dict(op=which,id=i,x=a.get('x'),target=a.get('target'),gold0=int(gold0),gold=int(p.gold),cost=paid,realCost=cost_observed,
                     state=state_engine(p,spent),uses={label(s.pet):s.pet.ability_counter for s in team if not s.empty},
                     offers=[[label(s.obj),s.slot_type,real_food_name(s.obj) if s.slot_type=='food' and not getattr(s.obj,'opaque',False) else s.obj.name,s.cost,s.frozen] for s in shop],
                     uids=[None if s.empty else label(s.pet) for s in reversed(team.slots)],unc=shop.unc,gen=shop.generated.copy(),stock=shop.native_items.copy()))
  out=state_engine(p,spent)
  return dict(state=out,unc=shop.unc,genIssues=shop.id_problems,costErrors=cost_errors,log=log)
 except Unresolved:raise
 except Exception as err:
  return dict(error=type(err).__name__+': '+str(err),at=index,op=which,partial=state_engine(p,spent),unc=shop.unc,genIssues=shop.id_problems,costErrors=cost_errors,log=log)

def differences(want,got):
 if not got.get('state'):return ['action']
 s=got['state'];ds=[]
 board,real=s['board'],want['board']
 for field,inds in [('species/order',(0,)),('level/xp',(1,2)),('permanent',(3,4)),('temporary',(5,6)),('perk',(7,))]:
  values=lambda b:[tuple(v[j] for j in inds) if v else None for v in b]
  if values(board)!=values(real):ds.append(field)
 for k in ('frozen','spent','gold','gain'):
  if s[k]!=want[k]:ds.append(k)
 if got.get('costErrors'):ds.append('food price')
 return ds
def settle(row,n,cap,details=False):
 k=str(n);split=row['split'];eid=[row['match'],row['seat'],n];r=dict(key=eid,split=split,source=row['source'])
 if str(n-1) not in row['states']:return dict(r,skip='no preceding snapshot')
 before=row['states'][str(n-1)];after=row['states'][k];acts=row['actions'][k]
 steps,names,inferred,shapeIssues,problems=compile_turn(row,n,'near-right')
 kinds={int(i):v[0] for i,v in row['known'].items()};kinds.update({i:'pet' for i in inferred})
 ps={v['name'] for v in before['board']+after['board'] if v}
 frozen_names={v['name'] for v in before['shop']+after['shop'] if v['f']}
 direct=set()
 fs=set();ops=set()
 for a,(op,_) in zip(acts,steps):
  ops.add(op)
  if a['t']==8 and a['spell']:fs.add(a['spell']['name'])
  if a['t'] in (6,7,9) and a['id'] in names:ps.add(names[a['id']]);direct.add(names[a['id']])
  if a['t'] in (7,8) and a.get('target') in names:direct.add(names[a['target']])
 r.update(pets=sorted(ps),foods=sorted(fs),ops=sorted(ops),actions=len(acts),boardSize=sum(v is not None for v in before['board']),
          inferred=len(inferred),shapeIssues=shapeIssues,direct=sorted(direct),frozenPetNames=sorted(frozen_names))
 if not acts or acts[0]['t']!=4 or acts[-1]['t'] not in (11,12):return dict(r,skip='missing start/end action')
 if any(a['t']==8 and not a.get('spell') for a in acts):return dict(r,skip='missing food response')
 if any(name.startswith('#') for name in fs):return dict(r,skip='unmapped purchased food enum',missing=sorted(n for n in fs if n.startswith('#')))
 for a,(op,_) in zip(acts,steps):
  if a['t'] in (6,7) and a['id'] not in names:return dict(r,skip='unidentified purchased/merged pet',missing=a['id'])
 if problems:return dict(r,skip='ambiguous UI action',why=problems)
 want=expected(after);best=None;seen=set();ever=set();seedbase=zlib.crc32('/'.join(map(str,eid)).encode());first=None;completed=False
 for attempt in range(cap):
  seed=(seedbase+attempt*2654435761)%2**32
  try:out=engine_turn(before,after,acts,names,kinds,seed)
  except Unresolved as e:return dict(r,skip=str(e))
  ds=differences(want,out);ever.update(set(['species/order','level/xp','permanent','temporary','perk','frozen','spent','gold','gain'])-set(ds) if 'state' in out else [])
  if 'state' in out:completed=True
  sig=json.dumps(out.get('state',[out.get('error'),out.get('at')]),sort_keys=True);seen.add(sig)
  if not ds or ('state'in out and all(x not in ds for x in ('species/order','level/xp','permanent','temporary','perk'))):ever.add('board')
  if first is None:first=ds
  score=(int('action' in ds),len(ds),sum(x!=y for x,y in zip(want['board'],out.get('state',out.get('partial'))['board'])), -out.get('at',0))
  if best is None or score<best[0]:best=(score,ds,seed,out)
  if not ds:break
 ds,seed,out=best[1:];r.update(reproduced=not ds,boardMatch='board'in ever,ever=sorted(ever),diff=ds,
            tries=attempt+1,distinct=len(seen),seed=seed,first=first,completed=completed,uncDiff=out['unc']-after['unc'],genIssues=out['genIssues'],costErrors=out['costErrors'])
 if ds or details:
  r.update(actual=want,result=out.get('state') or out.get('partial'),error=out.get('error'),at=out.get('at'),op=out.get('op'))
 return r

def process(args):
 row,cap,split,restriction=args
 return [settle(row,int(n),cap) for n in row['states'] if int(n)>=2 and (not split or row['split']==split)
         and (restriction is None or (row['match'],row['seat'],int(n)) in restriction)]
if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--seeds',type=int,default=1);parser.add_argument('--workers',type=int,default=6);parser.add_argument('--split')
 parser.add_argument('--limit',type=int);parser.add_argument('--out',required=True);parser.add_argument('--retry')
 a=parser.parse_args()
 rows=[json.loads(l) for l in open('seats.jsonl')]
 if a.limit:rows=rows[:a.limit]
 restriction=None
 if a.retry:
  restriction={tuple(r['key']) for r in map(json.loads,open(a.retry)) if not r.get('skip') and not r.get('reproduced')}
 import time
 start=time.monotonic();total=collections.Counter()
 with open(a.out,'w',buffering=1) as f,mp.Pool(a.workers) as pool:
  for block in pool.imap_unordered(process,[(row,a.seeds,a.split,restriction) for row in rows]):
   for r in block:
    total['turns']+=1;total['skip' if r.get('skip') else 'match' if r['reproduced'] else 'fail']+=1
    f.write(json.dumps(r,separators=(',',':'))+'\n')
 print(total,'seconds',time.monotonic()-start,flush=True)
