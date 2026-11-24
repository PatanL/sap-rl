import json,sys,collections as C
from pathlib import Path
sys.path.insert(0,'/audit')
from replay import Pet,Food,Player,Team,ShopSlot
from receipts import load_rows
from ordering import one as order_pairs
from shop_detail import one as detail
from mechanisms import spider_receipt
from sapai.effects import Swallow,SummonStoredPets
import multiprocessing as mp
def write(n,x):Path('handoff',n+'.json').write_text(json.dumps(x,indent=2,sort_keys=True))
def run(row):
 seat,rs,prior=row
 return order_pairs((seat,prior)),order_pairs((seat,rs)),detail((seat,rs))
def sales():
 out=[]
 for name in ('deer','spider','sheep'):
  for level,total in ((1,0),(2,2),(3,5)):
   pet=Pet(name);pet.set_total_experience(total)
   p=Player(team=Team([pet]),record_actions=False)
   pill=Food('sleeping-pill');p.shop.slots=[ShopSlot(pill)];p.buy_food(pill,pet)
   for s in list(p.team.slots):
    if s.empty:continue
    obj=s.pet;before=p.gold;p.sell(obj);out.append([name,level,obj.name,obj.level,obj._attack,obj._health,obj.status,p.gold-before])
 for lvl,exp in [(2,2),(3,5)]:
  whale=Pet('whale');whale.set_total_experience(exp);target=Pet('fish')
  p=Player(team=Team([target,whale]),record_actions=False)
  Swallow(whale,[0,1],[p.team,Team()],fixed_targets=[target]);p.team.remove(target)
  summoned,_=SummonStoredPets(whale,[0,1],[p.team,Team()],te_idx=[0,1])
  for obj in summoned:
   start=p.gold;p.sell(obj);out.append(['whale-release',lvl,obj.name,obj.level,obj._attack,obj._health,obj.status,p.gold-start])
 return out
if __name__=='__main__':
 seats=list(map(json.loads,open('seats.jsonl')))
 def by(label):
  d=C.defaultdict(dict)
  for (m,p,n),r in load_rows(label)[0].items():d[m,p][n]=r
  return d
 b,o=by('v31t'),by('v31')
 with mp.Pool(6) as pool:results=list(pool.imap_unordered(run,[(s,b[s['match'],s['seat']],o[s['match'],s['seat']]) for s in seats]))
 oldsample,current,remaining=[[i for tup in results for i in tup[j]] for j in range(3)]
 def score(s):return dict(pairs=len(s),ordered=sum(r['ids']==r['native'] for r in s),reversed_ordinary=sum(r['ids']!=r['native'] and r['ids'][0]>r['ids'][1] for r in s))
 oldremaining={tuple(r['key']):r for r in map(json.loads,open('remaining-shop.jsonl'))}
 cols=['ids','content','petOrder','foodOrder','fullyOrdered']
 compare={}
 for col in cols:
  counts=C.Counter()
  for r in remaining:
   prev=oldremaining.get(tuple(r['key']))
   if prev:counts[str((prev[col],r[col]))]+=1
  compare[col]=dict(counts)
 write('order',dict(same_seeds=score(oldsample),new_witnesses=score(current),common_turn_comparisons=compare,food_fd=Food('honey').fd,remaining=len(remaining)))
 write('spider',spider_receipt());write('summons',sales())
