"""Compare ordering rules on co-generated normal food offers, using their serialized IDs."""
import collections as C,itertools,json,multiprocessing as mp
from case import details
from receipts import load_rows
from replay import data
TIERS={v['name']:v['tier'] for v in data['foods'].values()}
def one(arg):
 row,rr=arg;out=[]
 for n,r in rr.items():
  if not r.get('completed'):continue
  s=row['states'][str(n)];food=s['food']
  if len(food)!=2 or any(v['name'] not in TIERS for v in food):continue
  d=details(row,n,r['seed']);native=d['run'];log=native['log']
  if native.get('error') or native['genIssues'] or native['unc']!=s['unc']:continue
  old=0;last={}
  for ev in log:
   fresh=ev['gen'][old:];old=len(ev['gen'])
   if ev['op'] in ('roll','start'):last={i:name for i,kind,how,name in fresh if kind=='food' and how=='roll'}
  if {v['id']:v['name'] for v in food}!=last:continue
  x,y=food
  out.append(dict(key=r['key'],split=r['split'],ids=[x['id'],y['id']],names=[x['name'],y['name']],tiers=[TIERS[x['name']],TIERS[y['name']]],
                 frozen=[x['f'],y['f']],native=[v['uid'] for v in native['shopDetail'] if v['kind']=='food']))
 return out
def main():
 by=C.defaultdict(dict)
 for (m,p,n),r in load_rows()[0].items():by[m,p][n]=r
 seats=[json.loads(l) for l in open('seats.jsonl')];pairs=[]
 with mp.Pool(6) as pool:
  for block in pool.imap_unordered(one,[(row,by[row['match'],row['seat']]) for row in seats]):pairs+=block
 out={}
 for sp in ('all','dev','heldout'):
  xs=[x for x in pairs if sp=='all' or x['split']==sp];c=C.Counter()
  for x in xs:
   ids=x['ids'];lo,hi=x['tiers'];c['pairs']+=1;c['native_matches']+=x['native']==ids
   c['id_asc']+=ids==sorted(ids)
   c['tier_desc_id_asc']+=lo>hi or (lo==hi and ids[0]<ids[1])
   c['tier_asc_id_asc']+=lo<hi or (lo==hi and ids[0]<ids[1])
   if lo!=hi:
    c['unequal_tier_pairs']+=1;c['higher_tier_first']+=lo>hi
   else:
    c['equal_tier_pairs']+=1;c['equal_tier_uid_asc']+=ids[0]<ids[1]
   if ids[0]>ids[1]:c['draw_order_reversed']+=1
  out[sp]=c
 json.dump(dict(summary=out,examples=[x for x in pairs if x['native']!=x['ids']][:8]),open('food-order.json','w'),indent=2)
 print(out)
if __name__=='__main__':main()
