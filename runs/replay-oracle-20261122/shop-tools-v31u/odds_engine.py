"""Export probabilities actually used by native ordinary pet rolls."""
import json
from replay import Shop,data
out={}
for tier in range(1,7):
 shop=Shop(slots=[],turn=tier*2-1,fixed_rules=False)
 assert shop.tier_avail==tier
 p={data['pets'][str(k)]['name']:float(v) for k,v in zip(shop.avail_pets,shop.pp)}
 assert abs(sum(p.values())-1)<1e-6;out[tier]=p
json.dump(out,open('engine-odds.json','w'),indent=2)
