import sys,itertools,json
from pathlib import Path
sys.path[:0]=['/engine/sapai',str(Path(__file__).resolve().parent)]
from fixed_slots import rule_put
from sapai import Pet,Team,Food,ShopSlot,Player
out=[]
for deer,sheep,hog in itertools.permutations(range(5),3):
    for high in ('deer','sheep'):
        team=Team();real=[None]*5
        for pos,name in [(deer,'deer'),(sheep,'sheep'),(hog,'hedgehog')]:
            pet=Pet(name);pet._health=1
            if name=='deer':pet.status='status-extra-life'
            if name==high:pet._attack=9
            team[4-pos]=pet
        safe=next(x for x in range(5) if x not in (deer,sheep,hog))
        fish=Pet('fish');fish._health=20;team[4-safe]=fish;real[safe]='fish'
        p=Player(team=team,record_actions=False);f=Food('sleeping-pill');p.shop.slots=[ShopSlot(f)]
        p.buy_food(f,team[4-hog].pet)
        for name in (['deer','sheep'] if high=='deer' else ['sheep','deer']):
            for token in (['bus','deer'] if name=='deer' else ['ram','ram']):rule_put(real,deer if name=='deer' else sheep,token)
        got=[None if slot.empty else slot.pet.name[4:] for slot in reversed(team.slots)]
        out.append(dict(deer=deer,sheep=sheep,hedgehog=hog,higher=high,rule=real,native=got,match=real==got))
Path('shop/simultaneous.json').write_text(json.dumps(out,indent=2))
print(sum(r['match'] for r in out),'/',len(out))
