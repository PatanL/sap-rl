"""Actual native shop purchase vs independent fixed-slot rule over every five-square layout."""
import sys,os,json,itertools as it,collections as C
from pathlib import Path
sys.path.insert(0,'/engine/sapai')
from sapai import Pet,Team,Food,ShopSlot,Player
from sapai.shop import pet_tier_lookup_std

def rule_put(board,x,token):
    if board[x] is None:board[x]=token;return
    # Coordinates here are real replay x, with x=4 at the front.
    for direction in (1,-1):
        y=x
        while 0<=y<5 and board[y] is not None:y+=direction
        if 0<=y<5:
            while y!=x:board[y]=board[y-direction];y-=direction
            board[x]=token;return
def main():
    tally=C.defaultdict(C.Counter);errors=[]
    for source,perk,x,mask in it.product(['deer','sheep','spider','fish'],['none','status-extra-life'],range(5),range(16)):
        if source=='fish' and perk=='none':continue
        members=[None]*5;team=Team();pet=Pet(source);pet.status=perk
        team[4-x]=pet;members[x]='oldsource'
        for bit,y in enumerate(k for k in range(5) if k!=x):
            if mask&(1<<bit):
                other=Pet('ant');other.marker=f'old{y}';team[4-y]=other;members[y]=other.marker
        player=Player(team=team,record_actions=False);food=Food('sleeping-pill')
        player.shop.slots=[ShopSlot(food)];player.buy_food(food,pet)
        native=[]
        for slot in reversed(team.slots):
            if slot.empty:native.append(None);continue
            nm=slot.pet.name[4:]
            if source=='spider' and slot.pet.name in pet_tier_lookup_std[3]:nm='tier3'
            native.append(getattr(slot.pet,'marker',nm))
        expected=list(members);expected[x]=None
        for token in {'sheep':['ram','ram'],'deer':['bus'],'spider':['tier3'],'fish':[]}[source]+([source] if perk!='none' else []):rule_put(expected,x,token)
        label=source+(' / Mushroom' if perk!='none' else '')
        tally[label]['tested']+=1;tally[label]['exact']+=expected==native
        if expected!=native:errors.append(dict(source=source,perk=perk,x=x,previous=members,real_rule=expected,native=native))
    Path('shop/fixed-slots.json').write_text(json.dumps(dict(counts=tally,misses=errors),indent=2))
    print(tally)
if __name__=='__main__':main()
