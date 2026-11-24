import json
from sapai import Pet,Team,Player,Food,ShopSlot,Battle
def run():
 p=Player(team=Team([Pet('fish'),Pet('rabbit')]),record_actions=False)
 rabbit=p.team[1].pet
 def eat(name,obj):
  f=Food(name);p.shop.slots=[ShopSlot(f)];p.buy_food(f,obj)
 eat('apple',p.team[0].pet);eat('apple',p.team[0].pet)
 out={'after_two_apples':rabbit.ability_counter}
 p.end_turn();out['after_END']=rabbit.ability_counter
 b=Battle(p.team,Team([Pet('fish')]),seed=42,fast_training=True)
 out['battle_constructor']=b.t0[1].pet.ability_counter;b.start();out['battle_start']=b.t0[1].pet.ability_counter
 Battle(p.team,Team([Pet('fish')]),seed=42,fast_training=True).battle()
 out['original_after_battle']=rabbit.ability_counter;p.start_turn();out['next_shop']=rabbit.ability_counter
 # Reproduce mid-turn level-up reset (the triggering chocolate can subsequently spend a new use).
 rabbit.set_total_experience(1);eat('apple',p.team[0].pet);eat('apple',p.team[0].pet);out['before_levelup']=rabbit.ability_counter
 eat('chocolate',rabbit);out['after_levelup_chocolate']=rabbit.ability_counter
 return out
if __name__=='__main__':
 r=run();print(r);json.dump(r,open('handoff/lifecycle.json','w'),indent=2)
