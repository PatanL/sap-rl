from rust_boundary import Bridge
from runtime import Battle
from sapai import Pet,Team
api=Bridge();t=Team([Pet('wolverine')]);u=Team([Pet('fish')]);t[0].pet.friends_hurt_counter=2
assert api.battle_class()(t,u,seed=17,fast_training=True).battle()==Battle(t,u,seed=17,fast_training=True).battle()
assert api.fallbacks==1 and api.calls==0
print('Unsupported partial Wolverine counter used the Python reference.')
