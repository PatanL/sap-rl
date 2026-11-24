import json, sys
sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "/dev/null", "1", "1"] + sys.argv[3:]
import run_sapai
from sapai import Battle, Pet
import sapai.pets as P
fixture = next(f for f in json.load(open(sys.argv[2]))["fixtures"] if f["battleId"].startswith(sys.argv[6]))
orig = P.Pet.before_attack_trigger
def traced(self, trigger):
    out = orig(self, trigger)
    if self.ability.get("trigger") == "BeforeAttack":
        print("   before_attack", self.name, "activated", out[0], "counter", self.ability_counter, "max", self.ability.get("maxTriggers"), "stats", self.attack, self.health)
    return out
P.Pet.before_attack_trigger = traced
b = Battle(run_sapai.make_team(fixture["player"]), run_sapai.make_team(fixture["opponent"]), seed=1234567)
print([ (s.pet.name, s.pet.ability_counter, s.pet.ability.get("maxTriggers")) for s in b.t1 if not s.empty])
b.start(); i = 0
while b.attack(i):
    i += 1
    print("atk", i, [(s.pet.name[4:], s.pet.attack, s.pet.health) for s in b.t1 if not s.empty][:2])
