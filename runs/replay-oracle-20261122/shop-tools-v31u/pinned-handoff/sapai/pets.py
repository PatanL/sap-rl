# %%
import copy
import numpy as np
from sapai.data import data
from sapai.effects import (
    get_effect_function,
    RespawnPet,
    SummonPet,
    SummonRandomPet,
    SummonStoredPets,
)
from sapai.tiers import pet_tier_lookup
from sapai.rand import MockRandomState
from sapai import status

# %%


class Pet:
    """
    Pet class defines all properties and triggers for Pets during gameplay

    """

    def __init__(
        self, name="pet-none", shop=None, team=None, player=None, seed_state=None
    ):
        if len(name) != 0:
            if not name.startswith("pet-"):
                name = f"pet-{name}"
        self.seed_state = seed_state
        if self.seed_state is not None:
            self.rs = np.random.RandomState()
            self.rs.set_state(self.seed_state)
        else:
            ### Otherwise, set use
            self.rs = MockRandomState()

        self.eaten = False
        self.shop = shop
        self.team = team
        self.player = player

        ### Used only for goat
        self.ability_counter = 0
        self.friends_hurt_counter = 0
        self.sell_value = 1

        self.name = name
        if name not in data["pets"]:
            raise Exception(f"Pet {name} not found")
        fd = data["pets"][name]
        self.fd = fd
        self.override_ability = False
        self.override_ability_dict = {}
        self.swallowed_pets = []
        self.tier = data["pets"][name]["tier"]

        ### Overall stats that should be brought into a battle
        self._attack = fd["baseAttack"]
        self._health = fd["baseHealth"]

        # For tracking buffs that only last until the end of battle
        self._until_end_of_battle_attack_buff = 0
        self._until_end_of_battle_health_buff = 0
        self._hurt = 0
        self.status = "none"
        if "status" in self.fd:
            self.status = self.fd["status"]

        self.level = 1
        self.experience = 0

        #### Add pet to team if not already present
        if self.team is not None:
            if self._attack != "none":
                if self not in team:
                    team.append(self)

            if self.team.shop is None:
                if self.shop is not None:
                    self.team.shop = self.shop

        ### Make sure everything has been initialized together
        if player is not None:
            if self.team is not None:
                player.team = self.team
            if self.shop is not None:
                player.shop = self.shop

    @property
    def attack(self):
        if self._attack == "none":
            return 0
        return min(
            self._attack + self._until_end_of_battle_attack_buff, 50
        )

    @property
    def attack_damage(self):
        """Damage dealt by a normal attack after applying its attack perk."""

        return status.apply_attack_dict[self.status](self.attack)

    @property
    def health(self):
        if self._health == "none":
            return 0
        return min(self._health + self._until_end_of_battle_health_buff, 50)

    @property
    def ability(self):
        if self.override_ability:
            return self.override_ability_dict
        if f"level{self.level}Ability" in self.fd:
            return self.fd[f"level{self.level}Ability"]
        else:
            return empty_ability

    def get_damage(self, value):
        return status.apply_damage_dict[self.status](value)

    def hurt(self, value):
        self._health -= value
        # Fully blocked Melon/Coconut hits are not Hurt events. This also keeps
        # Wolverine's four-friend counter aligned with actual positive damage.
        if value > 0:
            self._hurt += 1

    def set_ability(self, ability_dict):
        self.override_ability = True
        self.override_ability_dict = ability_dict
        return

    def eat(self, food):
        """Returns bool of whether pet levelups"""
        if food.apply_until_end_of_battle:
            self._until_end_of_battle_attack_buff = min(
                self._until_end_of_battle_attack_buff + food.attack,
                max(0, 50 - self._attack),
            )
            self._until_end_of_battle_health_buff = min(
                self._until_end_of_battle_health_buff + food.health,
                max(0, 50 - self._health),
            )
        else:
            self._attack += food.attack
            self._health += food.health
            self._attack = min(self._attack, 50)
            self._health = min(self._health, 50)

        if food.status != "none":
            self.status = food.status
            # Eating any shop food is one food event.  A perk carried by that
            # food is not a second, independent perk-gain event for Rabbit or
            # Seal; Player.buy_food dispatches the single food event below.

        if food.name == "food-chocolate":
            return self.gain_experience()
        elif food.name == "food-sleeping-pill":
            self._health = -1000
        elif food.name == "food-canned-food":
            # Every current shop pet eats the Canned Food effect, but its
            # future-shop modifier is registered exactly once per purchase.
            if not food.eaten:
                food.shop.shop_attack += food.attack
                food.shop.shop_health += food.health
                food.eaten = True
        return False

    def can_eat(self, food):
        """Return whether this pet is a legal target for the given food."""

        food_name = food if isinstance(food, str) else food.name
        return not (food_name == "food-chocolate" and self.level >= 3)

    def init_battle(self):
        self.fhealth = int(self.health)
        self.fattack = int(self.attack)

    def combine(self, pet):
        raise Exception("Combine this pet with another pet")

    def gain_experience(self, amount=1):
        """
        Add permanent experience and return the number of newly crossed levels.

        ``experience`` is stored relative to the current level (0-1 at level
        one, 0-2 at level two). Level-three pets cannot gain more experience or
        the +1/+1 that accompanies experience gain.

        """
        if amount < 0:
            raise ValueError("Experience gain cannot be negative")
        if self.level >= 3:
            return 0
        old_level = self.level
        self._attack = min(50, self._attack + amount)
        self._health = min(50, self._health + amount)
        self.set_total_experience(self.total_experience + amount)
        level_ups = self.level - old_level
        # Base sell value is the pet's level. Cake bonuses already accumulated
        # in sell_value remain on top of that base value.
        self.sell_value += level_ups
        return level_ups

    @property
    def total_experience(self):
        """Experience measured from a fresh level-one copy."""

        if self.level == 1:
            return self.experience
        if self.level == 2:
            return 2 + self.experience
        if self.level == 3:
            return 5 + self.experience
        raise ValueError(f"Invalid pet level {self.level}")

    def set_total_experience(self, total):
        """Set absolute experience without firing abilities or changing stats."""

        total = min(int(total), 5)
        if total < 0:
            raise ValueError("Total experience cannot be negative")
        if total < 2:
            self.level = 1
            self.experience = total
        elif total < 5:
            self.level = 2
            self.experience = total - 2
        else:
            self.level = 3
            self.experience = 0

    def trigger_levelup_abilities(self, previous_level):
        """Fire this pet's ability once for every newly crossed level.

        SAP evaluates the ability at the level being left: level one when the
        pet becomes level two, then level two when it becomes level three.
        """

        final_level = self.level
        for ability_level in range(int(previous_level), final_level):
            self.level = ability_level
            self.levelup_trigger(self)
            self.ability_counter = 0
            self.friends_hurt_counter = 0
        self.level = final_level

    def prepare_start_turn(self):
        """Expire temporary state and reset per-turn counters before ordering."""

        # Parrot's copied ability lasts through one battle and is restored to
        # its native EndOfTurn copy ability at the next shop phase.
        if self.name == "pet-parrot":
            self.override_ability = False
            self.override_ability_dict = {}

        self._until_end_of_battle_attack_buff = 0
        self._until_end_of_battle_health_buff = 0
        self.ability_counter = 0
        self.friends_hurt_counter = 0

    def sot_trigger(self, trigger=None, reset_state=True):
        """
        Apply pet's start of turn ability

        Pets:
            ["dromedary", "swan", "caterpillar", "squirrel", "hatching-chick lvl3"]
        """

        if reset_state:
            self.prepare_start_turn()

        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "StartOfTurn":
            return activated, targets, possible

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = func(self, [0, pet_idx], [self.team], trigger)

        activated = True
        return activated, targets, possible

    def cat_trigger(self, trigger=None):
        """
        Apply pet's shop ability to the given shop when shop is rolled

        Pets:
            ["cat"]
        """
        activated = False
        targets = []
        possible = []
        ### Hard coded because for some reason the cat trigger is hurt but it
        ###   should be roll
        if self.name not in ["pet-cat"]:
            return activated, targets, possible

        if type(trigger).__name__ != "Food":
            raise Exception("Must input purchased food as trigger for cat")

        # Chocolate and perk-only foods have no attack/health component and do
        # not consume either of Cat's two uses.
        if trigger.base_attack == 0 and trigger.base_health == 0:
            return activated, targets, possible

        if self.ability_counter >= 2:
            return activated, targets, possible
        self.ability_counter += 1

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = func(self, [0, pet_idx], [self.team], trigger)

        activated = True
        return activated, targets, possible

    def sell_trigger(self, trigger=None, trigger_idx=None):
        """
        Apply pet's sell ability when a friend (or self) is self

        Pets:
            ["beaver", "duck", "pig", "shrimp", "owl"]
        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "Sell":
            return activated, targets, possible

        if type(trigger).__name__ != "Pet":
            raise Exception("Sell must be triggered by a Pet")

        ### Check if self has been sold is important
        if self.ability["triggeredBy"]["kind"] == "Self":
            if trigger != self:
                return activated, targets, possible

        ### Check if not selling self is important, only for shrimp
        if self.ability["triggeredBy"]["kind"] == "EachFriend":
            if trigger == self:
                return activated, targets, possible

        # Player.sell supplies a stable snapshot and removes the sold pet
        # once before dispatch. Preserve backwards-compatible direct calls by
        # performing that removal here only when no trigger index was given.
        if trigger_idx is None and self.team.check_friend(trigger):
            trigger_idx = self.team.get_idx(trigger)
            self.team.remove(trigger)

        if self is trigger:
            # The sold pet is no longer present, so target geometry uses its
            # original physical position.
            pet_idx = 0 if trigger_idx is None else int(trigger_idx)
        else:
            pet_idx = self.team.get_idx(self)

        func = get_effect_function(self)

        targets, possible = func(self, [0, pet_idx], [self.team], trigger)

        activated = True
        return activated, targets, possible

    def eats_shop_food_trigger(self, trigger=None):
        """
        Apply pet's ability when food is eaten

        Pets:
            ["beetle", "tabby-cat", "rabbit", "worm", "seal"]

        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] not in {
            "EatsShopFood",
            "EatsShopFoodOrGainsPerk",
        }:
            return activated, targets, possible

        if type(trigger).__name__ != "Pet":
            raise Exception("Buy food must input pet that ate as trigger")

        ### Check if food has been bought for self is important
        if self.ability["triggeredBy"]["kind"] == "Self":
            if trigger != self:
                return activated, targets, possible

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            self.ability_counter += 1

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = func(self, [0, pet_idx], [self.team], trigger)

        activated = True
        return activated, targets, possible

    def gains_perk_trigger(self, trigger=None):
        """Apply current Turtle Rabbit after a friendly gains a perk."""

        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] not in {
            "GainsPerk",
            "EatsShopFoodOrGainsPerk",
        }:
            return activated, targets, possible
        if type(trigger).__name__ != "Pet":
            raise Exception("Gained perk trigger must be a Pet")
        if self.team is None or not self.team.check_friend(trigger):
            return activated, targets, possible
        if (
            self.ability["triggeredBy"]["kind"] == "Self"
            and trigger is not self
        ):
            return activated, targets, possible
        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            self.ability_counter += 1
        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = func(
            self, [0, pet_idx], [self.team], trigger
        )
        return True, targets, possible

    def buy_food_trigger(self, trigger=None):
        """
        Apply pet's ability when food is bought

        Pets:
            ["ladybug", "sauropod"]

        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "BuyFood":
            return activated, targets, possible

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = func(self, [0, pet_idx], [self.team], trigger)

        activated = True
        return activated, targets, possible

    def buy_friend_trigger(self, trigger=None):
        """
        Apply pet's ability when a friend (or self) is bought

        Pets:
            ["otter", "crab", "snail", "buffalo", "chicken", "cow",
             "goat", "dragon", ]
        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] not in ["Buy", "BuyAfterLoss", "BuyTier1Animal"]:
            return activated, targets, possible

        if type(trigger).__name__ != "Pet":
            raise Exception("Buy food must input food target as triggered")

        ### Behavior for bought self and friend
        if self.ability["trigger"] == "Buy":
            if self.ability["triggeredBy"]["kind"] == "Self":
                if trigger != self:
                    return activated, targets, possible
            elif self.ability["triggeredBy"]["kind"] == "Player":
                ### Behavior for kind=self and kind=player is actually the
                ###   same and any distinction is unnecessary
                if trigger != self:
                    return activated, targets, possible
            elif self.ability["triggeredBy"]["kind"] == "EachFriend":
                ### If trigger is EachFriend, then the trigger cannot actually
                ###   be self
                if trigger == self:
                    return activated, targets, possible
            else:
                raise Exception(f"Ability unrecognized for {self}")

        ### Behavior for BuyTier1Animal
        if self.ability["trigger"] == "BuyTier1Animal":
            if trigger.name not in pet_tier_lookup[1]:
                return activated, targets, possible

        ### Behavior for BuyAfterLoss
        if self.ability["trigger"] == "BuyAfterLoss":
            if self.player is None:
                return activated, targets, possible
            if self.player.lf_winner != False:
                return activated, targets, possible

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = func(self, [0, pet_idx], [self.team], trigger)

        activated = True
        return activated, targets, possible

    def friend_summoned_trigger(self, trigger=None):
        """
        Apply pet's ability when a friend is summoned

        Pets:
            ["horse", "dog", "lobster", "turkey"]
        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "Summoned":
            return activated, targets, possible

        if type(trigger).__name__ != "Pet":
            raise Exception("Trigger must be a Pet")

        if trigger == self:
            ### Do not activate for summoning self
            return activated, targets, possible

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = tiger_func(
            func, False, self, [0, pet_idx], [self.team], trigger
        )

        activated = True
        return activated, targets, possible

    def levelup_trigger(self, trigger=None):
        """
        Apply pet's ability when a friend (or self) level-up

        Pets:
            ["fish", "octopus"]
        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "LevelUp":
            return activated, targets, possible

        if type(trigger).__name__ != "Pet":
            raise Exception("Trigger must be a Pet")

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = func(self, [0, pet_idx], [self.team], te=trigger)

        activated = True
        return activated, targets, possible

    def eot_trigger(self, trigger=None):
        """
        Apply pet's end-of-turn ability

        Pets:
            ["bluebird", "hatching-chick", "giraffe (start of turn)", "puppy", "tropical-fish",
             "bison", "llama", "penguin", "parrot", "monkey", "poodle",
             "tyrannosaurus", snail]
        """
        activated = False
        targets = []
        possible = []

        # Shop perks are independent from the pet's native ability.
        if self.status == "status-cake":
            self.sell_value += 1
        elif self.status == "status-bread":
            self._until_end_of_battle_health_buff = min(
                self._until_end_of_battle_health_buff + 7,
                max(0, 50 - self._health),
            )

        if not self.ability["trigger"].startswith("EndOfTurn"):
            return activated, targets, possible

        ### Check gold for puppy and tyrannosaurus
        if self.ability["trigger"] == "EndOfTurnWith3PlusGold":
            if self.player is not None:
                if self.player.gold >= 3:
                    pass
                else:
                    return activated, targets, possible
            else:
                return activated, targets, possible
        ### Check for bison
        elif self.ability["trigger"] == "EndOfTurnWithLvl3Friend":
            if self.team is not None:
                has_level_three_friend = any(
                    not slot.empty
                    and slot.pet is not self
                    and slot.pet.level == 3
                    for slot in self.team
                )
                if not has_level_three_friend:
                    return activated, targets, possible
            else:
                return activated, targets, possible
        ### Check for llama
        elif self.ability["trigger"] == "EndOfTurnWith4OrLessAnimals":
            if self.team is not None:
                if len(self.team) > 4:
                    return activated, targets, possible
            else:
                return activated, targets, possible
        elif self.ability["trigger"] == "EndOfTurnAfterLoss":
            if self.player is None:
                return activated, targets, possible
            if self.player.lf_winner != False:
                return activated, targets, possible
        else:
            if self.ability["trigger"] != "EndOfTurn":
                raise Exception(f"Unrecognized trigger {self.ability['trigger']}")

        # SAP-Calculator: only the first eligible Bison on a team works.
        if self.name == "pet-bison":
            eligible_bisons = []
            living_pets = [slot.pet for slot in self.team if not slot.empty]
            for candidate in living_pets:
                if candidate.name != "pet-bison":
                    continue
                if any(
                    friend is not candidate and friend.level == 3
                    for friend in living_pets
                ):
                    eligible_bisons.append(candidate)
            if not eligible_bisons or eligible_bisons[0] is not self:
                return activated, targets, possible

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = func(self, [0, pet_idx], [self.team], te=trigger)

        activated = True
        return activated, targets, possible

    def faint_trigger(
        self,
        trigger=None,
        te_idx=None,
        oteam=None,
        remove_before_effect=True,
        tiger_override=None,
    ):
        """
        Apply pet's ability associated with a friend (or self) fainting

        Pets:
            ["ant", "cricket", "flamingo", "hedgehog", "spider", "badger",
             "ox", "sheep", "turtle", "deer", "rooster", "microbe",
             "eagle", "shark", "fly", "mammoth"]
        """
        te_idx = te_idx or []

        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "Faint":
            return activated, targets, possible

        if type(trigger).__name__ != "Pet":
            raise Exception("Trigger must be a Pet")

        if len(te_idx) == 0:
            raise Exception("Index of triggering entity must be input")

        if (
            self.name in {"pet-shark", "pet-fly"}
            and self.health <= 0
            and self.ability["triggeredBy"]["kind"] == "EachFriend"
        ):
            # A dead Shark/Fly cannot observe somebody else's faint. Do not
            # block its temporary Honey/Mushroom Self override, which must
            # still summon after this pet is removed.
            return False, targets, possible

        if self.ability["triggeredBy"]["kind"] == "Self":
            if trigger != self:
                return activated, targets, possible
        elif self.ability["triggeredBy"]["kind"] == "FriendAhead":
            pet_ahead = self.team.get_ahead(self, n=1)
            if len(pet_ahead) == 0:
                return activated, targets, possible
            pet_ahead = pet_ahead[0]
            if trigger != pet_ahead:
                return activated, targets, possible
        elif self.ability["triggeredBy"]["kind"] == "EachFriend":
            if trigger == self or trigger.team is not self.team:
                # EachFriend never observes an enemy faint. The triggering pet
                # retains its owning Team after removal, so identity remains
                # available for post-removal Shark/Fly processing.
                return activated, targets, possible
        else:
            pass

        ### Check for Fly
        if self.name == "pet-fly":
            # print("fly activated2, trigger: ", trigger)
            # print("trigger.team: ", trigger.team)
            # print("te_idx: ", te_idx)
            try:
                te_idx[1] = self.team.get_idx(trigger)
            except:
                te_idx[1] = te_idx[1]
            # print("te_idx: ", te_idx)
            if trigger.name == "pet-zombie-fly":
                ### Do not activate if another zombie-fly faints
                return activated, targets, possible

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        func = get_effect_function(self)
        if self.team.check_friend(self):
            pet_idx = self.team.get_idx(self)
        else:
            pet_idx = te_idx

        if oteam is not None:
            teams = [self.team, oteam]
        else:
            teams = [self.team]

        if func in [RespawnPet, SummonPet, SummonRandomPet, SummonStoredPets]:
            ### API for SummonPet is slightly different
            targets, possible = tiger_func(
                func,
                remove_before_effect,
                self,
                [0, pet_idx],
                teams,
                trigger,
                te_idx,
                tiger_override=tiger_override,
            )
            if possible == [[]]:
                if "maxTriggers" in self.ability:
                    self.ability_counter -= 1
                return activated, targets, possible
        else:
            targets, possible = tiger_func(
                func,
                remove_before_effect,
                self,
                [0, pet_idx],
                teams,
                trigger,
                tiger_override=tiger_override,
            )
        # print(f"Faint trigger activated for {self.name}")
        # print(f"Targets: {targets}")
        # print(f"Possible: {possible}")

        activated = True
        return activated, targets, possible

    def sob_trigger(self, trigger):
        """
        Start of a battle trigger. Input trigger is the opponent's Team.

        Pets:
            ["mosquito", "bat", "whale", "dolphin", "skunk", "crocodile",
            "leopard", "caterpillar lvl3"]

        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "StartOfBattle":
            return activated, targets, possible

        if type(trigger).__name__ != "Team":
            raise Exception("Trigger must be a Team")

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = tiger_func(
            func, False, self, [0, pet_idx], [self.team, trigger], trigger
        )

        activated = True
        return activated, targets, possible

    def before_attack_trigger(self, trigger):
        """
        Apply pet's ability before attacking. Input trigger is the
        opponent's Team.

        Pets:
            ["elephant", "boar", "octopus"]
        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "BeforeAttack":
            return activated, targets, possible

        if type(trigger).__name__ != "Team":
            raise Exception("Trigger must be a Team")

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = tiger_func(
            func, False, self, [0, pet_idx], [self.team, trigger], trigger
        )

        activated = True
        return activated, targets, possible

    def after_attack_trigger(self, trigger):
        """
        Apply pet's ability after attacking. Input trigger is the
        opponent's Team.

        Pets:
            ["kangaroo","snake", "elephant]

        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "AfterAttack":
            return activated, targets, possible

        if type(trigger).__name__ != "Team":
            raise Exception("Trigger must be a Team")

        triggered_by = self.ability["triggeredBy"]["kind"]
        if triggered_by == "FriendAhead":
            # The only normal attacker is the front pet. The activating pet
            # must therefore be directly behind that attacker.
            slot_ahead = self.team.get_ahead(self, n=1)
            if len(slot_ahead) == 0:
                return activated, targets, possible
            if self.team.index(slot_ahead[0]) != 0:
                return activated, targets, possible
        elif triggered_by == "Self":
            # Battle dispatch only calls this branch for the pet that attacked
            # (current Turtle case: Elephant).
            pass
        else:
            raise ValueError(
                f"Unsupported AfterAttack trigger source {triggered_by}"
            )

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        ### Otherwise, the pet ahead is the pet the just attacked and the
        ###   ability after attack should be activated
        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = tiger_func(
            func, False, self, [0, pet_idx], [self.team, trigger], trigger
        )

        activated = True
        return activated, targets, possible

    def hurt_trigger(self, trigger):
        """
        Apply pet's ability after being hurt attacking. Input trigger is the
        opponent's Team. Only activate hurt trigger if the pet has health above
        0.

        There is no way to test if hurt_trigger should be activated within this
        function. Therefore, only call hurt trigger where appropriate during
        battle and shop phase.

        Pets:
            ["peacock", "blowfish", "camel", "gorilla"]

        """
        activated = False
        targets = []
        possible = []
        if self._hurt == 0:
            raise Exception("Called hurt trigger on pet that was not hurt")
        else:
            self._hurt -= 1

        if self.ability["trigger"] != "Hurt":
            return activated, targets, possible

        if type(trigger).__name__ != "Team":
            raise Exception("Trigger must be a Team")

        if self.ability["triggeredBy"]["kind"] == "Self":
            pass
        else:
            raise Exception("Only Self trigger available for hurt_trigger")

        ### Cannot call if health is less than zero because fainted
        # if self._health <= 0:
        #     return activated, targets, possible

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = tiger_func(
            func, False, self, [0, pet_idx], [self.team, trigger], trigger
        )

        activated = True
        return activated, targets, possible

    def friend_hurt_trigger(self, trigger, opponent_team):
        """Activate current Wolverine after every fourth friendly hurt event."""

        if self.ability["trigger"] != "FourFriendsHurt":
            return False, [], []
        if trigger is self or not self.team.check_friend(self):
            return False, [], []
        self.friends_hurt_counter += 1
        if self.friends_hurt_counter < 4:
            return False, [], []
        self.friends_hurt_counter = 0
        func = get_effect_function(self)
        pet_idx = self.team.get_idx(self)
        targets, possible = tiger_func(
            func, False, self, [0, pet_idx], [self.team, opponent_team], trigger
        )
        return True, targets, possible

    def knockout_trigger(self, trigger, tiger_override=None):
        """
        Apply pet's ability after knockout on opponent. Input trigger is the
        opponent's Team. Only activate trigger if the pet has health above 0.

        There is no way to test if knockout_trigger should be activated within
        this function. Therefore, only call knockout_trigger where appropriate
        during the battle phase.

        Pets:
            ["hippo", "rhino"]
        """
        activated = False
        targets = []
        possible = []
        if self.ability["trigger"] != "KnockOut":
            return activated, targets, possible

        if type(trigger).__name__ != "Team":
            raise Exception("Trigger must be a Team")

        if not self.team.check_friend(self):
            target_spec = self.ability.get("effect", {}).get(
                "target",
                self.ability.get("effect", {}).get("to", {}),
            )
            # A removed Hippo cannot buff itself back to life. Rhino and
            # other outward-facing queued KnockOut effects still resolve.
            if target_spec.get("kind") == "Self":
                return activated, targets, possible

        ### Cannot call if health is less than zero because fainted
        # if self._health <= 0:
        #     return activated, targets, possible

        if "maxTriggers" in self.ability:
            if self.ability_counter >= self.ability["maxTriggers"]:
                return activated, targets, possible
            else:
                self.ability_counter += 1

        func = get_effect_function(self)
        try:
            pet_idx = self.team.get_idx(self)
        except:
            pet_idx = 0
        targets, possible = tiger_func(
            func,
            False,
            self,
            [0, pet_idx],
            [self.team, trigger],
            trigger,
            tiger_override=tiger_override,
        )

        activated = True
        return activated, targets, possible

    def __repr__(self):
        return f"< {self.name} {self.attack}-{self.health} {self.status} {self.level}-{self.experience} >"

    def copy(self):
        copy_pet = Pet(self.name, self.shop, seed_state=self.seed_state)
        for key, value in self.__dict__.items():
            if key in {"override_ability_dict", "swallowed_pets"}:
                copy_pet.__dict__[key] = copy.deepcopy(value)
            else:
                copy_pet.__dict__[key] = value
        return copy_pet

    @property
    def state(self):
        #### Cannot get state for attached objects such as shop, team, or player
        ####   as this will lead to circular logic. Therefore, state should be
        ####   saved at the Player level if all info is desired.

        #### Ensure that state can be JSON serialized
        if getattr(self, "rs", False):
            if isinstance(self.rs, MockRandomState):
                seed_state = None
            else:
                seed_state = list(self.rs.get_state())
                seed_state[1] = seed_state[1].tolist()
        else:
            seed_state = None

        state_dict = {
            "type": "Pet",
            "name": self.name,
            "eaten": False,
            "shop": {},
            "team": {},
            "player": {},
            "ability_counter": self.ability_counter,
            "friends_hurt_counter": self.friends_hurt_counter,
            "sell_value": self.sell_value,
            "override_ability": self.override_ability,
            "override_ability_dict": self.override_ability_dict,
            "swallowed_pets": self.swallowed_pets,
            "attack": self._attack,
            "health": self._health,
            "until_end_of_battle_attack_buff": self._until_end_of_battle_attack_buff,
            "until_end_of_battle_health_buff": self._until_end_of_battle_health_buff,
            "status": self.status,
            "level": self.level,
            "experience": self.experience,
            "seed_state": seed_state,
        }

        return state_dict

    @classmethod
    def from_state(cls, state):
        name = state["name"]
        ### Initialize and reset defaults by hand
        pet = cls(name)
        pet.store = None
        pet.team = None
        pet.player = None

        ### Set internal from state
        pet.ability_counter = state["ability_counter"]
        pet.friends_hurt_counter = state.get("friends_hurt_counter", 0)
        pet.sell_value = state.get("sell_value", 1)
        pet.override_ability = state["override_ability"]
        pet.override_ability_dict = state["override_ability_dict"]
        pet.swallowed_pets = copy.deepcopy(state.get("swallowed_pets", []))
        pet._attack = state["attack"]
        pet._health = state["health"]
        pet._until_end_of_battle_attack_buff = state["until_end_of_battle_attack_buff"]
        pet._until_end_of_battle_health_buff = state["until_end_of_battle_health_buff"]
        pet.status = state["status"]
        pet.level = state["level"]
        pet.experience = state["experience"]

        ### Supply seed_state in state dict should be optional
        if "seed_state" in state:
            if state["seed_state"] is not None:
                pet.seed_state = state["seed_state"]
                pet.rs = np.random.RandomState()
                pet.rs.set_state(pet.seed_state)

        return pet


def tiger_func(func, te_fainted, *args, tiger_override=None):
    ### Check behind for tiger
    apet = args[0]
    if apet.team is None:
        ### Just run function
        targets, possible = func(*args)
        return targets, possible

    ### If checked on Bee status, then not repeated because this is a status
    ###   and not an ability
    if "pet" in apet.ability["effect"]:
        if apet.ability["effect"]["pet"] == "pet-bee":
            targets, possible = func(*args)
            return targets, possible
    ### Mushroom
    if "kind" in apet.ability["effect"]:
        if apet.ability["effect"]["kind"] in {
            "RespawnPet",
            "SummonStoredPets",
        }:
            targets, possible = func(*args)
            return targets, possible

    ### Capture Tiger support at event creation. Knock Out can resolve after
    # the source has fainted and left the team; SAP-Calculator carries that
    # original support pet on the queued event.
    if tiger_override is not None:
        pet_behind = tiger_override
        tiger_was_captured = True
    elif apet.team.check_friend(apet):
        behind_slots = apet.team.get_behind(apet, n=1)
        pet_behind = behind_slots[0].pet if behind_slots else None
        tiger_was_captured = False
    else:
        pet_behind = None
        tiger_was_captured = False
    ability_before_trigger = copy.deepcopy(apet.ability)
    override_before_trigger = apet.override_ability

    ### If the triggering entity is supposed to have fainted, remove it now
    if te_fainted:
        if apet.team.check_friend(args[3]):
            apet.team.remove(args[3])

    ### Run function
    targets, possible = func(*args)

    ### If not in a battle, then tiger doesnt trigger
    if not apet.team.battle:
        return targets, possible

    ### If there's no pet behind, then return
    if pet_behind is None:
        return targets, possible

    ### Check if tiger is behind
    if pet_behind.name != "pet-tiger":
        return targets, possible
    if not tiger_was_captured and pet_behind.health <= 0:
        ### If tiger died and has been removed don't run function again
        if not apet.team.check_friend(pet_behind):
            return targets, possible

    post_override = apet.override_ability
    post_override_dict = copy.deepcopy(apet.override_ability_dict)
    # The repeated ability resolves at the Tiger's level. Native abilities are
    # re-read at that level (important for Whale and multi-hit abilities),
    # while Parrot's copied source is reconstructed at Tiger level.
    original_level = apet.level
    apet.level = pet_behind.level
    try:
        copied_from = ability_before_trigger.get("_copiedFrom")
        if copied_from is not None:
            apet.override_ability = True
            apet.override_ability_dict = copy.deepcopy(
                data["pets"][copied_from][
                    f"level{pet_behind.level}Ability"
                ]
            )
        elif override_before_trigger:
            apet.override_ability = True
            apet.override_ability_dict = ability_before_trigger
        else:
            apet.override_ability = False
        repeat_func = get_effect_function(apet)
        if len(args) == 5:
            te_idx = [0, args[4][1] + len(targets)]
            temp_targets, temp_possible = repeat_func(
                args[0], args[1], args[2], args[3], te_idx
            )
        else:
            temp_targets, temp_possible = repeat_func(*args)
    finally:
        apet.level = original_level
        apet.override_ability = post_override
        apet.override_ability_dict = post_override_dict

    return [targets] + [temp_targets], [possible] + [temp_possible]


empty_ability = {
    "description": "none",
    "trigger": "none",
    "triggeredBy": {"kind": "none", "n": "none"},
    "effect": {
        "kind": "none",
        "attackAmount": "none",
        "healthAmount": "none",
        "target": {"kind": "none", "n": "none", "includingFuture": "none"},
        "untilEndOfBattle": "none",
        "pet": "none",
        "withAttack": "none",
        "withHealth": "none",
        "team": "none",
        "amount": "none",
        "status": "none",
        "to": {"kind": "none", "n": "none"},
        "copyAttack": "none",
        "copyHealth": "none",
        "from": {"kind": "none", "n": "none"},
        "effects": "none",
        "tier": "none",
        "baseAttack": "none",
        "baseHealth": "none",
        "percentage": "none",
        "shop": "none",
        "food": "none",
        "level": "none",
    },
    "maxTriggers": "none",
}

# %%
