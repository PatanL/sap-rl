import numpy as np
from sapai import data, status
from sapai.battle import (
    Battle,
    _PhaseLog,
    battle_phase_hurt_and_faint,
)
from sapai.effects import (
    RespawnPet,
    SummonPet,
    SummonRandomPet,
    SummonStoredPets,
    get_effect_function,
    get_target,
)

import sapai.shop
from sapai.shop import Shop, ShopSlot
from sapai.teams import Team, TeamSlot


def storeaction(func):
    def store_action(*args, **kwargs):
        player = args[0]
        action_name = str(func.__name__).split(".")[-1]
        targets = func(*args, **kwargs)
        if not player.record_actions:
            return None
        store_targets = []
        if targets is not None:
            for entry in targets:
                if getattr(entry, "state", False):
                    store_targets.append(entry.state)
        player.action_history.append((action_name, store_targets))

    ### Make sure that the func returned as the same name as input func
    store_action.__name__ = func.__name__

    return store_action


class Player:
    """
    Defines and implements all of the actions that a player can take. In particular
    each of these actions is directly tied to the actions reinforment learning
    models can take.

    Actions with the shop are based off of Objects and not based off of indices.
    There is a huge advantage to doing things this way. The index that a Pet/Food
    is in a shop is arbitrary. Therefore, when actions are based off the Object,
    The ML Agent will not have to learn the index invariances of the shop.

    The Player class is allowed to make appropriate changes to the Shop and
    Team. Therefore, Shops and Teams input into the Player class will not be
    static. The Player class is also responsible for checking all of the
    relevant Pet triggers when taking any action.

    """

    def __init__(
        self,
        shop=None,
        team=None,
        lives=6,
        default_gold=10,
        gold=10,
        turn=1,
        lf_winner=None,
        action_history=None,
        pack="StandardPack",
        seed_state=None,
        wins=0,
        last_battle_outcome=None,
        record_actions=True,
    ):
        action_history = action_history or []

        self.shop = shop
        self.team = team
        self.lives = lives
        self.default_gold = default_gold
        self.gold = gold
        self.pack = pack
        self.turn = turn
        self.wins = wins
        self.last_battle_outcome = last_battle_outcome
        self.record_actions = bool(record_actions)

        ### Default Parameters
        self._max_team = 5

        ### Keep track of outcome of last battle for snail
        self.lf_winner = lf_winner

        ### Initialize shop and team if not provided
        if self.shop is None:
            self.shop = Shop(pack=self.pack, seed_state=seed_state)
        if self.team is None:
            self.team = Team(seed_state=seed_state)

        if type(self.shop) == list:
            self.shop = Shop(self.shop, seed_state=seed_state)
        if type(self.team) == list:
            self.team = Team(self.team, seed_state=seed_state)

        ### Connect objects
        self._connect_objects()

        ### This stores the history of actions taken by the given player
        if len(action_history) == 0:
            self.action_history = []
        else:
            self.action_history = list(action_history)
        self.turn_ended = False

    def _connect_objects(self):
        """Restore Player/Team/Shop ownership after every construction path."""

        self.team.player = self
        self.team.shop = self.shop
        for slot in self.team:
            slot._pet.team = self.team
            slot._pet.player = self
            slot._pet.shop = self.shop
            # One authoritative stream drives every shop-phase random target.
            # Battle copies replace this with their battle-owned stream.
            slot._pet.rs = self.shop.rs
            slot._pet.seed_state = self.shop.rs.get_state()
        for slot in self.shop:
            slot.obj.player = self
            slot.obj.shop = self.shop
            slot.obj.rs = self.shop.rs
            slot.obj.seed_state = self.shop.rs.get_state()

    def _shop_trigger_order(self, pets):
        """Highest attack first; ties consume the player's seeded shop RNG."""

        ordered = []
        by_attack = {}
        for pet in pets:
            by_attack.setdefault(int(pet.attack), []).append(pet)
        for attack in sorted(by_attack, reverse=True):
            tied = by_attack[attack]
            if len(tied) > 1:
                indices = self.shop.rs.choice(
                    np.arange(len(tied)), size=len(tied), replace=False
                )
                tied = [tied[int(index)] for index in indices]
                self.shop._advance_seed_state()
            ordered.extend(tied)
        return ordered

    def _notify_friend_gained_perk(self, gained_pet):
        """Resolve Rabbit-like shop observers in attack-priority order."""

        eligible = [
            slot.pet
            for slot in self.team
            if not slot.empty
            and slot.pet.ability["trigger"]
            in {"GainsPerk", "EatsShopFoodOrGainsPerk"}
        ]
        for observer in self._shop_trigger_order(eligible):
            observer.gains_perk_trigger(gained_pet)

    def _resolve_shop_hurt_and_faint(self):
        """Resolve Pill and cascading shop events with stable pet identities."""

        opponent = Team()

        class ShopEventContext:
            fast_training = True
            update_pet_priority = staticmethod(Battle.update_pet_priority)

            def __init__(context, team, rng):
                context.t0 = team
                context.t1 = opponent
                context.rs = rng
                context._pending_post_removal = []
                context.pet_priority = context.update_pet_priority(
                    context.t0, context.t1, context.rs
                )

        context = ShopEventContext(self.team, self.shop.rs)
        phase_dict = {"shop": _PhaseLog(fast_training=True)}
        battle_phase_hurt_and_faint(
            context,
            "shop",
            [context.t0, context.t1],
            context.pet_priority,
            phase_dict,
        )
        # Summoned shop pets were created after the last ownership/RNG bind.
        self._connect_objects()

    @storeaction
    def start_turn(self, winner=None):
        ### Update turn count and gold
        self.turn_ended = False
        self.turn += 1
        self.gold = self.default_gold
        self.lf_winner = winner

        ### Roll shop
        self.shop.turn = self.turn
        self.shop.update_shop_rules(turn=self.turn)

        # Every roll creates new concrete objects. Reconnect ownership before
        # Canned Food or any pet/food effect can inspect its shop/player.
        self._connect_objects()

        # Temporary stats expire before StartTurn event priority is measured.
        pets = [slot.pet for slot in self.team if not slot.empty]
        for pet in pets:
            pet.prepare_start_turn()

        eligible = [
            pet for pet in pets if pet.ability["trigger"] == "StartOfTurn"
        ]
        for pet in self._shop_trigger_order(eligible):
            pet.sot_trigger(reset_state=False)

        return ()

    @storeaction
    def buy_pet(self, pet, team_idx=None):
        """Buy one pet from the shop"""
        if len(self.team) == self._max_team:
            raise Exception("Attempted to buy Pet on full team")

        if type(pet) == int:
            pet = self.shop[pet]

        if isinstance(pet, ShopSlot):
            pet = pet.obj

        if type(pet).__name__ != "Pet":
            raise Exception(f"Attempted to buy_pet using object {pet}")

        if team_idx is not None:
            if team_idx < 0 or team_idx >= self._max_team:
                raise Exception(f"Invalid team destination {team_idx}")
            if not self.team[team_idx].empty:
                raise Exception(f"Attempted to buy Pet into occupied slot {team_idx}")

        shop_idx = self.shop.index(pet)
        shop_slot = self.shop.slots[shop_idx]
        cost = shop_slot.cost

        if cost > self.gold:
            raise Exception(
                f"Attempted to buy Pet of cost {cost} with only {self.gold} gold"
            )

        ### Connect pet with current Player
        pet.team = self.team
        pet.player = self
        pet.shop = self.shop

        ### Make all updates
        self.gold -= cost
        if team_idx is None:
            self.team.append(pet)
        else:
            self.team[team_idx] = pet
        self.shop.buy(pet)
        # v31: "Summoned: Gain Peanut perk". Real shop Scorpions carry no
        # perk until bought; make the buy grant it regardless of shop state.
        if pet.name == "pet-scorpion":
            pet.status = "status-poison-attack"
        if pet.status in status.PERK_STATUSES:
            self._notify_friend_gained_perk(pet)
        # self.last_bought_pet = pet.name

        ### Check buy_friend triggers after purchase
        for slot in self.team:
            slot._pet.buy_friend_trigger(pet)

        ### Check summon triggers after purchse
        for slot in self.team:
            slot._pet.friend_summoned_trigger(pet)

        return (pet,)

    @storeaction
    def buy_food(self, food, team_pet=None):
        """
        Buy and feed one food from the shop

        team_pet is either the purchase target or empty for food effect target

        """
        if type(food) == int:
            food = self.shop[food]
            if food.slot_type != "food":
                raise Exception("Shop slot not food")
        if isinstance(food, ShopSlot):
            food = food.obj
        if type(food).__name__ != "Food":
            raise Exception(f"Attempted to buy_food using object {food}")

        if team_pet is None:
            targets, _ = get_target(food, [0, None], [self.team])
        else:
            if type(team_pet) == int:
                team_pet = self.team[team_pet]
            if isinstance(team_pet, TeamSlot):
                team_pet = team_pet._pet
            if not self.team.check_friend(team_pet):
                raise Exception(f"Attempted to buy food for Pet not on team {team_pet}")
            if type(team_pet).__name__ != "Pet":
                raise Exception(f"Attempted to buy_pet using object {team_pet}")
            if not team_pet.can_eat(food):
                raise ValueError(
                    f"Cannot feed {food.name} to level {team_pet.level} pet"
                )
            targets = [team_pet]

        shop_idx = self.shop.index(food)
        shop_slot = self.shop.slots[shop_idx]
        cost = shop_slot.cost

        if cost > self.gold:
            raise Exception(
                f"Attempted to buy Pet of cost {cost} with only {self.gold} gold"
            )

        ### Before feeding, check for cat
        for slot in self.team:
            if slot._pet.name != "pet-cat":
                continue
            slot._pet.cat_trigger(food)

        ### Make all updates
        self.gold -= cost
        self.shop.buy(food)
        if food.name == "food-canned-food" and not food.eaten:
            self.shop.shop_attack += food.attack
            self.shop.shop_health += food.health
            food.eaten = True
        for pet in targets:
            previous_level = pet.level
            levelups = pet.eat(food)
            ### Check for levelup triggers if appropriate
            if levelups:
                pet.trigger_levelup_abilities(previous_level)
                self.shop.levelup(levelups)

            ### After feeding, check for eats_shop_food triggers
            # v31: only pets on the team count. Canned Food also feeds shop
            # pets, and those spent Rabbit's three per-turn uses (real
            # replays: the uses were still available for board pets).
            if not self.team.check_friend(pet):
                continue
            for slot in self.team:
                slot._pet.eats_shop_food_trigger(pet)

        ### After feeding, check for buy_food triggers
        for slot in self.team:
            slot._pet.buy_food_trigger()

        self._resolve_shop_hurt_and_faint()

        return (food, targets)

    def check_summon_triggers(self, fainted_pet, pet_idx, activated, targets, possible):
        if not activated:
            return
        func = get_effect_function(fainted_pet)
        if func not in [
            RespawnPet,
            SummonPet,
            SummonRandomPet,
            SummonStoredPets,
        ]:
            return
        flattened_targets = []

        def collect(target):
            if isinstance(target, (list, tuple)):
                for nested in target:
                    collect(nested)
            elif type(target).__name__ == "Pet":
                flattened_targets.append(target)

        collect(targets)
        for temp_te in flattened_targets:
            for temp_slot in self.team:
                temp_pet = temp_slot.pet
                temp_pet.friend_summoned_trigger(temp_te)

    def check_status_triggers(self, fainted_pet, pet_idx):
        if fainted_pet.status not in ["status-honey-bee", "status-extra-life"]:
            return

        # The pet may already have been removed from its TeamSlot. Status
        # summons still resolve relative to that owning shop team/position.
        fainted_pet.team = self.team
        fainted_pet.player = self
        fainted_pet.shop = self.shop
        ability = data["statuses"][fainted_pet.status]["ability"]
        fainted_pet.set_ability(ability)
        te_idx = [0, pet_idx]
        activated, targets, possible = fainted_pet.faint_trigger(fainted_pet, te_idx)
        self.check_summon_triggers(fainted_pet, pet_idx, activated, targets, possible)

    @storeaction
    def sell(self, pet):
        """Sell one pet on the team"""
        if type(pet) == int:
            pet = self.team[pet]

        if isinstance(pet, TeamSlot):
            pet = pet._pet

        if type(pet).__name__ != "Pet":
            raise Exception(f"Attempted to sell Object {pet}")

        # Resolve from a stable pre-sale listener snapshot. The sold pet must
        # leave the board exactly once before target selection, while its own
        # Sell ability and every eligible friend-sold ability must still fire.
        # Previously, any unrelated sell-ability pet encountered first could
        # remove ``pet`` and prevent its own later slot from triggering.
        sell_listeners = [
            slot.pet for slot in self.team if not slot.empty
        ]
        sold_idx = self.team.get_idx(pet)
        self.team.remove(pet)
        for listener in sell_listeners:
            listener.sell_trigger(pet, trigger_idx=sold_idx)

        ### Add the pet's current sell value (Cake can increase it).
        self.gold += pet.sell_value

        return (pet,)

    @storeaction
    def sell_buy(self, team_pet, shop_pet):
        """Sell one team pet and replace it with one shop pet"""
        if type(shop_pet) == int:
            shop_pet = self.shop[shop_pet]
        if type(team_pet) == int:
            team_pet = self.team[team_pet]

        if isinstance(shop_pet, ShopSlot):
            shop_pet = shop_pet.obj
        if isinstance(team_pet, TeamSlot):
            team_pet = team_pet._pet

        if type(shop_pet).__name__ != "Pet":
            raise Exception(f"Attempted sell_buy with Shop item {shop_pet}")
        if type(team_pet).__name__ != "Pet":
            raise Exception(f"Attempted sell_buy with Team Pet {team_pet}")

        ### Activate sell trigger first
        self.sell(team_pet)

        ### Then attempt to buy shop pet
        self.buy_pet(shop_pet)

        return (team_pet, shop_pet)

    def freeze(self, obj):
        """Freeze one pet or food in the shop"""
        if isinstance(obj, ShopSlot):
            obj = obj.obj
            shop_idx = self.shop.index(obj)
        elif type(obj) == int:
            shop_idx = obj
        shop_slot = self.shop.slots[shop_idx]
        # Discover choices are linked only for purchasing. Freeze state is
        # per physical choice: retaining one while rolling its sibling is a
        # core shop-space decision in current P1.
        shop_slot.freeze()
        return (shop_slot,)

    def unfreeze(self, obj):
        """Unfreeze one pet or food in the shop"""
        if isinstance(obj, ShopSlot):
            obj = obj.obj
            shop_idx = self.shop.index(obj)
        elif type(obj) == int:
            shop_idx = obj
        shop_slot = self.shop.slots[shop_idx]
        shop_slot.unfreeze()
        return (shop_slot,)

    @storeaction
    def roll(self):
        """Roll shop"""
        if self.gold < 1:
            raise Exception("Attempt to roll without gold")
        self.shop.roll()
        self._connect_objects()
        self.gold -= 1
        return ()

    @staticmethod
    def combine_pet_stats(
        pet_to_keep, pet_to_be_merged, trigger_levelup_abilities=True
    ):
        """Merge two copy stacks and return the number of discover rewards.

        A pet's absolute experience is the number of extra copies represented
        by its stack. Combining therefore adds both stacks plus one experience
        for joining their base copies. Permanent stats gain once for each new
        experience point beyond the more experienced input stack.
        """

        keep_total = pet_to_keep.total_experience
        merged_total = pet_to_be_merged.total_experience
        keep_level = pet_to_keep.level
        merged_level = pet_to_be_merged.level
        combined_total = keep_total + merged_total + 1
        experience_gained = combined_total - max(keep_total, merged_total)
        previous_level = max(keep_level, merged_level)
        keep_sell_bonus = max(0, pet_to_keep.sell_value - pet_to_keep.level)
        merged_sell_bonus = max(
            0, pet_to_be_merged.sell_value - pet_to_be_merged.level
        )

        c_attack = min(50, (
            max(pet_to_keep._attack, pet_to_be_merged._attack)
            + experience_gained
        ))
        c_until_end_of_battle_attack = max(
            pet_to_keep._until_end_of_battle_attack_buff,
            pet_to_be_merged._until_end_of_battle_attack_buff,
        )
        c_health = min(50, (
            max(pet_to_keep._health, pet_to_be_merged._health)
            + experience_gained
        ))
        c_until_end_of_battle_health = max(
            pet_to_keep._until_end_of_battle_health_buff,
            pet_to_be_merged._until_end_of_battle_health_buff,
        )
        c_until_end_of_battle_attack = min(
            c_until_end_of_battle_attack, max(0, 50 - c_attack)
        )
        c_until_end_of_battle_health = min(
            c_until_end_of_battle_health, max(0, 50 - c_health)
        )
        cstatus = get_combined_status(pet_to_keep, pet_to_be_merged)

        pet_to_keep._attack = c_attack
        pet_to_keep._health = c_health
        pet_to_keep._until_end_of_battle_attack_buff = c_until_end_of_battle_attack
        pet_to_keep._until_end_of_battle_health_buff = c_until_end_of_battle_health
        pet_to_keep.status = cstatus
        pet_to_keep.set_total_experience(combined_total)
        pet_to_keep.sell_value = pet_to_keep.level + max(
            keep_sell_bonus, merged_sell_bonus
        )

        newly_crossed_levels = max(0, pet_to_keep.level - previous_level)
        if newly_crossed_levels and trigger_levelup_abilities:
            pet_to_keep.trigger_levelup_abilities(previous_level)

        # Combining two level-two pets makes a level-three pet but current SAP
        # intentionally does not grant another discover for that shortcut.
        if pet_to_keep.level == 3 and keep_level == merged_level == 2:
            return 0
        return newly_crossed_levels

    @storeaction
    def buy_combine(self, shop_pet, team_pet):
        """Combine two pets on purchase"""
        if type(shop_pet) == int:
            shop_pet = self.shop[shop_pet]
        if type(team_pet) == int:
            team_pet = self.team[team_pet]

        if isinstance(shop_pet, ShopSlot):
            shop_pet = shop_pet.obj
        if isinstance(team_pet, TeamSlot):
            team_pet = team_pet._pet

        if type(shop_pet).__name__ != "Pet":
            raise Exception(f"Attempted buy_combined with Shop item {shop_pet}")
        if type(team_pet).__name__ != "Pet":
            raise Exception(f"Attempted buy_combined with Team Pet {team_pet}")
        if team_pet.name != shop_pet.name:
            raise Exception(
                f"Attempted combine for pets {team_pet.name} and {shop_pet.name}"
            )
        if team_pet.level >= 3:
            raise Exception("Cannot combine into a level 3 pet")

        shop_idx = self.shop.index(shop_pet)
        shop_slot = self.shop.slots[shop_idx]
        cost = shop_slot.cost

        if cost > self.gold:
            raise Exception(
                f"Attempted to buy Pet of cost {cost} with only {self.gold} gold"
            )

        ### Make all updates
        self.gold -= cost
        self.shop.buy(shop_pet)

        levelup_rewards = Player.combine_pet_stats(team_pet, shop_pet)
        if levelup_rewards:
            self.shop.levelup(levelup_rewards)

        ### Check for buy_pet triggers
        for slot in self.team:
            slot._pet.buy_friend_trigger(team_pet)

        return shop_pet, team_pet

    @storeaction
    def combine(self, pet1, pet2):
        """Combine two pets on the team together"""
        if type(pet1) == int:
            pet1 = self.team[pet1]
        if type(pet2) == int:
            pet2 = self.team[pet2]

        if isinstance(pet1, TeamSlot):
            pet1 = pet1._pet
        if isinstance(pet2, TeamSlot):
            pet2 = pet2._pet

        if not self.team.check_friend(pet1):
            raise Exception(f"Attempted combine for Pet not on team {pet1}")
        if not self.team.check_friend(pet2):
            raise Exception(f"Attempted combine for Pet not on team {pet2}")

        if pet1.name != pet2.name:
            raise Exception(f"Attempted combine for pets {pet1.name} and {pet2.name}")
        if pet1.level >= 3 or pet2.level >= 3:
            raise Exception("Cannot combine a level 3 pet")

        previous_level = max(pet1.level, pet2.level)
        levelup_rewards = Player.combine_pet_stats(
            pet1, pet2, trigger_levelup_abilities=False
        )

        ### Remove pet2 from team
        idx = self.team.index(pet2)
        self.team[idx] = TeamSlot()

        # Level-up targeting observes the post-combine board; the consumed
        # source pet is no longer a legal Fish/random-friend target.
        if pet1.level > previous_level:
            pet1.trigger_levelup_abilities(previous_level)
        if levelup_rewards:
            self.shop.levelup(levelup_rewards)

        return pet1, pet2

    @storeaction
    def reorder(self, idx):
        """Reorder team"""
        if len(idx) != len(self.team):
            raise Exception("Reorder idx must match team length")
        unique = np.unique(idx)

        if len(unique) != len(self.team):
            raise Exception(f"Cannot input duplicate indices to reorder: {idx}")

        self.team = Team([self.team[x] for x in idx], seed_state=self.team.seed_state)

        return idx

    @storeaction
    def swap(self, source_idx, target_idx):
        """Move one physical slot to a destination, shifting intervening slots."""

        if source_idx == target_idx:
            raise Exception("Source and target positions must differ")
        if source_idx < 0 or source_idx >= self._max_team:
            raise Exception(f"Invalid source position {source_idx}")
        if target_idx < 0 or target_idx >= self._max_team:
            raise Exception(f"Invalid target position {target_idx}")
        if self.team[source_idx].empty:
            raise Exception(f"Attempted to move empty slot {source_idx}")
        moved = self.team.slots.pop(source_idx)
        self.team.slots.insert(target_idx, moved)
        return (moved,)

    @storeaction
    def end_turn(self):
        """End turn and move to battle phase"""
        eligible = [
            slot.pet
            for slot in self.team
            if not slot.empty
            and (
                slot.pet.ability["trigger"].startswith("EndOfTurn")
                or slot.pet.status in {"status-cake", "status-bread"}
            )
        ]
        for pet in self._shop_trigger_order(eligible):
            pet.eot_trigger()
        self.turn_ended = True
        return None

    @property
    def state(self):
        state_dict = {
            "type": "Player",
            "team": self.team.state,
            "shop": self.shop.state,
            "lives": self.lives,
            "default_gold": self.default_gold,
            "gold": self.gold,
            "lf_winner": self.lf_winner,
            "pack": self.pack,
            "turn": self.turn,
            "action_history": self.action_history,
            "wins": self.wins,
            "last_battle_outcome": self.last_battle_outcome,
            "record_actions": self.record_actions,
        }
        return state_dict

    @classmethod
    def from_state(cls, state):
        team = Team.from_state(state["team"])
        shop_type = state["shop"]["type"]
        shop_cls = getattr(sapai.shop, shop_type)
        shop = shop_cls.from_state(state["shop"])
        if "action_history" in state:
            action_history = state["action_history"]
        else:
            action_history = []
        return cls(
            team=team,
            shop=shop,
            lives=state["lives"],
            default_gold=state["default_gold"],
            gold=state["gold"],
            turn=state["turn"],
            lf_winner=state["lf_winner"],
            pack=state["pack"],
            action_history=action_history,
            wins=state["wins"],
            last_battle_outcome=state.get("last_battle_outcome"),
            record_actions=state.get("record_actions", True),
        )

    def __repr__(self):
        info_str = f"PACK:  {self.pack}\n"
        info_str += f"TURN:  {self.turn}\n"
        info_str += f"LIVES: {self.lives}\n"
        info_str += f"WINS:  {self.wins}\n"
        info_str += f"GOLD:  {self.gold}\n"
        print_str = "--------------\n"
        print_str += "CURRENT INFO: \n--------------\n" + info_str + "\n"
        print_str += "CURRENT TEAM: \n--------------\n" + self.team.__repr__() + "\n"
        print_str += "CURRENT SHOP: \n--------------\n" + self.shop.__repr__()
        return print_str


def get_combined_status(pet1, pet2):
    """
    Statuses are combined based on the tier that they come from.

    """
    status_tier = {
        0: ["status-weak", "status-poison-attack", "none"],
        1: ["status-honey-bee"],
        2: ["status-bone-attack"],
        3: ["status-garlic-armor", "status-cake"],
        4: ["status-splash-attack", "status-bread"],
        5: [
            "status-coconut-shield",
            "status-melon-armor",
            "status-steak-attack",
            "status-extra-life",
        ],
    }

    status_lookup = {}
    for key, value in status_tier.items():
        for entry in value:
            status_lookup[entry] = key

    ### If there is a tie in tier, then pet1 status is used
    max_idx = np.argmax([status_lookup[pet1.status], status_lookup[pet2.status]])

    return [pet1.status, pet2.status][max_idx]
