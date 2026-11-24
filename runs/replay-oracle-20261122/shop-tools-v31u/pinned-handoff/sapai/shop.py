# %%

from functools import partial
import numpy as np

from sapai.data import data
from sapai.lists import Slot, SAPList
from sapai.foods import Food
from sapai.pets import Pet
import sapai.foods
import sapai.pets
from sapai.rand import MockRandomState

# %%

MAX_SHOP_SLOTS = 10

################################################################################
#### Building optimized datastructures for construction shop states
################################################################################

pet_tier_lookup = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
pet_tier_lookup_std = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
for key, value in data["pets"].items():
    if value.get("hidden", False):
        continue
    if type(value["tier"]) == int:
        pet_tier_lookup[value["tier"]].append(key)
        if "StandardPack" in value["packs"]:
            pet_tier_lookup_std[value["tier"]].append(key)
pet_tier_avail_lookup = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
for key, value in pet_tier_lookup.items():
    for temp_key, temp_value in pet_tier_avail_lookup.items():
        if temp_key >= key:
            temp_value += value

food_tier_lookup = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
food_tier_lookup_std = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
for key, value in data["foods"].items():
    if type(value["tier"]) == int:
        food_tier_lookup[value["tier"]].append(key)
        if "StandardPack" in value.get("packs", []):
            food_tier_lookup_std[value["tier"]].append(key)
food_tier_avail_lookup = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
for key, value in food_tier_lookup.items():
    for temp_key, temp_value in food_tier_avail_lookup.items():
        if temp_key >= key:
            temp_value += value

turn_prob_pets_std = {}
turn_prob_pets_exp = {}
for i in np.arange(0, 12):
    turn_prob_pets_std[i] = {}
    turn_prob_pets_exp[i] = {}
for key, value in data["pets"].items():
    if "probabilities" not in value:
        continue
    if data["pets"][key]["probabilities"] == "none":
        continue
    for temp_dict in data["pets"][key]["probabilities"]:
        temp_turn = int(temp_dict["turn"].split("-")[-1])
        if "StandardPack" in temp_dict["perSlot"]:
            temp_std = temp_dict["perSlot"]["StandardPack"]
            turn_prob_pets_std[temp_turn][key] = temp_std
        if "ExpansionPack1" in temp_dict["perSlot"]:
            temp_exp = temp_dict["perSlot"]["ExpansionPack1"]
            turn_prob_pets_exp[temp_turn][key] = temp_exp
        else:
            ### Assumption, if expansion info not provided, use standard info
            temp_exp = temp_std
            turn_prob_pets_exp[temp_turn][key] = temp_exp

turn_prob_foods_std = {}
turn_prob_foods_exp = {}
for i in np.arange(0, 12):
    turn_prob_foods_std[i] = {}
    turn_prob_foods_exp[i] = {}
for key, value in data["foods"].items():
    if "probabilities" not in value:
        continue
    if data["foods"][key]["probabilities"] == "none":
        continue
    for temp_dict in data["foods"][key]["probabilities"]:
        if temp_dict == "none":
            continue
        temp_turn = int(temp_dict["turn"].split("-")[-1])
        if "StandardPack" in temp_dict["perSlot"]:
            temp_std = temp_dict["perSlot"]["StandardPack"]
            turn_prob_foods_std[temp_turn][key] = temp_std
        if "ExpansionPack1" in temp_dict["perSlot"]:
            temp_exp = temp_dict["perSlot"]["ExpansionPack1"]
            turn_prob_foods_exp[temp_turn][key] = temp_exp
        else:
            ### Assumption, if expansion info not provided, use standard info
            temp_exp = temp_std
            turn_prob_foods_exp[temp_turn][key] = temp_exp

# %%

# %%


class Shop(SAPList):
    def __init__(
        self,
        slots=None,
        turn=1,
        shop_attack=0,
        shop_health=0,
        pack="StandardPack",
        seed_state=None,
        fixed_rules=None,
        next_reward_group_id=1,
        paired_tape=None,
        paired_tape_seat=None,
    ):
        provided_slots = slots is not None
        slots = slots or []

        #### Setting up random state
        self.seed_state = seed_state
        if self.seed_state is not None:
            self.rs = np.random.RandomState()
            self.rs.set_state(self.seed_state)
        else:
            self.rs = MockRandomState()

        # Optional private simulator backend; absent for legacy execution.
        from sapai.paired_random_tape import RandomTape
        if (paired_tape is None) != (paired_tape_seat is None):
            raise ValueError("Tape and seat must be restored together")
        if paired_tape_seat is not None and (type(paired_tape_seat) is not int or paired_tape_seat not in (0, 1)):
            raise ValueError("Invalid tape seat")
        self._paired_tape = None if paired_tape is None else RandomTape.from_private_state(paired_tape)
        self._paired_tape_seat = paired_tape_seat

        # Current UI/action contract has ten physical shop slots. Pigeon stock
        # and linked level-up rewards share this hard capacity.
        self.max_slots = MAX_SHOP_SLOTS
        self.next_reward_group_id = int(next_reward_group_id)
        self.turn = turn
        self.pack = pack
        self.shop_attack = shop_attack  ### Keep track of can/chicken stats
        self.shop_health = shop_health  ### Keep track of can/chicken stats
        slot_class = partial(
            ShopSlot, pack=self.pack, turn=self.turn, seed_state=self.seed_state
        )

        if pack == "StandardPack":
            self.turn_prob_pets = turn_prob_pets_std
            self.turn_prob_foods = turn_prob_foods_std
        elif pack == "ExpansionPack1":
            self.turn_prob_pets = turn_prob_pets_exp
            self.turn_prob_foods = turn_prob_foods_exp
        else:
            raise Exception(f"Pack {pack} not valid")

        ### Initialize shop based on input turn or input slots
        rules = get_shop_rules(self.turn, pack=self.pack)
        if len(slots) == 0:
            self.pslots = rules[0]  ### Number pet slots for shop tier
            self.fslots = rules[1]  ### Number food slots for shop tiers
            if fixed_rules is None:
                self.fixed_rules = False
                nslots = self.pslots + self.fslots
            else:
                self.fixed_rules = fixed_rules
                nslots = 0
        else:
            temp_slots = []
            for slot in slots:
                if not isinstance(slot, Slot):
                    temp_slots.append(slot_class(slot))
                else:
                    temp_slots.append(slot)
            slots = temp_slots
            slot_types = [x.slot_type for x in slots]
            self.pslots = slot_types.count("pet")
            self.fslots = slot_types.count("food")
            if fixed_rules is None:
                self.fixed_rules = True
            else:
                self.fixed_rules = fixed_rules
            nslots = len(slots)
        self.tier_avail = rules[2]  ### Pet tier available
        self.levelup_tier = rules[3]  ### Level-up tier available
        self.avail_pets = rules[4]  ### Available pets for pet slots
        self.avail_foods = rules[5]  ### Avaliable foods for food slots
        self.pp = rules[6]  ### Probability of rolling each pet
        self.fp = rules[7]  ### Probability of rolling eaach food
        if not self.fixed_rules:
            # A serialized live shop may currently have fewer slots because
            # items were purchased. Base capacity comes from the turn rules,
            # not from the surviving serialized entries.
            self.pslots = rules[0]
            self.fslots = rules[1]

        super().__init__(slots, nslots=nslots, slot_class=slot_class)
        if not self.fixed_rules and not provided_slots:
            self.update_shop_rules(roll_only_empty=True)

    def enable_paired_tape(self, seed, seat):
        """Install a fresh future world without changing visible shop contents."""
        from sapai.paired_random_tape import RandomTape
        if type(self) is not Shop:
            raise TypeError("Paired tape backend supports only the versioned Shop serializer")
        if type(seat) is not int or seat not in (0, 1):
            raise ValueError("Explicit player seat required")
        tape = RandomTape(seed)
        ability_seed = int(tape.uniform(f"player/{seat}/ability-seed") * 2**32)
        self._paired_tape, self._paired_tape_seat = tape, seat
        # Player._connect_objects attaches every shop-phase ability to this
        # separate stream; pet/food offers never consume it in tape mode.
        self.rs = np.random.RandomState(ability_seed)
        self.seed_state = self.rs.get_state()

    def buy(self, obj):
        """Remove a purchased item, including its linked discover sibling."""
        if type(obj) == int:
            idx = obj
        else:
            idx = -1
            for iter_idx, slot in enumerate(self.slots):
                if slot.obj == obj:
                    idx = iter_idx
                    break

        if idx < 0:
            raise Exception(f"Unrecognized Shop Object {obj}")

        reward_group = self.slots[idx].reward_group
        if reward_group is None:
            new_slots = self.slots[:idx] + self.slots[idx + 1 :]
        else:
            new_slots = [
                slot
                for slot in self.slots
                if slot.reward_group != reward_group
            ]
        self._nslots = len(new_slots)
        self.slots = new_slots

    def index(self, obj):
        if isinstance(obj, ShopSlot):
            obj = obj.obj
        idx = -1
        for iter_idx, slot in enumerate(self.slots):
            if slot.obj == obj:
                idx = iter_idx
                break
        if idx < 0:
            raise Exception(f"Unrecognized Shop Object {obj}")
        return idx

    @property
    def pets(self):
        pet_slots = []
        for slot in self.slots:
            if slot.slot_type == "pet":
                pet_slots.append(slot.obj)
            elif slot.slot_type == "levelup":
                pet_slots.append(slot.obj)
            else:
                pass
        return pet_slots

    @property
    def foods(self):
        food_slots = []
        for slot in self.slots:
            if slot.slot_type == "food":
                food_slots.append(slot.obj)
        return food_slots

    def roll(self, team=None, roll_only_empty=False):
        """Roll eligible slots from legacy RNG or isolated paired-tape domains.

        Shop slots retain RNG fields for standalone backwards compatibility,
        but a Shop never consumes those private streams. Frozen slots consume
        no draw and cannot publish stale state back into the Shop.
        """
        team = team or []

        self.check_rules()

        for slot in self.slots:
            slot.turn = self.turn
            slot.pack = self.pack
            eligible = (
                not slot.frozen
                and (not roll_only_empty or slot.empty)
            )
            if not eligible:
                continue
            self._roll_slot(slot)
            if slot.slot_type == "pet":
                slot.obj._attack = min(
                    50, slot.obj._attack + self.shop_attack
                )
                slot.obj._health = min(
                    50, slot.obj._health + self.shop_health
                )
            elif slot.slot_type == "food":
                slot.cost = slot.obj.cost

        # Current desktop SAP compacts every surviving frozen pet to the
        # leftmost pet positions after a roll. Remaining pets are ordered by
        # tier descending; Python's stable sort preserves draw order for ties.
        # Do not infer or alter food ordering without a verified rule.
        pet_indices = [
            index
            for index, slot in enumerate(self.slots)
            if slot.slot_type in {"pet", "levelup"}
        ]
        pet_slots = [self.slots[index] for index in pet_indices]
        frozen_pets = [slot for slot in pet_slots if slot.frozen]
        rerolled_pets = sorted(
            (slot for slot in pet_slots if not slot.frozen),
            key=lambda slot: int(slot.obj.tier),
            reverse=True,
        )
        compacted_pets = frozen_pets + rerolled_pets
        for index, slot in zip(pet_indices, compacted_pets):
            self._slots[index] = slot

        # v31: freshly rolled foods are listed higher tier first, ties in
        # draw order (2,467 of 2,467 real pairs). Frozen foods keep their
        # positions; no rule for them has been verified.
        def _food_tier(slot):
            try:
                return int(slot.obj.fd["tier"])
            except (AttributeError, KeyError, TypeError, ValueError):
                return 0

        food_indices = [
            index
            for index, slot in enumerate(self.slots)
            if slot.slot_type == "food" and not slot.frozen
        ]
        rerolled_foods = sorted(
            (self.slots[index] for index in food_indices),
            key=_food_tier,
            reverse=True,
        )
        for index, slot in zip(food_indices, rerolled_foods):
            self._slots[index] = slot

        for team_slot in team:
            team_slot._pet.shop_ability(shop=self, trigger="roll")

    def _advance_seed_state(self):
        if isinstance(self.rs, MockRandomState):
            self.seed_state = None
        else:
            self.seed_state = self.rs.get_state()

    def _roll_slot(self, slot):
        """Draw one item for ``slot`` from ``self.rs`` and advance once."""

        if slot.frozen:
            raise RuntimeError("Frozen slots must not be passed to _roll_slot")
        if slot.slot_type == "levelup":
            slot.slot_type = "pet"
            slot.reward_group = None
        if slot.slot_type == "pet":
            choice = self._paired_tape.shop_draw(self._paired_tape_seat, "pet", self.avail_pets, weights=self.pp) if self._paired_tape is not None else self.rs.choice(
                self.avail_pets, size=(1,), replace=True, p=self.pp
            )[0]
            self._advance_seed_state()
            slot.obj = Pet(str(choice), shop=self, seed_state=self.seed_state)
        elif slot.slot_type == "food":
            choice = self._paired_tape.shop_draw(self._paired_tape_seat, "food", self.avail_foods, weights=self.fp) if self._paired_tape is not None else self.rs.choice(
                self.avail_foods, size=(1,), replace=True, p=self.fp
            )[0]
            self._advance_seed_state()
            slot.obj = Food(str(choice), shop=self, seed_state=self.seed_state)
            slot.cost = slot.obj.cost
        else:
            raise RuntimeError(
                f"Cannot roll physical shop slot type {slot.slot_type!r}"
            )
        # Mirror the current state for standalone ShopSlot serialization and
        # direct-use compatibility. Shop.roll never reads this state.
        slot.seed_state = self.seed_state
        if isinstance(self.rs, MockRandomState):
            slot.rs = MockRandomState()
        else:
            slot.rs = np.random.RandomState()
            slot.rs.set_state(self.seed_state)

    def freeze(self, idx):
        """
        Freeze a shop index

        """
        if idx >= len(self.slots):
            ### Just do nothing if attempting to freeze outside the range of
            ###   the current shop
            return
        self.slots[idx].freeze()

    def unfreeze(self, idx):
        """
        Unfreeze shop index

        """
        if idx >= len(self.slots):
            ### Just do nothing if attempting to unfreeze outside the range of
            ###   the current shop
            return
        self.slots[idx].unfreeze()

    def levelup(self, count=1):
        """
        Add two linked, distinct choices from one tier above the shop tier.

        Each choice costs the normal three gold. Buying either one removes the
        complete pair.

        """
        if count < 0:
            raise ValueError("Level-up reward count cannot be negative")
        newly_generated_slots = set()
        for _ in range(int(count)):
            overflow = max(0, len(self.slots) + 2 - self.max_slots)
            removable = [
                index
                for index in range(len(self.slots) - 1, -1, -1)
                if not self.slots[index].frozen
                and self.slots[index] not in newly_generated_slots
            ]
            # Conservatively preserve discover atomicity. If the full linked
            # pair cannot fit without replacing a frozen entry, consume no
            # shop entries and discard both reward choices.
            if len(removable) < overflow:
                continue
            remove_indices = set(removable[:overflow])
            surviving_slots = [
                slot
                for index, slot in enumerate(self.slots)
                if index not in remove_indices
            ]
            rules = get_shop_rules(self.turn, pack=self.pack)
            levelup_tier = rules[3]
            candidates = (
                pet_tier_lookup_std[levelup_tier]
                if self.pack == "StandardPack"
                else pet_tier_lookup[levelup_tier]
            )
            if len(candidates) < 2:
                raise RuntimeError(
                    f"Tier {levelup_tier} needs two distinct discover choices"
                )
            choices = self._paired_tape.tier_up(self._paired_tape_seat, candidates) if self._paired_tape is not None else self.rs.choice(candidates, size=2, replace=False)
            self._advance_seed_state()
            reward_group = self.next_reward_group_id
            self.next_reward_group_id += 1
            reward_slots = []
            for choice in choices:
                reward_pet = Pet(str(choice), seed_state=self.seed_state)
                reward_pet._attack = min(
                    50, reward_pet._attack + self.shop_attack
                )
                reward_pet._health = min(
                    50, reward_pet._health + self.shop_health
                )
                slot = ShopSlot(
                    reward_pet,
                    slot_type="levelup",
                    pack=self.pack,
                    turn=self.turn,
                    cost=3,
                    seed_state=self.seed_state,
                    reward_group=reward_group,
                )
                # A concrete Pet normally infers slot_type="pet"; retain the
                # discover marker explicitly for linkage and observations.
                slot.slot_type = "levelup"
                reward_slots.append(slot)
                newly_generated_slots.add(slot)
            expanded_slots = reward_slots + surviving_slots
            self._nslots = len(expanded_slots)
            self.slots = expanded_slots

    def update_shop_rules(self, turn=-1, roll_only_empty=False):
        if turn < 0:
            turn = self.turn

        ### Turn 11 is max shopd info
        if turn > 11:
            turn = 11

        rules = get_shop_rules(turn, pack=self.pack)
        self.pslots = rules[0]
        self.fslots = rules[1]
        self.tier_avail = rules[2]
        self.levelup_tier = rules[3]
        self.avail_pets = rules[4]
        self.avail_foods = rules[5]
        self.pp = rules[6]
        self.fp = rules[7]

        # Reconcile surplus stock/reward slots against the new base shop while
        # preserving every frozen entry. This prevents non-frozen Pigeon stock
        # from permanently increasing future shop size.
        self.check_rules()

        ### Roll all slots upon update of rules.
        self.roll(roll_only_empty=roll_only_empty)

    def next_turn(self):
        ### Update turn counter
        self.turn += 1

        ### Update rules of the shop to generate a shop state
        self.update_shop_rules()

        return self.slots

    def check_rules(self):
        """
        Used to ensure that rules of the current shop are satisfied before
        beforming a roll. This is because slots may be added when an animal
        on the team levels-up.
        """
        levelup_slots = [
            slot
            for slot in self.slots
            if slot.slot_type == "levelup" and slot.frozen
        ]
        pet_slots = [
            slot for slot in self.slots if slot.slot_type == "pet" and slot.frozen
        ]
        food_slots = [
            slot for slot in self.slots if slot.slot_type == "food" and slot.frozen
        ]
        frozen_count = len(levelup_slots) + len(pet_slots) + len(food_slots)
        if frozen_count > self.max_slots:
            raise ValueError("Invalid shop state: frozen entries already exceed the hard shop capacity")
        # Frozen discover choices occupy normal pet-shop capacity on future
        # rolls. Keeping both therefore exposes one fewer fresh pet than
        # keeping only one. If transient level-up overflow leaves more frozen
        # rewards than the normal pet quota, preserve every frozen choice and
        # expose no ordinary pet slots until capacity becomes available.
        ordinary_pet_capacity = max(0, self.pslots - len(levelup_slots))
        if len(pet_slots) > ordinary_pet_capacity:
            # Frozen ordinary pets are never discarded. Non-frozen entries
            # will simply receive no capacity below.
            ordinary_pet_capacity = len(pet_slots)
        # Pet/food quotas share one physical capacity. Stocked foods and
        # discover rewards can exceed their ordinary category quota while
        # frozen, but must never cause the other category to grow past ten.
        # Reserve ALL frozen entries before allocating any fresh slots.
        ordinary_pet_capacity = min(
            ordinary_pet_capacity,
            self.max_slots - len(levelup_slots) - len(food_slots),
        )
        food_capacity = min(
            max(self.fslots, len(food_slots)),
            self.max_slots - len(levelup_slots) - ordinary_pet_capacity,
        )
        for slot in self.slots:
            if (
                slot.slot_type == "pet"
                and not slot.frozen
                and len(pet_slots) < ordinary_pet_capacity
            ):
                pet_slots.append(slot)
            elif (
                slot.slot_type == "food"
                and not slot.frozen
                and len(food_slots) < food_capacity
            ):
                food_slots.append(slot)
        while len(pet_slots) < ordinary_pet_capacity:
            pet_slots.append(
                ShopSlot(
                    "pet",
                    turn=self.turn,
                    pack=self.pack,
                    seed_state=self.seed_state,
                )
            )
        while len(food_slots) < food_capacity:
            food_slots.append(
                ShopSlot(
                    "food",
                    turn=self.turn,
                    pack=self.pack,
                    seed_state=self.seed_state,
                )
            )
        # Verified current shop behavior: frozen pets compact left, but frozen
        # foods compact RIGHT within their category. Retain stable order within
        # each group; frozen objects consume no draws and must not be replaced.
        food_slots = ([slot for slot in food_slots if not slot.frozen]
                      + [slot for slot in food_slots if slot.frozen])
        rebuilt_slots = levelup_slots + pet_slots + food_slots
        self._nslots = len(rebuilt_slots)
        self.slots = rebuilt_slots

    def __len__(self):
        return len(self.slots)

    def __getitem__(self, idx):
        return self.slots[idx]

    def __setitem__(self, idx, obj):
        """
        __setitem__ should never be used

        """
        raise Exception("Cannot set item in shop directly")

    def append(self, obj):
        """
        Append should be used when adding an animal from a levelup
        """
        if len(self.slots) >= self.max_slots:
            ### Max slots already reached so cannot be added
            return

        add_slot = ShopSlot(
            obj, pack=self.pack, turn=self.turn, seed_state=self.seed_state
        )
        pslots = []
        fslots = []
        for iter_idx, slot in enumerate(self.slots):
            if slot.slot_type == "pet":
                pslots.append(slot)
            if slot.slot_type == "food":
                fslots.append(slot)

        new_slots = []
        new_slots += [x for x in pslots]
        new_slots += [add_slot]
        new_slots += [x for x in fslots]

        self.nslots = len(new_slots)
        self.slots = new_slots

    @property
    def state(self):
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
            "type": "Shop",
            "slots": [x.state for x in self.slots],
            "turn": self.turn,
            "shop_attack": self.shop_attack,
            "shop_health": self.shop_health,
            "pack": self.pack,
            "seed_state": seed_state,
            "fixed_rules": self.fixed_rules,
            "next_reward_group_id": self.next_reward_group_id,
        }
        if self._paired_tape is not None:
            state_dict["paired_tape"] = self._paired_tape.private_state()
            state_dict["paired_tape_seat"] = self._paired_tape_seat
        return state_dict

    @classmethod
    def from_state(cls, state):
        ### Supply seed_state in state dict should be optional
        if "seed_state" in state:
            seed_state = state["seed_state"]
        else:
            seed_state = None
        return cls(
            slots=[ShopSlot.from_state(x) for x in state["slots"]],
            turn=state["turn"],
            shop_attack=state["shop_attack"],
            shop_health=state["shop_health"],
            pack=state["pack"],
            seed_state=seed_state,
            fixed_rules=state["fixed_rules"],
            next_reward_group_id=state.get("next_reward_group_id", 1),
            paired_tape=state.get("paired_tape"),
            paired_tape_seat=state.get("paired_tape_seat"),
        )

    def __repr__(self):
        repr_str = ""
        for iter_idx, slot in enumerate(self.slots):
            repr_str += f"{iter_idx}: {slot} \n    "
        return repr_str


class ShopLearn(Shop):
    """
    Shop behavior designed for learning algorithms. In this shop, all Pets and
    Foods are presented for a given tier. In this way, the all possible can
    be considered and becaues the API is considered with a regular shop, the
    agent can automatically be executed with normal random shop behavior.
    Only disadvantage is that rolling behavior can not be learned by this method
    until transitioning to normal random shop behavior.

    Rolling behavior is emulated. For example, on the first round, if 3 pets
    are bought, then all pets in the shop are removed requiring that a roll
    takes place before another pet can be bought. Without this requirement,
    teams that could never be built in the real game are possible by buying
    and selling more than 3 pets without rolling a single time.

    """

    def __init__(self, *args, **kwargs):
        self.npet_bought = 0
        self.nfood_bought = 0
        self.nmax_levelup = 0
        self.nlevelup_bought = 0
        self.shop_names = {}
        super().__init__(*args, **kwargs)
        self.nslots = len(self.avail_pets) + len(self.avail_foods)

    def roll(self):
        ### Reset npet_bought and nfood_bough upon rolling
        self.npet_bought = 0
        self.nfood_bought = 0
        self.nlevelup_bought = 0

        ### Return the number of pets that can be bought from the next tier
        ###   to zero
        self.nmax_levelup = 0

        ### Rebuild ShopLearn
        self.update_shop_rules()

    def buy(self, obj):
        """
        Buy has slightly different behavior than normal. After buying a pet,
        it is never removed. However, all levelup slots are removed. What this
        does is that Pets from a higher tier that is available from a combining
        to levelup can only be considered or bought a single time. This provides
        very good emulation of realistic game behavior during the shopping
        phase.

        """
        ### Desired object is ShopSlot in this case
        if isinstance(obj, ShopSlot):
            pass
        elif type(obj) == int:
            obj = self.slots[obj]
        else:
            idx = -1
            for iter_idx, slot in enumerate(self.slots):
                if slot.obj == obj:
                    idx = iter_idx
                    obj = slot
                    break
            if idx < 0:
                raise Exception(f"Unrecognized Shop Object {obj}")

        if obj.slot_type == "pet":
            self.npet_bought += 1
        elif obj.slot_type == "food":
            self.nfood_bought += 1
        elif obj.slot_type == "levelup":
            self.nlevelup_bought += 1
        else:
            raise Exception(f"Unrecognized ShopSlot {obj}")

        ### Rebuild ShopLearn to remove all levelup ShopSlots
        self.update_shop_rules()

    def update_shop_rules(self, turn=-1, roll_only_empty=True):
        """
        Rebuilds ShopLearn using all Pets and Foods available for the givne turn

        """
        if turn < 0:
            turn = self.turn

        ### Turn 11 is max shopd info
        if turn > 11:
            turn = 11

        rules = get_shop_rules(turn, pack=self.pack)
        self.pslots = rules[0]
        self.fslots = rules[1]
        self.tier_avail = rules[2]
        self.levelup_tier = rules[3]
        self.avail_pets = rules[4]
        self.avail_foods = rules[5]
        self.pp = len(self.avail_pets)
        self.fp = len(self.avail_foods)

        ### Setup the shop slots
        self.shop_names = {}
        self.slots = []
        new_slots_pet = []
        new_slots_food = []
        new_slots_levelup = []

        ### Check if Pet purchase limit or Food purchase limit has been reached
        if self.npet_bought < self.pslots:
            for pet in self.avail_pets:
                new_slots_pet.append(
                    ShopSlot(
                        pet, pack=self.pack, turn=self.turn, seed_state=self.seed_state
                    )
                )
                self.shop_names[pet] = True
        if self.nfood_bought < self.fslots:
            for food in self.avail_foods:
                new_slots_food.append(
                    ShopSlot(
                        food, pack=self.pack, turn=self.turn, seed_state=self.seed_state
                    )
                )
                self.shop_names[food] = True
        if self.nlevelup_bought < self.nmax_levelup:
            if self.pack == "StandardPack":
                levelup_avail_pets = pet_tier_lookup_std[self.levelup_tier]
            else:
                levelup_avail_pets = pet_tier_lookup[self.levelup_tier]
            for pet in levelup_avail_pets:
                if pet not in self.shop_names:
                    temp_slot = ShopSlot(
                        slot_type="levelup",
                        pack=self.pack,
                        turn=self.turn,
                        seed_state=self.seed_state,
                    )
                    temp_slot.obj = Pet(pet, seed_state=self.seed_state)
                    new_slots_levelup.append(temp_slot)
                    self.shop_names[pet] = True

        self.slots = new_slots_pet + new_slots_levelup + new_slots_food

    def levelup(self, count=1):
        ### Add the number of level-up rewards that can be bought.
        self.nmax_levelup += int(count)
        ### Rebuild shop
        self.update_shop_rules()

    def check_rules(self):
        raise Exception("ShopLearn does not use check_rules")

    @property
    def state(self):
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
            "type": "ShopLearn",
            "slots": [x.state for x in self.slots],
            "turn": self.turn,
            "shop_attack": self.shop_attack,
            "shop_health": self.shop_health,
            "pack": self.pack,
            "seed_state": seed_state,
        }
        return state_dict

    @classmethod
    def from_state(cls, state):
        ### Supply seed_state in state dict should be optional
        if "seed_state" in state:
            seed_state = state["seed_state"]
        else:
            seed_state = None
        return cls(
            slots=[ShopSlot.from_state(x) for x in state["slots"]],
            turn=state["turn"],
            shop_attack=state["shop_attack"],
            shop_health=state["shop_health"],
            pack=state["pack"],
            seed_state=seed_state,
        )


class ShopSlotNoneItem:
    def __init__(self, seed_state=None):
        self.player = None
        self.name = "none"
        self.seed_state = seed_state
        self.obj = None
        if self.seed_state is not None:
            self.rs = np.random.RandomState()
            self.rs.set_state(self.seed_state)
        else:
            ### Otherwise, set use
            self.rs = MockRandomState()

    @property
    def state(self):
        if getattr(self, "rs", False):
            if isinstance(self.rs, MockRandomState):
                seed_state = None
            else:
                seed_state = list(self.rs.get_state())
                seed_state[1] = seed_state[1].tolist()
        else:
            seed_state = None
        return {"type": "ShopSlotNoneItem", "seed_state": seed_state}

    @classmethod
    def from_state(cls, state):
        if "seed_state" in state:
            return cls(state["seed_state"])
        else:
            return cls()

    def copy(self):
        return self


class ShopSlot(Slot):
    """
    Class for a slot in the shop

    """

    def __init__(
        self,
        obj=None,
        slot_type="none",
        frozen=False,
        turn=1,
        cost=3,
        pack="StandardPack",
        seed_state=None,
        reward_group=None,
    ):
        self.seed_state = seed_state
        if self.seed_state is not None:
            self.rs = np.random.RandomState()
            self.rs.set_state(self.seed_state)
        else:
            ### Otherwise, set use
            self.rs = MockRandomState()

        self.slot_type = slot_type
        self.turn = turn
        self.pack = pack
        self.frozen = frozen
        self.cost = cost
        self.reward_group = reward_group
        self.obj = ShopSlotNoneItem(seed_state=self.seed_state)

        if slot_type not in ["pet", "food", "levelup", "none"]:
            raise Exception(f"Unrecognized slot type {self.slot_type}")

        if obj is not None and type(obj) != str:
            if isinstance(obj, Pet):
                self.slot_type = (
                    "levelup" if slot_type == "levelup" else "pet"
                )
                self.obj = obj
            elif isinstance(obj, Food):
                self.slot_type = "food"
                self.obj = obj
                self.cost = obj.cost
            elif isinstance(obj, ShopSlot):
                self.slot_type = obj.slot_type
                self.turn = obj.turn
                self.pack = obj.pack
                self.frozen = obj.frozen
                self.cost = obj.cost
                if self.slot_type == "food":
                    self.cost = obj.obj.cost
                self.reward_group = obj.reward_group
                self.obj = obj.obj
        else:
            if type(obj) == str:
                if obj not in ["pet", "food", "levelup"]:
                    if obj in data["pets"]:
                        name = obj
                        self.slot_type = "pet"
                    elif obj in data["foods"]:
                        name = obj
                        self.slot_type = "food"
                    elif f"pet-{obj}" in data["pets"]:
                        name = obj
                        self.slot_type = "pet"
                    elif f"food-{obj}" in data["foods"]:
                        name = obj
                        self.slot_type = "food"
                    else:
                        raise Exception(f"Unrecognized ShopSlot Object {obj}")
                else:
                    self.slot_type = obj
                    name = "none"
            else:
                self.slot_type = "none"
                name = "none"

            if self.slot_type == "pet":
                self.obj = Pet(name, seed_state=self.seed_state)
            elif self.slot_type == "food":
                self.obj = Food(name, seed_state=self.seed_state)
                self.cost = self.obj.cost
            elif self.slot_type == "levelup":
                self.roll_levelup()
            elif self.slot_type == "none":
                self.obj = ShopSlotNoneItem(seed_state=self.seed_state)

    def __repr__(self):
        if self.frozen:
            fstr = "frozen"
        else:
            fstr = "not-frozen"
        if self.slot_type == "pet" or self.slot_type == "levelup":
            if self.obj.name == "pet-none":
                return f"< ShopSlot-{self.slot_type} {fstr} EMPTY >"
            else:
                pet_repr = str(self.obj)
                pet_repr = pet_repr[2:-2]
                return (
                    f"< ShopSlot-{self.slot_type} {fstr} {self.cost}-gold {pet_repr} >"
                )
        elif self.slot_type == "food":
            if self.obj.name == "food-none":
                return f"< ShopSlot-{self.slot_type} {fstr} EMPTY >"
            else:
                food_repr = str(self.obj)
                food_repr = food_repr[2:-2]
                return (
                    f"< ShopSlot-{self.slot_type} {fstr} {self.cost}-gold {food_repr} >"
                )
        elif self.slot_type == "none":
            return f"< ShopSlot-{None} {fstr} EMPTY >"
        else:
            raise Exception(f"Invalid ShopSlot type {self.slot_type}")

    @property
    def empty(self):
        """
        Returns if the given slot is empty
        """
        return self.obj.name in ["pet-none", "food-none", "none"]

    def freeze(self):
        """
        Freeze current slot such that shop rolls don't update the ShopSlot
        """
        if self.slot_type == "none":
            raise Exception("Cannot freeze an empty shop slot")
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def roll(self, avail=None, prob=None):
        avail = avail or []
        prob = prob or []

        if self.frozen:
            return
        if self.slot_type == "levelup":
            ### If roll is called on a levelup slot, then it should change to
            ###   a typical pet slot type. Deletion of levelup slots is handled
            ###   by the Shop.check_rules method when appropriate.
            self.slot_type = "pet"
            self.reward_group = None

        if len(avail) == 0:
            rules = get_shop_rules(self.turn, pack=self.pack)
            if self.slot_type == "pet":
                avail = rules[4]
                prob = rules[6]
            elif self.slot_type == "food":
                avail = rules[5]
                prob = rules[7]
            else:
                avail = ["none"]
                prob = [1]

        choice = self.rs.choice(avail, size=(1,), replace=True, p=prob)[0]
        self.seed_state = self.rs.get_state()
        if self.slot_type == "pet":
            self.obj = Pet(choice, seed_state=self.seed_state)
        elif self.slot_type == "food":
            self.obj = Food(choice, seed_state=self.seed_state)
        elif self.slot_type == "none":
            self.obj = ShopSlotNoneItem(seed_state=self.seed_state)

    def roll_levelup(self):
        rules = get_shop_rules(self.turn, pack=self.pack)
        levelup_tier = rules[3]
        if self.pack == "StandardPack":
            avail_pets = pet_tier_lookup_std[levelup_tier]
        else:
            avail_pets = pet_tier_lookup[levelup_tier]
        pet_choice = self.rs.choice(avail_pets, size=(1,), replace=True)[0]
        self.seed_state = self.rs.get_state()
        self.obj = Pet(pet_choice, seed_state=self.seed_state)

    @property
    def state(self):
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
            "type": "ShopSlot",
            "slot_type": self.slot_type,
            "obj": self.obj.state,
            "turn": self.turn,
            "pack": self.pack,
            "cost": self.cost,
            "frozen": self.frozen,
            "seed_state": seed_state,
            "reward_group": self.reward_group,
        }
        return state_dict

    @classmethod
    def from_state(cls, state):
        slot_type = state["slot_type"]
        if state["obj"]["type"] == "Pet":
            item_cls = getattr(sapai.pets, "Pet")
        elif state["obj"]["type"] == "Food":
            item_cls = getattr(sapai.foods, "Food")
        elif state["obj"]["type"] == "ShopSlotNoneItem":
            item_cls = ShopSlotNoneItem
        else:
            raise Exception("Unrecognized item state")
        obj = item_cls.from_state(state["obj"])
        turn = state["turn"]
        pack = state["pack"]
        cost = state["cost"]
        frozen = state["frozen"]
        ### Supply seed_state in state dict should be optional
        if "seed_state" in state:
            seed_state = state["seed_state"]
        else:
            seed_state = None

        ### This should no longer be possible
        # obj.rs = np.random.RandomState()
        # if seed_state is not None:
        #     obj.rs.set_state(state["item"]["seed_state"])

        return cls(
            obj=obj,
            slot_type=slot_type,
            frozen=frozen,
            turn=turn,
            cost=cost,
            pack=pack,
            seed_state=seed_state,
            reward_group=state.get("reward_group"),
        )


def get_shop_rules(turn, pack="StandardPack"):
    if pack == "StandardPack":
        turn_prob_pets = turn_prob_pets_std
        turn_prob_foods = turn_prob_foods_std
    elif pack == "ExpansionPack1":
        turn_prob_pets = turn_prob_pets_exp
        turn_prob_foods = turn_prob_foods_exp
    else:
        raise Exception(f"Pack {pack} not valid")

    if turn <= 0:
        raise Exception("Input turn must be greater than 0")

    ### Turn 11 is max shop info
    if turn > 11:
        turn = 11

    td = data["turns"][f"turn-{turn}"]
    fslots = td["foodShopSlots"]
    pslots = td["animalShopSlots"]
    tier_avail = td["tiersAvailable"]
    levelup_tier = td["levelUpTier"]

    if pack == "StandardPack":
        # SAP-Calculator's pinned Turtle Pack adds content which has no entry
        # in the legacy per-item probability tables.  Turtle Pack shop slots
        # are uniform over all unlocked rollable items, so build that pool
        # directly from the current roster.
        avail_pets = []
        avail_foods = []
        for tier in range(1, tier_avail + 1):
            avail_pets.extend(pet_tier_lookup_std[tier])
            avail_foods.extend(food_tier_lookup_std[tier])
        pp = np.full(len(avail_pets), 1.0 / len(avail_pets))
        fp = np.full(len(avail_foods), 1.0 / len(avail_foods))
        return pslots, fslots, tier_avail, levelup_tier, avail_pets, avail_foods, pp, fp

    temp_avail_pets = pet_tier_avail_lookup[tier_avail]
    temp_avail_foods = food_tier_avail_lookup[tier_avail]

    temp_prob_pets = turn_prob_pets[turn]
    pp = []
    avail_pets = []
    for temp_pet in temp_avail_pets:
        if temp_pet not in temp_prob_pets:
            continue
        pp.append(temp_prob_pets[temp_pet])
        avail_pets.append(temp_pet)

    temp_prob_foods = turn_prob_foods[turn]
    fp = []
    avail_foods = []
    for temp_food in temp_avail_foods:
        if temp_food not in temp_prob_foods:
            continue
        fp.append(temp_prob_foods[temp_food])
        avail_foods.append(temp_food)

    ### Ensure that the probabilities sum to 1...
    pp = np.array(pp)
    pp = pp / np.sum(pp)

    fp = np.array(fp)
    fp = fp / np.sum(fp)

    return pslots, fslots, tier_avail, levelup_tier, avail_pets, avail_foods, pp, fp


# %%
