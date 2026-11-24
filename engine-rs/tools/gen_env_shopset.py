"""Reference traces for `set_shop_item` (Python, frozen engine).

`place` below is the Python reference of the primitive: the edit made directly on
`player.shop` of `ParallelSelfPlayEnv().core`, the way the reference itself builds a rolled slot:

  * `Shop._roll_slot` with the choice given instead of drawn (a `levelup` slot becomes a `pet`
    slot and loses its reward group; `Pet(name, shop=shop, seed_state=shop.seed_state)` or
    `Food(...)`; a food slot takes the food's cost),
  * the stat line `Shop.roll` runs right after it for a pet (`+ shop_attack / shop_health`, capped
    at 50),
  * the four assignments `Player._connect_objects` makes for a shop item (`player`, `shop`,
    `rs = shop.rs`, `seed_state`), for the new item only: every roll of the reference is followed
    by `_connect_objects` (`Player.roll`, `start_turn`, the reset), so a rolled item always draws
    from the shop stream. The other items are not touched (a level-up reward that still holds its
    private copy of the stream keeps it),
  * `slot.frozen = frozen`.

Nothing is drawn. Matches are written in the trace format of `gen_env_ghost.py` (extended state
block, battle teams) with these additional `pre` entries, which `envcheck` replays:

  ["shop", seat, slot, name, frozen]   `place(...)`                    -> `Game::set_shop_item`
  ["state", dump]                      `canonical(env.core)` right after the edits of the step
  ["obs", [seat 0, seat 1]]            `env.observations()` right after the edits (bytes; like
                                       every observation it registers the team order as seen)
together with the existing ["gold", ...], ["team", ...] and ["legal"] entries. Every step with an
edit carries both full observations.

What a match does (seeds are 884400000 + index; index // 2 odd: the scripted preferences of
`gen_env_targeted.py`, else uniformly random legal play; index % 5 < 2: "curriculum" match):
  * at random steps (more often at the start of a seat's turn and right after its roll) a seat's
    shop gets one or more items: any of the 60 pets and 18 foods of the roll pools whatever the
    turn, the species of a team pet (to be combined), Canned Food, an item into a level-up reward
    slot, a frozen item, an item over a frozen one, the item the slot already holds; sometimes
    for the seat that has already ended its turn;
  * in curriculum matches, at most turn starts: `set_team` with a board of the ghost pool, then
    items into the shop, then `set_gold` (the intended use);
  * in one match of forty (index % 40 == 7) one seat is handed Canned Food and the gold for it at
    most of its steps and buys it, so that the shop's permanent bonus passes 40 and placed pets
    reach the cap of 50;
  * while level-up rewards are on offer, a Sleeping Pill is often placed next to them and the seat
    buys a reward and feeds it the pill (a Spider or Seal reward then draws from the private copy
    of the shop stream it holds until the next `_connect_objects`: the edit must not rebind it);
  * after an edit the seat's choice is biased towards the placed items: buy them, combine them,
    freeze and unfreeze them, feed and sell the pets that came from them.
The displaced objects are kept alive until the match ends: the reference's legality cache keys
shop items by `id()`, and a freed object's address can be handed to the next placed one (the
cache would then return the legal actions of the earlier item; Rust identities are never reused).

Usage (training image, tools/run_ref.sh):
  gen_env_shopset.py <out.jsonl.gz> <matches> <first index> <full obs every N>
"""
import gzip, io, json, random, sys, traceback
import numpy as np

import gen_env_golden as G
import gen_env_ghost as H
from sapai import Pet
from sapai.foods import Food
from sapai.data import data as ENGINE_DATA
from sapai.rand import MockRandomState, LazyRandomState
from sapai.shop import get_shop_rules
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import ActionType, FlatActionCodec as C
from sapai_gym1.rl.joint_action import JointAction, step_decisions

MAIN = __name__ == "__main__"
_RULES = get_shop_rules(11)
POOL_PETS = [str(name) for name in _RULES[4]]       # every pet a Turtle shop can roll (60)
POOL_FOODS = [str(name) for name in _RULES[5]]      # every food it can roll (18)
PET_TIER = {name: int(ENGINE_DATA["pets"][name]["tier"]) for name in POOL_PETS}
FOOD_TIER = {name: int(ENGINE_DATA["foods"][name]["tier"]) for name in POOL_FOODS}
T = ActionType


def place(player, index, name, frozen, keep):
    """The reference of `set_shop_item(game, seat, index, name, frozen)`. Raises (nothing changed)
    for a slot that does not exist, an item outside the roll pools, or the wrong slot type."""
    shop = player.shop
    if type(index) is not int or not 0 <= index < len(shop.slots):
        raise IndexError(f"shop slot {index} does not exist")
    slot = shop.slots[index]
    if name in PET_TIER:
        if slot.slot_type not in ("pet", "levelup"):
            raise ValueError(f"a pet cannot go into a {slot.slot_type} slot")
        keep.append(slot.obj)
        # Shop.roll (the loop head) and Shop._roll_slot, the choice given instead of drawn
        slot.turn = shop.turn; slot.pack = shop.pack
        if slot.slot_type == "levelup":
            slot.slot_type = "pet"
            slot.reward_group = None
        shop._advance_seed_state()
        slot.obj = Pet(name, shop=shop, seed_state=shop.seed_state)
        # Shop.roll, right after _roll_slot
        slot.obj._attack = min(50, slot.obj._attack + shop.shop_attack)
        slot.obj._health = min(50, slot.obj._health + shop.shop_health)
        if slot.cost != 3:
            raise RuntimeError(f"a pet slot with cost {slot.cost}")     # never seen: nothing changes a pet slot's cost
    elif name in FOOD_TIER:
        if slot.slot_type != "food":
            raise ValueError(f"a food cannot go into a {slot.slot_type} slot")
        keep.append(slot.obj)
        slot.turn = shop.turn; slot.pack = shop.pack
        shop._advance_seed_state()
        slot.obj = Food(name, shop=shop, seed_state=shop.seed_state)
        slot.cost = slot.obj.cost
    else:
        raise ValueError(f"{name!r} is not in the roll pools of the Turtle pack")
    slot.seed_state = shop.seed_state
    slot.rs = MockRandomState() if isinstance(shop.rs, MockRandomState) else LazyRandomState(shop.seed_state)
    # Player._connect_objects, for this item only
    obj = slot.obj
    obj.player = player
    obj.shop = shop
    obj.rs = shop.rs
    obj.seed_state = shop.rs.get_state()
    slot.frozen = bool(frozen)
    return obj


class Session:
    """One match: the edits, the biased choices and the bookkeeping of what happened to the placed items.
    Used by the trace generator below and by the lockstep check of python/verify_search.py."""

    def __init__(self, env, index, targeted, pool, seed):
        self.env, self.core, self.index = env, env.core, index
        self.rng = random.Random(seed)
        self.draw = np.random.default_rng(np.random.SeedSequence([seed, 20261123]))
        self.sequences, self.by_turn, self.last_turn = pool
        self.scripted = (index // 2) % 2 == 1
        self.curriculum = index % 5 < 2
        idle = self.rng.choice([0, 0, 4, 8])
        self.lives = [H.Live(targeted, self.rng, self.scripted, idle) for _ in (0, 1)]
        self.canfarm = (index // 40) % 2 if index % 40 == 7 else None     # this seat is fed Canned Food all match
        self.force = [None, None]
        self.chain = [None, None]            # [reward pet, pill]: buy the reward, then feed it the pill
        self.keep = []                       # displaced objects (see the module docstring)
        self.focus = [[], []]                # placed objects that may still be in the seat's shop
        self.grown = [[], []]                # team pets that came from a placed pet (bought, or combined with one)
        self.levels = [{}, {}]
        self.last_type = [None, None]
        self.snap = [None, None]
        self.stats = {"edits": {}, "where": {}, "fate": {}, "placed": {}, "bought": {}, "tierByTurn": {}, "decisions": 0,
                      "battles": 0, "editSteps": 0, "followed": 0}

    def note(self, table, key, count=1):
        self.stats[table][key] = self.stats[table].get(key, 0) + count

    def players(self):
        return (self.core.agent, self.core.opponent)

    # ---- edits -------------------------------------------------------------------------------

    def edits(self):
        """Apply this step's edits to the Python environment; returns the `pre` list (an "obs"
        entry holds the raw observation dicts)."""
        env, rng = self.env, self.rng
        active = env.active_players()
        counts = (self.core.shop_action_count_agent, self.core.shop_action_count_opponent)
        pre = []
        if self.canfarm is not None and active[self.canfarm] and rng.random() < 0.8:
            # Canned Food into a food slot, gold to pay for it, and `decide` buys it: the permanent
            # shop bonus grows by one per step, so that placed (and rolled) pets reach the cap of 50.
            seat = self.canfarm
            player = self.players()[seat]
            foods = [i for i, slot in enumerate(player.shop.slots) if slot.slot_type == "food"]
            if foods:
                i = rng.choice(foods)
                obj = place(player, i, "food-canned-food", False, self.keep)
                pre.append(["shop", seat, i, "food-canned-food", False])
                self.focus[seat].append(obj); self.force[seat] = obj
                self.note("edits", "shop-food"); self.note("edits", "kind:canfarm"); self.note("placed", "food-canned-food")
                self.note("where", "canned-food-farm")
                if player.gold < 3:
                    player.gold = 10
                    pre.append(["gold", seat, 10]); self.note("edits", "set_gold")
        for seat in (0, 1):
            # A level-up reward on offer still holds its private copy of the shop stream (it was
            # created after the last `_connect_objects`). A Sleeping Pill is placed next to it, the
            # seat buys the reward and feeds it the pill: a Spider or Seal reward then draws from
            # that private copy, which shows whether the edit left the other shop items alone.
            player = self.players()[seat]
            slots = player.shop.slots
            rewards = [slot.obj for slot in slots if slot.slot_type == "levelup"]
            foods = [i for i, slot in enumerate(slots) if slot.slot_type == "food"]
            if active[seat] and self.chain[seat] is None and rewards and foods and len(player.team) < 5 and rng.random() < 0.35:
                wanted = [obj for obj in rewards if obj.name in ("pet-spider", "pet-seal")]
                reward = rng.choice(wanted) if wanted else rng.choice(rewards)
                i = rng.choice(foods)
                pill = place(player, i, "food-sleeping-pill", False, self.keep)
                pre.append(["shop", seat, i, "food-sleeping-pill", False])
                self.focus[seat].append(pill); self.chain[seat] = [reward, pill]
                self.note("edits", "shop-food"); self.note("edits", "kind:pill-for-a-reward"); self.note("placed", "food-sleeping-pill")
                self.note("where", "pill-next-to-an-untouched-reward"); self.note("where", "rewards-present")
                if player.gold < 4:
                    player.gold = 6
                    pre.append(["gold", seat, 6]); self.note("edits", "set_gold")
        order = [0, 1] if rng.random() < 0.5 else [1, 0]
        for seat in order:
            start = active[seat] and counts[seat] == 0
            if not active[seat]:
                chance = 0.05
            elif start:
                chance = 0.75 if self.curriculum else 0.2
            elif self.last_type[seat] == int(T.ROLL):
                chance = 0.25
            else:
                chance = 0.06
            if active[seat] and any(slot.slot_type == "levelup" for slot in self.players()[seat].shop.slots):
                chance = max(chance, 0.3)       # level-up rewards on offer
            if rng.random() < chance:
                self.edit_seat(seat, pre, bool(active[seat]), bool(start))
        if pre:
            self.stats["editSteps"] += 1
            pre.append(["state", G.canonical(self.core)])
            if rng.random() < 0.5:
                pre.append(["obs", env.observations()])
        return pre

    def marker(self, pre, chance):
        if self.rng.random() < chance:
            G.legal_of(self.env)
            pre.append(["legal"]); self.note("edits", "legal-marker")

    def edit_seat(self, seat, pre, active, start):
        rng = self.rng
        player = self.players()[seat]
        shop = player.shop
        kind = "curriculum" if self.curriculum and start and rng.random() < 0.85 else rng.choice(
            ["random"] * 6 + ["dup"] * 4 + ["reward"] * 4 + ["canned"] * 2 + ["multi"] * 2 + ["frozen"] * 2 + ["thaw", "same"])
        if kind != "curriculum" and any(slot.slot_type == "levelup" for slot in shop.slots) and rng.random() < 0.5:
            kind = "reward"
        tags = []
        if kind == "curriculum" or rng.random() < 0.06:
            turn = int(player.turn)
            if rng.random() < 0.7:
                specs = rng.choice(self.by_turn[min(max(turn, 1), self.last_turn)])
            else:
                specs = rng.choice(self.by_turn[rng.choice(sorted(self.by_turn))])
            specs = [dict(spec) for spec in specs]
            if specs and rng.random() < 0.5:        # a pet below level 3 at the brink of a level, to be combined
                spec = rng.choice(specs); spec["experience"] = rng.choice([1, 4]) if spec["id"] in PET_TIER else spec["experience"]
            pre.append(["team", seat, H.swap_team(player, specs)])
            self.note("edits", "set_team"); tags.append("with-set_team")
            self.grown[seat] = []
            self.marker(pre, 0.3)
        slots = shop.slots
        pet_slots = [i for i, s in enumerate(slots) if s.slot_type in ("pet", "levelup")]
        reward_slots = [i for i, s in enumerate(slots) if s.slot_type == "levelup"]
        food_slots = [i for i, s in enumerate(slots) if s.slot_type == "food"]
        frozen_slots = [i for i, s in enumerate(slots) if s.frozen]
        team_names = [s.pet.name for s in player.team if not s.empty and s.pet.name in PET_TIER and s.pet.level < 3]

        def item_for(i):
            if slots[i].slot_type == "food":
                return rng.choice(POOL_FOODS)
            if team_names and rng.random() < 0.15:
                return rng.choice(team_names)
            return rng.choice(POOL_PETS)

        plan = []       # (slot, name, frozen)
        if kind == "curriculum" and slots:
            chosen = rng.sample(range(len(slots)), min(len(slots), rng.randint(1, 3)))
            if pet_slots and not any(i in pet_slots for i in chosen):
                chosen[0] = rng.choice(pet_slots)
            for n, i in enumerate(chosen):
                name = item_for(i)
                if n == 0 and i in pet_slots and team_names and rng.random() < 0.6:
                    name = rng.choice(team_names)
                plan.append((i, name, rng.random() < 0.2))
        elif kind == "dup" and pet_slots and team_names:
            for i in rng.sample(pet_slots, min(len(pet_slots), rng.choice([1, 1, 2]))):
                plan.append((i, rng.choice(team_names), rng.random() < 0.15))
        elif kind == "reward" and reward_slots:
            for i in rng.sample(reward_slots, min(len(reward_slots), rng.choice([1, 1, 2]))):
                plan.append((i, rng.choice(team_names) if team_names and rng.random() < 0.3 else rng.choice(POOL_PETS), rng.random() < 0.3))
        elif kind == "canned" and food_slots:
            plan.append((rng.choice(food_slots), "food-canned-food", rng.random() < 0.15))
        elif kind == "multi" and len(slots) >= 2:
            for i in rng.sample(range(len(slots)), rng.randint(2, len(slots))):
                plan.append((i, item_for(i), rng.random() < 0.25))
        elif kind == "frozen" and slots:
            i = rng.choice(range(len(slots)))
            plan.append((i, item_for(i), True))
        elif kind == "thaw" and frozen_slots:
            i = rng.choice(frozen_slots)
            plan.append((i, item_for(i), False))
        elif kind == "same" and slots:
            i = rng.choice(range(len(slots)))
            name = slots[i].obj.name
            plan.append((i, name if name in PET_TIER or name in FOOD_TIER else item_for(i), bool(slots[i].frozen) if rng.random() < 0.7 else not slots[i].frozen))
        elif slots:
            kind = "random"
            i = rng.choice(range(len(slots)))
            plan.append((i, item_for(i), rng.random() < 0.25))
        if plan:
            self.note("edits", "kind:" + kind)
        turn = int(shop.turn)
        tier_avail = int(get_shop_rules(turn)[2])
        for n, (i, name, frozen) in enumerate(plan):
            slot = slots[i]
            where = list(tags)
            where.append("seat-ended" if not active else "turn-start" if start else "mid-turn")
            if active and self.last_type[seat] == int(T.ROLL):
                where.append("right-after-a-roll")
            if any(s.frozen for k, s in enumerate(slots) if k != i):
                where.append("frozen-items-present")
            if any(s.slot_type == "levelup" for s in slots):
                where.append("rewards-present")
            if slot.slot_type == "levelup":
                where.append("into-reward-slot")
                if slot.frozen:
                    where.append("into-frozen-reward-slot")
                if sum(1 for s in slots if s.reward_group == slot.reward_group) == 2:
                    where.append("into-reward-slot-with-sibling")
            elif slot.slot_type == "pet" and any(s.slot_type == "levelup" for s in slots[i + 1:]):
                where.append("pet-slot-ahead-of-a-reward")
            if slot.frozen:
                where.append("over-frozen-item")
            if id(slot.obj) in (self.core._freeze_toggled_agent if seat == 0 else self.core._freeze_toggled_opponent):
                where.append("over-freeze-toggled-item")
            if frozen:
                where.append("placed-frozen")
            is_pet = name in PET_TIER
            if is_pet and (shop.shop_attack or shop.shop_health):
                where.append("canned-bonus>0")
                if shop.shop_attack + int(ENGINE_DATA["pets"][name]["baseAttack"]) > 50 or shop.shop_health + int(ENGINE_DATA["pets"][name]["baseHealth"]) > 50:
                    where.append("canned-bonus-capped-at-50")
            tier = PET_TIER[name] if is_pet else FOOD_TIER[name]
            if tier > tier_avail:
                where.append("out-of-tier-" + ("pet" if is_pet else "food"))
            if not is_pet and slot.cost != slot.obj.cost:
                where.append("over-discounted-food")
            if not is_pet and slot.obj.name not in FOOD_TIER:
                where.append("over-stocked-food")
            if slot.obj.name == name:
                where.append("same-item-as-before")
            if is_pet and name in team_names:
                where.append("species-of-a-team-pet")
            obj = place(player, i, name, frozen, self.keep)
            pre.append(["shop", seat, i, name, bool(frozen)])
            self.focus[seat].append(obj)
            self.note("edits", "shop-pet" if is_pet else "shop-food")
            self.note("placed", name)
            self.note("tierByTurn", f"{'pet' if is_pet else 'food'} tier {tier} on rule turn {min(turn, 11)}")
            for key in where:
                self.note("where", key)
            if n + 1 < len(plan):
                self.marker(pre, 0.1)
        if active and plan and (kind == "curriculum" or rng.random() < 0.5):
            self.marker(pre, 0.1)
            value = rng.choice([0, 3, 3, 3, 4, 5, 6, 6, 8, 10, 10, 13, 20])
            player.gold = value
            pre.append(["gold", seat, value]); self.note("edits", "set_gold")

    # ---- choices -----------------------------------------------------------------------------

    def snapshot(self, side):
        player = self.players()[side]
        return {"shop": [slot.obj for slot in player.shop.slots],
                "team": [None if slot.empty else slot.pet for slot in player.team]}

    def follow(self, side, options):
        """A legal action aimed at a placed item, or None."""
        snap = self.snap[side]
        placed = {i for i, obj in enumerate(snap["shop"]) if any(obj is f for f in self.focus[side])}
        grown = {i for i, pet in enumerate(snap["team"]) if pet is not None and any(pet is g for g in self.grown[side])}
        if not placed and not grown:
            return None
        picks, weights = [], []
        for flat in options:
            action = C.decode(flat); kind, source, target = action.as_tuple(); weight = 0
            if kind == int(T.BUY_PET) and source in placed:
                weight = 8 if target < 5 and snap["team"][target] is not None else 3
            elif kind == int(T.BUY_PET) and target in grown and snap["team"][target] is not None:
                weight = 3
            elif kind == int(T.BUY_FOOD) and source in placed:
                weight = 3
            elif kind == int(T.BUY_FOOD) and target in grown:
                weight = 1
            elif kind in (int(T.FREEZE), int(T.UNFREEZE)) and source in placed:
                weight = 1.5
            elif kind == int(T.SELL) and source in grown:
                weight = 0.3
            elif kind == int(T.COMBINE) and (source in grown or target in grown):
                weight = 4
            if weight:
                picks.append(flat); weights.append(weight)
        if not picks:
            return None
        self.stats["followed"] += 1
        return JointAction(C.decode(self.rng.choices(picks, weights)[0]), None)

    def decide(self, legal):
        decisions = []
        for side, options in enumerate(legal):
            self.snap[side] = self.snapshot(side)
            if options is None:
                decisions.append(None); continue
            decision = None
            if self.chain[side] is not None:
                reward, pill = self.chain[side]
                snap = self.snap[side]
                wanted = []
                for flat in options:
                    kind, source, target = C.decode(flat).as_tuple()
                    if kind == int(T.BUY_PET) and snap["shop"][source] is reward and target < 5 and snap["team"][target] is None:
                        wanted.append(flat)
                    elif kind == int(T.BUY_FOOD) and snap["shop"][source] is pill and target < 5 and snap["team"][target] is reward:
                        wanted.append(flat)
                if wanted:
                    decision = JointAction(C.decode(self.rng.choice(wanted)), None)
                else:
                    self.chain[side] = None
            if decision is None and self.force[side] is not None:
                wanted = [flat for flat in options if C.decode(flat).as_tuple()[0] == int(T.BUY_FOOD)
                          and self.snap[side]["shop"][C.decode(flat).as_tuple()[1]] is self.force[side]]
                if wanted and self.rng.random() < 0.9:
                    decision = JointAction(C.decode(wanted[0]), None)
                self.force[side] = None
            if decision is None and self.rng.random() < 0.5:
                decision = self.follow(side, options)
            if decision is None:
                decision = self.lives[side].decide(self.env, side, options)
            decisions.append(decision); self.stats["decisions"] += 1
        return decisions

    # ---- what happened to the placed items ---------------------------------------------------

    def after(self, result):
        step = result.result
        players = self.players()
        for side, plan in enumerate(result.primitive_actions):
            snap, player = self.snap[side], players[side]
            placed = lambda obj: any(obj is f for f in self.focus[side])
            grown = lambda pet: pet is not None and any(pet is g for g in self.grown[side])
            in_shop = lambda obj: any(slot.obj is obj for slot in player.shop.slots)
            for action in plan:
                kind, source, target = action.as_tuple()
                if kind == int(T.BUY_PET):
                    obj = snap["shop"][source]; onto = snap["team"][target] if target < 5 else None
                    if placed(obj):
                        self.note("bought", obj.name)
                        if onto is None:
                            self.note("fate", "pet bought"); self.grown[side].append(obj)
                        else:
                            self.note("fate", "pet bought and combined into a team pet"); self.grown[side].append(onto)
                    elif grown(onto):
                        self.note("fate", "a shop pet combined into a pet that came from a placed one")
                elif kind == int(T.BUY_FOOD):
                    obj = snap["shop"][source]
                    if placed(obj):
                        self.note("bought", obj.name); self.note("fate", "food bought")
                    if self.chain[side] is not None and obj is self.chain[side][1] and target < 5 and snap["team"][target] is self.chain[side][0]:
                        self.note("fate", "placed pill fed to a reward bought right after the edit: " + snap["team"][target].name)
                        self.chain[side] = None
                    if target < 5 and grown(snap["team"][target]):
                        self.note("fate", "a pet that came from a placed one was fed")
                elif kind == int(T.FREEZE) and placed(snap["shop"][source]):
                    self.note("fate", "frozen by an action")
                elif kind == int(T.UNFREEZE) and placed(snap["shop"][source]):
                    self.note("fate", "unfrozen by an action")
                elif kind == int(T.SELL) and grown(snap["team"][source]):
                    self.note("fate", "a pet that came from a placed one was sold")
                elif kind == int(T.COMBINE) and (grown(snap["team"][source]) or grown(snap["team"][target])):
                    self.note("fate", "a pet that came from a placed one combined on the team")
                    for pet in (snap["team"][source], snap["team"][target]):
                        if pet is not None and not grown(pet):
                            self.grown[side].append(pet)
                elif kind == int(T.ROLL):
                    for obj in snap["shop"]:
                        if placed(obj):
                            self.note("fate", "kept through a roll (frozen)" if in_shop(obj) else "rolled away")
                if plan and action is plan[-1]:
                    self.last_type[side] = kind
            if step.battle_resolved:
                self.chain[side] = None
            if step.battle_resolved and not step.terminated:
                for obj in self.focus[side]:
                    if in_shop(obj):
                        self.note("fate", "carried into the next turn (frozen)")
                    elif any(obj is o for o in snap["shop"]):
                        self.note("fate", "replaced at the next turn's roll")
                self.last_type[side] = None
            self.focus[side] = [obj for obj in self.focus[side] if in_shop(obj)]
            team = [slot.pet for slot in player.team if not slot.empty]
            self.grown[side] = [pet for pet in self.grown[side] if any(pet is t for t in team)]
            levels = {}
            for pet in self.grown[side]:
                before = self.levels[side].get(id(pet))
                if before is not None and pet.level > before:
                    self.note("fate", f"a pet that came from a placed one reached level {pet.level}")
                levels[id(pet)] = int(pet.level)
            self.levels[side] = levels
        if step.battle_resolved:
            self.stats["battles"] += 1


def pre_for_trace(pre):
    return [["obs", G.pack_obs(op[1], True)["obs"]] if op[0] == "obs" else op for op in pre]


def play(out, matches, first, full_every):
    targeted = H.load_targeted()
    pool = H.load_pool()
    H.selfplay_env.Battle = H.RecordingBattle
    total = {"edits": {}, "where": {}, "fate": {}, "placed": {}, "bought": {}, "tierByTurn": {}, "decisions": 0, "battles": 0,
             "editSteps": 0, "followed": 0, "steps": 0, "failed": [], "curriculum": 0, "scripted": 0, "turns": {}, "canfarm": 0, "maxBonus": 0}
    for index in range(first, first + matches):
        seed = 884400000 + index
        full = full_every > 0 and index % full_every == 0
        buffer = io.StringIO()
        steps = 0
        try:
            env = ParallelSelfPlayEnv(); obs = env.reset(seed=seed, paired_tape=True)
            G.header(buffer, seed, env, obs, full)
            session = Session(env, index, targeted, pool, seed)
            for step_index in range(8000):
                pre = session.edits()
                legal = G.legal_of(env)
                decisions = session.decide(legal)
                turn_before = int(env.core.turn)
                result = step_decisions(env, decisions)
                step = H.write_step(buffer, step_index, env, legal, decisions, result, full or bool(pre), pre_for_trace(pre), turn_before)
                session.after(result)
                steps += 1
                if step.terminated:
                    break
            else:
                raise RuntimeError("match did not end")
        except Exception as error:   # the reference itself failed: keep the evidence, skip the match
            total["failed"].append({"match": seed, "error": repr(error), "where": traceback.format_exc().strip().splitlines()[-6:]})
            print("reference failed on match", seed, repr(error), file=sys.stderr)
            continue
        out.write(buffer.getvalue())
        total["steps"] += steps
        total["curriculum"] += int(session.curriculum); total["scripted"] += int(session.scripted)
        total["canfarm"] += int(session.canfarm is not None)
        total["maxBonus"] = max(total["maxBonus"], *(int(p.shop.shop_attack) for p in session.players()))
        last = int(env.core.turn) - 1
        total["turns"][last] = total["turns"].get(last, 0) + 1
        for key, value in session.stats.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    total[key][k] = total[key].get(k, 0) + v
            else:
                total[key] += value
    for key in ("edits", "where", "fate", "placed", "bought", "tierByTurn", "turns"):
        total[key] = dict(sorted(total[key].items(), key=lambda kv: str(kv[0])))
    return total


if MAIN:
    OUT, MATCHES, FIRST, FULL_EVERY = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    with gzip.open(OUT, "wt", compresslevel=5) as out:
        summary = play(out, MATCHES, FIRST, FULL_EVERY)
    summary = {"matches": MATCHES, "first": FIRST, "written": MATCHES - len(summary["failed"]), **summary}
    print(json.dumps(summary))
