"""Biased reference matches for rule coverage (acceptance item 3).

Random legal play buys a bit of everything and levels nothing, so most
level-2/3 abilities, the shop overflow rules and the abilities that need a
particular neighbour are never exercised. Here each seat follows a small
scripted preference ("scenario"): which species to collect and combine, what
to feed them, what to sell or put to sleep with a Sleeping Pill and when,
whether to grab level-up rewards. Everything goes through the public
interface (`step_decisions` with legal actions only), so the files replay in
the Rust checker exactly like the random ones. Trace format: see
`gen_env_golden.py`; states carry the extended block (`--ext` there).

Usage (in the training image, same docker line as gen_env_golden.py):
  gen_env_targeted.py <out.jsonl.gz> <matches> <first index> <full obs every N> [scenario name]

The scenario "cowfull" (the Cow bought out of a shop of ten pet and reward
slots, see `cowfull_choose`) runs only when named; its summary line carries
reference-side counts of what RefillShops did under "cow".

Seeds are 883300000 + index. A match the reference cannot finish (an
exception inside the reference) is not written; it is reported on stderr and
in the final JSON line under "failed".
"""
import gzip, io, json, random, sys, traceback

import gen_env_golden as G
from sapai.data import data
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import ActionType, FlatActionCodec as C
from sapai_gym1.rl.joint_action import JointAction, step_decisions

G.EXT = True
OUT, MATCHES, FIRST, FULL_EVERY = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
ONLY = sys.argv[5] if len(sys.argv) > 5 else None

TIER = {name: entry["tier"] for name, entry in data["pets"].items()}
ROLLABLE = [n for n, e in data["pets"].items() if isinstance(e["tier"], int) and "StandardPack" in e.get("packs", []) and not e.get("hidden")]
TIER1 = [n for n in ROLLABLE if TIER[n] == 1]
FAINT = ["pet-ant", "pet-cricket", "pet-flamingo", "pet-hedgehog", "pet-rat", "pet-spider", "pet-badger", "pet-sheep",
         "pet-turtle", "pet-deer", "pet-rooster", "pet-mammoth", "pet-fly", "pet-shark", "pet-ox"]
HURT = ["pet-peacock", "pet-camel", "pet-blowfish", "pet-gorilla", "pet-wolverine"]
SELL = ["pet-beaver", "pet-duck", "pet-pig", "pet-pigeon"]
TURN = ["pet-swan", "pet-worm", "pet-giraffe", "pet-penguin", "pet-squirrel", "pet-snail", "pet-monkey", "pet-bison", "pet-parrot"]
BUY = ["pet-otter", "pet-cow", "pet-dragon", "pet-fish", "pet-horse", "pet-dog", "pet-turkey", "pet-cat", "pet-rabbit",
       "pet-seal", "pet-scorpion"]
PERKS = ["food-honey", "food-mushroom", "food-garlic", "food-meat-bone", "food-melon", "food-steak", "food-chili",
         "food-cake", "food-bread"]
PILL = "food-sleeping-pill"
TOKENS = {"pet-zombie-cricket", "pet-ram", "pet-chick", "pet-bus", "pet-bee"}


class Scenario:
    def __init__(self, name, want, **kw):
        self.name = name
        self.want = list(want)
        self.pill = set(kw.get("pill", ()))          # species to put to sleep
        self.pill_level = kw.get("pill_level", 1)    # ... once at this level
        self.need = set(kw.get("need", ()))          # ... and only while these species are on the team
        self.need_count = kw.get("need_count", 0)    # ... and at least this many pets are on the team
        self.sell = set(kw.get("sell", ()))
        self.sell_level = kw.get("sell_level", 1)
        self.foods = kw.get("foods", ())             # food names, or "any"
        self.perk_first = kw.get("perk_first", False)  # give the pill target a perk first when one is on offer
        self.tier1 = kw.get("tier1", False)          # buy every tier-1 pet (Dragon)
        self.levelup = kw.get("levelup", 0.0)        # probability of taking a level-up reward, then feeding it at once
        self.freeze = kw.get("freeze", 0.0)          # probability of freezing something before rolling
        self.eps = kw.get("eps", 0.08)               # probability of a uniformly random legal action
        self.hold = kw.get("hold", False)            # combine two team copies only when both are level 2
        self.hold_turn = kw.get("hold_turn", 0)      # before this turn: collect separate copies, combine nothing
        self.late = set(kw.get("late", ()))          # species bought only once want_level is reached
        self.place = kw.get("place", {})             # preferred team slot per species
        self.fixed_idle = kw.get("idle")             # overrides the match's idle turns for this seat's script
        self.freeze_kind = kw.get("freeze_kind")     # "food": random freezes touch foods only
        self.crowd = kw.get("crowd", 0)              # combine team copies / buy a Cow only with this many shop slots
        self.freeze_first = True                     # crowded-shop scripts: freeze before buying or combining
        self.idle = 0                                # turns to sit out first (set per match, see play)
        self.want_level = kw.get("want_level", {})   # pill only once these species have reached these levels
        self.fresh = None                            # team slot of a just-bought level-up reward

    def describe(self):
        return {k: (sorted(v) if isinstance(v, set) else v) for k, v in self.__dict__.items() if k != "fresh"}


def make_scenario(rng):
    kind = ONLY or rng.choice(["faint", "faint", "faint", "hurt", "hurt", "sell", "sell", "turn", "turn", "buy", "buy",
                               "levelup", "levelup", "bison", "dragon", "parrot", "overflow", "overflow", "wolverine",
                               "scorpion", "chocolate", "summons", "twostack"])
    level = rng.choice([1, 2, 2, 3, 3])
    if kind == "faint":
        s = rng.choice(FAINT)
        friends = rng.sample(["pet-ant", "pet-fish", "pet-horse", "pet-dog", "pet-turkey", "pet-ox", "pet-fly", "pet-shark",
                              "pet-rabbit", "pet-peacock", "pet-camel", "pet-tiger", "pet-seal", "pet-cricket"], 2)
        if s in ("pet-fly", "pet-shark", "pet-ox"):   # observers: someone else must faint in front of them
            front = rng.choice(["pet-cricket", "pet-ant", "pet-sheep", "pet-spider", "pet-hedgehog", "pet-deer"])
            return Scenario(kind + ":" + s, [front, s] + friends[:1], pill={front} | TOKENS, need={s}, pill_level=1,
                            foods=["food-chocolate", "food-honey", "food-mushroom"], perk_first=rng.random() < 0.5)
        return Scenario(kind + ":" + s, [s] + friends, pill={s}, pill_level=level, need=set(friends[:1]) if rng.random() < 0.6 else (),
                        foods=["food-chocolate"] + (["food-honey", "food-mushroom"] if rng.random() < 0.5 else []),
                        perk_first=rng.random() < 0.5)
    if kind == "hurt":
        h = rng.choice(HURT)
        source = rng.choice(["pet-hedgehog", "pet-hedgehog", "pet-badger"])
        extra = rng.sample(["pet-rabbit", "pet-ant", "pet-fish", "pet-camel", "pet-peacock", "pet-cricket"], 2)
        order = [source, h] + extra if source == "pet-hedgehog" or rng.random() < 0.5 else [h, source] + extra
        return Scenario(kind + ":" + h, order, pill={source}, need={h}, pill_level=rng.choice([1, 1, 2]), want_level={h: level},
                        foods=["food-chocolate", "food-garlic", "food-melon", "food-pear"])
    if kind == "wolverine":
        return Scenario(kind, ["pet-hedgehog", "pet-wolverine", "pet-hedgehog", "pet-fish", "pet-ant"], pill={"pet-hedgehog"},
                        need={"pet-wolverine"}, need_count=5, foods=["food-pear", "food-apple", "food-garlic"], eps=0.03,
                        hold=True, want_level={"pet-wolverine": rng.choice([1, 1, 2])})
    if kind == "deep":       # one species to level 3 in the late shop, then the event that fires its ability
        s = rng.choice(["pet-blowfish", "pet-camel", "pet-gorilla", "pet-peacock", "pet-mammoth", "pet-ox", "pet-wolverine",
                        "pet-turtle", "pet-fly", "pet-shark", "pet-cat", "pet-dragon", "pet-seal", "pet-turkey", "pet-rooster"])
        common = dict(foods=["food-chocolate"], eps=0.02, idle=10)
        if s in ("pet-blowfish", "pet-camel", "pet-gorilla", "pet-peacock"):
            return Scenario(kind + ":" + s, [s, "pet-hedgehog"], late={"pet-hedgehog"}, want_level={s: 3}, pill={"pet-hedgehog"},
                            need={s}, **common)
        if s == "pet-wolverine":
            return Scenario(kind + ":" + s, [s, "pet-hedgehog", "pet-hedgehog", "pet-fish", "pet-ant"], pill={"pet-hedgehog"},
                            late={"pet-hedgehog", "pet-fish", "pet-ant"}, want_level={s: rng.choice([2, 3])}, need={s},
                            need_count=5, hold=True, **common)
        if s in ("pet-ox", "pet-fly", "pet-shark", "pet-turkey"):
            front = rng.choice(["pet-cricket", "pet-sheep", "pet-ant"])
            return Scenario(kind + ":" + s, [s, front], late={front}, want_level={s: 3}, pill={front} | TOKENS, need={s},
                            place={front: 0, s: 1}, **common)
        if s in ("pet-cat", "pet-seal"):
            return Scenario(kind + ":" + s, [s, "pet-fish"], foods="any", eps=0.02, idle=10)
        if s == "pet-dragon":
            return Scenario(kind + ":" + s, [s], tier1=True, **common)
        return Scenario(kind + ":" + s, [s, "pet-fish"], late={"pet-fish"}, want_level={s: 3}, pill={s}, pill_level=3,
                        need={"pet-fish"}, place={s: 0, "pet-fish": 1}, **common)
    if kind == "crowdloose":  # as "crowd", but buy and combine before freezing: fresh rewards and stock stay displaceable
        pair = rng.sample(TIER1, 2)
        scenario = Scenario(kind, ["pet-pigeon", pair[0], pair[0], pair[1], pair[1], "pet-cow"], sell={"pet-pigeon"},
                            sell_level=rng.choice([1, 2, 3]), freeze=0.97, freeze_kind=rng.choice(["food", "food", None]),
                            hold=True, crowd=rng.choice([8, 9, 10]), levelup=0.0, eps=0.02, idle=rng.choice([4, 8, 9]))
        scenario.freeze_first = False
        return scenario
    if kind == "crowd":      # ten shop slots, most of them frozen: overflow, discarded rewards, squeezed capacity
        pair = rng.sample(TIER1, 2)
        return Scenario(kind, ["pet-pigeon", pair[0], pair[0], pair[1], pair[1], "pet-cow"], sell={"pet-pigeon"},
                        sell_level=rng.choice([1, 2, 3]), freeze=0.97, freeze_kind=rng.choice(["food", "food", None]),
                        hold=True, crowd=rng.choice([8, 9, 10]), levelup=0.0, eps=0.02, idle=rng.choice([4, 8, 9]))
    if kind == "cowcrowd":   # ten pet/reward slots, then a Cow: its two Milks must displace something
        pair = rng.sample(TIER1, 2)
        return Scenario(kind, [pair[0], pair[0], pair[1], pair[1], "pet-cow"], freeze=rng.choice([0.97, 0.8, 0.6]), freeze_kind="pet", hold=True,
                        crowd=6, levelup=0.0, eps=0.02, idle=rng.choice([4, 8]))
    if kind == "cowfull":    # ten pet/reward slots with the Cow among them (see cowfull_choose); only by name
        scenario = Scenario(kind, [], eps=0.0, idle=rng.choice([2, 3, 3, 4, 4, 5]))
        scenario.cf = dict(phase="stage", stack=rng.random() < 0.3, split=rng.random() < 0.4,
                           mode=rng.choice(["all"] * 8 + ["none"] * 2 + ["pets"] * 3 + ["one"] * 3 + ["toggle"] * 3 + ["few"] * 2),
                           guard=rng.random() < 0.5, food=rng.random() < 0.3, milk=rng.choice([0, 1, 1, 2]), burst=None)
        return scenario
    if kind == "sealreward":  # a Seal taken as a level-up reward on turns 7-8 and fed before anything rebinds its stream
        return Scenario(kind, rng.sample(TIER1, 3), hold_turn=7, levelup=1.0, foods="any", eps=0.02, idle=3)
    if kind == "sell":
        s = rng.choice(SELL)
        return Scenario(kind + ":" + s, [s, rng.choice(TIER1), rng.choice(ROLLABLE)], sell={s}, sell_level=level,
                        foods=["food-chocolate"], freeze=0.3 if s == "pet-pigeon" else 0.05)
    if kind == "overflow":   # fill the ten shop slots: frozen stock, Pigeon crumbs, Worm apples, Cow milk, level-up rewards
        return Scenario(kind, ["pet-pigeon", rng.choice(["pet-worm", "pet-cow", "pet-fish", "pet-ant"]), "pet-pigeon", "pet-cow"],
                        sell={"pet-pigeon"}, sell_level=rng.choice([1, 2, 3]), freeze=rng.choice([0.5, 0.9, 0.97]),
                        foods=["food-chocolate"], levelup=rng.choice([0.0, 0.3]), eps=0.03)
    if kind == "twostack":   # two separate level-2 stacks of one species, combined on the team
        s = rng.choice(TIER1 + ROLLABLE)
        return Scenario(kind + ":" + s, [s, s], hold=True, foods=["food-chocolate"], eps=0.03)
    if kind == "turn":
        s = rng.choice(TURN)
        friends = rng.sample(ROLLABLE, 2)
        return Scenario(kind + ":" + s, [friends[0], s, friends[1]] if s in ("pet-giraffe", "pet-snail", "pet-parrot") else [s] + friends,
                        foods=["food-chocolate", "food-apple"] + ([rng.choice(PERKS)] if rng.random() < 0.5 else []))
    if kind == "buy":
        s = rng.choice(BUY)
        foods = "any" if s in ("pet-cat", "pet-rabbit", "pet-seal") else ["food-chocolate", "food-milk", "food-canned-food"]
        return Scenario(kind + ":" + s, [s] + rng.sample(ROLLABLE, 2), foods=foods, levelup=0.3)
    if kind == "levelup":    # take discover rewards and use them before anything rebinds their random stream
        return Scenario(kind, rng.sample(TIER1, 2) + [rng.choice(["pet-spider", "pet-seal", "pet-ant"])], levelup=1.0,
                        foods=["food-chocolate", "food-honey", "food-apple", PILL, "food-cupcake"], eps=0.05)
    if kind == "bison":
        return Scenario(kind, [rng.choice(TIER1), "pet-bison", "pet-bison"], foods=["food-chocolate"], eps=0.04)
    if kind == "dragon":
        return Scenario(kind, ["pet-dragon"], tier1=True, foods=["food-chocolate"], eps=0.04)
    if kind == "parrot":
        return Scenario(kind, [rng.choice(ROLLABLE), "pet-parrot", "pet-parrot"], foods=["food-chocolate"], eps=0.1)
    if kind == "scorpion":
        return Scenario(kind, ["pet-scorpion", "pet-rabbit", "pet-seal"], pill={"pet-scorpion"}, need={"pet-rabbit"},
                        foods=["food-mushroom", "food-honey"], perk_first=True)
    if kind == "chocolate":
        return Scenario(kind, rng.sample(ROLLABLE, 2), foods=["food-chocolate"], levelup=0.5, eps=0.05)
    if kind == "summons":    # summoners in front of the pets that react to a summon
        front = rng.choice(["pet-cricket", "pet-sheep", "pet-deer", "pet-rooster", "pet-spider", "pet-fly"])
        return Scenario(kind + ":" + front, [front, rng.choice(["pet-horse", "pet-dog", "pet-turkey"]),
                                              rng.choice(["pet-horse", "pet-dog", "pet-turkey", "pet-fly"]), front],
                        pill={front} | TOKENS, pill_level=1, want_level={front: level}, need_count=3,
                        foods=["food-chocolate", "food-honey", "food-mushroom"],
                        perk_first=rng.random() < 0.5)
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------
# "cowfull": the Cow bought out of a shop whose ten slots are all pets and level-up rewards.
#
# RefillShops keeps the non-food slots and adds two Milks; with nine non-food slots left after
# the purchase it must drop the last unfrozen one, or stock a single Milk when all are frozen.
# Shop arithmetic (Shop.check_rules, Shop.levelup): a turn from 9 on starts with five pet slots
# and two foods; a level-up of a team pet prepends a reward pair and drops unfrozen slots from the
# end when the shop would exceed ten. So the seat needs three level-ups in the turn it buys the
# Cow, and at most one of them may cost a shop pet: 5 - 1 + 2 + 2 + 2 = 10. Two of them are free
# combines prepared on the team (a pet with one experience next to a fresh copy), the third is a
# team pet with one experience plus its copy (or a Chocolate) in the shop. The prepared team is
# carried over the turns until a Cow is offered, so battles are fought; a loss costs one life.
# Variant "stack": two prepared pairs and a team Cow with one experience; the shop Cow bought onto
# her levels her up (third reward pair) and her Buy ability then fires at level 2 on ten slots.
# Counts for the summary line come from a wrapper around the reference function (cf_hook).
CF = {"events": {}, "matches": {}, "funnel": {}}
CF_NOW = {"keys": set(), "events": [], "env": None}


def cf_note(key):
    CF["funnel"][key] = CF["funnel"].get(key, 0) + 1


def cf_hook():
    """Count what every RefillShops call does. Observation only: the reference function runs unchanged."""
    import sapai.effects as E
    original = E.func_dict["RefillShops"]

    def counted(apet, *args, **kwargs):
        shop = apet.shop
        if shop is not None:
            count = int(apet.ability["effect"].get("n", 2))
            nonfood = [slot for slot in shop if slot.slot_type != "food"]
            missing = max(0, len(nonfood) + count - shop.max_slots)
            loose = [slot for slot in nonfood if not slot.frozen]
            removed = loose[::-1][:missing]
            stocked = min(count, shop.max_slots - (len(nonfood) - len(removed)))
            keys = [f"milks-stocked:{stocked}-of-{count}"]
            if removed:
                keys.append(f"overflow-removes:{len(removed)}")
            if missing:
                core = CF_NOW["env"].core
                toggled = core._freeze_toggled_agent if apet.player is core.agent else core._freeze_toggled_opponent
                keys.append(f"full:level-{apet.level}:removes-{len(removed)}:milks-{stocked}")
                keys.append(f"full:unfrozen-{min(len(loose), 4)}{'+' if len(loose) >= 4 else ''}")
                for slot in removed:
                    keys.append("full:removed-" + slot.slot_type + ("-toggled" if id(slot.obj) in toggled else ""))
                    if slot.reward_group is not None:
                        siblings = sum(1 for other in nonfood if other.reward_group == slot.reward_group)
                        keys.append("full:removed-reward-" + ("of-pair" if siblings == 2 else "single"))
                if stocked < count:
                    keys.append(f"full:short-toggled-{sum(1 for slot in nonfood if id(slot.obj) in toggled)}")
            CF_NOW["events"].extend(keys)
        return original(apet, *args, **kwargs)

    E.func_dict["RefillShops"] = counted


def cf_mask(env, side, mask):
    """An END freeze mask chosen by the script, or 0 when it is not a legal plan."""
    if not mask:
        return 0
    try:
        G.expand_end_freeze(env.core, side, mask)
    except ValueError:
        return 0
    return mask


def cf_project(shop, steps):
    """Slot arithmetic of a plan: ("buy", slot), ("freeze", slot), ("lvl",) = Shop.levelup(1)."""
    slots = [[slot, slot.slot_type, bool(slot.frozen)] for slot in shop]
    for step in steps:
        if step[0] == "buy":
            slots = [x for x in slots if x[0] is not step[1]]
        elif step[0] in ("freeze", "thaw"):
            for x in slots:
                if x[0] is step[1]:
                    x[2] = step[0] == "freeze"
        else:
            over = max(0, len(slots) + 2 - 10)
            loose = [i for i in range(len(slots) - 1, -1, -1) if not slots[i][2]]
            if len(loose) < over:
                continue
            drop = set(loose[:over])
            slots = [[None, "levelup", False], [None, "levelup", False]] + [x for i, x in enumerate(slots) if i not in drop]
    return slots


def cowfull_choose(env, side, legal_list, sc, rng):
    core = env.core
    player = core.agent if side == 0 else core.opponent
    legal = set(legal_list)
    st = sc.cf
    count = core.shop_action_count_agent if side == 0 else core.shop_action_count_opponent
    shop = list(player.shop)[:10]
    team = [None if slot.empty else slot.pet for slot in player.team]
    gold, turn = player.gold, player.turn

    def index_of(slot):
        for i, other in enumerate(shop):
            if other is slot:
                return i
        return None

    def burn():
        if C.ROLL in legal:
            return C.ROLL
        if C.END_TURN in legal:
            return C.END_TURN
        return rng.choice(legal_list)

    # After the purchase: a free Milk or two, then uniformly random legal play to the end of the match.
    if st["phase"] == "after":
        milks = [i for i, slot in enumerate(shop) if slot.slot_type == "food" and slot.obj.name == "food-milk" and slot.cost <= gold]
        if milks and st["milk"] > 0:
            st["milk"] -= 1
            i = rng.choice(milks)
            feeds = sorted(a for a in legal if C.BUY_FOOD + i * 6 <= a < C.BUY_FOOD + i * 6 + 6)
            if feeds:
                return rng.choice(feeds)
        return rng.choice(legal_list)

    # Sit the first turns out with an empty team (both seats: draws), banking copies of one species.
    if turn <= sc.idle:
        frozen = [slot for slot in shop if slot.slot_type == "pet" and slot.frozen]
        bank = frozen[0].obj.name if frozen else None
        if len(frozen) < 3:
            names = [slot.obj.name for slot in shop if slot.slot_type == "pet"]
            for i, slot in enumerate(shop):
                if slot.slot_type != "pet" or slot.frozen or C.FREEZE + i not in legal:
                    continue
                if slot.obj.name == bank or (bank is None and (names.count(slot.obj.name) > 1 or count >= 3)):
                    return C.FREEZE + i
        return burn()

    by = {}
    for i, pet in enumerate(team):
        if pet is not None:
            by.setdefault(pet.name, []).append(i)

    def exps(name):
        return sorted((team[i].total_experience for i in by.get(name, [])), reverse=True)

    species = [name for name in by if name != "pet-cow"]
    pairs = [name for name in species if exps(name) == [1, 0]]
    singles = [name for name in species if exps(name) == [1]]
    seeds = [name for name in species if exps(name) == [0]]
    empty = [i for i, pet in enumerate(team) if pet is None]
    cow_team = by.get("pet-cow", [])
    cows = [slot for slot in shop if slot.slot_type == "pet" and slot.obj.name == "pet-cow"]
    foods = [slot for slot in shop if slot.slot_type == "food"]
    chocolates = [slot for slot in foods if slot.obj.name == "food-chocolate"]
    nonfood = [slot for slot in shop if slot.slot_type != "food"]

    # The purchase turn, once started: level-ups, then the freeze pattern, then the Cow.
    burst = st["burst"]
    if burst is not None:
        cow = burst["cow"]
        if index_of(cow) is None:          # displaced after all: give up on this Cow
            cf_note("burst-lost-cow")
            st["burst"] = None
            return burn()
        if burst["stage"] == "level":
            if burst["guard"]:
                burst["guard"] = False
                if not cow.frozen and C.FREEZE + index_of(cow) in legal:
                    return C.FREEZE + index_of(cow)
            trigger = burst["trigger"]
            if trigger is not None:
                burst["trigger"] = None
                i = index_of(trigger)
                if trigger.slot_type == "food":
                    action = C.BUY_FOOD + i * 6 + by[burst["single"]][0]
                else:
                    action = C.BUY_PET + i * 6 + by[burst["single"]][0]
                if action in legal:
                    return action
            for name in pairs:
                a, b = by[name]
                action = C.COMBINE + a * 5 + b
                if action in legal:
                    return action
            burst["stage"] = "vary"
            others = [slot for slot in nonfood if slot is not cow]
            can_freeze = [slot for slot in others if not slot.frozen and C.FREEZE + index_of(slot) in legal]
            can_thaw = [slot for slot in others if slot.frozen and C.UNFREEZE + index_of(slot) in legal]
            mode, todo = st["mode"], []
            if mode == "toggle" and not can_thaw:
                mode = "one"
            if mode == "all":
                todo = [("f", slot) for slot in can_freeze]
            elif mode == "pets":
                todo = [("f", slot) for slot in can_freeze if slot.slot_type == "pet"]
            elif mode == "one" and can_freeze:
                keep = rng.choice(can_freeze)
                todo = [("f", slot) for slot in can_freeze if slot is not keep]
            elif mode == "toggle":
                thaw = rng.sample(can_thaw, min(len(can_thaw), rng.choice([1, 1, 1, 2, 3])))
                todo = [("u", slot) for slot in thaw]
                if rng.random() < 0.7:
                    todo += [("f", slot) for slot in can_freeze]
            elif mode == "few":
                todo = [("f", slot) for slot in can_freeze if rng.random() < 0.5]
            if st["stack"] and st["food"]:   # frozen food: the Cow's own level-up must displace a pet or reward instead
                extra = [("f", slot) for slot in foods if not slot.frozen and C.FREEZE + index_of(slot) in legal]
                plan = [("freeze" if kind == "f" else "thaw", slot) for kind, slot in todo + extra]
                after = cf_project(shop, plan + [("buy", cow), ("lvl",)])
                if sum(1 for x in after if x[1] != "food") >= 9:
                    todo += extra
            rng.shuffle(todo)
            burst["todo"] = todo[:max(0, 47 - count)]
        if burst["stage"] == "vary":
            while burst["todo"]:
                kind, slot = burst["todo"].pop()
                i = index_of(slot)
                if i is None:
                    continue
                action = (C.FREEZE if kind == "f" else C.UNFREEZE) + i
                if action in legal:
                    return action
            burst["stage"] = "buy"
        # buy
        i = index_of(cow)
        action = C.BUY_PET + i * 6 + (cow_team[0] if st["stack"] else (empty[0] if empty else 0))
        st["burst"] = None
        if action in legal:
            st["phase"] = "after"
            cf_note(f"bought:{'stack' if st['stack'] else 'main'}:nonfood-{len(nonfood)}:turn-{min(turn, 15)}{'+' if turn >= 15 else ''}")
            cf_note("bought:mode-" + st["mode"])
            return action
        cf_note("burst-cannot-buy")
        return burn()

    staged = len(pairs) == 2 and ((st["stack"] and bool(cow_team) and team[cow_team[0]].total_experience == 1)
                                  or (not st["stack"] and len(singles) == 1)) or st.get("split_done", False)
    if staged and "staged" not in st:
        st["staged"] = turn
        cf_note(f"staged:turn-{min(turn, 12)}{'+' if turn >= 12 else ''}")
    if cows and "seen" not in st:
        st["seen"] = turn
        cf_note("cow-seen")

    # A Cow on offer.
    if cows:
        cow = cows[0]
        ci = index_of(cow)
        if st["stack"]:
            if not cow_team:
                if gold >= cow.cost and empty and C.BUY_PET + ci * 6 + empty[0] in legal:
                    return C.BUY_PET + ci * 6 + empty[0]
            elif team[cow_team[0]].total_experience == 0:
                if gold >= cow.cost and C.BUY_PET + ci * 6 + cow_team[0] in legal:
                    return C.BUY_PET + ci * 6 + cow_team[0]
            elif team[cow_team[0]].total_experience == 1 and pairs and gold >= cow.cost:
                for guard in ([True, False] if st["guard"] else [False, True]):
                    if guard and (cow.frozen or C.FREEZE + ci not in legal):
                        continue
                    before = cf_project(shop, ([("freeze", cow)] if guard else []) + [("lvl",)] * len(pairs))
                    if not any(x[0] is cow for x in before):
                        continue
                    after = cf_project(shop, ([("freeze", cow)] if guard else []) + [("lvl",)] * len(pairs) + [("buy", cow), ("lvl",)])
                    if sum(1 for x in after if x[1] != "food") >= 9:
                        st["burst"] = dict(stage="level", cow=cow, guard=guard, trigger=None, single=None, todo=[])
                        cf_note("burst-start:stack")
                        return cowfull_choose(env, side, legal_list, sc, rng)
        else:
            triggers = []
            if singles:
                triggers = [slot for slot in shop if slot.slot_type == "pet" and slot.obj.name == singles[0]] + chocolates
                rng.shuffle(triggers)
            options = [(t, guard) for t in triggers + [None] for guard in ([True, False] if st["guard"] else [False, True])]
            for trigger, guard in options:
                if gold < cow.cost + (trigger.cost if trigger else 0) or (guard and (cow.frozen or C.FREEZE + ci not in legal)):
                    continue
                plan = ([("freeze", cow)] if guard else []) + ([("buy", trigger), ("lvl",)] if trigger else []) + [("lvl",)] * len(pairs)
                after = cf_project(shop, plan)
                if any(x[0] is cow for x in after) and sum(1 for x in after if x[1] != "food") == 10:
                    st["burst"] = dict(stage="level", cow=cow, guard=guard, trigger=trigger, single=singles[0] if trigger else None, todo=[])
                    cf_note("burst-start:main")
                    return cowfull_choose(env, side, legal_list, sc, rng)
        if not cow.frozen and C.FREEZE + ci in legal:    # keep her for a later turn
            return C.FREEZE + ci

    # Stack variant: one experience for the team Cow from a Chocolate.
    if st["stack"] and chocolates:
        choc = chocolates[0]
        i = index_of(choc)
        if cow_team and team[cow_team[0]].total_experience == 0:
            if gold >= choc.cost and C.BUY_FOOD + i * 6 + cow_team[0] in legal:
                return C.BUY_FOOD + i * 6 + cow_team[0]
        if (not cow_team or team[cow_team[0]].total_experience == 0) and turn >= 9 and not choc.frozen and C.FREEZE + i in legal \
                and not any(slot.frozen for slot in chocolates):
            return C.FREEZE + i

    # Main variant without a copy for the third level-up: a Chocolate does it too.
    if not st["stack"] and len(pairs) == 2 and len(singles) == 1 and chocolates and turn >= 9:
        if not any(slot.frozen for slot in chocolates) and not chocolates[0].frozen and C.FREEZE + index_of(chocolates[0]) in legal \
                and not any(slot.slot_type == "pet" and slot.obj.name == singles[0] for slot in shop):
            return C.FREEZE + index_of(chocolates[0])

    # Prepare the team: copies onto single pets, a second pet for two species, new species while there is room.
    want_species = 2 if st["stack"] else 3
    names = [slot.obj.name for slot in shop if slot.slot_type == "pet"]
    for i, slot in enumerate(shop):
        if slot.slot_type != "pet" or slot.obj.name == "pet-cow":
            continue
        name = slot.obj.name
        if name in seeds:
            action = C.BUY_PET + i * 6 + by[name][0]
        elif name in singles and len(pairs) < 2 and empty and not st.get("split_done"):
            action = C.BUY_PET + i * 6 + empty[0]
        elif name in singles and not st["stack"]:      # the third level-up's copy stays in the shop
            held = any(other.frozen and other.obj.name == name for other in shop if other.slot_type == "pet")
            if not held and not slot.frozen and C.FREEZE + i in legal:
                return C.FREEZE + i
            continue
        else:
            continue
        if slot.cost <= gold and action in legal:
            return action
        if not slot.frozen and C.FREEZE + i in legal:
            return C.FREEZE + i
    if len(species) < want_species and empty:
        fresh = [(i, slot) for i, slot in enumerate(shop) if slot.slot_type == "pet" and slot.obj.name not in by
                 and slot.obj.name != "pet-cow" and slot.cost <= gold]
        fresh.sort(key=lambda item: (-names.count(item[1].obj.name), not item[1].frozen, item[0]))
        for i, slot in fresh:
            action = C.BUY_PET + i * 6 + empty[0]
            if action in legal:
                return action

    # Out of gold with a Cow kept for the next turn: optionally one prepared combine now, its
    # rewards and every pet frozen by the END plan, so the next turn starts with seven of ten.
    if gold == 0 and C.END_TURN in legal and cows and cows[0].frozen and staged and turn >= 9:
        ready = st["stack"] or any(slot.slot_type == "pet" and slot.frozen and slot.obj.name in singles for slot in shop)
        if st["split"] and not st.get("split_done") and len(pairs) == 2 and ready:
            a, b = by[pairs[0]]
            if C.COMBINE + a * 5 + b in legal:
                st["split_done"] = True
                cf_note("split")
                return C.COMBINE + a * 5 + b
        mask = 0
        for i, slot in enumerate(shop):
            if slot.slot_type != "food" and not slot.frozen and C.FREEZE + i in legal \
                    and (st.get("split_done") or rng.random() < 0.6):
                mask |= 1 << i
        st["mask"] = mask
        return C.END_TURN
    return burn()


def choose(env, side, legal_list, sc, rng):
    """One decision of the scripted preference. Returns a flat action id from `legal_list`."""
    if sc.name == "cowfull":
        return cowfull_choose(env, side, legal_list, sc, rng)
    core = env.core
    player = core.agent if side == 0 else core.opponent
    legal = set(legal_list)
    count = core.shop_action_count_agent if side == 0 else core.shop_action_count_opponent
    team = [None if slot.empty else slot.pet for slot in player.team]
    if player.turn <= sc.idle:
        # Sit the early turns out with an empty team (both seats do, so the battles are draws and
        # nobody loses a life): freeze what we want, burn the gold on rolls, end the turn.
        for index, slot in enumerate(list(player.shop)[:10]):
            keep = (slot.slot_type != "food" and slot.obj.name in sc.want) or (slot.slot_type == "food" and slot.obj.name == "food-chocolate")
            if keep and C.FREEZE + index in legal:
                return C.FREEZE + index
        if C.ROLL in legal and count < 46:
            return C.ROLL
        if C.END_TURN in legal:
            return C.END_TURN
        return rng.choice(legal_list)
    if rng.random() < sc.eps:
        return rng.choice(legal_list)
    names = [None if pet is None else pet.name for pet in team]
    shop = list(player.shop)[:10]
    gold = player.gold

    def food_slots(wanted):
        return [(i, s) for i, s in enumerate(shop) if s.slot_type == "food" and s.cost <= gold and wanted(s.obj.name)]

    # A just-bought level-up reward: feed it now, while its random stream is still its own.
    if sc.fresh is not None:
        slot_index, sc.fresh = sc.fresh, None
        pet = team[slot_index]
        if pet is not None:
            prefer = [PILL] if pet.name in ("pet-spider", "pet-ant") else []
            options = food_slots(lambda n: True)
            options.sort(key=lambda item: (item[1].obj.name not in prefer, rng.random()))
            for index, slot in options:
                action = C.BUY_FOOD + index * 6 + slot_index
                if action in legal:
                    return action

    cow_here = sc.freeze_kind == "pet" and any(x.slot_type != "food" and x.obj.name == "pet-cow" for x in shop)
    cow_nonfood = sum(1 for x in shop if x.slot_type != "food")
    # With the Cow on offer: run the held combines back to back (their rewards stay unfrozen), then buy her.
    cow_now = cow_here and ((gold >= 3 and cow_nonfood >= 10)
                            or (cow_nonfood >= sc.crowd and any(C.COMBINE <= a < C.SELL for a in legal)))
    if sc.crowd and sc.freeze_first and not cow_now and rng.random() < sc.freeze:   # crowded-shop script: freeze before anything can be displaced
        freezes = [a for a in legal if C.FREEZE <= a < C.UNFREEZE
                   and (sc.freeze_kind != "food" or shop[a - C.FREEZE].slot_type == "food")
                   and (sc.freeze_kind != "pet" or shop[a - C.FREEZE].slot_type != "food")]
        if freezes:
            return rng.choice(freezes)

    # Combine copies of wanted species.
    holding = player.turn < sc.hold_turn
    crowded = len(player.shop) >= sc.crowd
    if sc.freeze_kind == "pet":
        nonfood = sum(1 for x in shop if x.slot_type != "food")
        crowded = nonfood >= sc.crowd and any(x.slot_type != "food" and x.obj.name == "pet-cow" for x in shop)
    combines = [a for a in legal if C.COMBINE <= a < C.SELL and names[(a - C.COMBINE) // 5] in sc.want and not holding
                and not (sc.hold and not sc.crowd and min(team[(a - C.COMBINE) // 5].level, team[(a - C.COMBINE) % 5].level) < 2)
                and (crowded or not sc.crowd)]
    if combines:
        return rng.choice(combines)

    # Sell.
    for index, pet in enumerate(team):
        if pet is not None and pet.name in sc.sell and pet.level >= sc.sell_level and C.SELL + index in legal:
            return C.SELL + index

    # Sleeping Pill (optionally a perk first).
    ready = (sc.need <= set(n for n in names if n) and sum(1 for n in names if n) >= sc.need_count
             and all(any(p is not None and p.name == n and p.level >= lv for p in team) for n, lv in sc.want_level.items()))
    if ready and sc.pill:
        victims = [i for i, pet in enumerate(team) if pet is not None and pet.name in sc.pill and pet.level >= sc.pill_level]
        if victims:
            victim = victims[0]
            if sc.perk_first and team[victim].status == "none":
                for index, slot in food_slots(lambda n: n in ("food-honey", "food-mushroom")):
                    action = C.BUY_FOOD + index * 6 + victim
                    if action in legal and gold >= slot.cost + 1:
                        return action
            for index, slot in food_slots(lambda n: n == PILL):
                action = C.BUY_FOOD + index * 6 + victim
                if action in legal:
                    return action

    empty = [i for i, n in enumerate(names) if n is None]
    # Level-up rewards.
    if sc.levelup and empty and rng.random() < sc.levelup:
        rewards = sorted(enumerate(shop), key=lambda item: item[1].obj.name != "pet-seal")
        for index, slot in rewards:
            if slot.slot_type == "levelup" and slot.cost <= gold:
                action = C.BUY_PET + index * 6 + empty[0]
                if action in legal:
                    sc.fresh = empty[0]
                    return action

    # Wanted pets: onto a copy when there is one, else into the first empty slot.
    def wanted_pet(name):
        return name in sc.want or (sc.tier1 and TIER.get(name) == 1)

    quota = {name: sc.want.count(name) for name in sc.want}
    reached = all(any(p is not None and p.name == n and p.level >= lv for p in team) for n, lv in sc.want_level.items())
    for index, slot in enumerate(shop):
        if slot.slot_type not in ("pet", "levelup") or slot.cost > gold or not wanted_pet(slot.obj.name):
            continue
        name = slot.obj.name
        if name in sc.late and not reached:
            continue
        if sc.crowd and name == "pet-cow" and sum(1 for x in shop if x.slot_type != "food") < (10 if sc.freeze_kind == "pet" else 9):
            continue
        copies = [i for i, pet in enumerate(team) if pet is not None and pet.name == name and pet.level < 3]
        separate = holding or len([n for n in names if n == name]) < quota.get(name, 1 if not sc.tier1 else 5)
        if copies and not (separate and empty) and not holding:
            action = C.BUY_PET + index * 6 + rng.choice(copies)
            if action in legal:
                return action
        if empty:
            spot = sc.place.get(name)
            action = C.BUY_PET + index * 6 + (spot if spot in empty else empty[0])
            if action in legal:
                return action
        if holding:
            continue
        # Team full: make room by selling something that is not wanted.
        spare = [i for i, n in enumerate(names) if n is not None and not wanted_pet(n) and C.SELL + i in legal]
        if spare:
            return C.SELL + rng.choice(spare)
        if sc.tier1:
            spare = [i for i, n in enumerate(names) if n is not None and n not in sc.want and C.SELL + i in legal]
            if spare:
                return C.SELL + rng.choice(spare)

    # Foods.
    def wanted_food(name):
        return (sc.foods == "any" and name != PILL) or (sc.foods != "any" and name in sc.foods and name != PILL)

    for index, slot in food_slots(wanted_food):
        action = C.BUY_FOOD + index * 6 + 5
        if action in legal:
            return action
        targets = [i for i, n in enumerate(names) if n in sc.want] or [i for i, n in enumerate(names) if n]
        rng.shuffle(targets)
        targets.sort(key=lambda i: sc.want.index(names[i]) if names[i] in sc.want else 99)
        for target in targets:
            action = C.BUY_FOOD + index * 6 + target
            if action in legal:
                return action

    # Keep what we want but cannot pay for; sometimes freeze at random.
    for index, slot in enumerate(shop):
        keep = (slot.slot_type != "food" and wanted_pet(slot.obj.name)) or (slot.slot_type == "food" and (wanted_food(slot.obj.name) or slot.obj.name == PILL and sc.pill))
        if keep and slot.cost > gold and C.FREEZE + index in legal:
            return C.FREEZE + index
    if sc.freeze and rng.random() < sc.freeze:
        freezes = [a for a in legal if C.FREEZE <= a < C.UNFREEZE
                   and (sc.freeze_kind != "food" or shop[a - C.FREEZE].slot_type == "food")]
        if freezes and (rng.random() < 0.8 or sc.crowd):
            return rng.choice(freezes)
        unfreezes = [a for a in legal if C.UNFREEZE <= a < C.MOVE]
        if unfreezes and not sc.crowd:
            return rng.choice(unfreezes)

    if C.ROLL in legal and count < 46:
        return C.ROLL
    if C.END_TURN in legal:
        return C.END_TURN
    return rng.choice(legal_list)


def play(out):
    decisions_total, failed, used = 0, [], {}
    for index in range(FIRST, FIRST + MATCHES):
        seed = 883300000 + index
        full = FULL_EVERY > 0 and index % FULL_EVERY == 0
        rng = random.Random(seed)
        scenarios = [make_scenario(rng), make_scenario(rng)]
        fixed = [s.fixed_idle for s in scenarios if s.fixed_idle is not None]
        idle = max(fixed) if fixed else rng.choice([0, 0, 4, 8, 10, 10])
        for scenario in scenarios:
            scenario.idle = idle
        buffer = io.StringIO()
        CF_NOW["keys"], CF_NOW["events"], CF_NOW["env"] = set(), [], None
        try:
            env = ParallelSelfPlayEnv(); obs = env.reset(seed=seed, paired_tape=True)
            CF_NOW["env"] = env
            G.header(buffer, seed, env, obs, full)
            for step_index in range(8000):
                legal = G.legal_of(env); decisions = []
                for side, options in enumerate(legal):
                    if options is None:
                        decisions.append(None); continue
                    action = C.decode(choose(env, side, options, scenarios[side], rng)); mask = None
                    if action.action_type == int(ActionType.END_TURN):
                        cf = getattr(scenarios[side], "cf", None)
                        if cf is not None and cf["phase"] != "after":   # the script decides what stays frozen
                            mask = cf_mask(env, side, cf.pop("mask", 0))
                        else:
                            mask = G.random_mask(env, side, rng)
                    decisions.append(JointAction(action, mask)); decisions_total += 1
                step = G.write_step(buffer, step_index, env, legal, decisions, step_decisions(env, decisions), full)
                if step.terminated:
                    break
            else:
                raise RuntimeError("match did not end")
        except Exception as error:   # the reference itself failed: keep the evidence, skip the match
            failed.append({"match": seed, "scenarios": [s.name for s in scenarios], "error": repr(error),
                           "where": traceback.format_exc().strip().splitlines()[-6:]})
            print("reference failed on match", seed, repr(error), file=sys.stderr)
            continue
        out.write(buffer.getvalue())
        for key in CF_NOW["events"]:
            CF["events"][key] = CF["events"].get(key, 0) + 1
        for key in set(CF_NOW["events"]) | CF_NOW["keys"]:
            CF["matches"][key] = CF["matches"].get(key, 0) + 1
        for s in scenarios:
            used[s.name] = used.get(s.name, 0) + 1
        used[f"idle-{idle}"] = used.get(f"idle-{idle}", 0) + 1
    return decisions_total, failed, used


if ONLY == "cowfull":
    cf_hook()
with gzip.open(OUT, "wt", compresslevel=5) as out:
    total, failed, used = play(out)
summary = {"mode": "targeted", "matches": MATCHES, "first": FIRST, "decisions": total, "written": MATCHES - len(failed),
           "failed": failed, "scenarios": used}
if ONLY == "cowfull":   # reference-side counts of the Cow's refill: per call ("events") and per written match ("matches")
    summary["cow"] = {name: dict(sorted(table.items())) for name, table in CF.items()}
print(json.dumps(summary))
