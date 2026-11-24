"""Turtle Pack content overrides.

The legacy database bundled with :mod:`sapai.data` predates several Turtle
Pack revisions.  This module keeps the compatibility-oriented legacy data
file intact while replacing the rollable Turtle Pack content with current P1
data and behaviour.  robertley/SAP-Calculator commit
``d165eb0a02f8aa0b54d72ed1d5490a44390d07f4`` is pinned as the authority for
battle abilities and event ordering.  It is not authoritative for the shop
phase, which it does not model completely; shop mechanics are maintained from
verified current-game rules and expert-confirmed P1 behaviour.

The compact dictionaries below use sapai's existing effect vocabulary; new
effect kinds are implemented in ``sapai.effects``.
"""

SOURCE_OF_TRUTH = {
    "repository": "https://github.com/robertley/SAP-Calculator",
    "commit": "d165eb0a02f8aa0b54d72ed1d5490a44390d07f4",
    "battle": {
        "authority": "SAP-Calculator",
        "scope": "battle ability semantics and event ordering",
    },
    "shop": {
        "authority": "verified current-game and expert-confirmed Turtle P1 rules",
        "calculator_authoritative": False,
        "scope": "shop generation, economy, stocking, overflow, and trigger order",
    },
}
SOURCE_OF_TRUTH["replay_engine_sha256"] = '852b274e7eb21b6126bbe9c2a615acc54e009e7d7f29de6026ef5e1bbb402a6b'
SIMULATOR_REVISION = 'turtle-v31-replay-852b274e7eb2-20261123'

# Current SAP uses only one food shop slot on tier 2 (turns three and four).
# The pinned calculator snapshot still carries the legacy two-food schedule,
# so keep this explicit current-game correction next to the other Turtle Pack
# overrides rather than silently changing the bundled legacy data table.
CURRENT_TURTLE_TIER_TWO_FOOD_SLOTS = 1

# Hidden/generated pets are tier 1 for gameplay mechanics (most visibly
# Rhino's doubled Knock Out damage), but they must never enter ordinary shop
# or discover pools.  The legacy catalog represented "hidden" by storing the
# string ``Summoned`` in the tier field, conflating two independent concepts.
# Keep them explicitly hidden and give them their real numeric tier instead.
CURRENT_TURTLE_GENERATED_PETS = frozenset(
    {
        "pet-bee",
        "pet-zombie-cricket",
        "pet-dirty-rat",
        "pet-ram",
        "pet-bus",
        "pet-chick",
        "pet-zombie-fly",
    }
)


TURTLE_PET_STATS = {
    "pet-ant": (2, 2, 1),
    "pet-beaver": (3, 2, 1),
    "pet-cricket": (1, 3, 1),
    "pet-duck": (2, 2, 1),
    "pet-fish": (2, 3, 1),
    "pet-horse": (2, 1, 1),
    "pet-mosquito": (2, 2, 1),
    "pet-otter": (1, 4, 1),
    "pet-pig": (4, 1, 1),
    "pet-pigeon": (3, 2, 1),
    "pet-crab": (4, 1, 2),
    "pet-flamingo": (3, 2, 2),
    "pet-hedgehog": (4, 2, 2),
    "pet-kangaroo": (2, 2, 2),
    "pet-peacock": (2, 5, 2),
    "pet-rat": (3, 6, 2),
    "pet-snail": (2, 3, 2),
    "pet-spider": (2, 2, 2),
    "pet-swan": (1, 2, 2),
    "pet-worm": (1, 4, 2),
    "pet-badger": (6, 3, 3),
    "pet-camel": (3, 3, 3),
    "pet-dodo": (4, 2, 3),
    "pet-dog": (3, 2, 3),
    "pet-dolphin": (4, 3, 3),
    "pet-elephant": (3, 7, 3),
    "pet-giraffe": (1, 2, 3),
    "pet-ox": (1, 3, 3),
    "pet-rabbit": (1, 2, 3),
    "pet-sheep": (2, 2, 3),
    "pet-bison": (4, 4, 4),
    "pet-blowfish": (3, 6, 4),
    "pet-deer": (2, 2, 4),
    "pet-hippo": (4, 6, 4),
    "pet-parrot": (4, 2, 4),
    "pet-penguin": (2, 3, 4),
    "pet-skunk": (3, 5, 4),
    "pet-squirrel": (3, 5, 4),
    "pet-turtle": (2, 5, 4),
    "pet-whale": (3, 7, 4),
    "pet-armadillo": (4, 8, 5),
    "pet-cow": (4, 6, 5),
    "pet-crocodile": (8, 4, 5),
    "pet-monkey": (1, 2, 5),
    "pet-rhino": (6, 7, 5),
    "pet-rooster": (6, 5, 5),
    "pet-scorpion": (1, 3, 5),
    "pet-seal": (3, 8, 5),
    "pet-shark": (2, 2, 5),
    "pet-turkey": (3, 4, 5),
    "pet-boar": (10, 6, 6),
    "pet-cat": (4, 5, 6),
    "pet-dragon": (3, 8, 6),
    "pet-fly": (4, 4, 6),
    "pet-gorilla": (7, 10, 6),
    "pet-leopard": (10, 4, 6),
    "pet-mammoth": (4, 12, 6),
    "pet-snake": (8, 3, 6),
    "pet-tiger": (6, 4, 6),
    "pet-wolverine": (5, 7, 6),
}


def _ability(description, trigger, triggered_by, effect, max_triggers=None):
    ability = {
        "description": description,
        "trigger": trigger,
        "triggeredBy": {"kind": triggered_by},
        "effect": effect,
    }
    if max_triggers is not None:
        ability["maxTriggers"] = max_triggers
    return ability


def _modify(target, attack=0, health=0, **target_options):
    target_spec = {"kind": target, **target_options}
    return {
        "kind": "ModifyStats",
        "target": target_spec,
        "attackAmount": attack,
        "healthAmount": health,
        "untilEndOfBattle": False,
    }


def _levels(factory):
    return {f"level{level}Ability": factory(level) for level in (1, 2, 3)}


PET_ABILITY_OVERRIDES = {
    "pet-beaver": _levels(
        lambda level: _ability(
            f"Sell: Give two random friends +{level} attack.",
            "Sell",
            "Self",
            _modify("RandomFriend", attack=level, n=2),
        )
    ),
    "pet-cow": _levels(
        lambda level: _ability(
            f"Buy: Replace shop food with two free level-{level} Milk.",
            "Buy",
            "Self",
            {
                "kind": "RefillShops",
                "shop": "Food",
                "food": "food-milk",
                "n": 2,
            },
        )
    ),
    "pet-bison": _levels(
        lambda level: _ability(
            f"End turn: If this has a level 3 friend, gain +{2 * level} attack "
            f"and +{2 * level} health. Works for 1 Bison.",
            "EndOfTurnWithLvl3Friend",
            "Player",
            _modify("Self", attack=2 * level, health=2 * level),
        )
    ),
    "pet-crab": _levels(
        lambda level: _ability(
            f"Start of battle: Gain health equal to {25 * level}% of the most "
            "healthy friend.",
            "StartOfBattle",
            "Player",
            {"kind": "GainHealthPercentHighestFriend", "percentage": 25 * level},
        )
    ),
    "pet-deer": _levels(
        lambda level: _ability(
            f"Faint: Summon one {5 * level}/{3 * level} Bus with Chili.",
            "Faint",
            "Self",
            {
                "kind": "SummonPet",
                "pet": "pet-bus",
                "withAttack": 5 * level,
                "withHealth": 3 * level,
                "team": "Friendly",
                "n": 1,
            },
        )
    ),
    "pet-dragon": _levels(
        lambda level: _ability(
            f"Tier 1 friend bought: Give friends +{level} attack and +{level} "
            "health. Works 4 times per turn.",
            "BuyTier1Animal",
            "Player",
            _modify("AllOtherFriends", attack=level, health=level),
            max_triggers=4,
        )
    ),
    "pet-giraffe": _levels(
        lambda level: _ability(
            f"Start of turn: Give the nearest {level} friend(s) ahead +1 attack "
            "and +1 health.",
            "StartOfTurn",
            "Player",
            _modify("FriendAhead", attack=1, health=1, n=level),
        )
    ),
    "pet-hippo": _levels(
        lambda level: _ability(
            f"Knock out: Gain +{3 * level} attack and +{3 * level} health. "
            "Works 3 times per battle.",
            "KnockOut",
            "Self",
            _modify("Self", attack=3 * level, health=3 * level),
            max_triggers=3,
        )
    ),
    "pet-monkey": _levels(
        lambda level: _ability(
            f"End turn: Give the front-most friendly pet +{2 * level} attack "
            f"and +{2 * level} health.",
            "EndOfTurn",
            "Player",
            _modify(
                "FrontMostFriend", attack=2 * level, health=2 * level
            ),
        )
    ),
    "pet-ox": _levels(
        lambda level: _ability(
            f"Friend ahead faints: Gain Melon perk and +1 attack. Works {level} "
            "time(s) per turn.",
            "Faint",
            "FriendAhead",
            {
                "kind": "AllOf",
                "effects": [
                    {
                        "kind": "ApplyStatus",
                        "status": "status-melon-armor",
                        "to": {"kind": "Self"},
                    },
                    _modify("Self", attack=1),
                ],
            },
            max_triggers=level,
        )
    ),
    "pet-penguin": _levels(
        lambda level: _ability(
            f"Start of turn: Give two level 2 or higher friends +{level} attack "
            f"and +{level} health.",
            "StartOfTurn",
            "Player",
            _modify("Level2And3Friends", attack=level, health=level, n=2),
        )
    ),
    "pet-rabbit": _levels(
        lambda level: _ability(
            f"Friendly ate shop food or gained a perk: Give them +{level} "
            "health. Works 3 times per turn.",
            "EatsShopFoodOrGainsPerk",
            "EachFriend",
            _modify("TriggeringEntity", health=level),
            max_triggers=3,
        )
    ),
    "pet-rat": _levels(
        lambda level: _ability(
            f"Faint: Summon {level} 1/1 Dirty Rat(s) up front for the opponent.",
            "Faint",
            "Self",
            {
                "kind": "SummonPet",
                "pet": "pet-dirty-rat",
                "team": "Enemy",
                "n": level,
            },
        )
    ),
    "pet-rhino": _levels(
        lambda level: _ability(
            f"Knock out: Deal {4 * level} damage to the first enemy. Double "
            "against tier 1 pets.",
            "KnockOut",
            "Self",
            {
                "kind": "DealDamage",
                "target": {"kind": "FirstEnemy"},
                "amount": 4 * level,
                "doubleAgainstTier": 1,
            },
        )
    ),
    "pet-seal": _levels(
        lambda level: _ability(
            f"This pet ate shop food or gained a perk: Give up to three "
            f"random other friends +{level} attack.",
            "EatsShopFoodOrGainsPerk",
            "Self",
            _modify("RandomFriend", attack=level, n=3),
        )
    ),
    "pet-snail": _levels(
        lambda level: _ability(
            f"End turn: If you lost last battle, give the three nearest friends "
            f"ahead +{level} attack.",
            "EndOfTurnAfterLoss",
            "Player",
            _modify("FriendAhead", attack=level, n=3),
        )
    ),
    "pet-snake": _levels(
        lambda level: _ability(
            f"Friend ahead attacks: Deal {5 * level} damage to one random enemy. "
            "Works 5 times per battle.",
            "AfterAttack",
            "FriendAhead",
            {
                "kind": "DealDamage",
                "target": {"kind": "RandomEnemy", "n": 1},
                "amount": 5 * level,
            },
            max_triggers=5,
        )
    ),
    "pet-skunk": _levels(
        lambda level: _ability(
            f"Start of battle: Remove {33 * level}% health from the highest "
            "health enemy.",
            "StartOfBattle",
            "Player",
            {
                "kind": "ReduceHealth",
                "target": {"kind": "HighestHealthEnemy"},
                "percentage": 33 * level,
            },
        )
    ),
    "pet-squirrel": _levels(
        lambda level: _ability(
            f"Start of turn: Discount all shop food by {level} gold.",
            "StartOfTurn",
            "Self",
            {"kind": "DiscountFood", "amount": level},
        )
    ),
    "pet-swan": _levels(
        lambda level: _ability(
            f"Start of turn: Gain {level} gold.",
            "StartOfTurn",
            "Player",
            {"kind": "GainGold", "amount": level},
        )
    ),
    "pet-turkey": _levels(
        lambda level: _ability(
            f"Friend summoned: Give it +{3 * level} attack and +{level} health.",
            "Summoned",
            "EachFriend",
            _modify("TriggeringEntity", attack=3 * level, health=level),
        )
    ),
    "pet-worm": _levels(
        lambda level: _ability(
            f"Start of turn: Stock one 2-gold "
            f"{'Apple' if level == 1 else 'Better Apple' if level == 2 else 'Best Apple'}.",
            "StartOfTurn",
            "Self",
            {
                "kind": "StockFood",
                "food": "food-apple",
                "n": 1,
                "cost": 2,
                "attackAmount": level,
                "healthAmount": level,
            },
        )
    ),
}


NEW_TURTLE_PETS = {
    "pet-pigeon": {
        "name": "Pigeon",
        "id": "pet-pigeon",
        "tier": 1,
        "baseAttack": 3,
        "baseHealth": 2,
        "packs": ["StandardPack"],
        "probabilities": "none",
        **_levels(
            lambda level: _ability(
                f"Sell: Stock {level} free Bread Crumb(s).",
                "Sell",
                "Self",
                {"kind": "StockFood", "food": "food-bread-crumbs", "n": level},
            )
        ),
    },
    "pet-armadillo": {
        "name": "Armadillo",
        "id": "pet-armadillo",
        "tier": 5,
        "baseAttack": 4,
        "baseHealth": 8,
        "packs": ["StandardPack"],
        "probabilities": "none",
        **_levels(
            lambda level: _ability(
                f"Start of battle: Give ALL pets +{8 * level} health.",
                "StartOfBattle",
                "Player",
                _modify("All", health=8 * level),
            )
        ),
    },
    "pet-wolverine": {
        "name": "Wolverine",
        "id": "pet-wolverine",
        "tier": 6,
        "baseAttack": 5,
        "baseHealth": 7,
        "packs": ["StandardPack"],
        "probabilities": "none",
        **_levels(
            lambda level: _ability(
                f"Four friends hurt: Remove {3 * level} health from all enemies.",
                "FourFriendsHurt",
                "EachFriend",
                _modify("EachEnemy", health=-3 * level),
            )
        ),
    },
}


NEW_TURTLE_FOODS = {
    "food-cake": {
        "name": "Cake",
        "id": "food-cake",
        "tier": 3,
        "packs": ["StandardPack"],
        "probabilities": "none",
        "ability": {
            "description": "Give one pet the Cake perk.",
            "effect": {
                "kind": "ApplyStatus",
                "status": "status-cake",
                "target": {"kind": "EachFriend"},
            },
        },
    },
    "food-bread": {
        "name": "Bread",
        "id": "food-bread",
        "tier": 4,
        "packs": ["StandardPack"],
        "probabilities": "none",
        "ability": {
            "description": "Give one pet the Bread perk.",
            "effect": {
                "kind": "ApplyStatus",
                "status": "status-bread",
                "target": {"kind": "EachFriend"},
            },
        },
    },
    "food-bread-crumbs": {
        "name": "Bread Crumbs",
        "id": "food-bread-crumbs",
        "tier": "Token",
        "cost": 0,
        "packs": [],
        "probabilities": "none",
        "ability": {
            "description": "Give one pet +1 attack.",
            "effect": {
                "kind": "ModifyStats",
                "attackAmount": 1,
                "healthAmount": 0,
                "target": {"kind": "EachFriend"},
                "untilEndOfBattle": False,
            },
        },
    },
}


def apply_turtle_pack_overrides(data):
    """Mutate the legacy database to the pinned Turtle Pack revision."""

    data["pets"].update(NEW_TURTLE_PETS)
    data["foods"].update(NEW_TURTLE_FOODS)

    for turn in (3, 4):
        data["turns"][f"turn-{turn}"]["foodShopSlots"] = (
            CURRENT_TURTLE_TIER_TWO_FOOD_SLOTS
        )

    for pet_id, (attack, health, tier) in TURTLE_PET_STATS.items():
        pet = data["pets"][pet_id]
        pet["baseAttack"] = attack
        pet["baseHealth"] = health
        pet["tier"] = tier
        pet["packs"] = ["StandardPack"]

    for pet_id in CURRENT_TURTLE_GENERATED_PETS:
        pet = data["pets"][pet_id]
        pet["tier"] = 1
        pet["hidden"] = True

    for pet_id, abilities in PET_ABILITY_OVERRIDES.items():
        data["pets"][pet_id].update(abilities)

    # Scorpion's Summoned ability is equivalent to entering battle with Peanut
    # in sapai's single-ability/single-perk model.
    data["pets"]["pet-scorpion"]["status"] = "status-poison-attack"

    # Sloth remains constructible as a hidden pet, but is not rollable in the
    # current Turtle Pack.
    sloth_packs = data["pets"]["pet-sloth"].get("packs", [])
    data["pets"]["pet-sloth"]["packs"] = [
        pack for pack in sloth_packs if pack != "StandardPack"
    ]
