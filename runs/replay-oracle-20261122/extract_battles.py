#!/usr/bin/env python3
"""Turn archived SAP Library replay bundles into engine-neutral battle fixtures.

Each real battle record carries the game's seed, both pre-battle boards, the
outcome and the surviving board. Field conventions follow SAP-Calculator's
replay parser (replay-calc-parser.ts): a missing key means the default (Enu 0,
Poi.x 0), stats are Perm + Temp, Exp is total experience, and Poi.x == 4 is the
front of the team.

Stdlib only. Usage: extract_battles.py <corpus/matches dir> <calculator dir> <out.json> [raw dir ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

EQUIPMENT_TO_STATUS = {
    "Meat Bone": "status-bone-attack",
    "Coconut": "status-coconut-shield",
    "Mushroom": "status-extra-life",
    "Garlic": "status-garlic-armor",
    "Honey": "status-honey-bee",
    "Melon": "status-melon-armor",
    "Peanut": "status-poison-attack",
    "Chili": "status-splash-attack",
    "Steak": "status-steak-attack",
    # Shop-phase perks with no battle effect. Bread's temporary health is
    # already inside Hp.Temp. Perk id 41 is absent from the calculator table;
    # every pet carrying it shows the Bread +7 temporary health.
    "Cake": "status-cake",
    "#41": "status-bread",
}
REAL_OUTCOME = {1: "player", 2: "opponent", 3: "draw"}
# ability id -> pet id, learned from every non-Parrot pet in the corpus.
_OWNER_FILE = Path(__file__).with_name("ability_owner.json")
ABILITY_OWNER = json.loads(_OWNER_FILE.read_text()) if _OWNER_FILE.exists() else {}


def load_tables(calculator: Path):
    data = calculator / "src/assets/data"
    pets = {str(p["Id"]): p["Name"] for p in json.loads((data / "pets.json").read_text())}
    perks = {str(p["Id"]): p["Name"] for p in json.loads((data / "perks.json").read_text())}
    return pets, perks


def total_experience(pet: dict) -> int:
    exp = pet.get("Exp") or 0
    if exp > 0:
        return int(exp)
    return {2: 2, 3: 5}.get(pet.get("Lvl") or 1, 0)


def convert_board(board: dict, pets: dict, perks: dict, problems: list):
    """Return pets front-to-back, plus raw ids for later triage."""
    items = (board.get("Mins") or {}).get("Items") or []
    slots = {}
    for index, pet in enumerate(items):
        if not pet:
            continue
        position = (pet.get("Poi") or {}).get("x", index)
        if position in slots:
            problems.append(f"duplicate position {position}")
        enu = str(pet.get("Enu") or 0)
        name = pets.get(enu)
        if name is None:
            problems.append(f"unknown pet id {enu}")
            name = f"#{enu}"
        perk_id = pet.get("Perk")
        perk_name = None if perk_id is None else perks.get(str(perk_id), f"#{perk_id}")
        status = "none"
        if perk_name is not None:
            status = EQUIPMENT_TO_STATUS.get(perk_name)
            if status is None:
                problems.append(f"unsupported perk {perk_name}")
                status = "none"
        attack, health = pet.get("At") or {}, pet.get("Hp") or {}
        extra = sorted(set(pet) - {"Abil", "At", "Cosm", "Enu", "Exp", "Hp", "Id", "Loc",
                                   "Lvl", "Own", "Perk", "Poi", "Pri"})
        slots[position] = {
            "id": "pet-" + name.lower().replace(" ", "-"),
            "name": name,
            "attack": int(attack.get("Perm") or 0) + int(attack.get("Temp") or 0),
            "health": int(health.get("Perm") or 0) + int(health.get("Temp") or 0),
            "tempAttack": int(attack.get("Temp") or 0),
            "tempHealth": int(health.get("Temp") or 0),
            "experience": total_experience(pet),
            "level": pet.get("Lvl") or 1,
            "status": status,
            "perk": perk_name,
            "abilities": [[a.get("Enu") or 0, a.get("Lvl") or 1] for a in pet.get("Abil") or []],
            # Per-turn ability uses already spent in the shop phase (TrCo).
            "triggersConsumed": max([int(a.get("TrCo") or 0) for a in pet.get("Abil") or []] or [0]),
            "abilityExtras": [{k: v for k, v in a.items() if k not in ("Enu", "Lvl", "Nat")} for a in pet.get("Abil") or []],
            "extraKeys": extra,
            "extra": {key: pet[key] for key in extra},
        }
    team = [slots[position] for position in sorted(slots, reverse=True)]
    for pet in team:
        if pet["name"] == "Parrot":
            # The ability a Parrot recorded tells us what it copied: another
            # pet's ability id, or its own (it copied nothing this turn).
            enu = str(pet["abilities"][0][0]) if pet["abilities"] else None
            pet["parrotCopies"] = ABILITY_OWNER.get(enu) if ABILITY_OWNER else None
            if not ABILITY_OWNER:
                del pet["parrotCopies"]
    return team


def survivors(board: dict, pets: dict):
    out = []
    for index, pet in enumerate(((board or {}).get("Mins") or {}).get("Items") or []):
        if pet:
            out.append({"slot": index, "name": pets.get(str(pet.get("Enu") or 0), "?"),
                        "id": "pet-" + pets.get(str(pet.get("Enu") or 0), "?").lower().replace(" ", "-"),
                        "attack": ((pet.get("At") or {}).get("Perm") or 0) + ((pet.get("At") or {}).get("Temp") or 0),
                        "health": ((pet.get("Hp") or {}).get("Perm") or 0) + ((pet.get("Hp") or {}).get("Temp") or 0),
                        "rawAttack": pet.get("At"), "rawHealth": pet.get("Hp"),
                        "owner": pet.get("Own")})
    return out


def main():
    matches, calculator, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    pets, perks = load_tables(calculator)
    fixtures, seen, stats = [], {}, Counter()
    sources = []
    for bundle_path in sorted(matches.glob("*/BUNDLE.json")):
        bundle = json.loads(bundle_path.read_text())
        sources.append((str(bundle["metadata"].get("game_version")), bundle["perspectives"]))
    # Optional extra directories written by fetch_replays.py.
    for raw_dir in sys.argv[4:]:
        for raw_path in sorted(Path(raw_dir).glob("*.json")):
            raw_file = json.loads(raw_path.read_text())
            sources.append((str(raw_file["meta"].get("game_version")), raw_file["perspectives"]))
    stats["replay files"] = len(sources)
    for version, perspectives in sources:
        for perspective in perspectives["perspectives"]:
            raw = perspective["raw"]
            for action in raw["Actions"]:
                if not action.get("Battle"):
                    continue
                battle = json.loads(action["Battle"])
                stats["battle records"] += 1
                key = battle["Id"]
                if key in seen:
                    stats["duplicate battle id"] += 1
                    continue
                problems = []
                player = convert_board(battle["UserBoard"], pets, perks, problems)
                opponent = convert_board(battle["OpponentBoard"], pets, perks, problems)
                fixture = {
                    "battleId": key,
                    "matchId": raw["MatchId"],
                    "participationId": raw["ParticipationId"],
                    "turn": action["Turn"],
                    "gameVersion": version,
                    "seed": battle.get("Seed"),
                    "realOutcome": REAL_OUTCOME.get(battle.get("Outcome"), f"?{battle.get('Outcome')}"),
                    "player": player,
                    "opponent": opponent,
                    "survivors": survivors(battle.get("ResultingBoard"), pets),
                    "problems": problems,
                }
                seen[key] = fixture
                fixtures.append(fixture)
                stats["fixtures"] += 1
                stats["with problems"] += bool(problems)
    # The same fight is uploaded from both seats under different battle ids.
    by_fight = {}
    for fixture in fixtures:
        by_fight.setdefault((fixture["matchId"], fixture["turn"]), []).append(fixture)
    mirrored = 0
    for group in by_fight.values():
        if len(group) == 2:
            first, second = group
            strip = lambda team: [(p["id"], p["attack"], p["health"], p["experience"], p["status"]) for p in team]
            flip = {"player": "opponent", "opponent": "player", "draw": "draw"}
            consistent = (strip(first["player"]) == strip(second["opponent"])
                          and strip(first["opponent"]) == strip(second["player"])
                          and first["realOutcome"] == flip.get(second["realOutcome"]))
            first["mirrorConsistent"] = second["mirrorConsistent"] = consistent
            second["mirrorOf"] = first["battleId"]
            mirrored += consistent
            stats["seat pairs"] += 1
    stats["seat pairs consistent"] = mirrored
    stats["distinct fights"] = len(by_fight)
    stats.update({f"version {v}": n for v, n in Counter(f["gameVersion"] for f in fixtures).items()})
    stats.update({f"real {o}": n for o, n in Counter(f["realOutcome"] for f in fixtures).items()})
    problem_kinds = Counter(p for f in fixtures for p in set(f["problems"]))
    extra_keys = Counter(k for f in fixtures for side in ("player", "opponent") for p in f[side] for k in p["extraKeys"])
    output.write_text(json.dumps({"stats": stats, "problemKinds": problem_kinds,
                                  "extraPetKeys": extra_keys, "fixtures": fixtures}, indent=1))
    print(json.dumps(stats, indent=1))
    print("problems:", dict(problem_kinds))
    print("extra pet keys:", dict(extra_keys))


if __name__ == "__main__":
    main()
