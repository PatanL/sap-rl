#!/usr/bin/env python3
"""What lines do named strong players use, compared with the ranked field?

Reads SAP Library replay files (both seats per match), takes each seat's
end-of-turn boards and food purchases, and counts how often a set of known
multi-turn lines appears, for the named players versus everyone else.
Stdlib only.

Usage: analyse.py <calculator dir> <out.json> <name substring>[,<name>...] <raw dir> [raw dir ...]
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

CALC, OUT, TARGETS = Path(sys.argv[1]), sys.argv[2], [t.lower() for t in sys.argv[3].split(",")]
DATA = CALC / "src/assets/data"
PETS = {str(p["Id"]): p["Name"] for p in json.loads((DATA / "pets.json").read_text())}
FOOD = {str(p["Id"]): p["Name"] for p in json.loads((DATA / "food.json").read_text())}
FOOD["10"] = "Bread"
PERK = {str(p["Id"]): p["Name"] for p in json.loads((DATA / "perks.json").read_text())}
PERK["41"] = "Bread"


def board_of(bor):
    pets = []
    for index, pet in enumerate((bor.get("Mins") or {}).get("Items") or []):
        if not pet:
            continue
        pets.append({"x": (pet.get("Poi") or {}).get("x", index), "uni": pet["Id"]["Uni"],
                     "name": PETS.get(str(pet.get("Enu") or 0), "?"), "level": pet.get("Lvl") or 1,
                     "perk": PERK.get(str(pet.get("Perk"))) if pet.get("Perk") is not None else None,
                     "attack": (pet.get("At") or {}).get("Perm") or 0, "health": (pet.get("Hp") or {}).get("Perm") or 0})
    return sorted(pets, key=lambda p: -p["x"])  # front first


def seats(dirs):
    seen = set()
    for d in dirs:
        for path in sorted(Path(d).glob("*.json")):
            doc = json.loads(path.read_text())
            for per in doc["perspectives"]["perspectives"]:
                raw = per["raw"]
                key = raw["ParticipationId"]
                if key in seen:
                    continue
                seen.add(key)
                turns, foods, prev = {}, [], {}
                known = {}
                pending = []
                for action in raw["Actions"]:
                    kind = action["Type"]
                    if kind == 8 and action.get("Response"):
                        event = (json.loads(action["Response"]).get("Event") or {}).get("Event") or {}
                        spell, target = event.get("Spell") or {}, event.get("Target") or {}
                        pending.append((action["Turn"], FOOD.get(str(spell.get("Enu") or 0), "?"), target.get("Uni")))
                    elif kind == 0 and action.get("Build"):
                        bor = json.loads(action["Build"])["Bor"]
                        board = board_of(bor)
                        for pet in board:
                            known[pet["uni"]] = pet["name"]
                        for turn, food, uni in pending:
                            foods.append((turn, food, known.get(uni)))
                        pending = []
                        turns[bor.get("Tur")] = {"board": board, "wins": bor.get("Vic") or 0, "lives": bor.get("LoPs"),
                                                 "rolls": (bor.get("Rold") or 1) - 1, "tier": bor.get("Ti")}
                if not turns:
                    continue
                last = max(turns)
                battles = [json.loads(a["Battle"]).get("Outcome") for a in raw["Actions"] if a.get("Battle")]
                yield {"name": raw.get("UserName") or "", "match": raw["MatchId"], "turns": turns, "foods": foods,
                       "battles": battles, "won": int(raw.get("Outcome") == 1), "rawOutcome": raw.get("Outcome"), "last": last}


def behind(board, index):
    return board[index + 1]["name"] if index + 1 < len(board) else None


def lines(seat):
    found = {}
    def mark(key, turn):
        found.setdefault(key, turn)
    for turn in sorted(seat["turns"]):
        board = seat["turns"][turn]["board"]
        names = [p["name"] for p in board]
        for i, pet in enumerate(board):
            nxt = behind(board, i)
            if pet["name"] == "Elephant" and nxt == "Blowfish":
                mark("Elephant ahead of Blowfish", turn)
                if "Wolverine" in names:
                    mark("Elephant, Blowfish and Wolverine", turn)
            if pet["name"] == "Elephant" and nxt == "Camel":
                mark("Elephant ahead of Camel", turn)
            if pet["name"] == "Elephant" and nxt == "Peacock":
                mark("Elephant ahead of Peacock", turn)
            if pet["name"] == "Mammoth" and pet["perk"] == "Mushroom" and nxt == "Tiger":
                mark("Mushroom Mammoth with Tiger behind", turn)
            if pet["name"] == "Mammoth" and nxt == "Tiger":
                mark("Mammoth with Tiger behind", turn)
            if nxt == "Tiger":
                mark("Tiger behind " + pet["name"], turn)
        if "Wolverine" in names:
            mark("Wolverine on board", turn)
        if "Snail" in names and turn <= 3:
            mark("Snail by turn 3", turn)
        if "Horse" in names and "Cricket" in names and turn <= 2:
            mark("Horse and Cricket by turn 2", turn)
        if "Sheep" in names and turn <= 4:
            mark("Sheep by turn 4", turn)
        summoners = sum(n in ("Sheep", "Deer", "Whale", "Rooster", "Fly", "Cricket", "Spider") for n in names)
        if "Turkey" in names and summoners >= 2:
            mark("Turkey with two or more summoners", turn)
        if "Shark" in names and summoners >= 2:
            mark("Shark with two or more summoners", turn)
        if "Fly" in names and turn >= 9:
            mark("Fly late", turn)
        if "Cat" in names and ("Cow" in names or "Seal" in names or "Rabbit" in names):
            mark("Cat with Cow, Seal or Rabbit", turn)
        if names.count("Deer") >= 2:
            mark("Two or more Deer", turn)
        if "Dolphin" in names and turn >= 7:
            mark("Dolphin late", turn)
        if "Rhino" in names:
            mark("Rhino on board", turn)
        if "Scorpion" in names:
            mark("Scorpion on board", turn)
        if "Skunk" in names:
            mark("Skunk on board", turn)
        if "Dragon" in names:
            mark("Dragon on board", turn)
    for turn, food, target in seat["foods"]:
        if food == "Sleeping Pill":
            mark("Any Sleeping Pill", turn)
            mark("Pill on " + (target or "a pet bought that turn"), turn)
        if food == "Chocolate":
            mark("Chocolate", turn)
        if food == "Mushroom":
            mark("Mushroom on " + (target or "?"), turn)
        if food == "Melon":
            mark("Melon on " + (target or "?"), turn)
    battles = seat["battles"]
    if len(battles) >= 2 and battles[1] == 2 and (seat["turns"].get(1) or {}).get("lives") in (None, 0):
        pass
    return found


def main():
    groups = defaultdict(list)
    for seat in seats(sys.argv[4:]):
        who = "target" if any(t in seat["name"].lower() for t in TARGETS) else "field"
        seat["lines"] = lines(seat)
        groups[who].append(seat)
    out = {"targets": TARGETS, "seats": {k: len(v) for k, v in groups.items()},
           "names": dict(Counter(s["name"] for s in groups["target"])),
           "rawOutcome": {k: dict(Counter(str(s["rawOutcome"]) for s in v)) for k, v in groups.items()},
           "winRate": {k: sum(s["won"] for s in v) / max(1, len(v)) for k, v in groups.items()},
           "meanLastTurn": {k: sum(s["last"] for s in v) / max(1, len(v)) for k, v in groups.items()}}
    table = []
    keys = Counter(k for v in groups.values() for s in v for k in s["lines"])
    for key, total in keys.items():
        row = {"line": key}
        for who, v in groups.items():
            have = [s for s in v if key in s["lines"]]
            row[who] = {"share": len(have) / max(1, len(v)), "n": len(have),
                        "winWith": sum(s["won"] for s in have) / max(1, len(have)),
                        "medianFirstTurn": sorted(s["lines"][key] for s in have)[len(have) // 2] if have else None}
        table.append(row)
    out["lines"] = table
    # Pets on late boards (turn 10 and later) and rolls per turn.
    for label, lo, hi in (("pets turn 3-5", 3, 5), ("pets turn 6-9", 6, 9), ("pets turn 10+", 10, 99)):
        out[label] = {}
        for who, v in groups.items():
            c, n = Counter(), 0
            for s in v:
                for turn, t in s["turns"].items():
                    if lo <= turn <= hi:
                        n += 1
                        c.update({p["name"] for p in t["board"]})
            out[label][who] = {"boards": n, "share": {k: x / max(1, n) for k, x in c.items()}}
    out["rollsPerTurn"] = {who: sum(t["rolls"] for s in v for t in s["turns"].values()) / max(1, sum(len(s["turns"]) for s in v)) for who, v in groups.items()}
    food = {}
    for who, v in groups.items():
        c = Counter(f for s in v for _, f, _ in s["foods"]); n = max(1, len(v))
        food[who] = {k: x / n for k, x in c.items()}
    out["foodsPerGame"] = food
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("seats", "names", "rawOutcome", "winRate", "meanLastTurn", "rollsPerTurn")}, indent=1))


if __name__ == "__main__":
    main()
