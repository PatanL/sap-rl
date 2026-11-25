#!/usr/bin/env python3
"""Turn the session 3 constructed client tests into a fixtures file for both engines.

Reads runs/client-oracle-20261123/session3/constructed-readings.json (one row per real-client
capture) and the injected battle JSON of each capture (session2/fixtures/<capture>.json), converts
the boards with the replay oracle's own converter (extract_battles.convert_board) and keeps the
slot positions: an empty slot inside a board becomes a {"id": "pet-none"} placeholder, which
tools/gen_golden.py builds as an empty TeamSlot and the Rust engine skips.

Writes constructed-fixtures.json next to this file: one fixture per distinct test board
(battleId = test name), plus "captures": the real readings to check against (check_constructed.py).
Usage: make_constructed_fixtures.py
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent
CLIENT = ORACLE.parent / "client-oracle-20261123"
CALC = ORACLE.parent.parent / "SAP-Calculator"

spec = importlib.util.spec_from_file_location("extract_battles_v32", ORACLE / "extract_battles.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
PETS, PERKS = ex.load_tables(CALC)
KEEP = ("id", "attack", "health", "experience", "status", "triggersConsumed")


def board(raw, problems):
    """Front first, with placeholders for empty slots ahead of the last pet."""
    items = (raw.get("Mins") or {}).get("Items") or []
    positions = sorted(((p.get("Poi") or {}).get("x", i) for i, p in enumerate(items) if p), reverse=True)
    pets = ex.convert_board(raw, PETS, PERKS, problems)
    assert len(pets) == len(positions)
    out = [{"id": "pet-none"} for _ in range(5)]
    for position, pet in zip(positions, pets):
        out[4 - position] = {key: pet[key] for key in KEEP}
    while out and out[-1]["id"] == "pet-none":
        out.pop()
    return out


def main():
    readings = json.load(open(CLIENT / "session3" / "constructed-readings.json"))
    fixtures, by_key, captures = [], {}, []
    for row in readings:
        raw = json.loads((CLIENT / "session2" / "fixtures" / (row["capture"] + ".json")).read_text())
        if isinstance(raw, str):
            raw = json.loads(raw)
        problems = []
        player, opponent = board(raw["UserBoard"], problems), board(raw["OpponentBoard"], problems)
        assert not problems, (row["capture"], problems)
        # The spec in the readings file must describe the same board.
        for side, built in ((row["left"], player), (row["right"], opponent)):
            want = [None if s is None else
                    [s[1] + (s[3] if len(s) > 3 else {}).get("tatk", 0), s[2] + (s[3] if len(s) > 3 else {}).get("thp", 0)]
                    for s in side]
            assert want == [None if p["id"] == "pet-none" else [p["attack"], p["health"]] for p in built], row["capture"]
        key = json.dumps([player, opponent], sort_keys=True)
        if key not in by_key:
            name = row["test"]
            if any(f["battleId"] == name for f in fixtures):
                name = f"{row['test']}#{sum(f['battleId'].split('#')[0] == row['test'] for f in fixtures) + 1}"
            by_key[key] = name
            fixtures.append({"battleId": name, "player": player, "opponent": opponent})
        captures.append({"capture": row["capture"], "test": row["test"], "fixture": by_key[key],
                         "seed": row["seed"], "start": row["start"], "terminal": row["terminal"]})
    out = HERE / "constructed-fixtures.json"
    out.write_text(json.dumps({"fixtures": fixtures, "captures": captures}, indent=1))
    print(json.dumps({"captures": len(captures), "fixtures": len(fixtures),
                      "withGaps": sum(any(p["id"] == "pet-none" for p in f["player"] + f["opponent"]) for f in fixtures)}))


if __name__ == "__main__":
    main()
