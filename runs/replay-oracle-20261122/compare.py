#!/usr/bin/env python3
"""Compare two engine result files: fixed and newly broken fights, by dev/held-out split.

Usage: compare.py <fixtures.json> <before.json> <after.json>
"""
import hashlib, json, sys
from collections import Counter

def split_of(match_id):
    return "heldout" if int(hashlib.sha256(match_id.encode()).hexdigest(), 16) % 5 < 2 else "dev"

fixtures = {f["battleId"]: f for f in json.load(open(sys.argv[1]))["fixtures"]}
def load(path):
    rows = {r["battleId"]: r for r in json.load(open(path))["rows"] if "skipped" not in r}
    bad = {i for i, r in rows.items() if not r["outcomePossible"] or not r["statePossible"]}
    wrong = {i for i, r in rows.items() if not r["outcomePossible"]}
    return rows, bad, wrong
rows_a, bad_a, wrong_a = load(sys.argv[2]); rows_b, bad_b, wrong_b = load(sys.argv[3])
split = lambda ids: dict(Counter(split_of(fixtures[i]["matchId"]) for i in ids))
print(f"failing: {len(bad_a)} -> {len(bad_b)} of {len(rows_b)}   wrong winner: {len(wrong_a)} -> {len(wrong_b)}")
print("fixed      ", len(bad_a - bad_b), split(bad_a - bad_b))
print("new breaks ", len(bad_b - bad_a), split(bad_b - bad_a))
print("remaining  ", split(bad_b))
for i in sorted(bad_b - bad_a):
    f = fixtures[i]; print("  broke", i[:8], split_of(f["matchId"]), "turn", f["turn"], sorted({p["name"] for p in f["player"] + f["opponent"]}))
