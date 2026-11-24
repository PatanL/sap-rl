"""Training-side pool of ranked players' boards, as one sequence per player per match.

Split rule (same as runs/player-analysis-20261123/ghost.py): a match whose sha256(match id) % 5 < 2 is the
yardstick side and never enters this file. Output: ghost-train-pool.json
  sequences: [{"match", "participation", "boards": {"<turn>": [pet specs front to back]}}]
Pet spec keys kept: id, attack, health, experience, status, triggersConsumed, parrotCopies (when present).
Stdlib only. Usage: build_pool.py <out.json> <fixtures.json>...
"""
import hashlib, json, sys
from collections import Counter, defaultdict

KEEP = ("id", "attack", "health", "experience", "status", "triggersConsumed", "parrotCopies")


def side(match_id):
    return "yardstick" if int(hashlib.sha256(match_id.encode()).hexdigest(), 16) % 5 < 2 else "train"


out, paths = sys.argv[1], sys.argv[2:]
sequences, matches = defaultdict(dict), {"train": set(), "yardstick": set()}
meta = {}
for path in paths:
    for f in json.load(open(path))["fixtures"]:
        matches[side(f["matchId"])].add(f["matchId"])
        if side(f["matchId"]) != "train" or not f["player"]:
            continue
        key = f["participationId"]
        if f["turn"] in sequences[key]:
            continue
        meta[key] = f["matchId"]
        sequences[key][f["turn"]] = [{k: p[k] for k in KEEP if k in p} for p in f["player"]]
assert not (matches["train"] & matches["yardstick"])
rows = [{"match": meta[k], "participation": k, "boards": {str(t): b for t, b in sorted(v.items())}} for k, v in sorted(sequences.items())]
assert all(side(r["match"]) == "train" for r in rows)
turns = Counter(int(t) for r in rows for t in r["boards"])
json.dump({"split": "train", "rule": "sha256(matchId) % 5 >= 2", "fixtures": [p.split("/")[-1] for p in paths],
           "matches": {k: len(v) for k, v in matches.items()}, "sequences": rows}, open(out, "w"), separators=(",", ":"))
print(json.dumps({"sequences": len(rows), "boards": sum(turns.values()), "trainMatches": len(matches["train"]),
                  "yardstickMatches": len(matches["yardstick"]), "boardsByTurn": dict(sorted(turns.items())),
                  "lengths": dict(sorted(Counter(len(r["boards"]) for r in rows).items()))}))
