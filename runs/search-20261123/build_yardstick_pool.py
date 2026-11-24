"""Yardstick-side boards by turn (sha256(match id) % 5 < 2), for SCORING only. Never used inside search or training.
Usage: build_yardstick_pool.py <out.json> <fixtures.json>..."""
import hashlib, json, sys
KEEP = ("id", "attack", "health", "experience", "status", "triggersConsumed", "parrotCopies")
out, paths = sys.argv[1], sys.argv[2:]
by_turn, seen = {}, set()
for path in paths:
    for f in json.load(open(path))["fixtures"]:
        if int(hashlib.sha256(f["matchId"].encode()).hexdigest(), 16) % 5 >= 2 or not f["player"]: continue
        key = (f["participationId"], f["turn"])
        if key in seen: continue
        seen.add(key); by_turn.setdefault(str(f["turn"]), []).append([{k: p[k] for k in KEEP if k in p} for p in f["player"]])
json.dump({"split": "yardstick", "byTurn": by_turn}, open(out, "w"), separators=(",", ":"))
print({int(t): len(b) for t, b in sorted(by_turn.items(), key=lambda kv: int(kv[0]))})
