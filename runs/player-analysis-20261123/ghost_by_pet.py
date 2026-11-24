#!/usr/bin/env python3
"""Which pets on ranked players' boards beat the bot's boards most?

Same pairing as ghost.py (bot board against a ranked player's board of the
same turn), for a range of turns, but keeps each pair so the bot's score can
be split by which pets and perks the human board holds.

Usage: ghost_by_pet.py <bot battles.jsonl> <out.json> <first turn> <last turn> <pairs> <fixtures.json>...
"""
import json, os, random, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path

ENGINE = Path("/home/dzliu/pzliu/sap-rl/engine-rs")
BOT, OUT, LO, HI, PAIRS, FIXTURES = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6:]
KEEP = ("id", "attack", "health", "experience", "status", "triggersConsumed", "parrotCopies")
rng = random.Random(7)
trim = lambda board: [{k: p[k] for k in KEEP if k in p} for p in board]
human, bot, seen = defaultdict(list), defaultdict(list), set()
for path in FIXTURES:
    for f in json.loads(Path(path).read_text())["fixtures"]:
        key = (f["participationId"], f["turn"])
        if key in seen or not f["player"] or not LO <= f["turn"] <= HI:
            continue
        seen.add(key); human[f["turn"]].append(trim(f["player"]))
for line in open(BOT):
    row = json.loads(line)
    if LO <= row["turn"] <= HI:
        bot[row["turn"]].extend(trim(t) for t in row["teams"] if t)
fixtures = []
turns = [t for t in range(LO, HI + 1) if len(human[t]) >= 20 and len(bot[t]) >= 20]
for n in range(PAIRS):
    t = turns[n % len(turns)]
    fixtures.append({"battleId": str(n), "matchId": "g", "turn": t, "realOutcome": "player", "player": rng.choice(bot[t]), "opponent": rng.choice(human[t]), "survivors": []})
with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(OUT))) as tmp:
    src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
    src.write_text(json.dumps({"fixtures": fixtures}))
    subprocess.run([str(ENGINE / "target/release/oracle"), "--fixtures", str(src), "--data", str(ENGINE / "data"), "--samples", "8", "--threads", "5", "--out", str(dst)], check=True, stdout=subprocess.DEVNULL)
    rows = json.loads(dst.read_text())["rows"]
total = [0.0, 0]; by = defaultdict(lambda: [0.0, 0])
for f, row in zip(fixtures, rows):
    o = row["simOutcomes"]; s = (o["player"] + 0.5 * o["draw"]) / (o["player"] + o["opponent"] + o["draw"])
    total[0] += s; total[1] += 1
    tags = {p["id"][4:] for p in f["opponent"]} | {"perk:" + p["status"][7:] for p in f["opponent"] if p["status"] not in ("none", None)}
    for tag in tags:
        by[tag][0] += s; by[tag][1] += 1
mean = total[0] / total[1]
table = sorted(({"tag": k, "share": v[1] / total[1], "botScoreWith": v[0] / v[1], "botScoreWithout": (total[0] - v[0]) / max(1, total[1] - v[1])} for k, v in by.items() if v[1] >= 0.02 * total[1]), key=lambda r: r["botScoreWith"])
json.dump({"turns": [LO, HI], "pairs": total[1], "botScore": mean, "byHumanPet": table}, open(OUT, "w"), indent=1)
print(f"turns {LO}-{HI}: bot score {100*mean:.1f}% over {total[1]} pairs")
print(f"{'on the human board':24s} {'share':>6s} {'bot with':>9s} {'bot without':>12s}")
for r in table[:14] + table[-8:]:
    print(f"{r['tag']:24s} {100*r['share']:5.1f}% {100*r['botScoreWith']:8.1f}% {100*r['botScoreWithout']:11.1f}%")
