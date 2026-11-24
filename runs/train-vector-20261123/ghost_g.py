#!/usr/bin/env python3
"""Fight the bot's boards against real ranked players' boards, turn by turn.

For each turn number, boards the bot took into battle in its own games are
paired at random with boards ranked players took into battle on the same turn
(SAP Library replays), and each pair is simulated on the Rust engine. The
score is wins plus half the draws. Two controls use the same method: field
against field (must be 50%) and a named strong player against the field.

It measures board strength at equal turn numbers. It is not a match result:
neither side reacts to the other, and lives are ignored.

The ranked pool is split by match so that a yardstick is never trained on:
matches with sha256(match id) % 5 < 2 are the YARDSTICK side (about 40%), the
rest are reserved for use as training opponents. This tool scores on the
yardstick side unless GHOST_SPLIT=train or GHOST_SPLIT=all is set.

Usage: ghost.py <bot battles.jsonl> <out.json> <name substring> <pairs per turn> <fixtures.json>... -- <raw dir>...
"""
import hashlib, json, os, random, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path

# CPU fixture parsing and fight workers use cores also available to vector jobs.
# Serialize with them so GPU throughput measurements have uncontended CPUs.
import fcntl,time
reservation=open('/job/tmp/gpu.lock','a')
fcntl.flock(reservation,fcntl.LOCK_EX)
active_started=time.monotonic()
ENGINE = Path("/sap/engine-rs-pinned/v31g")
split = sys.argv.index("--")
BOT, OUT, TARGET, PAIRS = sys.argv[1], sys.argv[2], sys.argv[3].lower(), int(sys.argv[4])
FIXTURES, RAW = sys.argv[5:split], sys.argv[split + 1:]
KEEP = ("id", "attack", "health", "experience", "status", "triggersConsumed", "parrotCopies")
MAX_TURN, SAMPLES = 15, 8
rng = random.Random(20261123)


SPLIT = "yardstick"


def side(match_id):
    return "yardstick" if int(hashlib.sha256(match_id.encode()).hexdigest(), 16) % 5 < 2 else "train"


def trim(board):
    return [{k: p[k] for k in KEEP if k in p} for p in board]


names = {}
for directory in RAW:
    for path in Path(directory).glob("*.json"):
        for per in json.loads(path.read_text())["perspectives"]["perspectives"]:
            names[per["raw"]["ParticipationId"]] = per["raw"].get("UserName") or ""

pools = {"bot": defaultdict(list), "field": defaultdict(list), "target": defaultdict(list)}
seen = set()
for path in FIXTURES:
    for f in json.loads(Path(path).read_text())["fixtures"]:
        key = (f["participationId"], f["turn"])
        if key in seen or not f["player"]:
            continue
        if SPLIT != "all" and side(f["matchId"]) != SPLIT:
            continue
        seen.add(key)
        who = "target" if TARGET in names.get(f["participationId"], "").lower() else "field"
        pools[who][f["turn"]].append(trim(f["player"]))
for line in open(BOT):
    row = json.loads(line)
    for team in row["teams"]:
        if team:
            pools["bot"][row["turn"]].append(trim(team))

COMPARISONS = (("bot", "field"), ("bot", "target"), ("target", "field"), ("field", "field"))
fixtures, index = [], []
for a, b in COMPARISONS:
    for turn in range(1, MAX_TURN + 1):
        left, right = pools[a][turn], pools[b][turn]
        if len(left) < 20 or len(right) < 20:
            continue
        for n in range(PAIRS):
            fixtures.append({"battleId": f"{a}-{b}-{turn}-{n}", "matchId": "ghost", "turn": turn, "realOutcome": "player",
                             "player": rng.choice(left), "opponent": rng.choice(right), "survivors": []})
            index.append((a, b, turn))
with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(OUT))) as tmp:
    src, dst = Path(tmp) / "in.json", Path(tmp) / "out.json"
    src.write_text(json.dumps({"fixtures": fixtures}))
    subprocess.run([sys.executable, "/job/ghost_fights.py", "--fixtures", str(src), "--data", str(ENGINE / "data"),
                    "--samples", str(SAMPLES), "--threads", "5", "--out", str(dst)], check=True, stdout=subprocess.DEVNULL)
    rows = json.loads(dst.read_text())["rows"]
score = defaultdict(lambda: [0.0, 0, 0])
for (a, b, turn), row in zip(index, rows):
    if row.get("skipped"):
        score[(a, b, turn)][2] += 1
        continue
    o = row["simOutcomes"]; total = o["player"] + o["opponent"] + o["draw"]
    score[(a, b, turn)][0] += (o["player"] + 0.5 * o["draw"]) / total
    score[(a, b, turn)][1] += 1
out = {"split": SPLIT, "pairsPerTurn": PAIRS, "samplesPerPair": SAMPLES, "target": TARGET,
       "boards": {k: {t: len(v) for t, v in sorted(p.items()) if t <= MAX_TURN} for k, p in pools.items()}, "comparisons": {}}
print(f"{'turn':>4s} " + " ".join(f"{a+' v '+b:>16s}" for a, b in COMPARISONS))
for turn in range(1, MAX_TURN + 1):
    cells = []
    for a, b in COMPARISONS:
        s = score.get((a, b, turn))
        if s and s[1]:
            value = s[0] / s[1]
            out["comparisons"].setdefault(f"{a} v {b}", {})[turn] = {"score": value, "pairs": s[1], "skipped": s[2],
                                                                   "se": (value * (1 - value) / s[1]) ** 0.5}
            cells.append(f"{100*value:15.1f}%")
        else:
            cells.append(f"{'-':>16s}")
    print(f"{turn:4d} " + " ".join(cells))
out["active_s"]=time.monotonic()-active_started
json.dump(out, open(OUT, "w"), indent=1)
print("boards per turn:", {k: [v.get(t, 0) for t in (1, 3, 5, 8, 10, 12)] for k, v in out["boards"].items()}, "(turns 1,3,5,8,10,12)")
