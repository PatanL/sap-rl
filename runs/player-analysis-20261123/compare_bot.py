#!/usr/bin/env python3
"""Line counts for the bot's own games beside Max Wu and the ranked field.

Usage: compare_bot.py <bot seats.jsonl> <maxwu.json> <out.json>
Reuses the line detectors in analyse.py on seats written by bot_games.py.
"""
import importlib.util, json, sys
from collections import Counter

sys_argv = sys.argv[:]
sys.argv = ["analyse.py", "/nonexistent", "/dev/null", "x"]
spec = importlib.util.spec_from_file_location("analyse", __file__.replace("compare_bot.py", "analyse.py"))
try:
    analyse = importlib.util.module_from_spec(spec); spec.loader.exec_module(analyse)
except FileNotFoundError:
    # analyse.py reads the calculator's name tables at import; only lines() is needed here.
    src = open(spec.origin).read()
    start = src.index("def behind("); end = src.index("def main(")
    ns = {}; exec(src[start:end], ns); analyse = type("A", (), {"lines": staticmethod(ns["lines"])})
sys.argv = sys_argv
seats = []
for line in open(sys.argv[1]):
    seat = json.loads(line); seat["turns"] = {int(k): v for k, v in seat["turns"].items()}
    seat["lines"] = analyse.lines(seat); seats.append(seat)
human = json.load(open(sys.argv[2])); n = len(seats)
rows = []
for row in human["lines"]:
    have = [s for s in seats if row["line"] in s["lines"]]
    rows.append({"line": row["line"], "bot": len(have) / n, "botMedianTurn": sorted(s["lines"][row["line"]] for s in have)[len(have) // 2] if have else None,
                 "maxwu": row.get("target", {}).get("share"), "field": row.get("field", {}).get("share")})
extra = Counter(k for s in seats for k in s["lines"])
known = {r["line"] for r in rows}
for key, count in extra.items():
    if key not in known:
        rows.append({"line": key, "bot": count / n, "maxwu": 0.0, "field": 0.0})
out = {"seats": n, "meanLastTurn": sum(s["last"] for s in seats) / n, "lines": rows}
for label, lo, hi in (("pets turn 3-5", 3, 5), ("pets turn 6-9", 6, 9), ("pets turn 10+", 10, 99)):
    c, boards = Counter(), 0
    for s in seats:
        for turn, t in s["turns"].items():
            if lo <= turn <= hi:
                boards += 1; c.update({p["name"] for p in t["board"]})
    out[label] = {"boards": boards, "bot": {k: v / max(1, boards) for k, v in c.items()},
                  "maxwu": human[label].get("target", {}).get("share", {}), "field": human[label].get("field", {}).get("share", {})}
food = Counter(f for s in seats for _, f, _ in s["foods"])
out["foodsPerGame"] = {"bot": {k: v / n for k, v in food.items()}, "maxwu": human["foodsPerGame"].get("target", {}), "field": human["foodsPerGame"].get("field", {})}
out["rollsPerTurn"] = {"bot": sum(t["rolls"] for s in seats for t in s["turns"].values()) / max(1, sum(len(s["turns"]) for s in seats)), **human["rollsPerTurn"]}
json.dump(out, open(sys.argv[3], "w"), indent=1)
pct = lambda x: "  -  " if x is None else f"{100*x:5.1f}"
print(f"bot seats {n}, mean last turn {out['meanLastTurn']:.1f}; rolls/turn {out['rollsPerTurn']}")
print(f"{'line':44s} {'bot':>6s} {'MaxWu':>6s} {'field':>6s}")
for r in sorted(rows, key=lambda r: -max(r["bot"], r["maxwu"] or 0, r["field"] or 0)):
    if max(r["bot"], r["maxwu"] or 0, r["field"] or 0) >= 0.03:
        print(f"{r['line'][:44]:44s} {pct(r['bot'])} {pct(r['maxwu'])} {pct(r['field'])}")
for label in ("pets turn 3-5", "pets turn 6-9", "pets turn 10+"):
    b, f, m = out[label]["bot"], out[label]["field"], out[label]["maxwu"]
    names = sorted(set(b) | set(f), key=lambda k: -(b.get(k, 0) - f.get(k, 0)))
    print(label, "bot more than field:", ", ".join(f"{k} {100*b.get(k,0):.0f}/{100*f.get(k,0):.0f}" for k in names[:9]))
    print(label, "bot less than field:", ", ".join(f"{k} {100*b.get(k,0):.0f}/{100*f.get(k,0):.0f}" for k in names[::-1][:9]))
fb, ff = out["foodsPerGame"]["bot"], out["foodsPerGame"]["field"]
print("foods per game bot/MaxWu/field:", ", ".join(f"{k} {fb.get(k,0):.2f}/{out['foodsPerGame']['maxwu'].get(k,0):.2f}/{ff.get(k,0):.2f}" for k in sorted(set(fb) | set(ff), key=lambda k: -ff.get(k, 0))))
