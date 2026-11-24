#!/usr/bin/env python3
"""Compare two score sets written by score_all.sh.
Usage: compare_scores.py <tagA or 'v31g'> <tagB>   -> per file: not reproduced, wrong winner, fixed, broken ids."""
import json, sys
from pathlib import Path
RO = Path(__file__).resolve().parent.parent
FILES = ["340", "fresh", "fresh2", "fresh3", "fresh4"]
def load(tag, name):
    if tag == "v31g":
        p = RO / f"rust-v31g-5000-{name}.json"
    else:
        p = RO / "v32" / "scores" / f"{tag}-{name}.json"
    d = json.load(open(p))
    return {r["battleId"]: r for r in d["rows"]}
def failing(rows):
    return {b for b, r in rows.items() if not (r["outcomePossible"] and r["statePossible"])}
def wrong(rows):
    return {b for b, r in rows.items() if not r["outcomePossible"]}
def main():
    a, b = sys.argv[1:3]
    tot = [0, 0, 0, 0, 0]
    fixed_all, broken_all = [], []
    for name in FILES:
        ra, rb = load(a, name), load(b, name)
        fa, fb = failing(ra), failing(rb)
        fixed, broken = sorted(fa - fb), sorted(fb - fa)
        print(f"{name:7s} fights {len(rb):5d}  {a}: {len(fa):3d} ({len(wrong(ra))})  {b}: {len(fb):3d} ({len(wrong(rb))})  fixed {len(fixed)}  broken {len(broken)} {[x[:8] for x in broken]}")
        tot[0] += len(rb); tot[1] += len(fa); tot[2] += len(wrong(ra)); tot[3] += len(fb); tot[4] += len(wrong(rb))
        fixed_all += [x[:8] for x in fixed]; broken_all += [x[:8] for x in broken]
    print(f"TOTAL   fights {tot[0]:5d}  {a}: {tot[1]:3d} ({tot[2]})  {b}: {tot[3]:3d} ({tot[4]})  fixed {len(fixed_all)}  broken {len(broken_all)}")
    print("fixed:", " ".join(fixed_all))
    print("broken:", " ".join(broken_all))
    same = all(json.dumps(load(a, n), sort_keys=True) == json.dumps(load(b, n), sort_keys=True) for n in FILES)
    print("identical rows:", same)
if __name__ == "__main__":
    main()
