#!/usr/bin/env python3
"""Check an engine trace file of the constructed boards against the real-client movies.

For every session 3 capture the real side is the collapsed movie timeline
(runs/client-oracle-20261123/session2/captures/<capture>/timeline.json); the engine side is every
seed's round-by-round boards from a trace file written by engine-rs/tools/gen_golden.py for
constructed-fixtures.json. Same test as session 3's pair3.py: an engine seed "follows" the real
fight when each of its boards (after start of battle, after every attack) is found, in order, among
the real readings (attack/health of every pet, front to back, both sides). The two random
generators differ, so a capture passes when at least one seed follows it to the end; "all" means
every seed does. "lag" is the same with pair3's tolerance for the client's stale labels.

Usage: check_constructed.py <trace.jsonl> <tag> [baseline tag]
Writes constructed/<tag>.json; prints one line per rule of findings.json and the captures whose
result changed against the baseline.
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT = HERE.parent.parent / "client-oracle-20261123"


def parse(token):
    a, h = token.split("/")
    return (None if a == "?" else int(a), None if h == "?" else int(h))


def alive(board):
    return all(h is None or h > 0 for side in board for a, h in side)


def eq(real, eng, lag=False):
    if len(real) != len(eng):
        return False
    for (a, h), (ea, eh) in zip(real, eng):
        if not lag:
            if a is not None and a != ea:
                return False
            if h is not None and h != eh:
                return False
        else:
            if a is None or h is None:
                continue
            if (a, h) == (ea, eh):
                continue
            if a == ea and h > eh:
                continue  # a hit not yet on the label
            if a <= ea and h <= eh and h > 0:
                continue  # a buff not yet on the label
            return False
    return True


def follow(real, steps, lag=False):
    pos = 0
    hits = 0
    for step in steps:
        found = None
        for index in range(pos, len(real)):
            rb = real[index]
            if eq(rb[0], step[0], lag) and eq(rb[1], step[1], lag) and (not lag or alive(rb)):
                found = index
                break
        if found is None:
            break
        hits += 1
        pos = found
    return hits


def main():
    trace_path, tag = sys.argv[1:3]
    base_tag = sys.argv[3] if len(sys.argv) > 3 else None
    doc = json.load(open(HERE / "constructed-fixtures.json"))
    by_fixture = collections.defaultdict(list)
    for line in open(trace_path):
        row = json.loads(line)
        if "trace" not in row:
            by_fixture[row["battleId"]].append(None)
            continue
        steps = [tuple(tuple((s[1], s[2]) for s in side if s is not None) for side in boards)
                 for boards in row["trace"][1:]]
        by_fixture[row["battleId"]].append(steps)
    out = {}
    for cap in doc["captures"]:
        rows = json.load(open(CLIENT / "session2" / "captures" / cap["capture"] / "timeline.json"))
        real = [tuple(tuple(parse(t[2]) for t in r[side]) for side in ("P", "O")) for r in rows]
        seeds = by_fixture[cap["fixture"]]
        cache = {}
        full = lagfull = errors = 0
        best = (-1, None)
        for steps in seeds:
            if steps is None:
                errors += 1
                continue
            key = tuple(steps)
            if key not in cache:
                hits = follow(real, steps)
                cache[key] = (hits == len(steps), follow(real, steps, True) == len(steps), hits)
            ok, lagok, hits = cache[key]
            full += ok
            lagfull += lagok
            if hits > best[0]:
                best = (hits, len(steps))
        out[cap["capture"]] = {"test": cap["test"], "seeds": len(seeds), "errors": errors, "follow": full,
                               "followLag": lagfull, "distinct": len(cache), "bestStages": best[0],
                               "stages": best[1]}
    (HERE / "constructed").mkdir(exist_ok=True)
    (HERE / "constructed" / f"{tag}.json").write_text(json.dumps(out, indent=1))
    findings = json.load(open(CLIENT / "session3" / "findings.json"))
    print(f"{tag}: captures {len(out)}  some seed follows {sum(v['follow'] > 0 for v in out.values())}"
          f"  every seed follows {sum(v['follow'] == v['seeds'] for v in out.values())}"
          f"  some seed follows with label lag {sum(v['followLag'] > 0 for v in out.values())}"
          f"  engine errors {sum(v['errors'] for v in out.values())}")
    for rule in findings["rules"]:
        caps = [c for c in rule["supporting_captures"] if c in out]
        if not caps:
            continue
        some = sum(out[c]["follow"] > 0 for c in caps)
        every = sum(out[c]["follow"] == out[c]["seeds"] for c in caps)
        lag = sum(out[c]["followLag"] > 0 for c in caps)
        print(f"  {rule['id']:4s} captures {len(caps):3d}  some {some:3d}  every {every:3d}  some(lag) {lag:3d}")
    if base_tag:
        base = json.load(open(HERE / "constructed" / f"{base_tag}.json"))
        for name, now in out.items():
            was = base[name]
            a = (was["follow"] > 0, was["follow"] == was["seeds"], was["followLag"] > 0)
            b = (now["follow"] > 0, now["follow"] == now["seeds"], now["followLag"] > 0)
            if a != b:
                print(f"  changed {name}: follow {was['follow']}/{was['seeds']} -> {now['follow']}/{now['seeds']}"
                      f"  lag {was['followLag']} -> {now['followLag']}")


if __name__ == "__main__":
    main()
