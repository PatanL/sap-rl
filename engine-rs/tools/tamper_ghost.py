"""Tamper test for the optional trace keys of tools/gen_env_ghost.py: one value of a `pre` edit or
of a recorded battle team is altered in a copy of a trace, and `envcheck` must fail at that step.

Usage: tamper_ghost.py <envcheck binary> <work dir> <trace.jsonl.gz> [<trace.jsonl.gz> ...]
"""
import gzip, json, os, random, subprocess, sys

ENVCHECK, WORK = sys.argv[1], sys.argv[2]
os.makedirs(WORK, exist_ok=True)
rng = random.Random(7)


def load(path):
    matches, current = [], []
    for line in gzip.open(path, "rt"):
        if line.startswith('{"match"') and current:
            matches.append(current); current = []
        current.append(line)
    matches.append(current)
    return matches


def cases(lines):
    """Yield (name, step index, edit function on the parsed line)."""
    steps = [(k, json.loads(text)) for k, text in enumerate(lines) if k > 0]
    def pick(test):
        found = [(k, line) for k, line in steps if test(line)]
        return rng.choice(found) if found else None
    def has(op):
        return lambda line: any(e[0] == op for e in line.get("pre", []))
    def team_with_pet(line):
        return any(e[0] == "team" and any(e[2]) for e in line.get("pre", []))
    def edit_team(field, change):
        def apply(line):
            edit = next(e for e in line["pre"] if e[0] == "team" and any(e[2]))
            row = next(r for r in edit[2] if r)
            row[field] = change(row[field])
        return apply
    found = pick(team_with_pet)
    if found:
        yield "pre team: attack +1", found[0], edit_team(1, lambda v: v + 1 if v < 50 else v - 1)
        yield "pre team: health -1", found[0], edit_team(2, lambda v: v - 1 if v > 1 else v + 1)
        yield "pre team: perk changed", found[0], edit_team(4, lambda v: "status-garlic-armor" if v != "status-garlic-armor" else "none")
        yield "pre team: experience changed", found[0], edit_team(3, lambda v: 5 if v < 2 else 0)
        yield "pre team: species changed", found[0], edit_team(0, lambda v: "pet-fish" if v != "pet-fish" else "pet-ant")
        def drop(line):
            line["pre"] = [e for e in line["pre"] if e[0] != "team"]
        yield "pre team: edit removed", found[0], drop
    found = pick(lambda line: any(e[0] == "team" and any(r and r[0] == "pet-parrot" and r[6] for r in e[2]) for e in line.get("pre", [])))
    if found:
        def copies(line):
            for e in line["pre"]:
                if e[0] == "team":
                    for r in e[2]:
                        if r and r[0] == "pet-parrot" and r[6]:
                            r[6] = "pet-fish" if r[6] != "pet-fish" else "pet-ant"; return
        yield "pre team: Parrot copies another species", found[0], copies
    found = pick(lambda line: any(e[0] == "gold" and e[2] > 0 for e in line.get("pre", [])))
    if found:
        def gold(line):
            edit = next(e for e in line["pre"] if e[0] == "gold" and e[2] > 0)
            edit[2] += 1
        yield "pre gold: value +1", found[0], gold
    found = pick(has("gold"))
    if found:
        def no_gold(line):
            line["pre"] = [e for e in line["pre"] if e[0] != "gold"]
        yield "pre gold: edit removed", found[0], no_gold
    found = pick(has("rekey"))
    if found:
        def rekey(line):
            next(e for e in line["pre"] if e[0] == "rekey")[1] ^= 1
        yield "pre rekey: seed changed", found[0], rekey
        def no_rekey(line):
            line["pre"] = [e for e in line["pre"] if e[0] != "rekey"]
        yield "pre rekey: edit removed", found[0], no_rekey
    found = pick(lambda line: line.get("teams") and any(line["teams"][0]) and any(line["teams"][1]))
    if found:
        def team_field(side, field, change):
            def apply(line):
                row = next(r for r in line["teams"][side] if r)
                row[field] = change(row[field])
            return apply
        yield "battle teams: seat 0 attack +1", found[0], team_field(0, 1, lambda v: v + 1)
        yield "battle teams: seat 1 health +1", found[0], team_field(1, 2, lambda v: v + 1)
        yield "battle teams: seat 1 experience", found[0], team_field(1, 3, lambda v: (v + 1) % 6)
        yield "battle teams: seat 0 consumed triggers", found[0], team_field(0, 5, lambda v: v + 1)
        def turn(line):
            line["battleTurn"] += 1
        yield "battle turn +1", found[0], turn
        def carried(line):
            slot = next(i for i, r in enumerate(line["teams"][0]) if r)
            line["carried"][0][slot][0] += 1
        yield "carried counter +1", found[0], carried
        def swap(line):
            line["teams"].reverse(); line["carried"].reverse()
        if found[1]["teams"][0] != found[1]["teams"][1]:
            yield "battle teams: seats swapped", found[0], swap


caught = total = 0
for path in sys.argv[3:]:
    matches = load(path)
    tried = set()
    for attempt in range(40):
        lines = rng.choice(matches)
        seed = json.loads(lines[0])["match"]
        for name, index, edit in cases(lines):
            if name in tried:
                continue
            tried.add(name)
            line = json.loads(lines[index]); edit(line)
            copy = list(lines); copy[index] = json.dumps(line, separators=(",", ":")) + "\n"
            target = os.path.join(WORK, "tampered.jsonl.gz")
            with gzip.open(target, "wt") as out:
                out.writelines(copy)
            run = subprocess.run([ENVCHECK, "--report", "1", target], capture_output=True, text=True)
            first = next((l for l in run.stdout.splitlines() if l.startswith("MISMATCH")), "")
            step = line["i"]
            where = first.split(" step ")[1].split(":")[0] if " step " in first else None
            ok = run.returncode != 0 and where is not None and int(where) >= step
            total += 1; caught += ok
            print(("caught " if ok else "MISSED ") + f"{os.path.basename(path)} match {seed} step {step}: {name} -> {first[:150]}")
print(f"{caught} of {total} alterations caught")
sys.exit(0 if caught == total and total > 0 else 1)
