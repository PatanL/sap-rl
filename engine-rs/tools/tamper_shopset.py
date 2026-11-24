"""Tamper test for the `set_shop_item` trace entries of tools/gen_env_shopset.py: one value of a
["shop", seat, slot, name, frozen] edit (or of the state / observation recorded right after the
edits) is altered in a copy of a trace, and `envcheck` must fail at that step.

Two rounds per alteration of an edit:
  with     the line as written: the state dump recorded right after the edits must expose it;
  without  the ["state", ...] entry removed from the line (lines without an ["obs", ...] or
           ["legal"] entry, where the placed item is still in that slot after the step): the
           ordinary per-step comparisons (legal lists, state, observations) must expose it. The
           same line with the entry removed and nothing altered must still pass (control).

Usage: tamper_shopset.py <envcheck binary> <work dir> <trace.jsonl.gz> [<trace.jsonl.gz> ...]
"""
import base64, gzip, json, os, random, subprocess, sys, zlib

ENVCHECK, WORK = sys.argv[1], sys.argv[2]
os.makedirs(WORK, exist_ok=True)
rng = random.Random(11)
PETS = ["pet-fish", "pet-ant", "pet-blowfish", "pet-dragon"]
FOODS = ["food-apple", "food-garlic", "food-melon", "food-sleeping-pill"]


def load(path):
    matches, current = [], []
    for line in gzip.open(path, "rt"):
        if line.startswith('{"match"') and current:
            matches.append(current); current = []
        current.append(line)
    matches.append(current)
    return matches


def check(lines):
    target = os.path.join(WORK, "tampered-shopset.jsonl.gz")
    with gzip.open(target, "wt") as out:
        out.writelines(lines)
    run = subprocess.run([ENVCHECK, "--report", "1", target], capture_output=True, text=True)
    first = next((l for l in run.stdout.splitlines() if l.startswith("MISMATCH")), "")
    where = int(first.split(" step ")[1].split(":")[0]) if " step " in first else None
    return run.returncode, where, first


def shop_edits(line):
    return [e for e in line.get("pre", []) if e[0] == "shop"]


def survives(line):
    """The single shop edit of the line is still in its slot after the step, and nothing else was edited."""
    pre = line.get("pre", [])
    edits = shop_edits(line)
    if len(edits) != 1 or any(e[0] in ("obs", "legal", "team") for e in pre):
        return False
    _, seat, slot, name, frozen = edits[0]
    shop = line["state"]["seats"][seat]["shop"]
    return slot < len(shop) and shop[slot][1] == name and shop[slot][2] == frozen


def other(name):
    pool = PETS if name.startswith("pet-") else FOODS
    return next(n for n in pool if n != name)


def alterations(line):
    """(name, edit function) for the first shop edit of a parsed line."""
    edit = shop_edits(line)[0]
    def species(l):
        e = shop_edits(l)[0]; e[3] = other(e[3])
    yield "species changed", species
    def frozen(l):
        e = shop_edits(l)[0]; e[4] = not e[4]
    yield "frozen flag flipped", frozen
    def removed(l):
        first = shop_edits(l)[0]
        l["pre"] = [e for e in l["pre"] if e is not first]
    yield "edit removed", removed
    def seat(l):
        e = shop_edits(l)[0]; e[1] = 1 - e[1]
    yield "other seat", seat
    def slot(l):
        e = shop_edits(l)[0]; e[2] += 1 if e[2] == 0 else -1
    yield "neighbouring slot", slot


caught = total = controls = controls_ok = 0


def report(ok, path, seed, step, name, first, control=False):
    global caught, total, controls, controls_ok
    if control:
        controls += 1; controls_ok += ok
        print(("passes " if ok else "MISSED ") + f"{os.path.basename(path)} match {seed} step {step}: {name} -> {first[:140]}")
        return
    total += 1; caught += ok
    print(("caught " if ok else "MISSED ") + f"{os.path.basename(path)} match {seed} step {step}: {name} -> {first[:140]}")


for path in sys.argv[3:]:
    matches = load(path)
    done = set()
    for attempt in range(60):
        lines = rng.choice(matches)
        seed = json.loads(lines[0])["match"]
        steps = [(k, json.loads(text)) for k, text in enumerate(lines) if k > 0 and '"pre":[' in text[:40]]
        def changes(k, line):
            """The first shop edit puts another item (or another frozen flag) into the slot than it held."""
            e = shop_edits(line)[0]
            before = json.loads(lines[k - 1])
            shop = before.get("state", before.get("state0"))["seats"][e[1]]["shop"]
            return line["pre"][0] is e and e[2] < len(shop) and shop[e[2]][1] != e[3]
        steps = [(k, line) for k, line in steps if shop_edits(line) and changes(k, line)]
        if not steps:
            continue
        # Round "with": any line with a shop edit.
        for kind in ("pet", "food", "reward"):
            def fits(line):
                e = shop_edits(line)[0]
                if kind == "reward":
                    return False        # chosen below from the state before the step
                return e[3].startswith(kind)
            found = [(k, line) for k, line in steps if fits(line)]
            if kind == "reward":
                found = []
                for k, line in steps:
                    e = shop_edits(line)[0]
                    before = json.loads(lines[k - 1])
                    shop = before.get("state", before.get("state0"))["seats"][e[1]]["shop"]
                    if line["pre"][0] is e and e[2] < len(shop) and shop[e[2]][0] == "levelup":
                        found.append((k, line))
            if not found:
                continue
            k, _ = rng.choice(found)
            for name, edit in alterations(json.loads(lines[k])):
                label = f"{kind} edit, {name} (state recorded after the edits kept)"
                if label in done:
                    continue
                done.add(label)
                line = json.loads(lines[k]); edit(line)
                copy = list(lines); copy[k] = json.dumps(line, separators=(",", ":")) + "\n"
                code, where, first = check(copy)
                report(code != 0 and where == line["i"], path, seed, line["i"], label, first)
        # Round "without": the placed item is still there after the step; only the ordinary comparisons.
        found = [(k, line) for k, line in steps if survives(line)]
        for kind in ("pet", "food"):
            pick = [(k, line) for k, line in found if shop_edits(line)[0][3].startswith(kind)]
            if not pick:
                continue
            k, _ = rng.choice(pick)
            control = json.loads(lines[k]); control["pre"] = [e for e in control["pre"] if e[0] != "state"]
            copy = list(lines); copy[k] = json.dumps(control, separators=(",", ":")) + "\n"
            label = f"{kind} edit, control: state entry removed, nothing altered"
            if label not in done:
                done.add(label)
                code, where, first = check(copy)
                report(code == 0, path, seed, control["i"], label + " (must pass)", first or "no mismatch", control=True)
            for name, edit in alterations(control):
                label = f"{kind} edit, {name} (ordinary comparisons only)"
                if label in done:
                    continue
                done.add(label)
                line = json.loads(json.dumps(control)); edit(line)
                copy = list(lines); copy[k] = json.dumps(line, separators=(",", ":")) + "\n"
                code, where, first = check(copy)
                report(code != 0 and where == line["i"], path, seed, line["i"], label, first)
        # The recorded state and observation right after the edits.
        with_state = [(k, line) for k, line in steps if any(e[0] == "state" for e in line["pre"])]
        if with_state and "pre state" not in done:
            done.add("pre state")
            k, _ = rng.choice(with_state)
            line = json.loads(lines[k])
            e = shop_edits(line)[0]
            dump = next(x for x in line["pre"] if x[0] == "state")[1]
            row = dump["seats"][e[1]]["shop"][e[2]]
            row[3] += 1
            copy = list(lines); copy[k] = json.dumps(line, separators=(",", ":")) + "\n"
            code, where, first = check(copy)
            report(code != 0 and where == line["i"], path, seed, line["i"], "recorded state after the edits: cost of the placed slot +1", first)
        with_obs = [(k, line) for k, line in steps if any(e[0] == "obs" for e in line["pre"])]
        if with_obs and "pre obs" not in done:
            done.add("pre obs")
            k, _ = rng.choice(with_obs)
            line = json.loads(lines[k])
            entry = next(x for x in line["pre"] if x[0] == "obs")
            raw = bytearray(zlib.decompress(base64.b64decode(entry[1][0])))
            raw[0] ^= 1
            entry[1][0] = base64.b64encode(zlib.compress(bytes(raw), 6)).decode()
            copy = list(lines); copy[k] = json.dumps(line, separators=(",", ":")) + "\n"
            code, where, first = check(copy)
            report(code != 0 and where == line["i"], path, seed, line["i"], "recorded observation after the edits: one bit", first)
print(f"{caught} of {total} alterations caught, {controls_ok} of {controls} unaltered controls pass")
sys.exit(0 if caught == total and total > 0 and controls_ok == controls and controls > 0 else 1)
