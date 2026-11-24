#!/usr/bin/env python3
"""Tamper test for the environment checker (acceptance item 4).

Takes one reference trace, alters exactly one value per variant, and runs
`envcheck` on each altered copy. Every variant must fail and the checker must
name the altered match and step. The untouched copy must pass. Needs only the
standard library (run with the host python3).

Usage: tamper_test.py <envcheck binary> <trace.jsonl.gz> <scratch dir>
"""
import base64, gzip, json, os, subprocess, sys, zlib

BINARY, TRACE, SCRATCH = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(SCRATCH, exist_ok=True)
lines = [json.loads(l) for l in gzip.open(TRACE, "rt")]
starts = [i for i, l in enumerate(lines) if "match" in l]


def first(pred, begin=0):
    return next(i for i in range(begin, len(lines)) if pred(lines[i]))


def match_of(index):
    return lines[max(s for s in starts if s <= index)]["match"]


def flip_obs_byte(line, key, side, offset):
    raw = bytearray(zlib.decompress(base64.b64decode(line[key][side])))
    raw[offset] ^= 1
    line[key][side] = base64.b64encode(zlib.compress(bytes(raw), 6)).decode()


variants = []


def variant(name, index, edit, expect):
    variants.append((name, index, edit, expect))


full_step = first(lambda l: "obs" in l and l["i"] >= 20)
hash_step = first(lambda l: "obs_sha" in l and l.get("i", 0) >= 20)
battle_step = first(lambda l: l.get("battle") is True)
term_step = first(lambda l: l.get("term") is True)
team_step = first(lambda l: "state" in l and l["i"] >= 10 and any(p for p in l["state"]["seats"][0]["team"]))
slot = next(k for k, p in enumerate(lines[team_step]["state"]["seats"][0]["team"]) if p)
frozen_step = first(lambda l: "state" in l and any(s[2] for s in l["state"]["seats"][1]["shop"]))
fslot = next(k for k, s in enumerate(lines[frozen_step]["state"]["seats"][1]["shop"]) if s[2])
plan_step = first(lambda l: "plan" in l and len(l["plan"][0]) > 1)
variant("legal list: drop one action id", 37, lambda l: l["legal"][next(s for s in (0, 1) if l["legal"][s])].pop(), "legal")
variant("state: gold +1", 41, lambda l: l["state"]["seats"][1].__setitem__("gold", l["state"]["seats"][1]["gold"] + 1), "gold")
variant("state: a team pet's attack +1", team_step, lambda l: l["state"]["seats"][0]["team"][slot].__setitem__(1, l["state"]["seats"][0]["team"][slot][1] + 1), "team")
variant("state: a frozen flag cleared", frozen_step, lambda l: l["state"]["seats"][1]["shop"][fslot].__setitem__(2, False), "shop")
variant("state: seen-order count +1", 12, lambda l: l["state"]["seats"][0].__setitem__("seenOrders", l["state"]["seats"][0]["seenOrders"] + 1), "seenOrders")
variant("state: one toggled id removed", first(lambda l: "state" in l and l["state"]["seats"][0]["toggled"]), lambda l: l["state"]["seats"][0]["toggled"].pop(), "toggled")
variant("reward: +0.1 on a battle step", battle_step, lambda l: l["rew"].__setitem__(0, l["rew"][0] + 0.1), "rew")
variant("battle result changed", battle_step, lambda l: l.__setitem__("result", (l["result"] + 1) % 3), "result")
variant("battle flag set on a shop step", 3, lambda l: l.__setitem__("battle", True), "battle")
variant("terminated cleared on the last step", term_step, lambda l: l.__setitem__("term", False), "term")
variant("winner changed", term_step, lambda l: l.__setitem__("winner", (l["winner"] + 1) % 3), "winner")
variant("plan: one executed primitive changed", plan_step, lambda l: l["plan"][0][0].__setitem__(1, (l["plan"][0][0][1] + 1) % 10), "plan")
variant("observation: one bit of numeric flipped", full_step, lambda l: flip_obs_byte(l, "obs", 0, 4 * 8 * 21 + 5), "numeric")
variant("observation: one bit of target_mask flipped", full_step, lambda l: flip_obs_byte(l, "obs", 1, 4 * 8 * 21 + 4 * 21 * 16 + 21 + 9 + 99 + 100), "target_mask")
variant("observation: one bit of the last byte flipped", full_step, lambda l: flip_obs_byte(l, "obs", 1, -1), "opponent_history_board_mask")
variant("observation hash: one hex digit changed", hash_step, lambda l: l["obs_sha"].__setitem__(0, ("0" if l["obs_sha"][0][0] != "0" else "1") + l["obs_sha"][0][1:]), "obs_sha")
variant("reset observation: one bit flipped", starts[0], lambda l: flip_obs_byte(l, "obs0", 0, 2000), "obs0")
variant("action: a different legal action submitted", 5, None, "")


def run(path):
    done = subprocess.run([BINARY, "--report", "3", path], capture_output=True, text=True)
    return done.returncode, done.stdout


clean = os.path.join(SCRATCH, "untouched.jsonl")
with open(clean, "w") as out:
    for l in lines:
        out.write(json.dumps(l, separators=(",", ":")) + "\n")
code, text = run(clean)
print(f"untouched copy: exit {code}: {text.strip().splitlines()[-1]}")
ok = code == 0
for name, index, edit, expect in variants:
    line = json.loads(json.dumps(lines[index]))
    if edit is None:
        side = next(s for s in (0, 1) if line["legal"][s])
        other = next(a for a in line["legal"][side] if a not in (0, 151) and 152 <= a < 172 or 146 <= a < 151)
        # decode a freeze/unfreeze/sell id into its triple
        line["act"][side] = [4, other - 146, 5] if other < 151 else ([6, other - 152, 5] if other < 162 else [7, other - 162, 5])
        line["mask"][side] = 0
    else:
        edit(line)
    path = os.path.join(SCRATCH, "tampered.jsonl")
    with open(path, "w") as out:
        for i, l in enumerate(lines):
            out.write(json.dumps(line if i == index else l, separators=(",", ":")) + "\n")
    code, text = run(path)
    report = next((r for r in text.splitlines() if r.startswith("MISMATCH")), "")
    step = "reset" if "match" in lines[index] else f"step {lines[index]['i']}"
    named = f"match {match_of(index)} {step}:" in report and expect in report
    good = code != 0 and named
    ok = ok and good
    print(f"{'caught' if good else 'MISSED'}  {name:48s} at match {match_of(index)} {step}: exit {code}; {report[:150]}")
print("TAMPER TEST", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
