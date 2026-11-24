#!/usr/bin/env python3
"""Do standard generators, seeded with the game's recorded seed, predict the real random choice?"""
import json, collections, itertools

class DotNet:
    MBIG = 2147483647
    def __init__(self, seed):
        sub = 2147483647 if seed == -2147483648 else abs(seed)
        mj = 161803398 - sub; arr = [0] * 56; arr[55] = mj; mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55; arr[ii] = mk; mk = mj - mk
            if mk < 0: mk += self.MBIG
            mj = arr[ii]
        for _ in range(4):
            for i in range(1, 56):
                arr[i] -= arr[1 + (i + 30) % 55]
                # C# int wraparound
                arr[i] = (arr[i] + 2**31) % 2**32 - 2**31
                if arr[i] < 0: arr[i] += self.MBIG
        self.arr, self.a, self.b = arr, 0, 21
    def raw(self):
        self.a = self.a + 1 if self.a + 1 < 56 else 1
        self.b = self.b + 1 if self.b + 1 < 56 else 1
        v = self.arr[self.a] - self.arr[self.b]
        if v == self.MBIG: v -= 1
        if v < 0: v += self.MBIG
        self.arr[self.a] = v
        return v
    def unit(self): return self.raw() * (1.0 / self.MBIG)

class Unity:
    def __init__(self, seed):
        s = [seed & 0xFFFFFFFF]
        for _ in range(3): s.append((s[-1] * 1812433253 + 1) & 0xFFFFFFFF)
        self.s = s
    def raw(self):
        x, y, z, w = self.s
        t = (x ^ (x << 11)) & 0xFFFFFFFF
        w2 = (w ^ (w >> 19) ^ t ^ (t >> 8)) & 0xFFFFFFFF
        self.s = [y, z, w, w2]
        return w2
    def unit(self): return (self.raw() & 0x7FFFFF) / 8388607.0

MAPS = {
    "floor(u*n)": lambda g, n: int(g.unit() * n),
    "raw mod n": lambda g, n: g.raw() % n,
}
pts = []
for f in ("340", "fresh", "fresh2"):
    pts += json.load(open(f"points-{f}.json"))["points"]
usable = []
for p in pts:
    if not p["labels"] or str(p["realOption"]) not in p["labels"]:
        continue
    p["index"] = p["labels"].index(str(p["realOption"]))
    usable.append(p)
print(len(usable), "usable points;", sum(not p["shufflesBefore"] for p in usable), "with no engine tie-break before the choice")
chance = sum(1 / p["options"] for p in usable) / len(usable)
print("chance level: %.3f" % chance)
rows = []
for subset_name, subset in (("all", usable), ("no tie-break before", [p for p in usable if not p["shufflesBefore"]])):
    for gname, G in (("System.Random", DotNet), ("Unity xorshift", Unity)):
        for mname, M in MAPS.items():
            for skip in range(0, 6):
                for reverse in (False, True):
                    hit = 0
                    for p in subset:
                        g = G(p["seed"])
                        for _ in range(skip): g.raw()
                        k = M(g, p["options"])
                        if reverse: k = p["options"] - 1 - k
                        hit += (k == p["index"])
                    rows.append((hit / max(1, len(subset)), subset_name, gname, mname, skip, reverse, len(subset)))
rows.sort(reverse=True)
for r in rows[:10]:
    print("%.3f  %-20s %-15s %-11s skip=%d reverse=%s n=%d" % r)
print("...\nworst: %.3f" % rows[-1][0])
