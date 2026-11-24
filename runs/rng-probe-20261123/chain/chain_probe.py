"""Is the per-action Seed in a replay derived from the previous action's Seed by a standard generator?
Every action's response carries {"Seed", "Hash"}. For consecutive actions of one player this tests whether
Seed[i+1] is among the first K outputs of a generator seeded with Seed[i]. Stdlib only.
Usage: chain_probe.py <raw dir>... """
import glob, json, sys
from collections import Counter
MBIG = 2147483647; K = 64

def dotnet(seed, k):
    sub = MBIG if seed == -2147483648 else abs(seed); mj = (161803398 - sub) % (1 << 32)
    mj = 161803398 - sub
    a = [0] * 56; a[55] = mj; mk = 1
    for i in range(1, 55):
        ii = (21 * i) % 55; a[ii] = mk; mk = mj - mk
        if mk < 0: mk += MBIG
        mj = a[ii]
    for _ in range(4):
        for i in range(1, 56):
            a[i] -= a[1 + (i + 30) % 55]
            if a[i] < 0: a[i] += MBIG
    inext, inextp, out = 0, 21, []
    for _ in range(k):
        inext = inext + 1 if inext + 1 < 56 else 1; inextp = inextp + 1 if inextp + 1 < 56 else 1
        r = a[inext] - a[inextp]
        if r == MBIG: r -= 1
        if r < 0: r += MBIG
        a[inext] = r; out.append(r)
    return out

def unity(seed, k):
    M = 0xFFFFFFFF; s = [seed & M]
    for i in range(1, 4): s.append((s[-1] * 1812433253 + 1) & M)
    x, y, z, w = s; out = []
    for _ in range(k):
        t = (x ^ (x << 11)) & M; x, y, z = y, z, w; w = (w ^ (w >> 19) ^ t ^ (t >> 8)) & M; out.append(w)
    return out

pairs = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + '/*.json'))[:150]:
        for p in json.load(open(f))['perspectives']['perspectives']:
            acts = sorted(p['raw']['Actions'], key=lambda a: a['BuildChangeCount'] or 0); prev = None
            for a in acts:
                try: seed = json.loads(a['Response'])['Data']['Seed']
                except Exception: prev = None; continue
                if prev is not None: pairs.append((prev[0], seed, prev[1], a['Type'], a['Turn']))
                prev = (seed, a['Type'])
print('consecutive pairs', len(pairs), 'negative seeds', sum(s < 0 for _, s, *_ in pairs), 'max', max(s for _, s, *_ in pairs))
hits = {'dotnet': Counter(), 'unity mod': Counter(), 'unity mask': Counter(), 'unity raw': Counter()}; by_type = Counter(); tot = Counter()
for a, b, t0, t1, turn in pairs[:60000]:
    tot[t1] += 1
    o = dotnet(a, K)
    if b in o: hits['dotnet'][o.index(b)] += 1; by_type[t1] += 1
    u = unity(a, K)
    for name, f in (('unity mod', lambda v: v % MBIG), ('unity mask', lambda v: v & 0x7FFFFFFF), ('unity raw', lambda v: v)):
        m = [f(v) for v in u]
        if b in m: hits[name][m.index(b)] += 1
n = min(len(pairs), 60000)
for name, c in hits.items(): print(name, 'hits', sum(c.values()), 'of', n, 'by draw index', sorted(c.items())[:12])
print('dotnet hits by next action type', sorted(by_type.items()), 'totals', sorted(tot.items()))
