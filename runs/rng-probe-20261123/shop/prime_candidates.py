#!/usr/bin/env python3
"""The plain candidates, on all tier-1 rolls at once (dev + held-out together, no selection):
System.Random / old Mono / Unity xorshift, seeded with the roll's own seed or the previous action's seed,
one of the first 8 draws, reduced by mod 10 or floor(u*10), against pet slot 1, 2, 3.
Prints the cell purity (share of rolls explained by the best cell->species table): truth ~1.0, no rule ~0.13.
Also: the same table restricted to seeds <= 161803398 (where System.Random cannot overflow int32, so the port
is covered by the published small-seed vectors), how often the port without overflow differs on the logged
seeds, and the score a rule would get if it held on only 10% of the rolls.
Stdlib only. Usage: prime_candidates.py <rolls.jsonl>"""
import json, math, sys
from collections import Counter
import random
from generators import dotnet, mono_old, unity, dotnet_nowrap, MBIG
TIER1 = {0, 3, 17, 26, 32, 39, 47, 51, 59, 559}
rows = [r for r in map(json.loads, open(sys.argv[1]))
        if r['turn'] <= 2 and r['type'] == 5 and r['frozen_pets'] == 0 and r['prev_seed'] is not None
        and all(p is None or p in TIER1 for p in r['pets'])]
print('tier-1 rolls with 3 pets generated:', len(rows))
res = []
for gname, gen in (('System.Random', dotnet), ('old Mono Random', mono_old), ('Unity xorshift128', unity)):
    for sname in ('seed', 'prev_seed'):
        draws = [gen(r[sname], 8) for r in rows]
        R = draws[0][1]
        for k in range(8):
            for mname, f in (('mod 10', lambda v: v % 10), ('floor(u*10)', lambda v: v * 10 // R)):
                for slot in range(3):
                    pairs = [(f(d[0][k]), r['pets'][slot]) for d, r in zip(draws, rows) if r['pets'][slot] is not None]
                    n = len(pairs)
                    j = Counter(pairs); rc = Counter(c for c, _ in pairs); cc = Counter(l for _, l in pairs)
                    chi = n * (sum(v * v / (rc[a] * cc[b]) for (a, b), v in j.items()) - 1)
                    best = {}
                    for (a, b), v in j.items():
                        best[a] = max(best.get(a, 0), v)
                    res.append(((chi - 81) / math.sqrt(162), sum(best.values()) / n, n, f'{gname}({sname}) draw {k} {mname} -> pet slot {slot + 1}'))
res.sort(reverse=True)
print('hypotheses:', len(res), ' purity range: %.3f to %.3f' % (min(r[1] for r in res), max(r[1] for r in res)),
      ' z range: %.2f to %.2f' % (res[-1][0], res[0][0]))
for z, pur, n, d in res[:5]:
    print(f'  z {z:+.2f} purity {pur:.3f} n {n}  {d}')
print('for reference, zero-skip candidates:')
for z, pur, n, d in res:
    if ' draw 0 ' in d and 'slot 1' in d:
        print(f'  z {z:+.2f} purity {pur:.3f} n {n}  {d}')

# --- System.Random on the range where no int32 overflow can happen (seed <= 161803398) ---
SAFE = 161803398
print()
for sname in ('seed', 'prev_seed'):
    sub = [r for r in rows if r[sname] <= SAFE]
    res2 = []
    draws = [dotnet(r[sname], 8)[0] for r in sub]
    for k in range(8):
        for mname, f in (('mod 10', lambda v: v % 10), ('floor(u*10)', lambda v: v * 10 // MBIG)):
            for slot in range(3):
                pairs = [(f(d[k]), r['pets'][slot]) for d, r in zip(draws, sub) if r['pets'][slot] is not None]
                n = len(pairs)
                best = {}
                for (a, b), v in Counter(pairs).items():
                    best[a] = max(best.get(a, 0), v)
                res2.append((sum(best.values()) / n, n, f'draw {k} {mname} -> pet slot {slot + 1}'))
    # purity expected with no rule at this sample size: shuffle labels
    random.seed(1)
    null = []
    for _ in range(200):
        pairs = [(random.randrange(10), r['pets'][0]) for r in sub if r['pets'][0] is not None]
        best = {}
        for (a, b), v in Counter(pairs).items():
            best[a] = max(best.get(a, 0), v)
        null.append(sum(best.values()) / len(pairs))
    null.sort()
    print(f'System.Random({sname}) restricted to {sname} <= {SAFE}: {len(sub)} rolls, 48 hypotheses, purity '
          f'{min(x[0] for x in res2):.3f} to {max(x[0] for x in res2):.3f} (best: {max(res2)[2]}, n {max(res2)[1]}); '
          f'no-rule purity at this size: median {null[100]:.3f}, 99th percentile {null[197]:.3f}; truth ~1.0')

# --- how wrong is the port without overflow on the logged seeds? ---
seeds = [r['seed'] for r in rows] + [r['prev_seed'] for r in rows]
d64 = sum(dotnet(s, 64)[0] != dotnet_nowrap(s, 64)[0] for s in seeds)
d1 = sum(dotnet(s, 1)[0] != dotnet_nowrap(s, 1)[0] for s in seeds)
big = sum(s > SAFE for s in seeds)
print(f'port without int32 overflow vs port with it, on {len(seeds)} logged seeds: differs somewhere in the first 64 outputs for {d64} '
      f'({100*d64/len(seeds):.1f}%), in the first output for {d1} ({100*d1/len(seeds):.1f}%); seeds above {SAFE}: {100*big/len(seeds):.1f}%')

# --- sensitivity: a rule that holds on only 10% of the rolls ---
pool = sorted(TIER1)
zs = []
for rep in range(5):
    random.seed(100 + rep)
    pairs = []
    for r in rows:
        if r['pets'][0] is None:
            continue
        cell = dotnet(r['seed'], 3)[0][2] * 10 // MBIG
        lab = pool[cell] if random.random() < 0.10 else random.choice(pool)
        pairs.append((cell, lab))
    n = len(pairs)
    j = Counter(pairs); rc = Counter(c for c, _ in pairs); cc = Counter(l for _, l in pairs)
    chi = n * (sum(v * v / (rc[a] * cc[b]) for (a, b), v in j.items()) - 1)
    zs.append((chi - 81) / math.sqrt(162))
print(f'planted rule that holds on 10% of {n} rolls (rest random): z = ' + ', '.join(f'{z:.1f}' for z in zs) + '  (family maximum under chance: 5 to 7.5)')
