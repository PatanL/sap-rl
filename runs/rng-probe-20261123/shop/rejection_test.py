#!/usr/bin/env python3
"""Second order-free test: "index into a bigger list, redraw until eligible".  Stdlib only.

If the game picks slot 1 by drawing an index into a list of M entries (all pets of the game, of the pack, a
weighted table...) and redraws until the entry is allowed, then for every roll whose slot 1 shows species s
one particular index must occur among the first K draws. So for each list size M and species s:
coverage(M, s) = largest share of the rolls with that species that contain one common index among their first
K draws. Truth gives a share near 1 - (1 - eligible/M)^K for all ten species at one M; chance gives about K/M.
The score of M is the SMALLEST coverage over the ten species minus the chance share (a real M lifts all ten).
Scanned: M = 10..800, System.Random (scaled and mod), Unity xorshift128 (mod and scaled), MT19937 (mod and
scaled), each seeded with the roll's own seed and with the previous action's seed. Labels shuffled across
rolls give the chance level of the best score.
Usage: rejection_test.py <rolls.jsonl> <out.json>
"""
import json, random, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from generators import dotnet, unity, mt19937, MBIG

K = 48
M_RANGE = range(10, 801)
DEV_DIRS = {'raw', 'raw-fresh', 'raw-fresh2'}
TIER1 = {0, 3, 17, 26, 32, 39, 47, 51, 59, 559}
rows = []
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r['turn'] <= 2 and r['type'] == 5 and r['frozen_pets'] == 0 and r['seed'] is not None and r['prev_seed'] is not None \
            and r['pets'] and r['pets'][0] in TIER1:
        rows.append(r)
dev = [r for r in rows if r['dir'] in DEV_DIRS]
held = [r for r in rows if r['dir'] not in DEV_DIRS]
random.seed(3)
lab_dev = [r['pets'][0] for r in dev]
lab_perm = lab_dev[:]
random.shuffle(lab_perm)
lab_held = [r['pets'][0] for r in held]

STREAMS = {
    'dotnet scaled': (dotnet, 'scaled'), 'dotnet mod': (dotnet, 'mod'),
    'unity mod': (unity, 'mod'), 'unity scaled': (unity, 'scaled'),
    'mt19937 mod': (mt19937, 'mod'), 'mt19937 scaled': (mt19937, 'scaled'),
}
SRC = {'own seed': 'seed', 'previous seed': 'prev_seed'}


def score(cellsets, labels):
    cnt = defaultdict(Counter)
    n = Counter(labels)
    for cs, l in zip(cellsets, labels):
        cnt[l].update(cs)
    return min(cnt[l].most_common(1)[0][1] / n[l] for l in n)


def work(task):
    sname, src = task
    gen, how = STREAMS[sname]
    d_dev = [gen(r[SRC[src]], K) for r in dev]
    R = d_dev[0][1]
    d_dev = [d for d, _ in d_dev]
    d_held = [gen(r[SRC[src]], K)[0] for r in held]
    best = []
    best_perm = -1.0
    for M in M_RANGE:
        if how == 'mod':
            cs = [{v % M for v in d} for d in d_dev]
        else:
            cs = [{v * M // R for v in d} for d in d_dev]
        chance = 1 - (1 - 1 / M) ** K
        s = score(cs, lab_dev) - chance
        sp = score(cs, lab_perm) - chance
        best_perm = max(best_perm, sp)
        best.append((s, M))
    best.sort(reverse=True)
    top = []
    for s, M in best[:3]:
        if how == 'mod':
            cs = [{v % M for v in d} for d in d_held]
        else:
            cs = [{v * M // R for v in d} for d in d_held]
        top.append({'M': M, 'excess_min_coverage_dev': s, 'excess_min_coverage_held': score(cs, lab_held) - (1 - (1 - 1 / M) ** K)})
    return {'stream': sname, 'seed': src, 'top': top, 'best_excess_shuffled': best_perm}


if __name__ == '__main__':
    print('rolls dev', len(dev), 'held-out', len(held), 'K', K, 'M', M_RANGE)
    tasks = [(s, src) for s in STREAMS for src in SRC]
    with Pool(5) as pool:
        res = pool.map(work, tasks, chunksize=1)
    for r in res:
        t = r['top'][0]
        print(f"{r['stream']:15s} {r['seed']:14s} best M {t['M']:4d}: excess min-coverage dev {t['excess_min_coverage_dev']:+.3f} "
              f"held-out {t['excess_min_coverage_held']:+.3f}; best on shuffled labels {r['best_excess_shuffled']:+.3f}")
    print('(truth with M up to 800 would show an excess of +0.4 to +1.0 on dev AND held-out)')
    json.dump({'K': K, 'M': [M_RANGE.start, M_RANGE.stop - 1], 'dev': len(dev), 'held': len(held), 'results': res}, open(sys.argv[2], 'w'), indent=1)
