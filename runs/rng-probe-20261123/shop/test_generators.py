#!/usr/bin/env python3
"""Does a standard generator seeded with the logged seed predict the rolled shop?  Stdlib only.

Order-free test. For a hypothesis (generator, which seed, draw index, way of reducing a draw to a small
cell number) the cell must determine the species if the hypothesis is right, whatever the unknown pool
order is. So each hypothesis is scored by the chi-square of the (cell x species) table, as
z = (chi2 - dof) / sqrt(2 dof). A true hypothesis gives z in the hundreds; chance gives |z| of a few.
The whole family is also run on labels shuffled across rolls, to get the best-of-family score under chance.

Rows: tier-1 rolls (turns 1-2, one tier, 10 pets, 2 foods) from rolls.jsonl.
Usage: test_generators.py <rolls.jsonl> <stratum> <out.json>
  stratum: roll   = Type 5, no pet frozen (3 pets generated), food frozen or not
           roll0  = Type 5, nothing frozen (3 pets + 1 food generated); also tests the food
           start  = Type 4 at turn 2, no pet frozen
           genesis = first shop of the match (pass genesis.jsonl), against the turn-1 Type 4 seed
           first  = first roll of a turn (turns 1-2); adds the turn's Type 4 seed as a seed source
                    (a generator seeded once at turn start and consumed by the rolls)
           tier2  = Type 5 at turns 3-4 (two tiers, 20 pets), no pet frozen; windows/shuffles use a pool of 20
  CONTROL=1 in the environment replaces the species by synthetic ones from a known rule (positive control).
"""
import json, math, os, random, sys
from collections import Counter
from multiprocessing import Pool
from generators import GENERATORS, M32

K = 40                      # outputs per seed (draw indices 0..39 = up to 39 skipped draws)
DEV_DIRS = {'raw', 'raw-fresh', 'raw-fresh2'}
TIER1 = {0, 3, 17, 26, 32, 39, 47, 51, 59, 559}
N_PERM = 2
rolls_path, stratum, out_path = sys.argv[1:4]
MIN_TURN, MAX_TURN = (3, 4) if stratum == 'tier2' else (1, 2)
POOL = 20 if stratum == 'tier2' else 10

rows = []
for line in open(rolls_path):
    r = json.loads(line)
    if stratum == 'genesis':
        # genesis.jsonl: the first shop of a match; the only seed near it is the turn-1 Type 4 response seed
        # (which comes AFTER the shop exists). "previous" sources are therefore the same values here.
        if r['seed_after'] is None:
            continue
        r.update(turn=1, type='G', frozen_pets=0, frozen_foods=0, seed=r['seed_after'], prev_seed=r['seed_after'],
                 hash=r['hash_after'], prev_hash=r['hash_after'])
    if not (MIN_TURN <= r['turn'] <= MAX_TURN) or r['frozen_pets'] != 0 or r['seed'] is None or r['prev_seed'] is None:
        continue
    if stratum == 'tier2' and r['type'] != 5:
        continue
    if stratum == 'first' and (r['type'] != 5 or r['rolls_before'] != 0 or r['turn_seed'] is None):
        continue
    if stratum == 'roll' and r['type'] != 5:
        continue
    if stratum == 'roll0' and (r['type'] != 5 or r['frozen_foods'] != 0):
        continue
    if stratum == 'start' and r['type'] != 4:
        continue
    if stratum != 'tier2' and any(p is not None and p not in TIER1 for p in r['pets']):
        continue
    rows.append(r)
print(stratum, 'rows', len(rows), 'dev', sum(r['dir'] in DEV_DIRS for r in rows))
if os.environ.get('CONTROL'):
    # Positive control: keep the real seeds and which slots are known, but replace the species by ones
    # produced by a known rule with 2% label noise. The sweep must find these rules.
    #   pet slot j  = pool[(int)(Sample * 10)] with System.Random(own seed), draws 2,3,4
    #   food        = Unity xorshift(previous seed), draw 5, value mod 2
    from generators import dotnet, unity, MBIG
    pool = sorted(TIER1)
    random.seed(5)
    for r in rows:
        d = dotnet(r['seed'], 8)[0]
        r['pets'] = [None if p is None else (random.choice(pool) if random.random() < 0.02 else pool[d[2 + j] * 10 // MBIG]) for j, p in enumerate(r['pets'])]
        if r['foods'] and r['foods'][0] in (0, 40):
            r['foods'] = [(0, 40)[unity(r['prev_seed'], 8)[0][5] % 2]]
    print('CONTROL RUN: synthetic labels')

TARGETS = {}
for t in range(3):
    TARGETS[f'pet slot {t + 1}'] = [(i, r['pets'][t]) for i, r in enumerate(rows) if len(r['pets']) > t and r['pets'][t] is not None]
if stratum == 'tier2':
    common = {sp for sp, n in Counter(p for r in rows for p in r['pets'] if p is not None).items() if n >= 30}
    print('  species kept (seen at least 30 times):', len(common))
    for name in list(TARGETS):
        TARGETS[name] = [(i, l) for i, l in TARGETS[name] if l in common]
    TARGETS['food slot'] = [(i, r['foods'][0]) for i, r in enumerate(rows) if r['frozen_foods'] == 0 and r['foods'] and r['foods'][0] is not None]
elif stratum != 'roll':
    TARGETS['food slot'] = [(i, r['foods'][0]) for i, r in enumerate(rows) if r['frozen_foods'] == 0 and r['foods'] and r['foods'][0] in (0, 40)]
is_dev = [r['dir'] in DEV_DIRS for r in rows]
random.seed(20261123)
T = {}
for name, pairs in TARGETS.items():
    dev = [(i, l) for i, l in pairs if is_dev[i]]
    held = [(i, l) for i, l in pairs if not is_dev[i]]
    perms = []
    for _ in range(N_PERM):
        lab = [l for _, l in dev]
        random.shuffle(lab)
        perms.append(lab)
    T[name] = {'dev_idx': [i for i, _ in dev], 'dev_lab': [l for _, l in dev], 'held_idx': [i for i, _ in held],
               'held_lab': [l for _, l in held], 'perm_lab': perms}
    print(f'  target {name}: dev {len(dev)} held-out {len(held)}')

SOURCES = {
    'own seed (response of the generating action)': lambda r: r['seed'],
    'previous action seed': lambda r: r['prev_seed'],
    'own hash': lambda r: r['hash'] & M32,
    'previous hash (request BoardHash)': lambda r: (r['prev_hash'] or 0) & M32,
    'own seed xor previous seed': lambda r: r['seed'] ^ r['prev_seed'],
    'own seed + turn': lambda r: r['seed'] + r['turn'],
}
if stratum == 'first':
    SOURCES["turn start seed (this turn's Type 4 response)"] = lambda r: r['turn_seed']


def zscore(cells, labels):
    n = len(cells)
    rc, cc = Counter(cells), Counter(labels)
    joint = Counter(zip(cells, labels))
    s = 0.0
    for (a, b), v in joint.items():
        s += v * v / (rc[a] * cc[b])
    chi = n * (s - 1.0)
    dof = (len(rc) - 1) * (len(cc) - 1)
    if dof <= 0:
        return 0.0, 0.0
    purity = sum(max(v for (a, b), v in joint.items() if a == cell) for cell in rc) / n if len(rc) <= 12 else None
    return (chi - dof) / math.sqrt(2 * dof), purity


def mappings(col, R):
    """cell lists for one column of raw draws with range R."""
    m = {
        'mod 10': [v % 10 for v in col],
        'mod 60': [v % 60 for v in col],
        'mod 64': [v % 64 for v in col],
        'mod 77': [v % 77 for v in col],
        'mod 63': [v % 63 for v in col],
        'floor(u*10)': [v * 10 // R for v in col],
        'mod 20': [v % 20 for v in col],
        'floor(u*20)': [v * 20 // R for v in col],
        'floor(u*60)': [v * 60 // R for v in col],
        'floor(u*64)': [v * 64 // R for v in col],
    }
    if R == 1 << 32:
        m['low 23 bits scaled to 60 (Unity value)'] = [(v & 0x7FFFFF) * 60 >> 23 for v in col]
        m['bits 8..31 scaled to 60 (float mantissa)'] = [(v >> 8) * 60 >> 24 for v in col]
    return m


def fy_backward(d, R, n, mod):
    a = list(range(n))
    k = 0
    for i in range(n - 1, 0, -1):
        j = d[k] % (i + 1) if mod else min(i, d[k] * (i + 1) // R)
        k += 1
        a[i], a[j] = a[j], a[i]
    return a


def fy_forward(d, R, n, mod):
    a = list(range(n))
    k = 0
    for i in range(0, n - 1):
        j = i + (d[k] % (n - i) if mod else min(n - i - 1, d[k] * (n - i) // R))
        k += 1
        a[i], a[j] = a[j], a[i]
    return a


def work(gname):
    gen = GENERATORS[gname]
    res = []                 # (z_dev, purity, z_held, [z_perm...], description)
    n_tests = 0
    best_perm = [-1e9] * N_PERM
    sum_z = sum_z2 = 0.0

    def score(desc, cells):
        nonlocal n_tests, sum_z, sum_z2
        for tname, t in T.items():
            if tname == 'food slot' and desc.startswith(('FY', 'arg', 'rank')):
                continue
            cd = [cells[i] for i in t['dev_idx']]
            z, pur = zscore(cd, t['dev_lab'])
            n_tests += 1
            sum_z += z; sum_z2 += z * z
            for p in range(N_PERM):
                zp, _ = zscore(cd, t['perm_lab'][p])
                if zp > best_perm[p]:
                    best_perm[p] = zp
            if z > 3.0:
                zh, purh = zscore([cells[i] for i in t['held_idx']], t['held_lab'])
                res.append((z, pur, zh, purh, f'{gname} | {sname} | {desc} | {tname}'))

    for sname, sf in SOURCES.items():
        draws = []
        R = None
        for r in rows:
            d, R = gen(sf(r), K)
            draws.append(d)
        for k in range(K):
            col = [d[k] for d in draws]
            for mname, cells in mappings(col, R).items():
                score(f'draw {k} {mname}', cells)
        # order statistics of a window of 10 draws (OrderBy(random) over a 10-pet pool) and of 2 draws
        for s in range(0, 21):
            win = [d[s:s + POOL] for d in draws]
            order = [sorted(range(POOL), key=w.__getitem__) for w in win]
            score(f'argmin of draws {s}..{s+POOL-1}', [o[0] for o in order])
            score(f'argmax of draws {s}..{s+POOL-1}', [o[POOL - 1] for o in order])
            score(f'rank-2 smallest of draws {s}..{s+POOL-1}', [o[1] for o in order])
            score(f'rank-3 smallest of draws {s}..{s+POOL-1}', [o[2] for o in order])
        # Fisher-Yates shuffles of a 10-pet pool, first/last three elements
        for s in range(0, 9):
            for mod in (False, True):
                for fname, fy in (('backward', fy_backward), ('forward', fy_forward)):
                    perms = [fy(d[s:], R, POOL, mod) for d in draws]
                    for pos in (0, 1, 2, POOL - 3, POOL - 2, POOL - 1):
                        score(f'FY {fname} {"mod" if mod else "scaled"} skip {s} element {pos}', [p[pos] for p in perms])
    return gname, n_tests, sum_z, sum_z2, best_perm, sorted(res, reverse=True)[:40]


if __name__ == '__main__':
    with Pool(5) as pool:
        names = [g for g in GENERATORS if not os.environ.get('GENS') or any(x in g for x in os.environ['GENS'].split(','))]
        results = pool.map(work, names, chunksize=1)
    total = sum(r[1] for r in results)
    sz = sum(r[2] for r in results); sz2 = sum(r[3] for r in results)
    best_perm = [max(r[4][p] for r in results) for p in range(N_PERM)]
    allres = sorted((x for r in results for x in r[5]), reverse=True)
    print(f'hypotheses scored: {total}; z mean {sz/total:+.3f} sd {math.sqrt(sz2/total-(sz/total)**2):.3f}')
    print('best z in the whole family on SHUFFLED labels (chance level for the maximum):', ['%.2f' % b for b in best_perm])
    print('top hypotheses on dev (z dev, purity dev if <=12 cells, z held-out, purity held-out):')
    for z, pur, zh, purh, desc in allres[:15]:
        print(f'  z_dev {z:7.2f}  pur {pur if pur is None else round(pur, 3)}  z_held {zh:+6.2f}  pur_held {purh if purh is None else round(purh, 3)}  {desc}')
    per_gen = {}
    for g, n, _, _, bp, res in results:
        per_gen[g] = {'tests': n, 'best_z_dev': res[0][0] if res else None, 'best_desc': res[0][4] if res else None,
                      'best_z_held_of_that': res[0][2] if res else None, 'best_z_shuffled': bp}
    print('per generator: tests, best z dev (its held-out z), best z on shuffled labels')
    for g, v in per_gen.items():
        print(f'  {g:42s} {v["tests"]:6d}  {v["best_z_dev"] if v["best_z_dev"] is None else round(v["best_z_dev"], 2)} ({v["best_z_held_of_that"] if v["best_z_held_of_that"] is None else round(v["best_z_held_of_that"], 2)})  shuffled {[round(b, 2) for b in v["best_z_shuffled"]]}')
    json.dump({'stratum': stratum, 'rows': len(rows), 'targets': {k: {'dev': len(v['dev_idx']), 'held': len(v['held_idx'])} for k, v in T.items()},
               'hypotheses': total, 'z_mean': sz / total, 'z_sd': math.sqrt(sz2 / total - (sz / total) ** 2), 'best_z_shuffled': best_perm,
               'top': [{'z_dev': z, 'purity_dev': pur, 'z_held': zh, 'purity_held': purh, 'hypothesis': desc} for z, pur, zh, purh, desc in allres[:40]],
               'per_generator': per_gen, 'sources': list(SOURCES), 'K': K}, open(out_path, 'w'), indent=1)
