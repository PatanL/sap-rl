#!/usr/bin/env python3
"""Generator-free tests on tier-1 rolls (turns 1-2). Stdlib only.

1. Is a yes/no property of the shop (food = Honey; pet slot 1 = species X) an affine function over GF(2)
   of the seed bits (as it would be for one output bit of any LFSR/xorshift-type generator seeded linearly),
   or a quadratic one?  Random subsets of (unknowns + 20) dev rolls are solved exactly; a solution is then
   scored on held-out rolls. Under chance such a subset is solvable with probability 2^-20.
2. Single seed bits and pairs of bits against species (chi-square, Bonferroni).
Usage: bit_tests.py <rolls.jsonl> <out.json>
"""
import json, math, random, sys
from collections import Counter

DEV_DIRS = {'raw', 'raw-fresh', 'raw-fresh2'}
TIER1 = [0, 3, 17, 26, 32, 39, 47, 51, 59, 559]
rows = []
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r['turn'] <= 2 and r['type'] == 5 and r['frozen_pets'] == 0 and r['seed'] is not None and r['prev_seed'] is not None:
        rows.append(r)
random.seed(11)
out = {}

FEATS = {
    'own seed (31 bits)': (lambda r: r['seed'], 31),
    'previous seed (31 bits)': (lambda r: r['prev_seed'], 31),
    'own hash (32 bits)': (lambda r: r['hash'] & 0xFFFFFFFF, 32),
    'previous hash (32 bits)': (lambda r: (r['prev_hash'] or 0) & 0xFFFFFFFF, 32),
    'own seed + previous seed (62 bits)': (lambda r: r['seed'] | (r['prev_seed'] << 31), 62),
}


def quad(v, nb):
    """all degree<=2 monomials of the nb low bits of v, packed into an int."""
    bits = [(v >> i) & 1 for i in range(nb)]
    x = v & ((1 << nb) - 1)
    pos = nb
    for i in range(nb):
        if bits[i]:
            for j2 in range(i + 1, nb):
                if bits[j2]:
                    x |= 1 << pos
                pos += 1
        else:
            pos += nb - i - 1
    return x, nb + nb * (nb - 1) // 2


def solve(eqs, F):
    """eqs: ints with features in bits 0..F-1, constant term feature at bit F (always 1), label at bit F+1.
    Returns solution mask over F+1 coefficient bits, or None if inconsistent."""
    piv = {}
    LAB = 1 << (F + 1)
    for e in eqs:
        while e & (LAB - 1):
            p = (e & (LAB - 1)).bit_length() - 1
            if p in piv:
                e ^= piv[p]
            else:
                piv[p] = e
                break
        else:
            if e & LAB:
                return None
    # back-substitute with free variables = 0
    sol = 0
    for p in sorted(piv):
        e = piv[p]
        val = (e >> (F + 1)) & 1
        rest = e & ((1 << p) - 1)
        val ^= bin(rest & sol).count('1') & 1
        if val:
            sol |= 1 << p
    return sol


def run(name, label_fn, row_filter, feats, degree):
    f, nb = FEATS[feats]
    data = []
    for r in rows:
        if not row_filter(r):
            continue
        v = f(r)
        if degree == 2:
            x, F = quad(v, nb)
        else:
            x, F = v, nb
        data.append((x | (1 << F), label_fn(r), r['dir'] in DEV_DIRS))
    dev = [(x, l) for x, l, d in data if d]
    held = [(x, l) for x, l, d in data if not d]
    need = F + 1 + 20
    if len(dev) < need:
        return {'test': name, 'features': feats, 'degree': degree, 'skipped': f'needs {need} dev rolls, have {len(dev)}'}
    solved = 0
    best = None
    trials = 30 if degree == 1 else 6
    for _ in range(trials):
        sub = random.sample(dev, need)
        sol = solve([x | (l << (F + 1)) for x, l in sub], F)
        if sol is not None:
            solved += 1
            acc = sum((bin(x & sol).count('1') & 1) == l for x, l in held) / len(held)
            best = max(best or 0, acc)
    base = max(sum(l for _, l in held), len(held) - sum(l for _, l in held)) / len(held)
    return {'test': name, 'features': feats, 'degree': degree, 'unknowns': F + 1, 'dev': len(dev), 'held': len(held), 'subsets_tried': trials,
            'subsets_solvable': solved, 'best_heldout_accuracy': best, 'heldout_majority_rate': base}


res = []
food_filter = lambda r: r['frozen_foods'] == 0 and r['foods'] and r['foods'][0] in (0, 40)
for feats in FEATS:
    for deg in (1, 2):
        if deg == 2 and FEATS[feats][1] > 32:
            continue
        res.append(run('food slot is Honey', lambda r: int(r['foods'][0] == 40), food_filter, feats, deg))
for sp in TIER1:
    for feats in FEATS:
        res.append(run(f'pet slot 1 is species {sp}', lambda r, sp=sp: int(r['pets'][0] == sp), lambda r: r['pets'][0] in TIER1, feats, 1))
for x in res:
    print(x)
out['gf2'] = res
print('subsets solvable in total:', sum(x.get('subsets_solvable', 0) for x in res), 'of', sum(x.get('subsets_tried', 0) for x in res))

# single bits and bit pairs vs species
def z_table(cells, labels):
    n = len(cells)
    rc, cc = Counter(cells), Counter(labels)
    j = Counter(zip(cells, labels))
    chi = n * (sum(v * v / (rc[a] * cc[b]) for (a, b), v in j.items()) - 1)
    dof = (len(rc) - 1) * (len(cc) - 1)
    return (chi - dof) / math.sqrt(2 * dof) if dof > 0 else 0.0


bit_res = {}
for feats in ('own seed (31 bits)', 'previous seed (31 bits)', 'own hash (32 bits)', 'previous hash (32 bits)'):
    f, nb = FEATS[feats]
    for tname, sel, lab in (('pet slot 1', lambda r: r['pets'][0] in TIER1, lambda r: r['pets'][0]),
                            ('food slot', food_filter, lambda r: r['foods'][0])):
        rr = [r for r in rows if sel(r)]
        labels = [lab(r) for r in rr]
        vals = [f(r) for r in rr]
        zs = []
        for i in range(nb):
            zs.append((z_table([(v >> i) & 1 for v in vals], labels), f'bit {i}'))
            for j2 in range(i + 1, nb):
                zs.append((z_table([((v >> i) & 1) * 2 + ((v >> j2) & 1) for v in vals], labels), f'bits {i},{j2}'))
        zs.sort(reverse=True)
        random.shuffle(labels)
        zperm = max(max(z_table([(v >> i) & 1 for v in vals], labels) for i in range(nb)),
                    max(z_table([((v >> i) & 1) * 2 + ((v >> j2) & 1) for v in vals], labels) for i in range(nb) for j2 in range(i + 1, nb)))
        bit_res[f'{feats} vs {tname}'] = {'rolls': len(rr), 'tests': len(zs), 'best_z': zs[0][0], 'best': zs[0][1], 'best_z_shuffled_labels': zperm}
        print(f'{feats} vs {tname}: {len(rr)} rolls, {len(zs)} single-bit and bit-pair tables, best z {zs[0][0]:.2f} ({zs[0][1]}), best on shuffled labels {zperm:.2f}')
out['bits'] = bit_res
json.dump(out, open(sys.argv[2], 'w'), indent=1)
