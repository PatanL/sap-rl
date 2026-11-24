#!/usr/bin/env python3
"""Statistics of the per-action Seed / Hash logged in Super Auto Pets replays. Stdlib only.
Usage: seed_stats.py <out.json> <raw dir>...
"""
import glob, json, math, os, random, sys
from collections import Counter, defaultdict

M31 = 1 << 31
MBIG = 2147483647
out = {}


def J(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


recs = []        # (dir, match, pid, turn, type, bcc, seed, hash, req_boardhash, time)
end_hash = []    # (type-11 hash, snapshot Bor.Hash)
battle = []      # (match, turn, battle seed, type-11 seed, type-4 seed of this turn)
for d in sys.argv[2:]:
    for f in sorted(glob.glob(d + '/*.json')):
        j = json.load(open(f))
        for per in j['perspectives']['perspectives']:
            acts = sorted(per['raw']['Actions'], key=lambda a: a['BuildChangeCount'] or 0)
            last11 = {}
            last4 = {}
            for a in acts:
                rs = J(a.get('Response')).get('Data') or {}
                if 'Seed' in rs:
                    rq = (J(a.get('Request')).get('Data') or {})
                    recs.append((os.path.basename(d), j['meta']['id'], per['participationId'], a['Turn'], a['Type'], a['BuildChangeCount'],
                                 rs['Seed'], rs.get('Hash', 0), rq.get('BoardHash'), a.get('CreatedOn')))
                    if a['Type'] == 11:
                        last11[a['Turn']] = (rs['Seed'], rs.get('Hash', 0))
                    if a['Type'] == 4:
                        last4[a['Turn']] = rs['Seed']
                if a['Type'] == 0:
                    bu = J(a.get('Build')) if isinstance(a.get('Build'), str) else (a.get('Build') or {})
                    ba = J(a.get('Battle')) if isinstance(a.get('Battle'), str) else (a.get('Battle') or {})
                    if a['Turn'] in last11:
                        end_hash.append((last11[a['Turn']][1], (bu.get('Bor') or {}).get('Hash', 0)))
                        battle.append((j['meta']['id'], a['Turn'], ba.get('Seed', 0), last11[a['Turn']][0], last4.get(a['Turn'])))

seeds = [r[6] for r in recs]
N = len(seeds)
print('seed-bearing actions', N)
out['n_seeds'] = N
out['min'], out['max'] = min(seeds), max(seeds)
out['negative'] = sum(s < 0 for s in seeds)
print('min', min(seeds), 'max', max(seeds), 'negative', out['negative'], ' (2^31-1 =', MBIG, ')')


def chi2_uniform(counts, n_bins, n):
    e = n / n_bins
    return sum((counts.get(i, 0) - e) ** 2 / e for i in range(n_bins))


def z_of(chi2, dof):
    return (chi2 - dof) / math.sqrt(2 * dof)


# 1. uniformity
c = Counter(s >> 23 for s in seeds)
x = chi2_uniform(c, 256, N)
print(f'top-8-bit bins: chi2 {x:.1f} on 255 dof (z {z_of(x, 255):+.2f})')
out['chi2_top8'] = [x, 255]
bits = [sum((s >> b) & 1 for s in seeds) / N for b in range(31)]
worst = max(range(31), key=lambda b: abs(bits[b] - 0.5))
print(f'bit balance: worst bit {worst} mean {bits[worst]:.4f} (z {(bits[worst]-0.5)*2*math.sqrt(N):+.2f}); 31 bits tried')
out['bit_means'] = bits
for m in (2, 3, 5, 7, 10, 16, 60, 64, 1000):
    c = Counter(s % m for s in seeds)
    x = chi2_uniform(c, m, N)
    print(f'seed mod {m}: chi2 {x:.1f} on {m-1} dof (z {z_of(x, m-1):+.2f})')
out['mean_over_2^31'] = sum(seeds) / N / M31
print('mean / 2^31 =', out['mean_over_2^31'], '(uniform: 0.5, sd %.5f)' % (1 / math.sqrt(12 * N)))
by_type = defaultdict(list)
for r in recs:
    by_type[r[4]].append(r[6])
for t in sorted(by_type):
    v = by_type[t]
    print(f'  type {t}: n {len(v)} mean/2^31 {sum(v)/len(v)/M31:.4f}')

# 2. duplicates
pos = defaultdict(list)
for i, s in enumerate(seeds):
    pos[s].append(i)
dups = [v for v in pos.values() if len(v) > 1]
exp = N * (N - 1) / 2 / MBIG
print(f'duplicate seed values: {len(dups)} (pairs {sum(len(v)*(len(v)-1)//2 for v in dups)}); birthday expectation for uniform 31-bit: {exp:.1f}')
out['duplicate_seeds'] = {'n': len(dups), 'expected_uniform': exp, 'pairs': []}
for v in dups:
    desc = [dict(dir=recs[i][0], match=recs[i][1][:8], pid=recs[i][2][-6:], turn=recs[i][3], type=recs[i][4], bcc=recs[i][5], hash=recs[i][7], time=recs[i][9]) for i in v]
    out['duplicate_seeds']['pairs'].append({'seed': seeds[v[0]], 'where': desc})
    print('   seed', seeds[v[0]], [(x['match'], x['pid'], 't%d' % x['turn'], 'type%d' % x['type'], x['hash']) for x in desc])
hashes = [r[7] for r in recs]
hc = Counter(hashes)
hd = sum(1 for v in hc.values() if v > 1)
print(f'duplicate Hash values: {hd}; birthday expectation for uniform 32-bit: {N*(N-1)/2/2**32:.1f}; zero hashes {hc.get(0, 0)}')
out['duplicate_hashes'] = {'n': hd, 'expected_uniform': N * (N - 1) / 2 / 2 ** 32}

# 3. chain properties
chain = Counter()
pairs = []
prev = None
for r in recs:
    if prev is not None and prev[2] == r[2]:
        pairs.append((prev[6], r[6], prev, r))
        if r[8] is not None:
            chain['request BoardHash == previous response Hash' if r[8] == prev[7] else 'request BoardHash != previous response Hash'] += 1
    prev = r
print(dict(chain))
out['hash_chain'] = dict(chain)
eh = Counter(a == b for a, b in end_hash)
print('end-of-turn snapshot Bor.Hash == Type 11 response Hash:', dict(eh))
out['snapshot_hash_equals_type11_hash'] = {str(k): v for k, v in eh.items()}
bs = Counter()
for m, t, b, s11, s4 in battle:
    bs['battle seed == type 11 seed'] += (b == s11)
    bs['battle seed == type 4 seed'] += (b == s4)
    bs['battle seed negative'] += (b < 0)
    bs['n'] += 1
print(dict(bs))
out['battle_seed'] = dict(bs)
# same seed for both players in a turn?
sm = defaultdict(list)
for r in recs:
    if r[4] in (4, 11):
        sm[(r[1], r[3], r[4])].append(r[6])
same = Counter(len(set(v)) == 1 for v in sm.values() if len(v) == 2)
print('type 4/11 seed identical for the two players of a match-turn:', dict(same))

# 4. consecutive seeds
P = len(pairs)
print('consecutive pairs (same player)', P)
a = [p[0] / M31 for p in pairs]; b = [p[1] / M31 for p in pairs]
ma, mb = sum(a) / P, sum(b) / P
cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / P
va = sum((x - ma) ** 2 for x in a) / P; vb = sum((y - mb) ** 2 for y in b) / P
corr = cov / math.sqrt(va * vb)
print(f'serial correlation {corr:+.5f} (null sd {1/math.sqrt(P):.5f})')
out['serial_corr'] = corr
c = Counter(((y - x) % M31) >> 23 for x, y, *_ in pairs)
x2 = chi2_uniform(c, 256, P)
print(f'difference mod 2^31, top-8-bit bins: chi2 {x2:.1f} / 255 (z {z_of(x2, 255):+.2f})')
c = Counter(((y - x) % MBIG) * 256 // MBIG for x, y, *_ in pairs)
x2 = chi2_uniform(c, 256, P)
print(f'difference mod 2^31-1, 256 bins: chi2 {x2:.1f} / 255 (z {z_of(x2, 255):+.2f})')
pc = sum(bin(x ^ y).count('1') for x, y, *_ in pairs) / P
print(f'mean popcount of xor {pc:.4f} (independent: 15.5, sd {math.sqrt(31/4/P):.4f})')
c = Counter((x ^ y) >> 23 for x, y, *_ in pairs)
x2 = chi2_uniform(c, 256, P)
print(f'xor, top-8-bit bins: chi2 {x2:.1f} / 255 (z {z_of(x2, 255):+.2f})')
c = Counter(((x >> 27) << 4) | (y >> 27) for x, y, *_ in pairs)
x2 = chi2_uniform(c, 256, P)
print(f'joint (top 4 bits of seed, top 4 bits of next): chi2 {x2:.1f} / 255 (z {z_of(x2, 255):+.2f})')
c = Counter(((x & 15) << 4) | (y & 15) for x, y, *_ in pairs)
x2 = chi2_uniform(c, 256, P)
print(f'joint (low 4 bits of seed, low 4 bits of next): chi2 {x2:.1f} / 255 (z {z_of(x2, 255):+.2f})')
most = Counter((y - x) % M31 for x, y, *_ in pairs).most_common(3)
print('most common differences:', most)
most = Counter(x ^ y for x, y, *_ in pairs).most_common(3)
print('most common xors:', most)

# seed vs hash relation (same action, and seed vs previous hash)
c = Counter(((r[6] >> 27) << 4) | ((r[7] >> 28) & 15) for r in recs)
x2 = chi2_uniform(c, 256, N)
print(f'joint (top 4 bits of seed, top 4 bits of own hash): chi2 {x2:.1f} / 255 (z {z_of(x2, 255):+.2f})')
c = Counter(((r[6] & 15) << 4) | (r[7] & 15) for r in recs)
x2 = chi2_uniform(c, 256, N)
print(f'joint (low 4 bits of seed, low 4 bits of own hash): chi2 {x2:.1f} / 255 (z {z_of(x2, 255):+.2f})')
eq = sum(1 for r in recs if r[6] == (r[7] & 0x7FFFFFFF) or r[6] == abs(r[7]))
print('seed == |hash| or hash & 0x7fffffff:', eq)

# 5. linear congruential relations seed' = (a*seed + c) mod m, classic constants (and their 31-bit truncations)
LCG = {
    'ANSI C / glibc TYPE_0 (1103515245,12345,2^31)': (1103515245, 12345, 1 << 31),
    'ANSI C (1103515245,12345,2^32)': (1103515245, 12345, 1 << 32),
    'MSVC (214013,2531011,2^32)': (214013, 2531011, 1 << 32),
    'Numerical Recipes (1664525,1013904223,2^32)': (1664525, 1013904223, 1 << 32),
    'Borland (22695477,1,2^32)': (22695477, 1, 1 << 32),
    'MINSTD (16807,0,2^31-1)': (16807, 0, MBIG),
    'MINSTD2 (48271,0,2^31-1)': (48271, 0, MBIG),
    'Park-Miller 69621': (69621, 0, MBIG),
    'RANDU (65539,0,2^31)': (65539, 0, 1 << 31),
    'Unity init (1812433253,1,2^32)': (1812433253, 1, 1 << 32),
    'MT init-like (1812433253,0,2^32)': (1812433253, 0, 1 << 32),
    'Knuth (6364136223846793005,1442695040888963407,2^64)': (6364136223846793005, 1442695040888963407, 1 << 64),
    'java (0x5DEECE66D,0xB,2^48)': (0x5DEECE66D, 0xB, 1 << 48),
    'glibc rand_r-like (1103515245,12345,2^32)>>16': (1103515245, 12345, 1 << 32),
    'cc65 (16843009,826366247,2^32)': (16843009, 826366247, 1 << 32),
    'Delphi (134775813,1,2^32)': (134775813, 1, 1 << 32),
    'VB (1140671485,12820163,2^24)': (1140671485, 12820163, 1 << 24),
    'drand48 as java': (0x5DEECE66D, 0xB, 1 << 48),
}
lcg_hits = {}
for name, (aa, cc, mm) in LCG.items():
    h = 0
    for x, y, *_ in pairs:
        s = x
        for step in range(4):                       # up to 4 steps of the recurrence
            s = (aa * s + cc) % mm
            if y in (s, s & 0x7FFFFFFF, s % MBIG, (s >> 16) & 0x7FFFFFFF, s >> 1, (s >> 32) & 0x7FFFFFFF, s >> 33, (s >> 17)):
                h += 1
                break
    lcg_hits[name] = h
print('LCG relation hits (seed -> next seed, up to 4 steps, 8 output truncations) over', P, 'pairs:')
print('  ', lcg_hits)
out['lcg_hits'] = lcg_hits
out['pairs'] = P
# generic test for ANY multiplicative/affine relation mod 2^31-1 or mod 2^31:
# if s' = a*s + c (mod m) for fixed (a, c), then for two pairs (s1,s1'),(s2,s2') and (s3,s3') the implied a agrees.
def implied_a(m, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    try:
        a1 = (y1 - y2) * pow(x1 - x2, -1, m) % m
        a2 = (y2 - y3) * pow(x2 - x3, -1, m) % m
    except ValueError:
        return None
    return a1 == a2
random.seed(7)
cons = Counter()
xy = [(p[0], p[1]) for p in pairs]
for _ in range(20000):
    tri = random.sample(xy, 3)
    r = implied_a(MBIG, tri)
    if r is not None:
        cons['mod 2^31-1: triples consistent with one affine map'] += r
        cons['mod 2^31-1: triples tested'] += 1
print(dict(cons), '(a fixed affine map mod the prime 2^31-1 would make every triple consistent)')
out['affine_triples'] = dict(cons)

# 6. is the seed time-derived?  compare seed with timestamp (same-second / close actions)
def tsec(t):
    # '2026-11-05T02:21:47.138129+00:00'
    try:
        import datetime
        return datetime.datetime.fromisoformat(t).timestamp()
    except Exception:
        return None
tt = [(tsec(r[9]), r[6]) for r in recs[:60000]]
tt = [(t, s) for t, s in tt if t is not None]
c = Counter((int(t * 1000) & 15) << 4 | (s & 15) for t, s in tt)
x2 = chi2_uniform(c, 256, len(tt))
print(f'joint (low 4 bits of ms timestamp, low 4 bits of seed): chi2 {x2:.1f} / 255 (z {z_of(x2, 255):+.2f}) [timestamp bins themselves need not be uniform]')
mt = sum(t for t, s in tt) / len(tt); ms = sum(s for t, s in tt) / len(tt)
num = sum((t - mt) * (s - ms) for t, s in tt)
den = math.sqrt(sum((t - mt) ** 2 for t, s in tt) * sum((s - ms) ** 2 for t, s in tt))
print(f'correlation(seed, timestamp) {num/den:+.4f} over {len(tt)} actions')
out['corr_seed_time'] = num / den

json.dump(out, open(sys.argv[1], 'w'), indent=1)
