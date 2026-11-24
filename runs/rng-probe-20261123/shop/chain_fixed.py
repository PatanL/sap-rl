#!/usr/bin/env python3
"""Chain test, redone: is Seed[i+1] among the first outputs of a generator seeded with Seed[i]?
../chain/chain_probe.py ported System.Random without C#'s int32 wrap-around, which is wrong for seeds above
161803398 (92% of the logged ones). This redoes it with the wrapped port, adds old Mono, and runs the other
generators of generators.py on a sample. Also tests the reverse direction and a two-step gap.
Stdlib only. Usage: chain_fixed.py <out.json> <raw dir>..."""
import glob, json, sys
from multiprocessing import Pool
from generators import GENERATORS, dotnet, mono_old, MBIG

K_MAIN, K_OTHER, N_OTHER = 64, 16, 20000


def load(dirs):
    pairs = []
    for d in dirs:
        for f in sorted(glob.glob(d + '/*.json')):
            for p in json.load(open(f))['perspectives']['perspectives']:
                acts = sorted(p['raw']['Actions'], key=lambda a: a['BuildChangeCount'] or 0)
                prev = prev2 = None
                for a in acts:
                    try:
                        seed = json.loads(a['Response'])['Data']['Seed']
                    except Exception:
                        continue
                    if prev is not None:
                        pairs.append((prev, seed, prev2))
                    prev2, prev = prev, seed
    return pairs


def main_chunk(chunk):
    h = {'dotnet next': 0, 'dotnet reverse': 0, 'dotnet two-back': 0, 'mono_old next': 0, 'mono_old reverse': 0}
    for a, b, a2 in chunk:
        o = dotnet(a, K_MAIN)[0]
        h['dotnet next'] += b in o
        h['dotnet reverse'] += a in dotnet(b, K_MAIN)[0]
        if a2 is not None:
            h['dotnet two-back'] += b in dotnet(a2, K_MAIN)[0]
        h['mono_old next'] += b in mono_old(a, K_MAIN)[0]
        h['mono_old reverse'] += a in mono_old(b, K_MAIN)[0]
    return h


def other_chunk(args):
    name, chunk = args
    gen = GENERATORS[name]
    hits = 0
    for a, b, _ in chunk:
        o, R = gen(a, K_OTHER)
        if any(b == v or b == (v & 0x7FFFFFFF) or b == (v >> 1) or b == v % MBIG for v in o):
            hits += 1
    return name, hits


if __name__ == '__main__':
    pairs = load(sys.argv[2:])
    print('consecutive seed pairs (same player):', len(pairs))
    n = len(pairs)
    chunks = [pairs[i::25] for i in range(25)]
    with Pool(5) as pool:
        res = pool.map(main_chunk, chunks)
        tot = {k: sum(r[k] for r in res) for k in res[0]}
        exp = n * K_MAIN / MBIG
        print(f'System.Random (wrapped port), first {K_MAIN} outputs; chance expectation {exp:.4f} hits per test')
        for k, v in tot.items():
            print(f'   {k}: {v} hits of {n}')
        sample = pairs[:: max(1, n // N_OTHER)]
        others = pool.map(other_chunk, [(g, sample) for g in GENERATORS if 'System.Random' not in g])
    print(f'other generators, first {K_OTHER} outputs (raw, low 31 bits, >>1, mod 2^31-1), {len(sample)} pairs:')
    for g, h in others:
        print(f'   {g}: {h}')
    json.dump({'pairs': n, 'K': K_MAIN, 'system_random': tot, 'others_pairs': len(sample), 'others_K': K_OTHER, 'others': dict(others)},
              open(sys.argv[1], 'w'), indent=1)
