#!/usr/bin/env python3
"""What the logged Hash is. Stdlib only. Usage: hash_checks.py <genesis.jsonl> <raw dir>..."""
import glob, json, sys
from collections import Counter, defaultdict
gen = [json.loads(l) for l in open(sys.argv[1])]
g = defaultdict(list)
for r in gen:
    g[(tuple(r['pets']), tuple(r['foods']))].append(r)
pairs = same = 0
for v in g.values():
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            pairs += 1
            same += v[i]['hash_after'] == v[j]['hash_after']
print(f'players whose first shop is identical (same 3 pets in the same order, same food): {pairs} pairs; '
      f'same Hash after the turn-1 start action in {same} of them')
recs = []
for d in sys.argv[2:]:
    for f in sorted(glob.glob(d + '/*.json')):
        j = json.load(open(f))
        for per in j['perspectives']['perspectives']:
            prev = None
            for a in sorted(per['raw']['Actions'], key=lambda a: a['BuildChangeCount'] or 0):
                try:
                    dd = json.loads(a['Response'])['Data']
                except Exception:
                    continue
                rq = json.loads(a['Request'])
                rec = (dd['Hash'], dd['Seed'], per['participationId'][-6:], a['Turn'], a['Type'], a['BuildChangeCount'],
                       sorted(k for k in rq.get('Data', {}) if k not in ('BuildId', 'BoardHash')), prev)
                recs.append(rec)
                prev = dd['Hash']
by = defaultdict(list)
for r in recs:
    by[r[0]].append(r)
d = [v for v in by.values() if len(v) > 1]
print('duplicate Hash values:', len(d), '(chance for uniform 32-bit: %.1f)' % (len(recs) * (len(recs) - 1) / 2 / 2 ** 32))
for v in d:
    print('   ', v[0][0], [(r[2], 'turn %d' % r[3], 'type %d' % r[4], 'bcc %d' % r[5], 'seed %d' % r[1], 'unchanged from previous' if r[7] == r[0] else '') for r in v])
unch = Counter(r[4] for r in recs if r[7] == r[0])
print('actions whose response Hash equals the previous response Hash (state unchanged), by type:', dict(unch))
