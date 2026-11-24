#!/usr/bin/env python3
"""Step 4: what the data can say about determinism. Stdlib only.
Usage: determinism.py <rolls.jsonl> <seed_stats.json>"""
import json, sys
from collections import Counter, defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
ss = json.load(open(sys.argv[2]))
print('generation events in verified turns:', len(rows), ' distinct seeds among them:', len({r['seed'] for r in rows}))
by = defaultdict(list)
for r in rows:
    by[r['seed']].append(r)
d = [v for v in by.values() if len(v) > 1]
print('seed values shared by two generation events in the dataset:', len(d))
for v in d:
    for r in v:
        print('   seed', r['seed'], 'turn', r['turn'], 'type', r['type'], 'pets', r['pets'], 'foods', r['foods'], r['match'][:8])
print('all actions: duplicate seed values', ss['duplicate_seeds']['n'], 'expected by chance for uniform 31-bit values %.1f' % ss['duplicate_seeds']['expected_uniform'])
both = [p for p in ss['duplicate_seeds']['pairs'] if all(w['type'] in (4, 5) for w in p['where'])]
print('of these, pairs where both actions generate a shop (Type 4 or 5):', len(both))
for p in both:
    print('   ', p['seed'], [(w['match'], 'turn %d' % w['turn'], 'type %d' % w['type']) for w in p['where']])
    for w in p['where']:
        hit = [r for r in rows if r['seed'] == p['seed'] and r['match'].startswith(w['match']) and r['turn'] == w['turn']]
        print('       in dataset:', [(r['turn'], r['pets'], r['foods']) for r in hit] or 'no (turn not verified)')
same_turn = [p for p in both if len({w['turn'] for w in p['where']}) == 1]
print('same seed AND same turn:', len(same_turn), '-> the determinism check has no usable case in this corpus')
# two players of one match: any seed in common?
m = defaultdict(lambda: defaultdict(set))
for r in rows:
    m[r['match']][r['pid']].add(r['seed'])
common = sum(1 for v in m.values() if len(v) == 2 and set.intersection(*v.values()))
print('matches in which the two players share any generation seed:', common, 'of', sum(1 for v in m.values() if len(v) == 2))
