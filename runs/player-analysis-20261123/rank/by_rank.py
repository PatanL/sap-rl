"""Does a player's rank predict the strength of their boards? Boards of the top and bottom rank quarters are paired at
random on the same turn and fought on the Rust engine (ghost_fights.py). Stdlib only.
Usage: by_rank.py build <pairs.json> | by_rank.py report <pairs.json> <out.json>"""
import glob, json, random, sys, statistics
O = '/home/dzliu/pzliu/sap-rl/runs/replay-oracle-20261122/'
KEEP = ("id", "attack", "health", "experience", "status", "triggersConsumed", "parrotCopies")
TURNS = (3, 5, 7, 10, 12); PAIRS = 3000
mode, path = sys.argv[1], sys.argv[2]
if mode == 'build':
    rank = {}
    for d in ('raw', 'raw-fresh', 'raw-fresh2', 'raw-fresh3', 'raw-fresh4'):
        for f in glob.glob(O + d + '/*.json'):
            x = json.load(open(f)); m = x['meta']
            for p in x['perspectives']['perspectives']:
                rank[p['participationId']] = m['player_rank'] if p['participationId'] == m['participation_id'] else m['opponent_rank']
    values = sorted(v for v in rank.values() if isinstance(v, (int, float)))
    lo, hi = values[len(values) // 4], values[3 * len(values) // 4]
    boards = {t: {'top': [], 'bottom': [], 'mid': []} for t in TURNS}; seen = set()
    for name in ('fixtures-340.json', 'fixtures-fresh-all.json', 'fixtures-fresh2-all.json', 'fixtures-fresh3-all.json', 'fixtures-fresh4-all.json'):
        for f in json.load(open(O + name))['fixtures']:
            key = (f['participationId'], f['turn'])
            if key in seen or f['turn'] not in TURNS or not f['player'] or f['participationId'] not in rank: continue
            seen.add(key); r = rank[f['participationId']]
            boards[f['turn']]['top' if r >= hi else 'bottom' if r <= lo else 'mid'].append([{k: p[k] for k in KEEP if k in p} for p in f['player']])
    rng = random.Random(20261123); fixtures = []; index = []
    for t in TURNS:
        for a, b in (('top', 'bottom'), ('top', 'mid'), ('mid', 'bottom'), ('mid', 'mid')):
            for _ in range(PAIRS):
                fixtures.append({'player': rng.choice(boards[t][a]), 'opponent': rng.choice(boards[t][b])}); index.append([t, a, b])
    json.dump({'fixtures': fixtures, 'index': index, 'cut': [lo, hi], 'counts': {t: {k: len(v) for k, v in boards[t].items()} for t in TURNS},
               'ranks': [values[0], values[len(values) // 2], values[-1]], 'players': len(values)}, open(path, 'w'))
    print('rank quarters: bottom <=', lo, ' top >=', hi, ' boards', {t: {k: len(v) for k, v in boards[t].items()} for t in TURNS})
else:
    d = json.load(open(path)); rows = json.load(open(sys.argv[3]))['rows']; acc = {}
    for (t, a, b), r in zip(d['index'], rows):
        o = r['simOutcomes']; n = sum(o.values()); acc.setdefault((a, b), {}).setdefault(t, []).append((o['player'] + .5 * o['draw']) / n)
    for (a, b), per in acc.items():
        print(f'{a} v {b}:', {t: f'{100 * statistics.mean(v):.1f}' for t, v in per.items()}, 'mean %.1f' % (100 * statistics.mean(statistics.mean(v) for v in per.values())),
              '+-%.1f' % (196 * statistics.mean(statistics.pstdev(v) / len(v) ** .5 for v in per.values())))
