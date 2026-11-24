"""Second yardstick: the bot's recorded boards against boards of TOP-QUARTER ranked players only (rating >= the pool's
75th percentile), yardstick side of the split only (sha256(match id) % 5 < 2), same turns and pairing method as the
main ranked-board index. Stdlib only.
Usage: top_index.py build <bot battles.jsonl> <pairs.json>  |  top_index.py report <pairs.json> <fights out.json>"""
import glob, hashlib, json, random, statistics, sys
O = '/home/dzliu/pzliu/sap-rl/runs/replay-oracle-20261122/'
KEEP = ("id", "attack", "health", "experience", "status", "triggersConsumed", "parrotCopies")
TURNS = (3, 5, 7, 10, 12); PAIRS = 4000
if sys.argv[1] == 'build':
    rank = {}
    for d in ('raw', 'raw-fresh', 'raw-fresh2', 'raw-fresh3', 'raw-players'):
        for f in glob.glob(O + d + '/*.json'):
            x = json.load(open(f)); m = x['meta']
            for p in x['perspectives']['perspectives']:
                rank[p['participationId']] = m['player_rank'] if p['participationId'] == m['participation_id'] else m['opponent_rank']
    values = sorted(v for v in rank.values() if isinstance(v, (int, float))); cut = values[3 * len(values) // 4]
    top = {t: [] for t in TURNS}; field = {t: [] for t in TURNS}; seen = set()
    for name in ('fixtures-340.json', 'fixtures-fresh-all.json', 'fixtures-fresh2-all.json', 'fixtures-fresh3-all.json', 'fixtures-players.json'):
        for f in json.load(open(O + name))['fixtures']:
            key = (f['participationId'], f['turn'])
            if key in seen or f['turn'] not in TURNS or not f['player']: continue
            if int(hashlib.sha256(f['matchId'].encode()).hexdigest(), 16) % 5 >= 2: continue      # yardstick side only
            seen.add(key); board = [{k: p[k] for k in KEEP if k in p} for p in f['player']]
            field[f['turn']].append(board)
            if rank.get(f['participationId'], 0) >= cut: top[f['turn']].append(board)
    bot = {t: [] for t in TURNS}
    for line in open(sys.argv[2]):
        row = json.loads(line)
        if row['turn'] in TURNS: bot[row['turn']] += [[{k: p[k] for k in KEEP if k in p} for p in team] for team in row['teams'] if team]
    rng = random.Random(20261123); fixtures = []; index = []
    for t in TURNS:
        for name, pool in (('top', top[t]), ('field', field[t])):
            for _ in range(PAIRS):
                fixtures.append({'player': rng.choice(bot[t]), 'opponent': rng.choice(pool)}); index.append([t, name])
    json.dump({'fixtures': fixtures, 'index': index, 'cut': cut, 'top': {t: len(v) for t, v in top.items()}, 'field': {t: len(v) for t, v in field.items()}}, open(sys.argv[3], 'w'))
else:
    d = json.load(open(sys.argv[2])); rows = json.load(open(sys.argv[3]))['rows']; acc = {}
    for (t, name), r in zip(d['index'], rows):
        o = r['simOutcomes']; acc.setdefault(name, {}).setdefault(t, []).append((o['player'] + .5 * o['draw']) / sum(o.values()))
    out = {name: {t: statistics.mean(v) for t, v in per.items()} for name, per in acc.items()}
    print(' '.join(f"{name}: index {100 * statistics.mean(per.values()):.2f} {[round(100 * per[t], 1) for t in TURNS]}" for name, per in out.items()), '| rating cut', d['cut'], 'top boards per turn', list(d['top'].values()))
