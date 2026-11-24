"""Yardstick-side boards by turn from top-quarter players only (rating >= the pool's 75th percentile). Scoring only."""
import glob, hashlib, json, sys
O = '/home/dzliu/pzliu/sap-rl/runs/replay-oracle-20261122/'; KEEP = ("id", "attack", "health", "experience", "status", "triggersConsumed", "parrotCopies")
rank = {}
for d in ('raw', 'raw-fresh', 'raw-fresh2', 'raw-fresh3', 'raw-players'):
    for f in glob.glob(O + d + '/*.json'):
        x = json.load(open(f)); m = x['meta']
        for p in x['perspectives']['perspectives']:
            rank[p['participationId']] = m['player_rank'] if p['participationId'] == m['participation_id'] else m['opponent_rank']
values = sorted(v for v in rank.values() if isinstance(v, (int, float))); cut = values[3 * len(values) // 4]
by_turn, seen, players = {}, set(), set()
for name in ('fixtures-340.json', 'fixtures-fresh-all.json', 'fixtures-fresh2-all.json', 'fixtures-fresh3-all.json', 'fixtures-players.json'):
    for f in json.load(open(O + name))['fixtures']:
        if int(hashlib.sha256(f['matchId'].encode()).hexdigest(), 16) % 5 >= 2 or not f['player']: continue
        key = (f['participationId'], f['turn'])
        if key in seen or rank.get(f['participationId'], 0) < cut: continue
        seen.add(key); players.add(f['participationId']); by_turn.setdefault(str(f['turn']), []).append([{k: p[k] for k in KEEP if k in p} for p in f['player']])
json.dump({'split': 'yardstick', 'ratingCut': cut, 'byTurn': by_turn}, open(sys.argv[1], 'w'), separators=(',', ':'))
print('rating cut', cut, 'players', len(players), {int(t): len(b) for t, b in sorted(by_turn.items(), key=lambda kv: int(kv[0]))})
