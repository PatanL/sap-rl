import json,hashlib,itertools
from pathlib import Path
from collections import Counter
R=Path('../replay-oracle-20261122')
names={}
for p in R.glob('raw*/*.json'):
    for c in json.loads(p.read_text())['perspectives']['perspectives']:
        r=c['raw'];names[r['ParticipationId']]=r.get('UserName','') or ''
options={}
for p in R.glob('fixtures*.json'):
    if 'slotcheck' in str(p):continue
    options[p.name]={(r['participationId'],r['turn']) for r in json.loads(p.read_text())['fixtures'] if r['player'] and int(hashlib.sha256(r['matchId'].encode()).hexdigest(),16)%5<2 and 'maxwu' not in names[r['participationId']].lower()}
goal=json.load(open('../player-analysis-20261123/yardstick-baseline.json'))['fieldBoardsPerTurn']
for n in range(1,8):
 for comb in itertools.combinations(options,n):
    s=set.union(*(options[k] for k in comb));cnt=Counter(t for _,t in s)
    if all(cnt[int(t)]==v for t,v in goal.items()):
        print(comb);Path('results/fixtures.json').write_text(json.dumps(list(comb)));quit()
print({k:sum(t==1 for _,t in v) for k,v in options.items()})
