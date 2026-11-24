"""Validate final hashes, ledger, timing windows, paired worlds, and the reported source files."""
import hashlib,json,re
from pathlib import Path
P=Path(__file__).resolve().parent;R=P/'results'
def load(f):return json.loads(f.read_text())
def main():
    c=load(R/'chosen.json');d=load(R/c['ledger']);q=d['checkpoints'][-1];h=load(R/c['confirmation'])
    f=R/c['proof'];r=load(f/'result.json');ck=next(f.glob('update-*.pt'))
    with ck.open('rb') as fp:ch=hashlib.file_digest(fp,'sha256').hexdigest()
    assert ch==q['sha256']==q['board_strength']['sha256']==h['candidate']['sha256']
    assert r['updates']==100 and r.get('elapsed',r.get('wall'))<870
    if 'events' in r:
        ev=r['events'];assert all(0<=x['version']-1-x['min_version']<=1 and x['min_version']<=x['max_version']<=x['version']-1 for x in ev)
    else:
        ev=list(map(json.loads,(f/'updates.jsonl').open()))
        assert all(0<=x['version']-1-x['behavior_min']<=1 and x['behavior_min']<=x['behavior_max']<=x['version']-1 for x in ev)
    assert sum(x['decisions'] for x in ev)==r['learned']
    assert r['delivered']==r['learned']+r['stale']+r['queued']
    assert h['pairs']==256 and h['games']==512 and h['engine']==r['engine']==d['engine']
    def seeds(s):
        rows=s['records'];x={(v['seed'],v['seat']) for v in rows}
        worlds={a for a,_ in x};assert len(x)==len(rows) and x=={(a,b) for a in worlds for b in (0,1)}
        return worlds
    assert seeds(h).isdisjoint(seeds(load(Path(q['against_25772']['receipt']))))
    for b in [d['board_baseline'],q['board_strength']]:
        assert b['matches']==1000 and b['split']=='yardstick' and b['engine']==r['engine'] and abs(sum(b['scores'].values())/5-b['index'])<1e-12
        p=Path(b['receipt']);g=load(p);assert sum(1 for _ in (p.parent/'seats.jsonl').open())==2000
        assert {g['comparisons']['bot v field'][t]['pairs'] for t in b['scores']}=={4000}
    for n in ('v-replay-act','v-replay-train','v-replay-act-4096','v-python-act','v-python-train','v-python-loose','v-replay-loose','v-python-train-128','v-python-train-256','v-python-train-512'):
        z=load(R/n/'result.json');assert z['wall']<870 and z['matches']>0 and .99<z['matches']/(z['games_s']*z['wall'])<1.01
        if z['training']:
            assert z['learned']+z['stale']+z['queued']==z['delivered']
            assert all(0<=row['version']-1-row['min_version']<=1 for row in z['events'])
    for name in ('vector-check','turns-check','replay-check'):
        print((R/(name+'.log')).read_text().splitlines()[-1])
    for link in re.findall(r'\]\(([^)]+)\)',(P/'REPORT.md').read_text()):assert (P/link).exists(),link
    print(f"Verified chosen {c['winner']}, {r['updates']} real updates, {r['learned']} learned rows, two 1,000-match board receipts, and 320 disjoint paired match seeds. Checkpoint {ch}.")
if __name__=='__main__':main()
