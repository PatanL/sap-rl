"""Validate linkage of the completed throughput, strength and outside-yardstick receipts."""
import hashlib
import json
import re
from pathlib import Path

R=Path('results')
def read(path):return json.loads(path.read_text())
def verify():
    digest='ddf5a8f22f0c39067e7c96a6054f01b5c87b624373b4fc1879715469470f4372'
    a=read(R/'parity-production.json')
    n=a['counts'];assert a['exact'] and a['decision_target']==8192 and n['matches']>=256 and n['selected_batches']>=3
    for check in ['parity-production.json','parity-256.json']:
        n=read(R/check)['counts']
        assert n['seat_decisions']==n['live_decisions']+n['frozen_decisions']
        assert n['live_decisions']==n['delivered']+n['pending']
        assert n['delivered']==n['learner_decisions']+n['queued']+n['stale']
    selected=read(R/'selection.json')['games']
    release=Path('../train-live-20261123/GPU-FREE')
    for folder in [*R.glob('act-*'),*R.glob('train-*'),R/'proof']:
        if not folder.is_dir():continue
        q=read(folder/'result.json')
        assert q['library_sha256']==digest and q['wall']<870 and q['matches']>0 and q['seat_s']>0
        for row in q['events']:
            assert row['version']-2<=row['min_version']<=row['max_version']<=row['version']-1
        if q['training']:assert q['delivered']==q['learned']+q['queued']+q['stale']
    proof=read(R/'proof/result.json');assert proof['updates']==100 and proof['games']==selected
    ck=next((R/'proof').glob('update-*.pt'))
    with ck.open('rb') as f:ch=hashlib.file_digest(f,'sha256').hexdigest()
    strength=read(R/'strength.json');boards=read(R/'board.json')
    assert strength['library_sha256']==digest
    assert strength['candidate']['sha256']==boards['sha256']==ch
    assert strength['pairs']==2048 and strength['batch']==4096 and strength['games']==4096
    rows=strength['records'];seeds={r['seed'] for r in rows}
    assert len(seeds)==2048 and {(r['seed'],r['seat']) for r in rows}=={(seed,s) for seed in seeds for s in (0,1)}
    assert boards['split']=='yardstick' and boards['batch']==128 and boards['matches']==1000
    assert set(boards['scores'])=={'3','5','7','10','12'}
    index=sum(boards['scores'].values())/5;assert abs(boards['index']-index)<1e-12
    g=read(R/'ghost.json');assert all(g['comparisons']['bot v field'][k]['pairs']==4000 for k in boards['scores'])
    rehearsal=read(R/'board-rehearsal.json');assert rehearsal['matches']==1000 and rehearsal['hidden_fields_from_archive']
    for doc in ['REPORT.md','interface.md','signal.md']:
        for match in re.findall(r'\]\(([^)]+)\)',Path(doc).read_text()):assert Path(match).exists(),match
    print(f'Verified {proof["learned"]:,} learned rows in 100 updates, {len(rows):,} paired-game records and {boards["matches"]:,} greedy yardstick matches. Checkpoint SHA-256 {ch}.')
if __name__=='__main__':verify()
