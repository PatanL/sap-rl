"""Test ledger state transitions without touching production files or consuming GPU time."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import watch
def test():
    with tempfile.TemporaryDirectory(dir=watch.JOB/'tmp') as t:
        root=Path(t);run=root/'runs'/'isolated';run.mkdir(parents=True);path=run.parent/'progression.json'
        for k in (1,2,3,4):(run/f'update-{k:06d}.pt').write_bytes(bytes([k]))
        def score(a,b,mode,pairs,tag):
            k=int(Path(a).stem.split('-')[-1])
            bounds={'screen':[(.51,.65),(.51,.62),(.35,.43),(.45,.58)],'confirm':[(.46,.63),(.55,.71),(.3,.42),(.4,.6)]}[mode][k-1]
            receipt=root/(tag+'.json');watch.atomic(receipt,dict(records=[dict(seed=100*k+int(mode=='confirm'))]))
            return dict(ci95=bounds,score=sum(bounds)/2,candidate_update=k,receipt=str(receipt))
        with patch.object(watch,'ROOT',root),patch.object(watch,'scoring',score),patch.object(watch,'panel',return_value={'test':{'unavailable':True}}):
            watch.observe(run,path);first=watch.read(path);watch.observe(run,path);assert watch.read(path)==first
            assert [r['confirmed_champion'] for r in first['checkpoints']]==[False,True,False,False]
            assert [r['confirmed_regression'] for r in first['checkpoints']]==[False,False,True,False]
            assert first['champion'].endswith('000002.pt')
            assert first['checkpoints'][3]['confirmation'] is None
if __name__=='__main__':test();print('Screen, independent confirmation, champion change, regression, and idempotent resume pass')
