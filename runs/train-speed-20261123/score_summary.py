"""Common-world delta vs parent, distinct from the direct fresh-world old/new duel."""
from pathlib import Path
import json,random
P=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text())
def pairs(r):
    scores={}
    for g in r['records']:scores.setdefault(g['seed'],{})[g['seat']]=g['score']
    assert all(set(s)=={0,1} for s in scores.values())
    return {k:(v[0]+v[1])/2 for k,v in scores.items()}
def interval(x):
    rng=random.Random(90154);boot=sorted(sum(rng.choices(x,k=len(x)))/len(x) for _ in range(20000));return [boot[500],boot[19499]]
if __name__=='__main__':
    old=read(P/read(P/'results/incumbent.json')['eval']);new=read(P/'results/proof-vs-parent.json');head=read(P/'results/proof-vs-old.json')
    x=pairs(new);y=pairs(old)
    assert x.keys()==y.keys() and not x.keys()&pairs(head).keys()
    ds=[x[k]-y[k] for k in sorted(x)]
    out=dict(parent_screen=new['score'],parent_ci95=new['ci95'],old_score=old['score'],old_ci95=old['ci95'],
             paired_delta=sum(ds)/len(ds),paired_delta_ci95=interval(ds),direct=head['score'],direct_ci95=head['ci95'],
             proof_update=new['candidate']['update'],old_update=old['candidate']['update'])
    (P/'results/strength.json').write_text(json.dumps(out,indent=2));print(out)
