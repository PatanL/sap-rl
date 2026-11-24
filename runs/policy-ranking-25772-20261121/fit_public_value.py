"""Single prespecified diagnostic: fit on blocks 0-7, evaluate on 8-15.

The candidate END observations come from played branches. Comparing them
retrospectively measures evaluator quality; deployment would need separately
simulated candidate shops with no access to the actual future tape.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


def load_blocks(directory, indices):
    blocks = []
    for i in indices:
        with np.load(directory / f'block-{i:04d}.npz') as data:
            assert data['checkpoint'].item() == '0669109e9b6ee2d4f0309a3fec83039fc8a7d74f74e3c9f629d2f7026e515d49'
            blocks.append({k:data[k] for k in data.files if k!='checkpoint'})
    return {k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def fit(train):
    # Each match origin receives equal weight, including its two branches.
    keep=train['turn']<=8
    origins=train['origin'][keep]
    _, idx, counts = np.unique(origins,return_inverse=True,return_counts=True)
    weights=1/counts[idx]
    x=train['features'][keep].astype(np.float64)
    mean=np.average(x,axis=0,weights=weights)
    scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),0.1)
    weights *= len(weights)/weights.sum()
    model=Ridge(alpha=len(weights)).fit((x-mean)/scale,train['outcome'][keep],sample_weight=weights)
    return mean,scale,model


def evaluate(test, scores):
    effects=[]
    changed=0
    positive=negative=0
    for origin in np.unique(test['origin']):
        ids=np.flatnonzero((test['origin']==origin)&(test['turn']==test['target']))
        if not len(ids):
            continue
        assert len(ids)==2 and test['sample'][ids].tolist()==[False,True]
        first,second=ids
        change=scores[second]>scores[first]
        changed+=int(change)
        diff=float(test['outcome'][second]-test['outcome'][first])
        positive+=int(diff>0)
        negative+=int(diff<0)
        effects.append(diff if change else 0.)
    effect=np.array(effects)
    rng=np.random.default_rng(20261121)
    interval=np.quantile(rng.choice(effect,(10000,len(effect))).mean(axis=1),[.025,.975])
    return effect,dict(eligible=len(effect), pickedAlternative=changed, alternativeWins=positive,
                alternativeLosses=negative, meanScoreDelta=effect.mean(),
                pairedOrigin95=interval.tolist())


def main():
    root=Path(__file__).resolve().parent
    train=load_blocks(root/'public-data',range(8))
    test=load_blocks(root/'public-data',range(8,16))
    assert set(train['origin']).isdisjoint(test['origin'])
    mean,scale,model=fit(train)
    scores=model.predict((test['features']-mean)/scale)
    ridge_effect,ridge_report=evaluate(test,scores)
    critic_effect,critic_report=evaluate(test,test['critic'])
    paired=ridge_effect-critic_effect
    diff_interval=np.quantile(np.random.default_rng(20261121).choice(paired,(10000,len(paired))).mean(axis=1),[.025,.975])
    report=dict(trainOrigins=len(np.unique(train['origin'])),testOrigins=len(np.unique(test['origin'])),
                trainEndDecisions=len(train['turn']),testEndDecisions=len(test['turn']),
                ridge=ridge_report,frozenCritic=critic_report,
                ridgeMinusFrozen=dict(meanScoreDelta=paired.mean(),pairedOrigin95=diff_interval.tolist()))
    np.savez(root/'public-value-ridge.npz',mean=mean,scale=scale,coef=model.coef_,intercept=model.intercept_)
    (root/'public-value-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
