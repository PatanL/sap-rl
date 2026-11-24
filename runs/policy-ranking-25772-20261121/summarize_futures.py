"""Score the fixed six-future selection rule on six independent futures."""
import json
from pathlib import Path
import numpy as np


def interval(deltas):
    draws=np.random.default_rng(20261121).choice(deltas,(20000,len(deltas)))
    return dict(mean=deltas.mean(),origin95=np.quantile(draws.mean(1),[.025,.975]).tolist())


def main():
    folder=Path(__file__).resolve().parent/'repeated-futures-v1'
    rows=[json.loads((folder/f'{i:03d}.json').read_text()) for i in range(32)]
    assert len({r['origin'] for r in rows})==32
    rows=[r for r in rows if r['eligible']]
    seeds=[s for r in rows for s in r['seeds']]
    assert len(set(seeds))==len(seeds)==12*len(rows)
    effects=[]
    ridge=[]
    critic=[]
    for r in rows:
        c,s=(r['branches'][k] for k in ('control','sample'))
        effects.append(np.array(s['scores'])-c['scores'])
        ridge.append(s['ridge']>c['ridge'])
        critic.append(s['critic']>c['critic'])
    effects=np.array(effects)
    select=effects[:,:6].mean(axis=1)
    confirm=effects[:,6:].mean(axis=1)
    # Ties always retain the incumbent. All origins are the units of inference.
    picked=select>0
    report=dict(eligible=len(rows),completedContinuations=effects.size*2,
                selected=int(picked.sum()),selectionGain=np.maximum(select,0).mean(),
                independentConfirmation=interval(picked*confirm),
                confirmedPositive=int(np.sum(picked & (confirm>0))),
                confirmedNegative=int(np.sum(picked & (confirm<0))),
                confirmedTied=int(np.sum(picked & (confirm==0))),
                ridgeAllFutures=interval(np.array(ridge)*effects.mean(1)),
                frozenCriticAllFutures=interval(np.array(critic)*effects.mean(1)),
                unconditionalSample=interval(effects.mean(1)),
                splitCorrelation=float(np.corrcoef(select,confirm)[0,1]) if select.std() and confirm.std() else None)
    # Report replication among genuinely distinct choices as well. If two
    # shops executed identical actions, exact-zero margins in both halves can
    # make aggregate agreement look stronger than it is.
    distinct=np.array([r['branches']['control']['trace']!=r['branches']['sample']['trace'] for r in rows])
    report['sameTraceOrigins']=int((~distinct).sum())
    report['sameTraceWorldDisagreements']=int(np.count_nonzero(effects[~distinct]))
    if distinct.any():
        differences=effects[distinct]
        within=differences.var(axis=1,ddof=1).mean()
        observed=differences.mean(axis=1).var(ddof=1) if len(differences)>1 else None
        # Method of moments on the deliberately prospectively sampled roots.
        # A negative estimate is possible on a small sample; retain its sign.
        report['distinctShops']=dict(count=len(differences),
            pairedWorldOutcomeVariance=within,
            estimatedBetweenOriginEffectVariance=observed-within/differences.shape[1] if observed is not None else None,
            selectPositive=int(np.sum(select[distinct]>0)),
            nonzeroInBoth=int(np.sum((select[distinct]!=0)&(confirm[distinct]!=0))),
            sameDirectionNonzero=int(np.sum(select[distinct]*confirm[distinct]>0)),
            oppositeDirection=int(np.sum(select[distinct]*confirm[distinct]<0)))
    (folder/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
