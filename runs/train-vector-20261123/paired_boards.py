"""Independent matched-world check of the greedy ranked-field index, bootstrapped by self-play match."""
from collections import defaultdict
from pathlib import Path
import argparse,hashlib,json,multiprocessing as mp
import numpy as np
import ghost_fights as sim
TURNS=(3,5,7,10,12)
def tables(file):
    m=defaultdict(dict)
    for line in Path(file).open():
        r=json.loads(line);t=r['turn']
        if t in TURNS:m[int(r['match'].split('-')[-1])][t]=[sim.pack(x) if x else None for x in r['teams']]
    return m
def passes(task):
    world,turn,seat,a,b,draws=task
    def one(team):
        if team is None:return 0.,0
        hits=0.
        for other in draws:
            r=sim.score((team,other))['simOutcomes'];hits+=(r['player']+.5*r['draw'])/len(sim.SEEDS)
        return hits/len(draws),1
    return world,turn,*one(a),*one(b)
def main():
    p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');p.add_argument('--out',default='results/paired-board-check.json');args=p.parse_args()
    import fcntl
    reservation=open('/job/tmp/gpu.lock','a');fcntl.flock(reservation,fcntl.LOCK_EX)
    a=tables(args.a);b=tables(args.b);assert a.keys()==b.keys() and len(a)==1000
    base=Path('/sap/runs/replay-oracle-20261122');names={}
    for d in ('raw','raw-fresh','raw-fresh2','raw-fresh3','raw-players'):
        for f in (base/d).glob('*.json'):
            for e in json.loads(f.read_text())['perspectives']['perspectives']:
                r=e['raw'];names[r['ParticipationId']]=r.get('UserName','') or ''
    field=defaultdict(list);seen=set()
    for name in json.loads(Path('results/fixtures.json').read_text()):
        for r in json.loads((base/name).read_text())['fixtures']:
            t=r['turn'];key=(r['participationId'],t)
            if t not in TURNS or key in seen or not r['player'] or int(hashlib.sha256(r['matchId'].encode()).hexdigest(),16)%5>=2:continue
            seen.add(key)
            if 'maxwu' not in names[key[0]].lower():field[t].append(sim.pack(r['player']))
    sim.SEEDS=[(1234567+i*2654435761)%2**32 for i in range(8)]
    rng=np.random.default_rng(826767);jobs=[];worlds=sorted(a)
    for i,w in enumerate(worlds):
        for j,t in enumerate(TURNS):
            for s in (0,1):
                l=a[w].get(t,[None,None])[s];r=b[w].get(t,[None,None])[s]
                if l is None and r is None:continue
                draws=[field[t][ix] for ix in rng.integers(len(field[t]),size=16)]
                jobs.append((i,j,s,l,r,draws))
    data=np.zeros((1000,5,4));ctx=mp.get_context('fork')
    with ctx.Pool(5,initializer=sim.init) as pool:
        for i,j,sa,na,sb,nb in pool.imap_unordered(passes,jobs,chunksize=32):data[i,j]+=[sa,na,sb,nb]
    def means(d):
        totals=d.sum(axis=-3);return np.stack([(totals[...,0]/totals[...,1]).mean(axis=-1),(totals[...,2]/totals[...,3]).mean(axis=-1)],axis=-1)
    lo,hi=means(data);deltas=[]
    for _ in range(100):
        vals=means(data[rng.integers(1000,size=(100,1000))]);deltas.extend(vals[:,1]-vals[:,0])
    out=dict(parent=lo,candidate=hi,delta=hi-lo,ci95=np.quantile(deltas,[.025,.975]).tolist(),worlds=len(worlds),draws_per_board=16,seeds_per_pair=8,pool={k:len(v) for k,v in field.items()})
    Path(args.out).write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
