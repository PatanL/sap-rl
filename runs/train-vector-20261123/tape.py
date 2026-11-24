"""Record every state, initial to terminal inclusive, of a fixed number of sampled parent games."""
import json
from pathlib import Path
import numpy as np,torch
from vector_env import *
from actor_wave import WaveActor
from opponents import Opponents
from learner import prepare
from runtime import verify_engine
def main():
    n=64;example=selfplay_env.ParallelSelfPlayEnv().reset(seed=123,paired_tape=True)[0]
    o=allocate(n,example,True);env=PythonVector(n,o,6);seeds=np.arange(n)+834491
    def snapshot(i):
        return {**{'obs_'+k:v[i].copy() for k,v in o['obs'].items()},**{k:v[i].copy() for k,v in o.items() if k!='obs'}}
    try:
        env.reset(np.arange(n),seeds,o);info=verify_engine()
        torch.set_num_threads(1);torch.manual_seed(873)
        parent=Path('/sap/checkpoints/update-025772.pt');policy,*_=prepare(parent,None,'parent',info,873);model=policy.cuda().eval()
        pool=Opponents(model,parent,.5,4,2*n,873,25772,info,None)
        actor=WaveActor(model,pool,n);actor.reset(np.arange(n));frames=[[snapshot(i)] for i in range(n)];dec=np.zeros(n,'i4')
        while not o['terminated'].all():
            alive=np.flatnonzero(~o['terminated']);dec+=o['active'].sum(axis=1).astype('i4')
            actor.act(o);env.step(actor.actions,actor.masks,o)
            for i in alive:frames[i].append(snapshot(i))
        out=Path('results/tape');out.mkdir(exist_ok=True)
        lens=np.array(list(map(len,frames)));np.save(out/'ends',lens.cumsum()-1);np.save(out/'starts',lens.cumsum()-lens)
        for key in frames[0][0]:np.save(out/key,np.stack([row[key] for game in frames for row in game]))
        (out/'receipt.json').write_text(json.dumps(dict(games=n,seat_decisions=dec.tolist(),steps=(lens-1).tolist(),engine=info['payload_sha256'])))
    finally:env.close()
if __name__=='__main__':main()
