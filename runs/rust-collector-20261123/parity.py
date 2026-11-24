"""Lockstep CPU acceptance of the real trajectory boundary, including the FIFO selection."""
import argparse
import collections
from dataclasses import fields
from pathlib import Path
import json

import numpy as np
import torch
from runtime import verify_engine, selfplay_env
from evaluate import load
from opponents import Opponents, LIVE
from vector_env import PythonVector, allocate
from trajectories import Turns
from collector_g import Queue
from rust_adapter import RustVector

PARENT=Path('/sap/checkpoints/update-025772.pt')
POLICY=Path('/sap/runs/train-live-20261123/g-hist-frozen50/update-026022.pt')

def same(a,b,key):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),key
        for k in a:same(a[k],b[k],key+'.'+k)
        return
    if isinstance(a,torch.Tensor):a=a.numpy();b=b.numpy()
    x,y=np.asarray(a),np.asarray(b)
    if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():
        pos=np.flatnonzero(x.ravel()!=y.ravel())
        i=pos[0] if len(pos) else 0
        raise AssertionError((key,x.dtype,y.dtype,x.shape,y.shape,i,x.ravel()[i],y.ravel()[i],x.tobytes()[:8],y.tobytes()[:8]))

class CPUActor:
    def __init__(self,model,n,info):
        self.model=model
        self.pool=Opponents(model,PARENT,.5,4,2*n,7501,26022,info,None)
        self.memory=model.initial_memory(2*n,'cpu')
        self.actions=np.zeros((n,2,3),'i4');self.masks=np.zeros((n,2),'i4');self.live_log=np.zeros(2*n,'f4')
    def reset(self,indices):
        self.pool.reset_games(indices);self.memory.view(len(self.actions),2,-1)[indices]=0
    def act(self,o):
        active=np.flatnonzero(o['active'].ravel());owner=self.pool.owners[active]
        self.actions.fill(0);self.masks.fill(0)
        for key in np.unique(owner):
            slots=active[owner==key];idx=torch.from_numpy(slots)
            ob={k:torch.from_numpy(v.reshape(-1,*v.shape[2:])[slots]) for k,v in o['obs'].items()}
            policy=self.model if key==LIVE else self.pool.models[key]
            moves=policy.act(ob,memory=self.memory.index_select(0,idx))
            self.memory.index_copy_(0,idx,moves.memory.float())
            self.actions.reshape(-1,3)[slots]=moves.actions.numpy()
            self.masks.ravel()[slots]=moves.end_freeze_mask.numpy()
            self.live_log[slots]=moves.log_prob.float().numpy()
        return active,active[owner==LIVE]

def segments(a,b):
    assert len(a)==len(b)
    for left,right in zip(a,b):
        for field in fields(left):same(getattr(left,field.name),getattr(right,field.name),'turn.'+field.name)

@torch.inference_mode()
def main(n,goal,production=False):
    torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    info=verify_engine()
    example=selfplay_env.ParallelSelfPlayEnv().reset(seed=42,paired_tape=True)[0]
    p=allocate(n,example,True);py=PythonVector(n,p,2);rs=RustVector(n,2)
    try:
        outs=[p,rs.output];envs=[py,rs]
        model,desc=load(POLICY,'cpu')
        actors=[CPUActor(model,n,info) for _ in outs];rings=[Turns(o,n,model.config.memory_dim) for o in outs]
        queues=[Queue(),Queue()];rng=np.random.default_rng(613);ids=np.arange(n)
        def reset(slots):
            seeds=rng.integers(0,2**62,size=len(slots))
            for env,o,actor in zip(envs,outs,actors):env.reset(slots,seeds,o);actor.reset(slots)
        reset(ids);torch.manual_seed(250536)
        tally=collections.Counter();owners=collections.Counter()
        version=26022; publish_at=None
        while tally['matches']<goal:
            same(*outs,'pre')
            if (production and tally['waves']==publish_at) or (not production and tally['waves'] and tally['waves']%13==0):
                version+=1;publish_at=None
                if (version-26022)%(2 if production else 5)==0:
                    for actor in actors:actor.pool.snapshot(version,None)
            gen=torch.get_rng_state()
            batches=[]
            for actor,o,ring,env in zip(actors,outs,rings,envs):
                live=np.flatnonzero(o['active'].ravel() & (actor.pool.owners==LIVE))
                start=live[ring.lengths[live]==0]
                ring.new_initial=actor.memory[start].numpy().copy()
                torch.set_rng_state(gen);active,live=actor.act(o)
                ring.push(o,actor,live,version)
                env.step(actor.actions,actor.masks,o)
                batches.append(ring.finish(o,actor.pool))
                if actor is actors[0]:end_rng=torch.get_rng_state()
                else:same(end_rng,torch.get_rng_state(),'sampler')
            same(*outs,'post')
            for key in ('actions','masks','live_log','memory'):same(getattr(actors[0],key),getattr(actors[1],key),key)
            same(actors[0].pool.owners,actors[1].pool.owners,'episode_owners')
            segments(*batches)
            for q,turns in zip(queues,batches):q.add(turns)
            selected=[q.pop(version,8192 if production else 1024,1) for q in queues] if publish_at is None else [None,None]
            assert (selected[0] is None)==(selected[1] is None)
            if selected[0]:
                if production:publish_at=tally['waves']+3
                l,r=selected
                assert l[1:]==r[1:];segments(l[0],r[0])
                tally['selected_turns']+=len(l[0]);tally['learner_decisions']+=sum(t.length for t in l[0])
                tally['selected_batches']+=1
            own=actors[0].pool.owners[active]
            owners.update(dict(zip(*[arr.tolist() for arr in np.unique(own,return_counts=True)])))
            tally.update(waves=1,seat_decisions=len(active),live_decisions=len(live),frozen_decisions=len(active)-len(live),
                         delivered_turns=len(batches[0]),delivered=sum(t.length for t in batches[0]),battles=int(p['battle_resolved'].sum()))
            finish=np.flatnonzero(p['terminated']);tally['matches']+=len(finish)
            if len(finish):reset(finish)
        same(*outs,'after_reset');assert queues[0].stale==queues[1].stale and queues[0].rows==queues[1].rows
        tally.update(stale=queues[0].stale,queued=queues[0].rows,pending=int(rings[0].lengths.sum()))
        assert tally['delivered']==tally['learner_decisions']+tally['stale']+tally['queued']
        assert tally['live_decisions']==tally['delivered']+tally['pending']
        out=dict(counts=dict(tally),assignment_decisions=dict(owners),policy=desc,observation_fields=list(p['obs']),
                 segment_fields=[k.name for k in fields(batches[0][0])] if batches[0] else list(__import__('sapai_gym1.rl.dense_trajectory',fromlist=['DenseTurnSegment']).DenseTurnSegment.__dataclass_fields__),
                 exact=True,decision_target=8192 if production else 1024,final_version=version,engine=info['payload_sha256'])
        Path('results/parity-production.json' if production else f'results/parity-{goal}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
    finally:py.close();rs.close()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--games',type=int,default=48);a.add_argument('--completed',type=int,default=256)
    a.add_argument('--production',action='store_true');args=a.parse_args();main(args.games,args.completed,args.production)
