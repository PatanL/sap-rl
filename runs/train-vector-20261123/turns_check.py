"""Independent transition reconstruction on heterogeneous real turns, including frozen and ended seats."""
from types import SimpleNamespace
from vector_env import *
from trajectories import Turns
from opponents import LIVE
import torch,json
def main():
    n=8;rng=np.random.default_rng(4834);width=3
    example=selfplay_env.ParallelSelfPlayEnv().reset(seed=45,paired_tape=True)[0]
    o=allocate(n,example,True);env=PythonVector(n,o,3)
    from sapai_gym1.rl.action_schema import FlatActionCodec
    refs=[selfplay_env.ParallelSelfPlayEnv() for _ in range(n)];seeds=np.arange(n)+5961
    real=[e.reset(seed=int(s),paired_tape=True) for e,s in zip(refs,seeds)]
    pool=SimpleNamespace(owners=np.where(np.arange(2*n)%4==3,-2,LIVE))
    seg=Turns(o,n,width);manual=[[] for _ in range(n*2)];initial=np.zeros((2*n,width),'f4');mem=initial.copy()
    actor=SimpleNamespace(actions=np.zeros((n,2,3),'i4'),masks=np.zeros((n,2),'i4'),live_log=np.zeros(2*n,'f4'))
    count=steps=0
    try:
        env.reset(np.arange(n),seeds,o)
        for tick in range(150):
            active=o['active'].copy();allids=np.flatnonzero(active.ravel());live=allids[pool.owners[allids]==LIVE]
            for i,e in enumerate(refs):
                if o['terminated'][i]:continue
                actions=[]
                for side,yes in enumerate(active[i]):
                    if not yes:actions.append(None);continue
                    a=FlatActionCodec.decode(int(rng.choice(tuple(e.core._avail_actions(side)))))
                    actor.actions[i,side]=a.as_tuple();actions.append(a)
                real[i]=step_with_end_freeze(e,*actions,mask_ids=[0,0]).result.observations
            actor.live_log[allids]=rng.normal(size=len(allids));version=77+tick//60
            new=live[seg.lengths[live]==0];seg.new_initial=mem[new].copy();initial[new]=mem[new];mem[allids]=rng.normal(size=(len(allids),width))
            for s in live:
                pair=divmod(s,2)
                manual[s].append(({k:v[pair].copy() for k,v in o['obs'].items()},actor.actions[pair].copy(),actor.live_log[s],version))
            seg.push(o,actor,live,version);env.step(actor.actions,actor.masks,o)
            for t in seg.finish(o,pool):
                s=t.slot_id;pair=divmod(s,2);truth=manual[s];assert len(truth)==t.length
                for k,v in t.observations.items():
                    np.testing.assert_array_equal(v.numpy(),np.stack([r[0][k] for r in truth]))
                    np.testing.assert_array_equal(t.bootstrap_observation[k].numpy(),real[pair[0]][pair[1]][k])
                np.testing.assert_array_equal(t.initial_memory.numpy(),initial[s]);np.testing.assert_array_equal(t.actions.numpy(),np.stack([r[1] for r in truth]))
                np.testing.assert_array_equal(t.behavior_log_probs.numpy(),[r[2] for r in truth]);np.testing.assert_array_equal(t.behavior_versions.numpy(),[r[3] for r in truth])
                assert t.terminal==o['terminated'][pair[0]] and t.rewards[-1]==o['rewards'][pair] and not t.rewards[:-1].any()
                assert t.discounts[-1]==(0 if t.terminal else .99) and (t.discounts[:-1]==1).all()
                outcome=o['battle_result'][pair[0]];assert t.battle_outcome==(2 if outcome==2 else int(outcome!=pair[1]))
                steps+=t.length;count+=1;truth.clear()
        print(json.dumps(dict(turns=count,seat_steps=steps,frozen_excluded=True)))
    finally:env.close()
if __name__=='__main__':main()
