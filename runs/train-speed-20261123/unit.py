"""CPU-only tests for version gating, virtual reductions, and completed-turn transport."""
from types import SimpleNamespace
from dataclasses import replace
import torch
from collector_async import Queue,MeasuredCollector,Meter
from worker_profile import actors
from fixture import PARENT,get_fixture
from equivalence import execute,rel
from learner import prepare
from runtime import verify_engine
from opponents import Opponents

def queue_test():
    def s(*versions):return SimpleNamespace(length=len(versions),behavior_versions=torch.tensor(versions))
    q=Queue();a,b,c=s(2,3),s(3,4),s(4,4)
    q.add([a,b]);assert q.pop(4,3,1) is None and q.stale==2 and q.rows==2
    q.add([c]);batch,lo,hi=q.pop(4,3,1);assert batch==[b,c] and (lo,hi)==(3,4) and not q.rows
    q.add([s(9)])
    try:q.pop(6,1,1)
    except ValueError:pass
    else:raise AssertionError('Accepted a future policy version')

def reductions():
    seq=get_fixture(256)
    for j,s in enumerate(seq):
        boot={k:v.clone() for k,v in s.bootstrap_observation.items()}
        if j%3==0:boot['opponent_history_board_mask'][-1]=False
        if j%5==0:boot['opponent_history_mask'][-1]=False
        seq[j]=replace(s,bootstrap_observation=boot,battle_outcome=None if j%4==0 else s.battle_outcome)
    info=verify_engine();lhs,g1,d1=execute(False,'cpu',info,seq,'fp32',384);rhs,g2,d2=execute(True,'cpu',info,seq,'fp32',384)
    assert abs(lhs['loss']-rhs['loss'])<2e-5 and rel(g1,g2)['relative_l2']<3e-4 and rel(d1,d2)['relative_l2']<1e-3
def transport():
    info=verify_engine();model,*_=prepare(PARENT,None,'parent',info,991);del _;model.eval().requires_grad_(False)
    v=25772
    with actors(2,3,model.config.memory_dim,991,v,True) as (pages,pipes):
        pool=Opponents(model,PARENT,.5,4,12,991,v,info,None)
        c=MeasuredCollector(pages,pipes,model,model.config.memory_dim,'cpu',pool,meter=Meter(),profile=True)
        for g in c.groups:assert not c.receive(g);c.dispatch(g,v)
        for k in range(80):
            for g in c.groups:
                segments=c.receive(g)
                assert all((s.behavior_versions<=v).all() for s in segments)
                c.dispatch(g,v)
            if k==3:v+=1
        for g in c.groups:c.receive(g)
        assert c.live_steps+c.frozen_steps==c.seat_decisions
        assert c.live_steps==c.delivered+sum(c.pending.values()) and c.delivered>100 and c.frozen_steps>0
if __name__=='__main__':
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    queue_test();reductions();transport();print('PASS version gating, heterogeneous auxiliary denominators, frozen-only exclusion and transport conservation')
