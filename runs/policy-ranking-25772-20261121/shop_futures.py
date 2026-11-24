"""Test repeatability of terminal preferences between two completed shops.

The fork occurs after the decisions for the target shop have been selected and
before its battle. Future seeds are paired between branches. The simulator
retains the true unseen opponent for labeling; policies see public inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from plan_common import install, load_actor, seed_of


def public_digest(obs):
    digest = hashlib.sha256()
    for key, value in sorted(obs.items()):
        digest.update(key.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def decisions(games, model, *, propose):
    from sapai_gym1.rl.joint_action import JointAction
    from sapai_gym1.rl.action_schema import AutoregressiveAction
    slots, obs, memories = [], [], []
    for i,g in enumerate(games):
        both=g['env'].observations()
        for seat,active in enumerate(g['env'].active_players()):
            if active:
                slots.append((i,seat))
                obs.append(both[seat])
                memories.append(g['memory'][seat])
    inputs={k:torch.as_tensor(np.stack([o[k] for o in obs])) for k in obs[0]}
    f=model.encode_decision(inputs,torch.stack(memories),forecast=False)
    greedy=model.sample_joint_action(f,deterministic=True)
    sample=model.sample_joint_action(f,deterministic=False) if propose else None
    pairs=[[None,None] for _ in games]
    for j,(i,seat) in enumerate(slots):
        g=games[i]
        target=propose and g['env'].core.turn==g['turn'] and seat==g['seat']
        response=sample if target and g['branch']=='sample' else greedy
        a=JointAction(AutoregressiveAction(*response.actions[j].tolist()),response.end_freeze_mask[j].item())
        g['memory'][seat]=response.memory[j]
        pairs[i][seat]=a
        if target:
            if not g['trace']:
                g['publicRoot']=public_digest(obs[j])
            g['trace'].append(a.to_dict())
            if a.action_type==0:
                bits=[float((a.end_freeze_mask>>k)&1) for k in range(10)]
                g['feature']=np.concatenate([f.context[j].numpy(),bits])
                g['critic']=model.value_from_encoded(f)[j].item()
    return pairs


def one_origin(index, actor, ridge, futures):
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    from sapai_gym1.rl.joint_action import step_decisions
    from sapai_gym1.rl.distributed import _completed_match_result
    namespace='shop-future-pilot-v1'
    seed=seed_of(namespace,'origin',index)
    torch.manual_seed(seed_of(namespace,'proposal',index))
    turn=1+index%8
    seat=(index%2)^((index//8)%2)
    pending=[]
    forks={}
    for branch in ('control','sample'):
        env=ParallelSelfPlayEnv()
        env.reset(seed=seed,paired_tape=True)
        pending.append(dict(env=env,branch=branch,turn=turn,seat=seat,trace=[],
                            memory=actor.model.initial_memory(2,'cpu')))
    for _ in range(2500):
        if not pending:
            break
        pairs=decisions(pending,actor.model,propose=True)
        next_pending=[]
        for g,pair in zip(pending,pairs):
            if g['env'].core.turn==turn and all(a is None or a.action_type==0 for a in pair):
                forks[g['branch']]=(g,pair)
            elif not step_decisions(g['env'],pair).result.terminated:
                next_pending.append(g)
        pending=next_pending
    assert not pending
    if not forks:
        return dict(origin=seed,index=index,turn=turn,eligible=False)
    assert len(forks)==2 and forks['control'][0]['publicRoot']==forks['sample'][0]['publicRoot']
    # The held-out, already fitted evaluators are scored before any outcomes.
    record=dict(origin=seed,index=index,turn=turn,seat=seat,eligible=True,branches={})
    bank=[seed_of(namespace,'continuation',index,i) for i in range(futures)]
    assert len(set(bank))==futures
    record['seeds']=bank
    live=[]
    def advance(g,pair):
        if step_decisions(g['env'],pair).result.terminated:
            winner=_completed_match_result(g['env'].core)
            score=.5 if winner==2 else float(winner==seat)
            outcomes=record['branches'][g['branch']]['scores']
            assert outcomes[g['i']] is None
            outcomes[g['i']]=score
            return False
        return True
    for branch,(g,pair) in forks.items():
        x=g['feature']
        pred=((x-ridge['mean'])/ridge['scale'])@ridge['coef']+ridge['intercept']
        record['branches'][branch]=dict(trace=g['trace'],critic=g['critic'],ridge=float(pred),
                                       publicRoot=g['publicRoot'],scores=[None]*futures)
        original=tuple(public_digest(o) for o in g['env'].observations())
        for i,world in enumerate(bank):
            env=g['env'].clone()
            env.redeterminize(world)
            assert original==tuple(public_digest(o) for o in env.observations())
            child=dict(env=env,branch=branch,i=i,memory=g['memory'].clone())
            if advance(child,pair):
                live.append(child)
    for _ in range(2500):
        if not live:
            break
        pairs=decisions(live,actor.model,propose=False)
        live=[g for g,pair in zip(live,pairs) if advance(g,pair)]
    assert not live and all(None not in b['scores'] for b in record['branches'].values())
    return record


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start',type=int,required=True)
    ap.add_argument('--count',type=int,required=True)
    args=ap.parse_args()
    install(cuda=False)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_num_threads(2)
    actor=load_actor(device='cpu')
    root=Path(__file__).resolve().parent
    out=root/'repeated-futures-v1'
    out.mkdir(exist_ok=True)
    with np.load(root/'public-value-ridge.npz') as r:
        ridge=dict(r)
    for index in range(args.start,args.start+args.count):
        path=out/f'{index:03d}.json'
        if path.exists():
            raise FileExistsError(path)
        row=one_origin(index,actor,ridge,12)
        row['checkpoint']=actor.owner
        temp=path.with_suffix('.partial')
        temp.write_text(json.dumps(row))
        temp.replace(path)
        print('completed',index,'eligible',row['eligible'],flush=True)


if __name__=='__main__':
    main()
