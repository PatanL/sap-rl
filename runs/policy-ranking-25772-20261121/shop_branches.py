"""Estimate outcome variation from one sampled shop turn on fresh paired worlds.

The policy sees public observations only. Plans and scores here are development
diagnostics; a single continuation per branch cannot label a best plan.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from plan_common import install, load_actor, seed_of


@torch.inference_mode()
def run(dataset=None, blocks=2, start=0):
    install(cuda=False)
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    from sapai_gym1.rl.joint_action import JointAction, step_decisions
    from sapai_gym1.rl.action_schema import AutoregressiveAction
    from sapai_gym1.rl.distributed import _completed_match_result
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    actor = load_actor(device='cpu')
    model = actor.model
    rows = []
    if dataset:
        dataset.mkdir(parents=True, exist_ok=True)
    for block in range(start,start+blocks):
        output = dataset / f'block-{block:04d}.npz' if dataset else None
        namespace = 'shop-value-public-v1' if dataset else 'shop-branch-development-25772-20261121'
        if output and output.exists():
            raise FileExistsError(output)
        torch.manual_seed(seed_of(namespace, 'policy', block) if dataset else seed_of('shop-branch-policy-20261121', block))
        games = []
        for offset in range(8):
            index = block * 8 + offset
            seed = seed_of(namespace, index)
            for branch in (('control','sample') if dataset else ('control','duplicate','sample')):
                env = ParallelSelfPlayEnv()
                env.reset(seed=seed, paired_tape=True)
                games.append(dict(env=env, seed=seed, branch=branch, side=(index%2 ^ (block%2)) if dataset else index%2,
                                  turn=1+offset, memory=model.initial_memory(2,'cpu'),
                                  trace=[], ends=[], root=None, outcome=None))
        for tick in range(2500):
            slots, observations, memories = [], [], []
            for i, game in enumerate(games):
                if game['outcome'] is not None:
                    continue
                env = game['env']
                obs = env.observations()
                for seat, active in enumerate(env.active_players()):
                    if active:
                        slots.append((i,seat))
                        observations.append(obs[seat])
                        memories.append(game['memory'][seat])
            if not slots:
                break
            batch = {k:torch.as_tensor(np.stack([o[k] for o in observations])) for k in observations[0]}
            f = model.encode_decision(batch,torch.stack(memories),forecast=False)
            greedy = model.sample_joint_action(f,deterministic=True)
            sampled = model.sample_joint_action(f,deterministic=False)
            actions = [[None,None] for g in games]
            for j,(i,seat) in enumerate(slots):
                g=games[i]
                target=g['env'].core.turn==g['turn'] and seat==g['side']
                response=sampled if target and g['branch']=='sample' else greedy
                a=JointAction(AutoregressiveAction(*response.actions[j].tolist()),response.end_freeze_mask[j].item())
                actions[i][seat]=a
                if target:
                    if dataset and not g['trace']:
                        g['root'] = {k:v.copy() for k,v in observations[j].items()}
                    g['trace'].append(a.to_dict())
                if dataset and seat==g['side'] and a.action_type==0:
                    mask = a.end_freeze_mask
                    bits = [float((mask >> k) & 1) for k in range(10)]
                    feature = np.concatenate([f.context[j].numpy(),bits]).astype(np.float32)
                    g['ends'].append((g['env'].core.turn,feature,model.value_from_encoded(f)[j].item()))
                g['memory'][seat]=response.memory[j]
            for g,pair in zip(games,actions):
                if g['outcome'] is not None:
                    continue
                env=g['env']
                if step_decisions(env,pair).result.terminated:
                    result=_completed_match_result(env.core)
                    g['outcome']=0.5 if result==2 else float(result==g['side'])
        assert all(g['outcome'] is not None for g in games)
        if dataset:
            records = []
            for control,sample in zip(games[::2],games[1::2]):
                assert (control['root'] is None)==(sample['root'] is None)
                for k,v in (control['root'] or {}).items():
                    assert np.array_equal(v,sample['root'][k]), (control['seed'],k)
                for g in (control,sample):
                    assert sum(t==g['turn'] for t,_,_ in g['ends'])==int(g['root'] is not None)
                    for t,feature,critic in g['ends']:
                        records.append((g['seed'], g['branch']=='sample', t, g['turn'], feature, critic,g['outcome']))
            fields = ['origin','sample','turn','target','features','critic','outcome']
            temp = output.with_suffix('.partial.npz')
            np.savez_compressed(temp, checkpoint=actor.owner, **dict(zip(fields,map(np.array,zip(*records)))))
            temp.replace(output)
            print(output.name,'games',len(games),'public END decisions',len(records),flush=True)
            continue
        for i in range(0,len(games),3):
            control,duplicate,sample=games[i:i+3]
            assert control['outcome']==duplicate['outcome'] and control['trace']==duplicate['trace']
            rows.append(dict(seed=control['seed'],side=control['side'],turn=control['turn'],
                             baseline=control['outcome'],sample=sample['outcome'],
                             baselineShop=control['trace'],sampleShop=sample['trace']))
        print('completed origins',len(rows),flush=True)
    if dataset:
        return
    effects=[r['sample']-r['baseline'] for r in rows]
    summary=dict(checkpoint=actor.owner, origins=len(rows), completedGames=len(rows)*3,
                 changedShop=sum(r['baselineShop']!=r['sampleShop'] for r in rows),
                 positive=sum(x>0 for x in effects),negative=sum(x<0 for x in effects),
                 meanEffect=float(np.mean(effects)))
    Path('/out/shop-results.json').write_text(json.dumps(dict(summary=summary,origins=rows),indent=2))
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--blocks',type=int,default=2)
    parser.add_argument('--start',type=int,default=0)
    args=parser.parse_args()
    run(args.dataset,args.blocks,args.start)
