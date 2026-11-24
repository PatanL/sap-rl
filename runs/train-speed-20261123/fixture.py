"""Produce a real parent-policy CPU fixture for correctness tests; no learner or benchmark."""
from dataclasses import fields
from pathlib import Path
import json,os
import numpy as np
import torch
from runtime import verify_engine
from learner import prepare
from sapai_gym1.rl.dense_trajectory import DenseTurnSegment
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.end_freeze import step_with_end_freeze
from sapai_gym1.rl.action_schema import AutoregressiveAction
PARENT=Path('/sap/checkpoints/update-025772.pt')
def get_fixture(n=None):
    raw=torch.load(os.environ.get('SPEED_BATCH','results/batch.pt'),map_location='cpu',weights_only=True)['turns']
    seq=[]
    for v in raw:
        seq.append(DenseTurnSegment(**v))
        if n is not None and sum(s.length for s in seq)>=n:break
    return seq
@torch.inference_mode()
def main():
    info=verify_engine();torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    model,*_=prepare(PARENT,None,'parent',info,207);model.eval().requires_grad_(False);del _
    rng=np.random.default_rng(16914);torch.manual_seed(30422)
    def fresh():
        env=ParallelSelfPlayEnv();obs=env.reset(seed=int(rng.integers(2**60)),paired_tape=True)
        h=model.initial_memory(2,'cpu')
        return dict(env=env,obs=obs,h=h,initial=h.clone(),turns=[[],[]])
    g=[fresh() for _ in range(24)];rows=[];n=0;terminal=0
    while n<8192 or not terminal:
        req=[(i,s) for i,d in enumerate(g) for s,active in enumerate(d['env'].active_players()) if active]
        inp={k:torch.from_numpy(np.stack([g[i]['obs'][s][k] for i,s in req])) for k in g[0]['obs'][0]}
        out=model.act(inp,memory=torch.stack([g[i]['h'][s] for i,s in req]))
        moves=[[None,None] for _ in g];masks=[[0,0] for _ in g]
        for j,(i,s) in enumerate(req):
            a=out.actions[j].tolist();m=int(out.end_freeze_mask[j]);moves[i][s]=AutoregressiveAction(*a);masks[i][s]=m
            g[i]['turns'][s].append((g[i]['obs'][s],a,m,float(out.log_prob[j])));g[i]['h'][s]=out.memory[j]
        for i,d in enumerate(g):
            ret=step_with_end_freeze(d['env'],*moves[i],mask_ids=masks[i]).result;d['obs']=ret.observations
            if not ret.battle_resolved:continue
            for s,t in enumerate(d['turns']):
                l=len(t);reward=torch.zeros(l);reward[-1]=ret.rewards[s];disc=torch.ones(l);disc[-1]=0 if ret.terminated else .99
                rows.append(dict(observations={k:torch.from_numpy(np.stack([x[0][k] for x in t])) for k in t[0][0]},
                    actions=torch.tensor([x[1] for x in t]),freeze_masks=torch.tensor([x[2] for x in t],dtype=torch.int16),
                    behavior_log_probs=torch.tensor([x[3] for x in t]),behavior_versions=torch.full((l,),25772),
                    initial_memory=d['initial'][s].clone(),bootstrap_observation={k:torch.tensor(v) for k,v in ret.observations[s].items()},
                    rewards=reward,discounts=disc,terminal=ret.terminated,battle_outcome=2 if ret.battle_result==2 else int(ret.battle_result!=s),slot_id=2*i+s))
                n+=l;t.clear()
            d['initial']=d['h'].clone()
            if ret.terminated:terminal+=1;g[i]=fresh()
    torch.save(dict(turns=rows,decisions=n,terminals=terminal,engine=info['payload_sha256']), 'results/batch.pt')
    print(json.dumps(dict(decisions=n,turns=len(rows),terminals=terminal)))
if __name__=='__main__':main()
