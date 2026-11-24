"""Capture ordered public action inputs, battle seeds and native results for paired backend comparisons."""
from pathlib import Path
import json
import numpy as np
import torch
from fixture import PARENT
from learner import prepare
from runtime import verify_engine,selfplay_env,Battle
from sapai_gym1.rl.end_freeze import step_with_end_freeze
from sapai_gym1.rl.action_schema import AutoregressiveAction
@torch.inference_mode()
def main():
    info=verify_engine();torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.manual_seed(712)
    policy,*_=prepare(PARENT,None,'parent',info,712);del _;policy.eval().requires_grad_(False)
    m=64;games=[selfplay_env.ParallelSelfPlayEnv() for _ in range(m)]
    seeds=[(3<<51)+i for i in range(m)];obs=[e.reset(seed=s,paired_tape=True) for e,s in zip(games,seeds)]
    memory=policy.initial_memory(2*m,'cpu');tapes=[[] for _ in games];done=set();battle=[]
    class Log(Battle):
        def __init__(self,*a,seed,**kw):self.tape_seed=seed;super().__init__(*a,seed=seed,**kw)
        def battle(self):
            v=super().battle();battle.append([self.tape_seed,v]);return v
    selfplay_env.Battle=Log
    for tick in range(2500):
        if len(done)==m:break
        req=[(i,s) for i,e in enumerate(games) if i not in done for s,v in enumerate(e.active_players()) if v]
        ix=[2*i+s for i,s in req]
        inp={k:torch.tensor(np.stack([obs[i][s][k] for i,s in req])) for k in obs[0][0]}
        out=policy.act(inp,memory=memory[ix]);memory[ix]=out.memory
        pairs=[[None,None] for _ in games]
        for j,(i,s) in enumerate(req):pairs[i][s]=[*out.actions[j].tolist(),int(out.end_freeze_mask[j])]
        for i,e in enumerate(games):
            if i in done:continue
            aa=pairs[i];args=[None if x is None else AutoregressiveAction(*x[:3]) for x in aa];masks=[0 if x is None else x[3] for x in aa]
            r=step_with_end_freeze(e,*args,mask_ids=masks).result;obs[i]=r.observations
            tapes[i].append([aa,battle.pop() if r.battle_resolved else None,r.terminated,list(r.rewards)])
            if r.terminated:done.add(i)
    assert len(done)==m
    Path('results/tape.json').write_text(json.dumps(dict(seeds=seeds,games=tapes,engine=info['payload_sha256']),separators=(',',':')))
    print('captured',m,'complete matches')
if __name__=='__main__':main()
