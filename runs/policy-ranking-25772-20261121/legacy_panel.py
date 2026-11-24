"""Paired native-action checkpoint evaluation on common frozen v30 rules."""
import json
from pathlib import Path

import numpy as np
import torch
from plan_common import install, load_actor, seed_of, sha

FILES = {
    'v22': ('/residents/v22.pt','8f0cd760bdf3ea8258fec84ad6e93882054ee82755032808308737324aff6e5c'),
    'v25': ('/residents/v25.pt','7c60acec58b56080bf240a85444163a1d72dba7bee25ac7f54e3dc8302d60994'),
}


@torch.inference_mode()
def main():
    install(cuda=False)
    from sapai_gym1.rl.model import EntityTransformerConfig,EntityTransformerPolicy
    from sapai_gym1.rl.causal_transport import validate_checkpoint_context
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    from sapai_gym1.rl.joint_action import step_decisions,JointAction
    from sapai_gym1.rl.action_schema import AutoregressiveAction
    from sapai_gym1.rl.distributed import _completed_match_result
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    actor=load_actor(device='cpu')
    results={}
    for name,(path,digest) in FILES.items():
        assert sha(path)==digest
        ck=torch.load(path,map_location='cpu',weights_only=True)
        validate_checkpoint_context(ck,allow_observation_transfer=True)
        old=EntityTransformerPolicy(EntityTransformerConfig(**ck['config'])).eval()
        old.load_state_dict(ck['model'],strict=True)
        assert not old.config.end_freeze
        games=[]
        for i in range(8):
            seed=seed_of('25772-legacy-compatibility-v1',i)
            for seat in (0,1):
                env=ParallelSelfPlayEnv()
                env.reset(seed=seed,paired_tape=True)
                games.append(dict(env=env,seed=seed,seat=seat,score=None,steps=0,
                                  memory=actor.model.initial_memory(2,'cpu')))
        for tick in range(2500):
            live=[g for g in games if g['score'] is None]
            if not live:
                break
            pairs=[[None,None] for _ in live]
            groups=[[],[]]
            for i,g in enumerate(live):
                obs=g['env'].observations()
                for s,active in enumerate(g['env'].active_players()):
                    if active:
                        groups[int(s!=g['seat'])].append((i,s,obs[s],g['memory'][s]))
            for policy,requests in zip((actor.model,old),groups):
                if not requests:
                    continue
                batch={k:torch.tensor(np.stack([r[2][k] for r in requests])) for k in requests[0][2]}
                out=policy.act(batch,memory=torch.stack([r[3] for r in requests]),deterministic=True)
                for j,(i,s,_,_) in enumerate(requests):
                    mask=None if out.end_freeze_mask is None else out.end_freeze_mask[j].item()
                    pairs[i][s]=JointAction(AutoregressiveAction(*out.actions[j].tolist()),mask)
                    live[i]['memory'][s]=out.memory[j]
            for g,pair in zip(live,pairs):
                g['steps']+=1
                if step_decisions(g['env'],pair).result.terminated:
                    result=_completed_match_result(g['env'].core)
                    g['score']=.5 if result==2 else float(result==g['seat'])
        assert all(g['score'] is not None for g in games)
        results[name]=dict(legacySHA=digest,originalSimulator=ck.get('simulator_revision'),
                           score=np.mean([g['score'] for g in games]),
                           games=[{k:g[k] for k in ('seed','seat','score','steps')} for g in games])
        print(name,results[name]['score'],flush=True)
    Path('/out/legacy-panel.json').write_text(json.dumps(dict(policy=actor.owner,results=results),indent=2))


if __name__=='__main__':
    main()
