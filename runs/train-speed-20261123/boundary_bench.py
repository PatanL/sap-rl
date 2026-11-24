"""Fixed actions and seeds, no network. Paired timing of real battle replacement and ideal shop bypass."""
import argparse,hashlib,json,time,statistics,os
from pathlib import Path
import numpy as np
from runtime import selfplay_env,Battle
from rust_boundary import Bridge
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.end_freeze import step_with_end_freeze

def digest_obs(h,obs):
    for d in obs:
        for k in sorted(d):h.update(d[k].tobytes())
def run(tape,mode,verify=False,limit=None):
    bridge=Bridge();counts=0;steps=0;seats=0;bh=hashlib.sha256();oh=hashlib.sha256()
    def cached(t0,t1,seed=None,**kw):
        class Result:
            def battle(self):
                assert expected is not None and seed==expected[0];return expected[1]
        return Result()
    base={'python':Battle,'rust':bridge.battle_class(),'cached':cached}[mode.split('-')[0]] if mode!='replay' else None
    selfplay_env.Battle=base
    start=time.perf_counter()
    for seed,actions in zip(tape['seeds'][:limit],tape['games'][:limit]):
        if mode!='replay':
            env=selfplay_env.ParallelSelfPlayEnv()
            if mode.endswith('-noobs'):env.observations=lambda:(None,None)
            obs=env.reset(seed=seed,paired_tape=True)
            if verify:digest_obs(oh,obs)
        for aa,expected,terminal,reward in actions:
            steps+=1;seats+=sum(a is not None for a in aa)
            if mode=='replay':
                # Ideal trace-return boundary: recorded step metadata, with encoding excluded in both arms.
                outcome=None if expected is None else expected[1];is_terminal=terminal;got_reward=reward
            else:
                args=[None if a is None else AutoregressiveAction(*a[:3]) for a in aa]
                masks=[0 if a is None else a[3] for a in aa]
                r=step_with_end_freeze(env,*args,mask_ids=masks).result
                outcome=r.battle_result;is_terminal=r.terminated;got_reward=r.rewards
                if verify:digest_obs(oh,r.observations)
            if outcome!=(expected[1] if expected else None) or terminal!=is_terminal or list(got_reward)!=reward:
                raise ValueError((seed,steps,aa,expected,outcome,reward,got_reward))
            if outcome is not None:bh.update(bytes((outcome,)));counts+=1
    elapsed=time.perf_counter()-start
    return dict(s=elapsed,steps=steps,seats=seats,battles=counts,outcome_sha=bh.hexdigest(),obs_sha=oh.hexdigest(),rust_calls=bridge.calls,fallbacks=bridge.fallbacks)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--limit',type=int)
    a=p.parse_args();t=json.load(open('results/tape.json'))
    if a.test:
        r1=run(t,'python',True,a.limit);r2=run(t,'rust',True,a.limit);r3=run(t,'cached',True,a.limit)
        assert r1['obs_sha']==r2['obs_sha']==r3['obs_sha']
        Path('results/boundary-parity.json').write_text(json.dumps(dict(matches=len(t['games'][:a.limit]),battles=r1['battles'],seats=r1['seats'],rust_calls=r2['rust_calls'],fallbacks=r2['fallbacks'],public_obs_digest=r2['obs_sha']),indent=2))
    else:
        assert Path('/sap/runs/train-live-20261123/STOPPED').exists()
        os.sched_setaffinity(0,{max(os.sched_getaffinity(0))})
        modes=['python','rust','cached','cached-noobs','replay'];out={m:[] for m in modes}
        for sweep in (modes,modes[::-1],modes):
            for mode in sweep:out[mode].append(run(t,mode,False,a.limit))
        Path('results/boundary-bench.json').write_text(json.dumps(out,indent=2))
        print({m:statistics.median(r['s'] for r in v) for m,v in out.items()})
