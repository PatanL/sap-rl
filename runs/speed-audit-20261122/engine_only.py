from functools import lru_cache
import json, os, random, time
from plan_common import install
install(cuda=False)
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import FlatActionCodec as C
from sapai_gym1.rl.end_freeze import step_with_end_freeze

@lru_cache(maxsize=16384)
def subsets(eligible,budget):
    return tuple(m for m in range(1<<eligible.bit_length()) if (m & ~eligible)==0 and m.bit_count()<=budget)

def run(env,mode,index):
    seed=221122000+index
    rng=random.Random(seed)
    env.reset(seed=seed,paired_tape=True)
    d=b=expanded=0
    for tick in range(4000):
        actions=[]; masks=[]
        for s,active in enumerate(env.active_players()):
            if not active:
                actions.append(None);masks.append(0);continue
            keys=tuple(env.core._avail_actions(s))
            mask=0
            if mode=='joint' and C.END_TURN in keys:
                pl=(env.core.agent,env.core.opponent)[s]
                eligible=sum(1<<i for i,slot in enumerate(pl.shop) if not slot.empty and
                    (C.UNFREEZE+i if slot.frozen else C.FREEZE+i) in keys) if pl.gold==0 else 0
                count=(env.core.shop_action_count_agent,env.core.shop_action_count_opponent)[s]
                opts=subsets(eligible,max(0,50-count))
                pick=rng.randrange(len(keys)+len(opts)-1)
                if pick>=len(keys): code=C.END_TURN;mask=opts[pick-len(keys)+1]
                else:code=keys[pick]
            else:code=rng.choice(keys)
            actions.append(C.decode(code));masks.append(mask);d+=1;expanded+=1+mask.bit_count()
        step=(step_with_end_freeze(env,*actions,mask_ids=masks).result if mode=='joint' else env.step(*actions))
        b+=int(step.battle_resolved)
        if step.terminated:return d,b,expanded
    raise RuntimeError('incomplete match')

def main():
    import sys
    raw='--core-only' in sys.argv
    env=ParallelSelfPlayEnv()
    if raw:
        # Neither the sampler nor the reference transition consumes these
        # returned numpy arrays. Retain legality calls and all transitions.
        env.observations=lambda:(None,None)
    results={}
    for mode in ('primitive','joint'):
        run(env,mode,1000)
        blocks=[]
        for j in range(4):
            start=time.perf_counter()
            rows=[run(env,mode,16*j+i) for i in range(16)]
            blocks.append(dict(seconds=time.perf_counter()-start,matches=len(rows),decisions=sum(x[0] for x in rows),battles=sum(x[1] for x in rows),expanded=sum(x[2] for x in rows)))
        results[mode]=blocks
        print(mode,blocks,flush=True)
    open('engine-core-only.json' if raw else 'engine-only.json','w').write(json.dumps(dict(affinity=list(os.sched_getaffinity(0)),results=results),indent=2))

if __name__=='__main__':main()
