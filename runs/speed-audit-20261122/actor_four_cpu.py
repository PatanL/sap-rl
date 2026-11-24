"""Time the pinned actor/IPC service without a learner, optimization or checkpoint writes."""
import copy,inspect,json,os,threading,time
from pathlib import Path
from queue import Empty
import torch
from plan_common import install,load_actor
install(cuda=False)
from sapai_gym1.rl import distributed as D
from sapai_gym1.rl import selfplay_env as S
from sapai_gym1.rl.entity_observation import EntityObservationEncoder
REF_ACTOR=D._actor_main
WARM=96
CYCLES=1024
clock=time.perf_counter

def audited_actor(*args):
    if args[0]!=0:
        reset=S.ParallelSelfPlayEnv.reset
        def paired(self,*a,**kw):
            kw['paired_tape']=True
            return reset(self,*a,**kw)
        S.ParallelSelfPlayEnv.reset=paired
        return REF_ACTOR(*args)
    stop=inspect.signature(REF_ACTOR).bind(*args).arguments['stop_event']
    rows=[]; current=None
    def timed(obj, name, category, *, reset=False):
        old=getattr(obj,name)
        def wrap(*a,**kw):
            if reset:kw['paired_tape']=True
            t=clock();out=old(*a,**kw);elapsed=clock()-t
            if current is not None:
                current[category]+=elapsed
                if category=='reset':current['matches']+=1
                if category=='battle':current['battles']+=1
            return out
        setattr(obj,name,wrap)
    # Count construction+simulation, and avoid timing each ability.
    B=S.Battle
    class TimedBattle(B):
        def __init__(self,*a,**kw):
            t=clock();super().__init__(*a,**kw)
            if current is not None:current['battle_init']+=clock()-t
    S.Battle=TimedBattle
    timed(TimedBattle,'battle','battle')
    timed(EntityObservationEncoder,'encode','obs')
    timed(S.ParallelSelfPlayEnv,'reset','reset',reset=True)
    timed(D,'step_with_end_freeze','step')
    timed(D,'_write_observation','stage')
    def begin():
        nonlocal current
        now=clock()
        if current is not None:
            current['end']=now;rows.append(current)
        if len(rows)==WARM+CYCLES:
            current=None; stop.set(); return False
        current=dict(start=now,step=0.,battle=0.,battle_init=0.,obs=0.,reset=0.,stage=0.,battles=0,matches=0)
        return True
    src=inspect.getsource(REF_ACTOR)
    def insert(a,b):
        nonlocal src
        assert src.count(a)==1,a
        src=src.replace(a,b)
    insert('        shared.active.zero_()', '        if not begin(): return\n        shared.active.zero_()')
    insert('        shared.request_event.set()', "        current['decisions']=sum(sum(active) for active in active_by_game)\n        current['sent']=clock()\n        shared.request_event.set()")
    insert('        if shared.telemetry_enabled:\n            completed_at', "        current['received']=clock()\n        if shared.telemetry_enabled:\n            completed_at")
    ns={**vars(D),'begin':begin,'clock':clock,'getcurrent':lambda:current}
    src=src.replace("current['","getcurrent()['")
    exec(compile(src,'<timed frozen actor>','exec'),ns)
    ns['_actor_main'](*args)
    Path('four-actor-cycles.json').write_text(json.dumps(rows))

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    assert os.sched_getaffinity(0)=={16,17,18,19}
    model=load_actor(device='cpu').model
    obs=S.ParallelSelfPlayEnv().reset(seed=75002,paired_tape=True)[0]
    D._actor_main=audited_actor
    runtime=D.DistributedSelfPlayRuntime(model,4,obs,'cpu',seed=62011,games_per_actor=3,
        queue_size=48,discount_mode='per_turn',lean_policy_execution=True,inference_batch_delay_ms=2.)
    # Resident copy of released parent, routed separately from the main replica.
    from plan_common import sha
    receipt=json.loads(Path('/residents/receipt.json').read_text())
    residents={1:model}
    for i,name in enumerate(('terminal-response','strategy'),2):
        path=Path('/residents')/(name+'.pt')
        assert sha(path)==receipt['residents'][name]
        policy=copy.deepcopy(model)
        policy.load_state_dict(torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model'])
        residents[i]=policy.eval().requires_grad_(False)
    runtime.set_policy_pool(residents,main_probability=3.)
    forwards=[]; focal={}
    serve=runtime.batcher._serve
    def track(ready):
        nonlocal focal
        focal={}
        if 0 in ready:
            shared=runtime.actor_buffers[0]
            for a,p in zip(shared.active.tolist(),shared.policy_ids.tolist()):
                if a: focal[p]=focal.get(p,0)+1
        return serve(ready)
    runtime.batcher._serve=track
    for owner,policy in runtime.batcher.resident_policies.items():
        fn=policy.act
        def measured(obs,*a,_fn=fn,_owner=owner,**kw):
            start=clock();out=_fn(obs,*a,**kw);end=clock()
            forwards.append([start,end,len(obs['type_mask']),_owner,focal.get(_owner,0)]);return out
        policy.act=measured
    runtime.start()
    try:
        while runtime.processes[0].is_alive():
            try:runtime.trajectory_ring.read_dense(4,.3)
            except Empty:pass
        runtime.processes[0].join()
        if runtime.processes[0].exitcode:raise RuntimeError(runtime.processes[0].exitcode)
    finally:runtime.stop()
    Path('four-actor-forwards.json').write_text(json.dumps(forwards))
    rows=json.loads(Path('four-actor-cycles.json').read_text());assert len(rows)==WARM+CYCLES
    assert len(forwards)==runtime.batcher.inference_batches
    all_calls=len(forwards)
    all_items=sum(f[2] for f in forwards)
    forwards=[f for f in forwards if f[-1]]
    idx=0;blocks=[]
    for i,r in enumerate(rows):
        times=[]
        while idx<len(forwards) and forwards[idx][0]<r['received']:
            start,end,n,owner,focal_items=forwards[idx]
            assert r['sent']<=start<=end<=r['received']
            times.append((end-start,focal_items));idx+=1
        assert times and sum(n for _,n in times)==r['decisions']
        if i<WARM:continue
        b=(i-WARM)//256
        if b==len(blocks):blocks.append(dict())
        block=blocks[b];wall=r['end']-r['start']
        forward=sum(t for t,_ in times)
        battle=r['battle']+r['battle_init']; obs=r['obs']
        shop=r['step']+r['reset']-battle-obs
        ipc=r['received']-r['sent']-forward+r['stage']
        other=wall-shop-battle-obs-forward-ipc
        assert min(shop,battle,obs,forward,ipc,other)>=-1e-7
        for k,v in dict(wall=wall,shop=shop,battle=battle,obs=obs,forward=forward,ipc=ipc,other=other,
                        decisions=r['decisions'],matches=r['matches'],battles=r['battles'],groups=len(times),staging=r['stage']).items():
            block[k]=block.get(k,0)+v
    assert idx==len(forwards)
    print(blocks)
    Path('four-actor-summary.json').write_text(json.dumps(dict(warmup=WARM,cycles=CYCLES,blocks=blocks,allCallsIncludingWarmup=all_calls,allItemsIncludingWarmup=all_items),indent=2))
if __name__=='__main__':main()
