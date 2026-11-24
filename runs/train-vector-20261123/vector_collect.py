"""Train or time full-width vector acting. Replay steps may only exercise a disposable zero-rate learner."""
import argparse,collections,copy,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch,numpy as np
from runtime import verify_engine,selfplay_env,SIMULATOR_REVISION,OBSERVATION_REVISION,PARENT_SHA,SOURCE_OF_TRUTH
from learner import prepare,optimizer_for
from learner_fast import update_fast
from learner_ops import enable_learner
from opponents import Opponents,LIVE
from collector_g import Queue,GPUUtil
from vector_env import allocate,PythonVector,RecordedVector
from actor_wave import WaveActor
from trajectories import Turns
clock=time.perf_counter
def run(a):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.save and (a.backend=='replay' or not a.train):raise ValueError('Only real, trained checkpoints may be saved')
    info=verify_engine();n=a.games;ids=np.arange(n)
    example=selfplay_env.ParallelSelfPlayEnv().reset(seed=42,paired_tape=True)[0];o=allocate(n,example,a.backend=='python')
    env=PythonVector(n,o,a.workers) if a.backend=='python' else RecordedVector(n,'results/tape')
    try:
        _,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(8*2**30/total)
        parent=Path('/sap/checkpoints/update-025772.pt')
        model,state,rates,sem,start,rows,primitives=prepare(parent,None,'parent',info,a.seed)
        rng=np.random.default_rng(a.seed);env.reset(ids,rng.integers(2**62,size=n),o)
        model.cuda();actor_model=copy.deepcopy(model).eval().requires_grad_(False)
        pool=Opponents(actor_model,parent,.5,4,2*n,a.seed,start,info,None)
        actor=WaveActor(actor_model,pool,n);actor.reset(ids)
        if a.train:
            opt=optimizer_for(model,state,rates,1 if a.backend=='python' else 0)
            enable_learner(model);turns=Turns(o,n,model.config.memory_dim)
        del state
        version=start;queue=Queue();future=None;learned=seat=live=delivered=matches=waves=0;events=[];measured=False
        times=collections.Counter();learner_wall=0.;captures=[];torch.cuda.manual_seed(a.seed+start)
        stream=torch.cuda.Stream()
        def optimize(batch):
            t=clock()
            with torch.cuda.stream(stream):
                stats=update_fast(model,opt,batch,sem,'cuda',1536,retry_oom=True)
            stream.synchronize();return stats,clock()-t
        def publish():
            nonlocal future,version,learned,learner_wall,primitives
            t=clock();stats,elapsed=future.result();torch.cuda.current_stream().wait_stream(stream)
            actor_model.load_state_dict(model.state_dict());torch.cuda.current_stream().synchronize()
            times['wait_publish']+=clock()-t;future=None;version+=1;learner_wall+=elapsed;learned+=stats['decisions'];primitives+=stats['primitive_actions']
            events.append(dict(version=version,min_version=lo,max_version=hi,seconds=clock()-begin,decisions=stats['decisions'],stale=queue.stale))
            if (version-start)%250==0:pool.snapshot(version,a.out/f'update-{version:06d}.pt')
        a.out.mkdir(parents=True,exist_ok=False)
        with GPUUtil() as gpu,ThreadPoolExecutor(1) as executor:
            begin=clock();sample=None
            while True:
                if future and (future.done() or queue.rows>=8192*a.backlog):publish()
                if a.train and not future and (not a.updates or version-start<a.updates):
                    sel=queue.pop(version,8192,1)
                    if sel:
                        batch,lo,hi=sel;stream.wait_stream(torch.cuda.current_stream());future=executor.submit(optimize,batch)
                if (a.updates and version-start>=a.updates) or clock()-begin>=a.seconds:break
                t=clock()
                if a.train:
                    flags=o['active'].ravel() & (pool.owners==LIVE)
                    turns.start_memory(actor,np.flatnonzero(flags))
                times['turn_capture']+=clock()-t;t=clock()
                all_ids,live_ids=actor.act(o);times['actor']+=clock()-t
                t=clock()
                if a.train:turns.push(o,actor,live_ids,version)
                times['turn_capture']+=clock()-t
                t=clock();env.step(actor.actions,actor.masks,o);times['engine']+=clock()-t
                t=clock()
                if a.train:
                    complete=turns.finish(o,pool);queue.add(complete);delivered+=sum(x.length for x in complete)
                finish=np.flatnonzero(o['terminated']);matches+=len(finish)
                times['turn_capture']+=clock()-t;t=clock()
                if len(finish):
                    env.reset(finish,rng.integers(2**62,size=len(finish)),o);actor.reset(finish)
                times['reset']+=clock()-t
                seat+=len(all_ids);live+=len(live_ids);waves+=1
                if sample is None and matches>=n and clock()-begin<a.seconds-10:
                    sample=(clock(),seat,matches,learned,queue.stale)
            if future:publish()
            duration=clock()-begin
            if a.train:
                assert learned+queue.stale+queue.rows==delivered and delivered+sum(turns.lengths)==live
            result=dict(backend=a.backend,training=a.train,games=n,workers=a.workers,wall=duration,matches=matches,waves=waves,seat=seat,live=live,
                seat_s=seat/duration,games_s=matches/duration,learned_s=learned/duration,learned=learned,delivered=delivered,updates=version-start,
                stale=queue.stale,stale_share=queue.stale/max(1,learned+queue.stale),queued=queue.rows,backlog=a.backlog,pending_stale=sum(s.length for s,minimum,maximum in queue.q if version-minimum>1),
                gpu=float(np.mean([v for ts,v in gpu.samples if ts>=begin])),main_seconds=times,learner_wall=learner_wall,
                actor_timing=actor.timing,mean_active_seats=seat/waves,policy_batch_percentiles=np.percentile(actor.batches,[10,50,90,99]).tolist(),peak_gib=torch.cuda.max_memory_reserved()/2**30,
                steady=None if sample is None else dict(seconds=clock()-sample[0],seat_s=(seat-sample[1])/(clock()-sample[0]),games_s=(matches-sample[2])/(clock()-sample[0]),learned_s=(learned-sample[3])/(clock()-sample[0])),
                events=events,seed=a.seed,engine=info['payload_sha256'])
            (a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)
            if a.save:
                from dataclasses import asdict
                from sapai_gym1.rl.causal_transport import checkpoint_context_metadata
                f=a.out/f'update-{version:06d}.pt';tmp=f.with_suffix('.tmp')
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),config=asdict(model.config),update=version,transitions=rows+learned,learner_primitive_actions=primitives,simulator_revision=SIMULATOR_REVISION,engine_payload_sha256=info['payload_sha256'],source_of_truth=SOURCE_OF_TRUTH,observation_revision=OBSERVATION_REVISION,causal_context=checkpoint_context_metadata(model.config),training_semantics=sem,wide_arm='parent',parent_checkpoint_sha256=PARENT_SHA,collector='vector-g',max_behavior_lag=1),tmp);tmp.replace(f)
    finally:env.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['python','replay'],required=True);p.add_argument('--train',action='store_true');p.add_argument('--games',type=int,default=1024);p.add_argument('--workers',type=int,default=12);p.add_argument('--seconds',type=float,default=150);p.add_argument('--updates',type=int,default=0);p.add_argument('--save',action='store_true');p.add_argument('--out',type=Path,required=True);p.add_argument('--backlog',type=float,default=.5);p.add_argument('--seed',type=int,default=2577223)
    run(p.parse_args())
