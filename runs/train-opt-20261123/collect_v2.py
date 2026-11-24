"""Training collector on the Rust environment (pinned v31g-env2), from runs/rust-collector-20261123/collect_rust.py.
Added here: ranked players' boards as opponents (--ghost-share, ghost_rust.py), --frozen-share, --resume,
--checkpoint-every, a stall alarm (--stall seconds without a new update ends the process with code 3), and the
three speed options of collect_opt.py:
  --cap N        padded-row cap of the weight update (was fixed at 1536)
  --compact      learner evaluates only valid decisions (compact_sequence, exact in FP32)
  --compile      entity transformer through torch.compile for the learner, the acting copy and frozen opponents
"""
import argparse,collections,copy,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch,numpy as np
from runtime import verify_engine,SIMULATOR_REVISION,OBSERVATION_REVISION,PARENT_SHA,SOURCE_OF_TRUTH
from learner import prepare,optimizer_for
from learner_fast import update_fast
from learner_ops import enable_learner
from opponents import Opponents,LIVE
import os,threading
from ghost_rust import GhostOpponents,GhostWaveActor,Ghosts,GHOST
from collector_g import Queue,GPUUtil
from rust_adapter import RustVector,LIB_SHA
from actor_wave import WaveActor
from trajectories import Turns
clock=time.perf_counter
def run(a):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.save and not a.train:raise ValueError('Only real, trained checkpoints may be saved')
    info=verify_engine();n=a.games;ids=np.arange(n)
    env=RustVector(n,a.workers)
    o=env.output
    try:
        _,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(8*2**30/total)
        parent=Path('/sap/checkpoints/update-025772.pt')
        model,state,rates,sem,start,rows,primitives=prepare(parent,a.resume,a.arm,info,a.seed)
        if a.parent_rates:rates=dict(base=2e-6,opponent_history=1e-4,opponent_adapter=1e-5,end_freeze=5e-5)   # the 14.35M champion's saved rates
        if a.fresh_optimizer:state=None
        if a.rate_scale!=1. or a.base_lr or a.parent_rates:
            rates={k:v*a.rate_scale for k,v in rates.items()}
            if a.base_lr:rates['base']=a.base_lr
            for group,field in [('opponent_history','opponent_module_learning_rate'),('opponent_adapter','opponent_adapter_learning_rate'),('end_freeze','end_freeze_learning_rate')]:sem[field]=rates[group]
            print('learning rates',rates,flush=True)
        rng=np.random.default_rng([a.seed,start]);env.reset(ids,rng.integers(2**62,size=n),o)
        model.cuda();actor_model=copy.deepcopy(model).eval().requires_grad_(False)
        if a.compile:
            import fast_encoder,opponents
            fast_encoder.enable(actor_model)   # deep copies made by the pool keep this class and get their own wrapper
        pool=GhostOpponents(actor_model,(parent if a.arm=='parent' else a.resume),(.5 if (a.arm=='parent' or a.resume) else 0.),4,2*n,a.seed,start,info,a.resume,ghost_share=a.ghost_share,frozen_share=a.frozen_share)
        actor=GhostWaveActor(actor_model,pool,n);actor.reset(ids)
        ghosts=Ghosts(env.native,pool,n,a.seed+start)
        if a.ghost_share:print('ghost pool boards accepted by set_team:',ghosts.validate_all(),flush=True)
        ghosts.assign(ids);by_type=collections.Counter();progress=[clock()]
        if a.train:
            opt=optimizer_for(model,state,rates,1)
            enable_learner(model);turns=Turns(o,n,model.config.memory_dim)
            if a.compile:
                fast_encoder.enable(model)
                # Compile both graphs here, in the main thread, before the learner thread exists: once without
                # gradients (acting) and once with (learning), at two batch sizes so the shapes stay symbolic.
                example={k:torch.from_numpy(v.reshape(-1,*v.shape[2:])) for k,v in o['obs'].items()}
                for rows in (2,64,96):
                    piece={k:v[:rows].cuda() for k,v in example.items()}
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):actor_model.encode_entities(piece)
                    with torch.autocast('cuda',dtype=torch.bfloat16):model.encode_entities(piece).float().sum().backward()
                    # third variant: the learner's bootstrap value call (trainable weights, no gradient)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):model.encode_entities(piece)
                model.zero_grad(set_to_none=True);print('compiled encoder warmed up',flush=True)
        del state
        version=start;queue=Queue();future=None;learned=seat=live=delivered=matches=waves=battles=0;events=[]
        times=collections.Counter();learner_wall=0.;torch.cuda.manual_seed(a.seed+start)
        stream=torch.cuda.Stream()
        def optimize(batch):
            t=clock()
            with torch.cuda.stream(stream):
                stats=update_fast(model,opt,batch,sem,'cuda',a.cap,compact=a.compact,retry_oom=True)
            stream.synchronize();return stats,clock()-t
        def checkpoint(path):
            from dataclasses import asdict
            from sapai_gym1.rl.causal_transport import checkpoint_context_metadata
            record = dict(
                model=model.state_dict(), optimizer=opt.state_dict(), config=asdict(model.config),
                update=version, transitions=rows+learned, learner_primitive_actions=primitives,
                simulator_revision=SIMULATOR_REVISION, engine_payload_sha256=info['payload_sha256'],
                source_of_truth=SOURCE_OF_TRUTH, observation_revision=OBSERVATION_REVISION,
                causal_context=checkpoint_context_metadata(model.config), training_semantics=sem,
                wide_arm=a.arm, parent_checkpoint_sha256=PARENT_SHA, rust_environment_sha256=LIB_SHA,
                collector='rust-vector-g-v2', actor_sampling='stochastic policy.act, no search',
                max_behavior_lag=a.max_lag, reference_frozen_share=a.frozen_share, ghost_share=a.ghost_share, ghost_pool_sha256=(__import__('ghost_rust').POOL_SHA256 if a.ghost_share else None), physical_rows=a.cap, reference_rows=128, learner_compact=a.compact, encoder_compiled=a.compile,
            )
            tmp = path.with_suffix('.pt.tmp')
            torch.save(record,tmp)
            tmp.replace(path)

        def publish():
            nonlocal future,version,learned,learner_wall,primitives
            t=clock();stats,elapsed=future.result();torch.cuda.current_stream().wait_stream(stream)
            actor_model.load_state_dict(model.state_dict());torch.cuda.current_stream().synchronize()
            times['wait_publish']+=clock()-t;future=None;version+=1;learner_wall+=elapsed;learned+=stats['decisions'];primitives+=stats['primitive_actions']
            with open(a.out/'updates.jsonl','a') as fp:fp.write(json.dumps(dict(version=version,seconds=clock()-begin,learner_s=elapsed,decisions=stats['decisions'],stale=queue.stale,loss=stats['loss'],grad_norm=stats['grad_norm']))+'\n')
            events.append(dict(version=version,min_version=lo,max_version=hi,seconds=clock()-begin,decisions=stats['decisions'],turns=stats['turns'],stale=queue.stale))
            progress[0]=clock()
            if (version-start)%a.checkpoint_every==0:
                path=a.out/f'update-{version:06d}.pt'
                if a.save:checkpoint(path)
                pool.snapshot(version,path)
        a.out.mkdir(parents=True,exist_ok=False)
        (a.out/'args.json').write_text(json.dumps(vars(a),default=str,indent=2))
        with GPUUtil() as gpu,ThreadPoolExecutor(1) as executor:
            begin=clock();sample=None;progress[0]=clock()
            def watchdog():
                while True:
                    time.sleep(5)
                    if a.train and clock()-progress[0]>a.stall:
                        print(f'STALL: no update for {a.stall} s at version {version}; exiting',flush=True);import faulthandler,sys;faulthandler.dump_traceback(file=sys.stdout,all_threads=True);sys.stdout.flush();os._exit(3)
            threading.Thread(target=watchdog,daemon=True).start()
            while True:
                if future and (future.done() or queue.rows>=a.target*a.backlog):publish()
                if a.train and not future and (not a.updates or version-start<a.updates):
                    sel=queue.pop(version,a.target,a.max_lag)
                    if sel:
                        batch,lo,hi=sel;stream.wait_stream(torch.cuda.current_stream());future=executor.submit(optimize,batch)
                if (a.updates and version-start>=a.updates) or clock()-begin>=a.seconds:break
                t=clock()
                if a.train:
                    flags=o['active'].ravel() & (pool.owners==LIVE)
                    turns.start_memory(actor,np.flatnonzero(flags))
                times['turn_capture']+=clock()-t;t=clock()
                all_ids,live_ids=actor.act(o);times['actor']+=clock()-t
                t=clock();ghosts.before_step(o,actor)
                other=pool.owners[live_ids^1];by_type['ghost']+=int((other==GHOST).sum());by_type['mirror']+=int((other==LIVE).sum());by_type['frozen']+=int(((other!=GHOST)&(other!=LIVE)).sum());times['ghost']+=clock()-t
                t=clock()
                if a.train:turns.push(o,actor,live_ids,version)
                times['turn_capture']+=clock()-t
                t=clock();env.step(actor.actions,actor.masks,o);times['engine']+=clock()-t
                t=clock();ghosts.after_step(o);times['ghost']+=clock()-t
                t=clock()
                if a.train:
                    complete=turns.finish(o,pool);queue.add(complete);delivered+=sum(x.length for x in complete)
                battles+=int(np.count_nonzero(o['battle_resolved']));finish=np.flatnonzero(o['terminated']);matches+=len(finish)
                times['turn_capture']+=clock()-t;t=clock()
                if len(finish):
                    env.reset(finish,rng.integers(2**62,size=len(finish)),o);actor.reset(finish);ghosts.assign(finish)
                times['reset']+=clock()-t
                seat+=len(all_ids);live+=len(live_ids);waves+=1
                if sample is None and matches>=n and clock()-begin<a.seconds-10:
                    sample=(clock(),seat,matches,learned,queue.stale)
            if future:publish()
            duration=clock()-begin
            if a.train:
                assert learned+queue.stale+queue.rows==delivered and delivered+sum(turns.lengths)==live
            result=dict(backend=a.backend,training=a.train,games=n,workers=a.workers,wall=duration,matches=matches,battles=battles,waves=waves,seat=seat,live=live,
                seat_s=seat/duration,games_s=matches/duration,learned_s=learned/duration,learned=learned,delivered=delivered,updates=version-start,
                stale=queue.stale,stale_share=queue.stale/max(1,learned+queue.stale),queued=queue.rows,backlog=a.backlog,pending_stale=sum(s.length for s,minimum,maximum in queue.q if version-minimum>a.max_lag),
                gpu=float(np.mean([v for ts,v in gpu.samples if ts>=begin])),main_seconds=times,learner_wall=learner_wall,
                actor_timing=actor.timing,mean_active_seats=seat/waves,policy_batch_percentiles=np.percentile(actor.batches,[10,50,90,99]).tolist(),peak_gib=torch.cuda.max_memory_reserved()/2**30,
                steady=None if sample is None else dict(seconds=clock()-sample[0],seat_s=(seat-sample[1])/(clock()-sample[0]),games_s=(matches-sample[2])/(clock()-sample[0]),learned_s=(learned-sample[3])/(clock()-sample[0])),
                events=events,assignment=dict(pool.assignments),live_decisions_by_opponent=dict(by_type),ghost=ghosts.stats,library_sha256=LIB_SHA,seed=a.seed,engine=info['payload_sha256'])
            if a.save and version>start:
                tail=a.out/f'update-{version:06d}.pt'
                if not tail.exists():
                    checkpoint(tail)
            (a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)
    finally:env.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['rust'],default='rust');p.add_argument('--train',action='store_true');p.add_argument('--games',type=int,default=1024);p.add_argument('--workers',type=int,default=12);p.add_argument('--seconds',type=float,default=150);p.add_argument('--updates',type=int,default=0);p.add_argument('--save',action='store_true');p.add_argument('--out',type=Path,required=True);p.add_argument('--backlog',type=float,default=.5);p.add_argument('--seed',type=int,default=2577223);p.add_argument('--cap',type=int,default=1536);p.add_argument('--compact',action='store_true');p.add_argument('--compile',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--checkpoint-every',type=int,default=250);p.add_argument('--ghost-share',type=float,default=0.);p.add_argument('--frozen-share',type=float,default=.5);p.add_argument('--stall',type=float,default=240.);p.add_argument('--max-lag',type=int,default=1);p.add_argument('--arm',choices=['parent','fresh'],default='parent');p.add_argument('--target',type=int,default=8192);p.add_argument('--base-lr',type=float,default=0.);p.add_argument('--parent-rates',action='store_true');p.add_argument('--fresh-optimizer',action='store_true');p.add_argument('--rate-scale',type=float,default=1.)
    run(p.parse_args())
