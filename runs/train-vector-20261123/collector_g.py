"""Single-owner actor/coordinator and single-flight V-trace learner with an immutable publication boundary."""
import argparse,collections,copy,hashlib,json,math,os,threading,time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from runtime import verify_engine,SIMULATOR_REVISION
from learner import prepare,optimizer_for,update as reference_update,write_checkpoint,max_segment_lag
from frozen_wide import Collector as NativeCollector
from opponents import Opponents,LIVE
from worker_profile import actors
from learner_fast import update_fast
clock=time.perf_counter

class Queue:
    def __init__(self):
        self.q=collections.deque();self.rows=0;self.stale=0
    def add(self,segs):
        for s in segs:
            v=s.behavior_versions
            self.q.append((s,int(v.min()),int(v.max())));self.rows+=s.length
    def pop(self,version,target,lag):
        if self.rows<target:return None
        chosen=[];used=0
        while self.q and used<target:
            s,lo,hi=self.q.popleft();self.rows-=s.length
            if hi>version or lo<0:raise ValueError('Invalid behavior version')
            if version-lo>lag:self.stale+=s.length;continue
            chosen.append((s,lo,hi));used+=s.length
        if used<target:
            self.q.extendleft(reversed(chosen));self.rows+=used;return None
        return [x[0] for x in chosen],min(x[1] for x in chosen),max(x[2] for x in chosen)

class Meter:
    def __init__(self):
        self.events=[];self.learner=[];self.waves=[];self.worker=collections.Counter();self.counts=collections.Counter()
    def add(self,k,a,b):
        if b>a:self.events.append((a,b,k))
    def result(self,start,end):
        # Main intervals are nonoverlapping. Charge their intersection with the learner as overlap.
        spans=sorted(self.events);complete=[];last=start
        for a,b,k in spans:
            a=max(last,a,start);b=min(b,end)
            if b<=a:continue
            if a>last:complete.append((last,a,'other'))
            complete.append((a,b,k));last=b
        if last<end:complete.append((last,end,'other'))
        main=collections.Counter();overlap=collections.Counter();i=0;ls=sorted(self.learner)
        for a,b,k in complete:
            main[k]+=b-a
            while i<len(ls) and ls[i][1]<=a:i+=1
            j=i
            while j<len(ls) and ls[j][0]<b:
                overlap[k]+=max(0,min(b,ls[j][1])-max(a,ls[j][0]));j+=1
        # Count only waves wholly inside a learner call; partially overlapped waves are additional.
        j=0;inside=0;inside_seats=0
        for a,b,n in self.waves:
            while j<len(ls) and ls[j][1]<a:j+=1
            if j<len(ls) and ls[j][0]<=a and b<=ls[j][1]:
                inside+=1;inside_seats+=n
        return dict(fully_overlapped_inference_waves=inside,seats_fully_served_during_updates=inside_seats,
                    main_seconds=main,overlap_with_learner_seconds=overlap,learner_seconds=sum(b-a for a,b in ls),
                    worker_seconds=self.worker,counts=self.counts)

class MeasuredCollector(NativeCollector):
    def __init__(self,*args,meter,profile,priority=False):
        super().__init__(*args);self.meter=meter;self.profile=profile;self.service_stream=torch.cuda.Stream(priority=-1) if priority else None
    def receive(self,group):
        if not self.profile:
            t=clock();s=super().receive(group);self.meter.add('worker_wait_and_copy',t,clock());return s
        segs=[]
        for i in group:
            t=clock()
            if not self.pipes[i].poll(120):raise TimeoutError(f'Worker {i} timed out')
            msg=self.pipes[i].recv();end=clock()
            if msg[0]=='error':raise RuntimeError(msg[1])
            n,ns,nr,matches,live,frozen,pending,spans,parts,counts=msg
            for s,e,k in spans:
                self.meter.worker[k]+=max(0.,e-max(s,getattr(self.meter,'begin',s)))
            self.meter.counts.update(counts)
            cur=t
            for s,e,k in spans:
                s=max(cur,s,t);e=min(end,e)
                if e<=s:continue
                self.meter.add('ipc_wait',cur,s);self.meter.add(k,s,e);cur=e
            self.meter.add('ipc_wait',cur,end)
            self.live_steps+=live;self.frozen_steps+=frozen;self.pending[i]=pending;self.ready[i]=n
            own=self.pages[i].drain(ns,nr)
            if any(self.pool.owners[s.slot_id]!=LIVE for s in own):raise RuntimeError('Frozen trajectory reached learner')
            self.delivered+=nr;self.matches+=matches;segs.extend(own)
            self.meter.add('drain',end,clock())
        return segs
    def dispatch(self,group,version):
        t=clock();before=self.seat_decisions
        if self.service_stream is None:super().dispatch(group,version)
        else:
            self.service_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.service_stream):super().dispatch(group,version)
            torch.cuda.current_stream().wait_stream(self.service_stream)
        end=clock()
        self.meter.add('inference_service',t,end);self.meter.waves.append((t,end,self.seat_decisions-before))

class GPUUtil:
    def __enter__(self):
        import pynvml as nv
        nv.nvmlInit();self.nv=nv;self.dev=nv.nvmlDeviceGetHandleByIndex(0);self.samples=[];self.stop=threading.Event()
        def poll():
            while not self.stop.is_set():
                self.samples.append((clock(),int(nv.nvmlDeviceGetUtilizationRates(self.dev).gpu)));self.stop.wait(.5)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
    def __exit__(self,*_):
        self.stop.set();self.thread.join();self.nv.nvmlShutdown()

def run(a):
    if not Path('/sap/runs/train-live-20261123/STOPPED').exists():raise RuntimeError('Operator has not released GPU')
    if min(a.target,a.updates,a.games,a.cap,a.checkpoint_every)<1 or a.backlog<=0 or a.workers<2:raise ValueError('Invalid configuration')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    _,mem=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(float(os.environ.get('SAP_GPU_GIB','8'))*2**30/mem)
    info=verify_engine();parent=Path('/sap/checkpoints/update-025772.pt')
    policy,state,rates,sem,start,rows,primitives=prepare(parent,a.resume,'parent',info,a.seed)
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'args.json').write_text(json.dumps({**vars(a),"geometry":os.environ.get("SPEED_GEOMETRY","sorted"),"fused_loss":os.environ.get("SPEED_FUSED_LOSS","0")},default=str,indent=2))
    version=start;highwater=0;backpressure=0;queue=Queue();meter=Meter();learned=0;events=[];future=None
    with actors(a.workers,a.games,policy.config.memory_dim,a.seed,start,a.profile,a.battle_backend,a.out if a.audit_battles else None) as (pages,pipes), ThreadPoolExecutor(1) as ex, GPUUtil() as util:
        torch.cuda.manual_seed(start+a.seed);policy.cuda().train();opt=optimizer_for(policy,state,rates,1);del state
        actor=copy.deepcopy(policy).eval().requires_grad_(False)
        if a.fast_actor:
            from fast_actor import enable
            enable(actor)
            torch.distributions.Distribution.set_default_validate_args(False)
        pool=Opponents(actor,parent,.5,4,len(pages)*pages[0].slots,a.seed,start,info,a.resume)
        col=MeasuredCollector(pages,pipes,actor,policy.config.memory_dim,'cuda',pool,meter=meter,profile=a.profile,priority=a.priority_inference)
        stream=torch.cuda.Stream()
        if os.environ.get('SPEED_FUSED_LOSS')=='1':
            from learner_ops import enable_learner
            enable_learner(policy)
        cap=a.cap
        def learn(batch):
            nonlocal cap
            t=clock()
            with torch.cuda.stream(stream):
                if a.learner=='old':m=reference_update(policy,opt,batch,sem,'cuda',128)
                else:
                    m=update_fast(policy,opt,batch,sem,'cuda',cap,compact=a.compact,retry_oom=True)
                    cap=m['effective_cap']
            stream.synchronize()
            return m,t,clock()
        def checkpoint(path):
            from dataclasses import asdict
            from runtime import SOURCE_OF_TRUTH,OBSERVATION_REVISION,PARENT_SHA
            from sapai_gym1.rl.causal_transport import checkpoint_context_metadata
            record=dict(model=policy.state_dict(),optimizer=opt.state_dict(),config=asdict(policy.config),update=version,
                transitions=rows,learner_primitive_actions=primitives,simulator_revision=SIMULATOR_REVISION,
                engine_payload_sha256=info['payload_sha256'],source_of_truth=SOURCE_OF_TRUTH,observation_revision=OBSERVATION_REVISION,
                causal_context=checkpoint_context_metadata(policy.config),training_semantics=sem,wide_arm='parent',parent_checkpoint_sha256=PARENT_SHA,
                rust_library_sha256='d8cd765d898fb98d879dc7f89c77b32c705efbb10166588dacc401607d7fb58f',collector='two-cohort-single-flight-v31g',actor_sampling='stochastic policy.act, no search',native_rust_scorer=False,
                battle_backend=a.battle_backend,learner=a.learner,concurrent=a.concurrent,physical_rows=cap,reference_rows=128,
                max_behavior_lag=a.max_lag,reference_frozen_share=.5)
            temp=path.with_suffix('.pt.tmp');torch.save(record,temp);temp.replace(path)
        def publish():
            nonlocal version,rows,primitives,learned,future
            t=clock();m,ls,le=future.result();meter.add('await_learner',t,clock());meter.learner.append((ls,le))
            t=clock();actor.load_state_dict(policy.state_dict());torch.cuda.current_stream().synchronize()
            version+=1;rows+=m['decisions'];learned+=m['decisions'];primitives+=m['primitive_actions']
            if (version-start)%a.checkpoint_every==0:
                f=a.out/f'update-{version:06d}.pt'
                if a.save:checkpoint(f)
                pool.snapshot(version,f)
            meter.add('publish',t,clock())
            r=dict(version=version,learner_s=le-ls,elapsed=clock()-begin,stale=queue.stale,
                   delivered=col.delivered,seat=col.seat_decisions,behavior_min=lo,behavior_max=hi,**m)
            events.append(r)
            with (a.out/'updates.jsonl').open('a') as fp:fp.write(json.dumps(r)+'\n')
            future=None
        # Exclude spawn/load from throughput; all games still start on turn one.
        for g in col.groups:assert not col.receive(g)
        meter.events.clear();begin=clock();meter.begin=begin
        for g in col.groups:col.dispatch(g,version)
        while version-start<a.updates:
            for g in col.groups:
                queue.add(col.receive(g))
                highwater=max(highwater,queue.rows)
                if future is not None and (future.done() or queue.rows>=a.target*a.backlog):
                    backpressure+=int(not future.done())
                    publish()
                if future is None and version-start<a.updates and clock()-begin<a.seconds:
                    selection=queue.pop(version,a.target,a.max_lag)
                    if selection is not None:
                        batch,lo,hi=selection
                        if a.capture_at and version-start+1==a.capture_at:
                            from dataclasses import fields
                            def own(x):
                                if isinstance(x,dict):return {k:own(v) for k,v in x.items()}
                                return x.clone() if isinstance(x,torch.Tensor) else x
                            saved=[{f.name:own(getattr(s,f.name)) for f in fields(s)} for s in batch]
                            torch.save(dict(turns=saved,engine=info['payload_sha256'],behavior_min=lo,behavior_max=hi),a.out/'batch.pt')
                            del saved
                        if a.fingerprints and version-start<8:
                            with (a.out/'segments.jsonl').open('a') as fp:
                                sig=[]
                                for s in batch:
                                    h=hashlib.blake2b(digest_size=12)
                                    for t in (s.actions,s.freeze_masks,s.behavior_versions,s.initial_memory,s.behavior_log_probs,
                                              s.rewards,s.observations['numeric'][0]):h.update(t.numpy().tobytes())
                                    sig.append([s.slot_id,s.length,h.hexdigest()])
                                fp.write(json.dumps([version,sig],separators=(',',':'))+'\n')
                        stream.wait_stream(torch.cuda.current_stream())
                        future=ex.submit(learn,batch)
                        if not a.concurrent:publish()
                col.dispatch(g,version)
                if version-start>=a.updates or clock()-begin>=a.seconds:break
            if clock()-begin>=a.seconds:break
        if future is not None:publish()
        for g in col.groups:queue.add(col.receive(g))
        end=clock();stats=meter.result(begin,end)
        if learned+queue.stale+queue.rows!=col.delivered or col.delivered+sum(col.pending.values())!=col.live_steps:raise RuntimeError('Live decision accounting')
        if col.seat_decisions!=col.frozen_steps+col.live_steps:raise RuntimeError('Lost seat decisions')
        tail=a.out/f'update-{version:06d}.pt'
        if a.save and version>start and not tail.exists():checkpoint(tail)
        gpu=[v for t,v in util.samples if begin<=t<=end]
        result=dict(version=version,engine=info['payload_sha256'],updates=len(events),elapsed=end-begin,learned=learned,delivered=col.delivered,
            seat=col.seat_decisions,live=col.live_steps,frozen=col.frozen_steps,pending=sum(col.pending.values()),queued=queue.rows,stale=queue.stale,
            matches=col.matches,queue_highwater=highwater,backpressure_hits=backpressure,assignment=pool.assignments,gpu_util_mean=float(np.mean(gpu)) if gpu else None,gpu_samples=len(gpu),
            peak_bytes=torch.cuda.max_memory_reserved(),learned_s=learned/(end-begin),collected_s=col.delivered/(end-begin),
            seat_s=col.seat_decisions/(end-begin),stale_share=queue.stale/max(1,learned+queue.stale),**stats)
        (a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--resume',type=Path);p.add_argument('--checkpoint-every',type=int,default=250)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--games',type=int,default=96);p.add_argument('--target',type=int,default=8192)
    p.add_argument('--priority-inference',action='store_true');p.add_argument('--capture-at',type=int,default=0);p.add_argument('--fingerprints',action='store_true')
    p.add_argument('--compact',action='store_true');p.add_argument('--fast-actor',action='store_true')
    p.add_argument('--battle-backend',choices=['python','rust'],default='python');p.add_argument('--audit-battles',action='store_true')
    p.add_argument('--concurrent',action='store_true');p.add_argument('--learner',choices=['old','fast'],default='fast')
    p.add_argument('--cap',type=int,default=512);p.add_argument('--updates',type=int,required=True);p.add_argument('--max-lag',type=int,default=1)
    p.add_argument('--backlog',type=float,default=1.);p.add_argument('--seconds',type=float,default=810);p.add_argument('--seed',type=int,default=2577223)
    p.add_argument('--save',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
