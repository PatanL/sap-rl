"""Collect complete recurrent turns, update the learner and publish new actor weights."""
import argparse,collections,copy,json,os,time
from pathlib import Path
import torch
from runtime import verify_engine,SIMULATOR_REVISION,OBSERVATION_REVISION,PARENT_SHA
from wide import Collector,actors
from learner import prepare,optimizer_for,update,write_checkpoint,weight_hash,max_segment_lag

def save(path,record):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(path)

def train(args):
    info=verify_engine()
    dry=args.mode=='dry'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode!='inspect':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        free,total=torch.cuda.mem_get_info()
        limit=float(os.environ.get('SAP_GPU_GIB','1.5'))*1024**3
        torch.cuda.set_per_process_memory_fraction(min(1.,limit/total))
        print(f'CUDA free at startup {free}, allocator cap {int(limit)}',flush=True)
    policy,state,rates,semantics,version,rows,primitives=prepare(args.parent,args.resume,args.arm,info,args.seed)
    params=sum(p.numel() for p in policy.parameters())
    if params!={'parent':14346925,'fresh':1349741}[args.arm]:raise ValueError('Unexpected policy architecture')
    if args.mode=='inspect':
        opt=optimizer_for(policy,state,rates,1)
        if args.verify_save:
            import tempfile
            with tempfile.TemporaryDirectory(dir='/work/tmp') as scratch:
                ck=Path(scratch)/'untrained-roundtrip.pt'
                write_checkpoint(ck,policy,opt,args.arm,info,semantics,version,rows,primitives)
                restored,ostate,rr,sem,v,r,p=prepare(args.parent,ck,args.arm,info,args.seed)
                opt2=optimizer_for(restored,ostate,rr,1.)
                assert (v,r,p)==(version,rows,primitives) and weight_hash(restored)==weight_hash(policy)
                chk=opt.state_dict();other=opt2.state_dict();assert chk['param_groups']==other['param_groups']
                assert chk['state'].keys()==other['state'].keys()
                for i,st in chk['state'].items():
                    for k,val in st.items():
                        assert torch.equal(val,other['state'][i][k]) if isinstance(val,torch.Tensor) else val==other['state'][i][k]
                print('CPU checkpoint round trip: weights, Adam state, counters and v31 identity verified')
        print(json.dumps(dict(arm=args.arm,revision=SIMULATOR_REVISION,parameters=params,update=version,transitions=rows,
                              optimizer_states=len(opt.state),rates=rates,engine=info['payload_sha256'])))
        return
    if args.workers<2 or args.games<1 or args.transitions<1 or args.max_padded_rows<1:raise ValueError('Invalid collector or learner budget')
    if args.updates<1 or (dry and args.updates>2):raise ValueError('Dry run supports at most two updates')
    if args.lr_scale<=0 or args.checkpoint_every<1 or args.max_lag<0:raise ValueError('Invalid optimizer or checkpoint setting')
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'RUN.json').open('w') as f:
        json.dump(dict(mode=args.mode,arm=args.arm,policy_parameters=params,parent_checkpoint_sha256=PARENT_SHA,
            transferred_weights_and_optimizer=(args.arm=='parent' or args.resume is not None),
            initialized_from=str(args.resume or args.parent) if (args.arm=='parent' or args.resume) else None,
            simulator_revision=SIMULATOR_REVISION,observation_revision=OBSERVATION_REVISION,engine_sha256=info['payload_sha256'],
            training_semantics=semantics,battle_backend=info['battle_backend'],policy_rates={k:(0 if dry else v*args.lr_scale) for k,v in rates.items()},
            start_version=version,cohorts=2,workers=args.workers,games_per_worker=args.games,seed=args.seed,updates=args.updates,
            decision_target=args.transitions,microbatch_padded_rows=args.max_padded_rows,max_lag=args.max_lag,opponents='live both seats'),f,indent=2)
    start=version;clock=time.monotonic();buffer=collections.deque();accepted=stale=0;events=[];sampled_after=0
    with actors(args.workers,args.games,policy.config.memory_dim,args.seed,version) as (pages,pipes):
        device='cuda'
        torch.cuda.manual_seed(args.seed+version)
        policy.to(device).train();opt=optimizer_for(policy,state,rates,0 if dry else args.lr_scale);del state
        actor=copy.deepcopy(policy).eval().requires_grad_(False)
        col=Collector(pages,pipes,actor,policy.config.memory_dim,device)
        for group in col.groups:
            assert not col.receive(group)
            col.dispatch(group,version)
        pre_hash=weight_hash(policy) if dry else None
        while version<start+args.updates or (dry and not sampled_after):
            for group in col.groups:
                segs=col.receive(group)
                if version==start+args.updates:
                    # Prove the new version actually produced a completed learner-format turn.
                    sampled_after+=sum(int((s.behavior_versions==version).sum()) for s in segs)
                buffer.extend(segs)
                batch=[];n=0
                if version<start+args.updates and sum(s.length for s in buffer)>=args.transitions:
                    while buffer and n<args.transitions:
                        s=buffer.popleft()
                        if s.behavior_versions.max().item()>version:raise RuntimeError('Future policy version')
                        if max_segment_lag(s,version)>args.max_lag:stale+=s.length;continue
                        n+=s.length;batch.append(s)
                    if n<args.transitions:
                        buffer.extendleft(reversed(batch));batch=[]
                if batch:
                    versions=torch.cat([s.behavior_versions for s in batch])
                    before_steps=[float(st['step']) for st in opt.state.values() if 'step' in st]
                    metrics=update(policy,opt,batch,semantics,device,args.max_padded_rows)
                    accepted+=n;rows+=n;primitives+=metrics['primitive_actions'];version+=1
                    actor.load_state_dict(policy.state_dict(),strict=True)
                    event=dict(version=version,behavior_min=int(versions.min()),behavior_max=int(versions.max()),
                               adam_steps_min_before=min(before_steps,default=0),adam_steps_max_before=max(before_steps,default=0),
                               adam_steps_max_after=max(float(st['step']) for st in opt.state.values() if 'step' in st),
                               stale=stale,elapsed_s=time.monotonic()-clock,**metrics)
                    with (args.output/'updates.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                    events.append(event)
                    if not dry:events=events[-1:]
                    if not dry and ((version-start)%args.checkpoint_every==0 or version==start+args.updates):
                        write_checkpoint(args.output/f'update-{version:06d}.pt',policy,opt,args.arm,info,semantics,version,rows,primitives)
                col.dispatch(group,version)
                if version==start+args.updates and (not dry or sampled_after):break
        # Drain each outstanding wave once; no new inference after this point.
        for group in col.groups:buffer.extend(col.receive(group))
        unused=sum(s.length for s in buffer)
        if accepted+stale+unused!=col.delivered:raise RuntimeError('Learner delivery accounting mismatch')
        end_hash=weight_hash(policy) if dry else None
        if dry and (end_hash!=pre_hash or weight_hash(actor)!=end_hash):raise RuntimeError('Zero-LR parameter preservation failed')
        save(args.output/'RESULT.json',dict(arm=args.arm,mode=args.mode,simulator_revision=SIMULATOR_REVISION,engine_sha256=info['payload_sha256'],
            updates=version-start,final_version=version,trained_rows=accepted,unused_delivered=unused,discarded_stale=stale,
            delivered=col.delivered,seat_decisions=col.seat_decisions,complete_matches_collected=col.matches,parameters=params,
            new_version_rows_observed_after_last_step=sampled_after,before_weights=pre_hash,after_weights=end_hash,
            elapsed_s=time.monotonic()-clock,peak_gpu_reserved=torch.cuda.max_memory_reserved(),events=events))
        print(json.dumps(events[-1]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['inspect','dry','train']);p.add_argument('--arm',choices=['parent','fresh'],required=True)
    p.add_argument('--verify-save',action='store_true');p.add_argument('--parent',type=Path,default=Path('/weights/parent.pt'));p.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=6);p.add_argument('--games',type=int,default=96)
    p.add_argument('--seed',type=int,default=2577223);p.add_argument('--updates',type=int,default=1)
    p.add_argument('--transitions',type=int,default=4096);p.add_argument('--max-padded-rows',type=int,default=128)
    p.add_argument('--max-lag',type=int,default=1);p.add_argument('--lr-scale',type=float,default=1.);p.add_argument('--checkpoint-every',type=int,default=64)
    a=p.parse_args()
    if a.mode!='inspect' and a.output is None:p.error('--output is required')
    if a.mode=="dry":
        import signal
        signal.alarm(100)
    train(a)
