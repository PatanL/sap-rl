"""Frozen-policy shared-page collector benchmark. No optimizer or training."""
import argparse,ctypes,gc,json,multiprocessing as mp,os,time,traceback
from collections import Counter
from multiprocessing.connection import wait
import numpy as np
from common import *
from sapai_gym1.rl import selfplay_env as E
from sapai_gym1.rl.entity_observation import EntityObservationEncoder
from sapai_gym1.rl.end_freeze import step_with_end_freeze
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.dense_trajectory import OwnedTrajectoryBatch
clock=time.perf_counter

def alloc(ctx,shape,dtype):
    dtype=np.dtype(dtype); buf=ctx.RawArray(ctypes.c_ubyte,int(np.prod(shape))*dtype.itemsize)
    return np.frombuffer(buf,dtype).reshape(shape)

class Page:
    def __init__(self,ctx,g,o,h):
        self.slots=2*g; self.capacity=2*g*52
        a=lambda shape,dtype:alloc(ctx,shape,dtype)
        store=lambda n:{k:a((n,*v.shape),v.dtype) for k,v in o.items()}
        self.obs=store(self.slots);self.req=a((self.slots,3),'i4') # local slot, policy id, reset
        self.actions=a((self.slots,4),'i8');self.logp=a((self.slots,),'f4');self.mem=a((self.slots,h),'f4')
        self.traj=store(self.capacity);self.boot=store(self.slots);self.initial=a((self.slots,h),'f4')
        self.meta=a((self.slots,4),'i4') # length, terminal, battle label, local slot
        self.moves=a((self.capacity,4),'i8');self.vals=a((self.capacity,3),'f4') # behavior lp, reward, discount

def allocate_games(w,g,h):
    envs=[ParallelSelfPlayEnv() for _ in range(g)]; obs=[]; owners=[]; gens=np.zeros(g,'i8')
    # Independent env & opponent streams, identical between layouts with the same index.
    def reset(i):
        index=w*g+i; generation=int(gens[i]);gens[i]+=1
        seed=722110000+index*32768+generation
        # Balanced 50% mirror / 50% three frozen residents by slot and epoch.
        owners[i]=(0,1,0,2,0,3)[(index+generation)%6]
        return envs[i].reset(seed=seed,paired_tape=True)
    owners[:]=[0]*g
    for i in range(g):obs.append(reset(i))
    return envs,obs,owners,reset

def install_timers(spans=None):
    now=[0.,0.]
    orig=EntityObservationEncoder.encode
    def enc(*aa,**kk):
        t=clock();out=orig(*aa,**kk);end=clock();now[0]+=end-t
        if spans is not None:spans['obs'].append((t,end))
        return out
    EntityObservationEncoder.encode=enc
    B=E.Battle
    class TimedBattle(B):
        def __init__(self,*aa,**kk):
            t=clock();super().__init__(*aa,**kk);now[1]+=clock()-t
        def battle(self):
            t=clock();out=super().battle();now[1]+=clock()-t;return out
    E.Battle=TimedBattle
    return now

def worker(w,g,h,page,conn,trace_time=False):
    try:
        torch.set_num_threads(1)
        spans={'engine':[],'obs':[]} if trace_time else None
        times=install_timers(spans);envs,obs,owners,reset=allocate_games(w,g,h)
        new=np.ones((g,2),'i4'); mem=np.zeros((g,2,h),'f4'); initial=mem.copy()
        histories=[[[],[]] for _ in envs]
        stats={};nseg=nrow=0;pending=0
        while True:
            t=clock();active=[]
            for i,env in enumerate(envs):
                for s,yes in enumerate(env.active_players()):
                    if yes:active.append((i,s))
            n=len(active); slots=[obs[i][s] for i,s in active]
            for key,arr in page.obs.items():np.stack([o[key] for o in slots],out=arr[:n])
            page.req[:n]=[(i*2+s,owners[i] if s else 0,new[i,s]) for i,s in active]
            stats['request_pack']=clock()-t
            if spans is not None:stats['spans']=spans
            stats['ready_at']=clock();conn.send((n,nseg,nrow,pending,stats))
            wait_start=clock();cmd=conn.recv();wait_secs=clock()-wait_start
            if cmd=='stop':break
            begin=clock()
            if spans is not None:spans['engine'].clear();spans['obs'].clear()
            is_training=0;matches=battles=0;scorehash=0;engine=0.;o0,b0=times
            nseg=nrow=0;pairs=[[None,None] for _ in envs];masks=[[0,0] for _ in envs]
            for j,(i,s) in enumerate(active):
                a=page.actions[j]
                pairs[i][s]=AutoregressiveAction(int(a[0]),int(a[1]),int(a[2]));masks[i][s]=int(a[3])
                if s==0 or owners[i]==0:
                    histories[i][s].append((obs[i][s],a.copy(),page.logp[j]));is_training+=1;pending+=1
                mem[i,s]=page.mem[j];new[i,s]=0
            for i,env in enumerate(envs):
                t=clock();step=step_with_end_freeze(env,*pairs[i],mask_ids=masks[i]).result;end=clock();engine+=end-t
                if spans is not None:spans['engine'].append((t,end))
                obs[i]=step.observations
                if not step.battle_resolved:continue
                battles+=1
                for s,buf in enumerate(histories[i]):
                    if not buf:continue
                    length=len(buf);assert length<=52 and buf[-1][1][0]==0
                    section=slice(nrow,nrow+length);assert nrow+length<=page.capacity
                    for key,arr in page.traj.items():
                        np.stack([r[0][key] for r in buf],out=arr[section]);page.boot[key][nseg]=obs[i][s][key]
                    np.stack([r[1] for r in buf],out=page.moves[section])
                    page.vals[section,0]=[r[2] for r in buf]
                    page.vals[section,1]=0;page.vals[section,2]=1
                    page.vals[nrow+length-1,1]=step.rewards[s];page.vals[nrow+length-1,2]=0. if step.terminated else .99
                    label=2 if step.battle_result==2 else int(step.battle_result!=s)
                    page.meta[nseg]=(length,int(step.terminated),label,i*2+s)
                    page.initial[nseg]=initial[i,s]
                    nseg+=1;nrow+=length;pending-=length;buf.clear()
                initial[i]=mem[i]
                if step.terminated:
                    matches+=1;scorehash+=(i+1)*(1+_completed_match_result(env.core))
                    t=clock();obs[i]=reset(i);end=clock();engine+=end-t
                    if spans is not None:spans['engine'].append((t,end))
                    mem[i]=0;initial[i]=0;new[i]=1
            stats=dict(observation=times[0]-o0,battle=times[1]-b0,engine=engine-(times[0]-o0),
                       active=n,eligible=is_training,matches=matches,battles=battles,scorehash=scorehash,
                       work=clock()-begin,wait=wait_secs)
    except BaseException:
        traceback.print_exc();conn.close();raise

def drain(page,ns,nr):
    if ns==0:
        assert nr==0;return None
    # Own all learner inputs before releasing this page. One copy per field per worker.
    arr=lambda x:torch.from_numpy(x.copy())
    val=page.vals[:nr];meta=page.meta[:ns];mov=page.moves[:nr]
    lengths=tuple(map(int,meta[:,0]));assert sum(lengths)==nr
    return OwnedTrajectoryBatch({k:arr(v[:nr]) for k,v in page.traj.items()},arr(mov[:,:3]),arr(val[:,0]),
        torch.zeros(nr,dtype=torch.long),arr(val[:,1]),arr(val[:,2]),arr(mov[:,3].astype(np.int16)),lengths,
        arr(page.initial[:ns]),{k:arr(v[:ns]) for k,v in page.boot.items()},
        arr(meta[:,1].astype(bool)),arr(meta[:,2].astype(np.int64)),tuple(map(int,meta[:,3])))

def serve(pages,ready,models,bank,worker_ids=None):
    t=clock();N=sum(r[0] for r in ready)
    req=np.concatenate([p.req[:r[0]] for p,r in zip(pages,ready)])
    offset=np.repeat([i*p.slots for i,p in zip(range(len(pages)) if worker_ids is None else worker_ids,pages)],[r[0] for r in ready]);slots=req[:,0]+offset
    order=np.argsort(req[:,1],kind='stable');reverse=np.argsort(order)
    ids=req[order,1];bounds=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,N]
    obs={k:torch.from_numpy(np.concatenate([p.obs[k][:r[0]] for p,r in zip(pages,ready)])[order]).to('cuda') for k in pages[0].obs}
    address=torch.from_numpy(slots[order]).cuda()
    memory=bank.index_select(0,address)
    memory.masked_fill_(torch.from_numpy(req[order,2].astype(bool)).cuda()[:,None],0.)
    torch.cuda.synchronize(); t1=clock()
    acts=[];lp=[];mem=[];sizes=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for a,b in zip(bounds,bounds[1:]):
            o=models[ids[a]].act({k:x[a:b] for k,x in obs.items()},memory=memory[a:b])
            acts.append(torch.cat((o.actions,o.end_freeze_mask[:,None]),1));lp.append(o.log_prob.float());mem.append(o.memory.float())
            sizes.append(int(b-a))
    torch.cuda.synchronize();t2=clock()
    with torch.no_grad():
        nxt=torch.cat(mem);bank.index_copy_(0,address,nxt)
        actions=torch.cat(acts).cpu().numpy()[reverse];logp=torch.cat(lp).cpu().numpy()[reverse];memout=nxt.cpu().numpy()[reverse]
    j=0
    for p,r in zip(pages,ready):
        n=r[0]; p.actions[:n]=actions[j:j+n];p.logp[:n]=logp[j:j+n];p.mem[:n]=memout[j:j+n];j+=n
    return sizes,t2-t1,(t1-t)+(clock()-t2),actions

def summary(stats,ticks,wall):
    tot=lambda k:sum(row[k] for row in stats)
    return dict(ticks=ticks,seconds=wall,matches=int(tot('matches')),battles=int(tot('battles')),
                seat=int(tot('active')),eligible=int(tot('eligible')),accepted=int(tot('accepted')),
                matches_s=tot('matches')/wall,seat_s=tot('active')/wall,eligible_s=tot('eligible')/wall,
                accepted_s=tot('accepted')/wall,forward_s=tot('forward'),routing_s=tot('routing'),drain_s=tot('drain'),
                critical_engine_s=tot('critical_engine'),critical_battle_s=tot('critical_battle'),critical_obs_s=tot('critical_obs'),
                critical_remaining_s=wall-tot('forward')-tot('critical_engine')-tot('critical_obs'),
                coordinator_wait_s=tot('coordinator_wait'),worker_engine_s=tot('worker_engine'),worker_obs_s=tot('worker_obs'),
                worker_work_s=tot('worker_work'),mean_policy_batch=tot('active')/tot('calls'),
                mean_items_per_wave=tot('active')/ticks,mean_calls_per_wave=tot('calls')/ticks)

def bench(args):
    ctx=mp.get_context('fork')
    torch.set_num_threads(1)
    sample=ParallelSelfPlayEnv().reset(seed=401,paired_tape=True)[0]
    h=SIZES[args.model][3]
    pages=[Page(ctx,args.games,sample,h) for _ in range(args.workers)]
    conns=[];procs=[]
    for i,p in enumerate(pages):
        local,remote=ctx.Pipe()
        proc=ctx.Process(target=worker,args=(i,args.games,h,p,remote));proc.start();remote.close()
        conns.append(local);procs.append(proc)
    try:
        setup_gpu()
        torch.manual_seed(9019)
        models=[model_for(args.model,seed=742+i,random_parent=args.random_parent or i!=0).eval().requires_grad_(False).cuda() for i in range(4)]
        if args.model=='parent' and not args.random_parent:
            for i,n in enumerate(('terminal-response','strategy'),2):
                ck=torch.load(Path('/residents')/(n+'.pt'),map_location='cpu',weights_only=True,mmap=True)
                models[i].load_state_dict(ck['model']);del ck
            models[1].load_state_dict(models[0].state_dict())
        torch.manual_seed(9019)
        bank=torch.zeros(args.workers*args.games*2,h,device='cuda')
        ready=[c.recv() for c in conns]
        warm=0;ticks=0;stats=[];sizes=[];tape=[];tapen=[];events=[];seen_example=False
        warm_matches=0;start=None;pending0=None;sample_batches=[];sample_rows=0;sizes_by_policy=[[] for _ in models]
        while True:
            if start is None and ticks>=args.warm_ticks and warm_matches>=args.warm_factor*args.workers*args.games:
                warm=ticks;start=clock();pending0=sum(r[3] for r in ready);print('measurement starts',warm,warm_matches,flush=True)
            wave=clock()
            batch,fw,rt,acts=serve(pages,ready,models,bank)
            if args.tape:
                tape.append(acts.astype(np.int16));tapen.append([r[0] for r in ready])
            if start is not None:
                group_ids=sorted(set(int(x) for page,r in zip(pages,ready) for x in page.req[:r[0],1]))
                for k,v in zip(group_ids,batch):sizes_by_policy[k].append(v)
            sent=clock()
            for c in conns:c.send('go')
            ready=[c.recv() for c in conns];received=clock()
            metrics=[r[4] for r in ready]
            critical=max(metrics,key=lambda m:m['ready_at'])
            total=lambda k:sum(m[k] for m in metrics)
            nrows=sum(r[2] for r in ready)
            for page,r in zip(pages,ready):
                owned=drain(page,r[1],r[2])
                if owned is not None and start is not None and sample_rows<4096:
                    sample_rows+=r[2];sample_batches.append(owned)
                if owned is not None and not seen_example:
                    ss=owned.segments();assert sum(s.length for s in ss)==r[2]
                    for s in ss:
                        assert s.actions[-1,0].item()==0 and torch.isfinite(s.behavior_log_probs).all()
                        assert set(s.observations)==set(sample) and s.discounts[-1].item()==(0 if s.terminal else np.float32(.99))
                    seen_example=True
            drained=clock();ticks+=1
            events.append([int(total(k)) for k in ('active','eligible','matches','battles','scorehash')])
            if start is None:warm_matches+=total('matches')
            else:
                sizes.extend(batch)
                stats.append(dict(wave_wall=drained-wave,forward=fw,routing=rt,coordinator_wait=received-sent,drain=drained-received,
                              critical_engine=critical['engine'],critical_battle=critical['battle'],critical_obs=critical['observation'],
                              worker_engine=total('engine'),worker_obs=total('observation'),worker_work=total('work'),
                              matches=total('matches'),battles=total('battles'),active=total('active'),eligible=total('eligible'),
                              accepted=nrows,calls=len(batch)))
                if drained-start>=args.seconds and len(stats)>=args.min_ticks:break
            if ticks>=2500:raise RuntimeError('Warmup/measurement failed to finish')
        wall=drained-start
        result=summary(stats,ticks-warm,wall)
        pending1=sum(r[3] for r in ready);assert result['eligible']-result['accepted']==pending1-pending0
        result.update(model=args.model,parameters=sum(p.numel() for p in models[0].parameters()),random_parent=args.random_parent,
                      workers=args.workers,gamesPerActor=args.games,warm_ticks=warm,warm_matches=warm_matches,
                      pendingAtStart=pending0,pendingAtEnd=pending1,precision='bf16',revision=SIMULATOR_REVISION,
                      workerAffinities=list(os.sched_getaffinity(0)),batchPercentiles=np.percentile(sizes,[10,50,90]).tolist(),
                      policyBatchMeans=[np.mean(v) if v else None for v in sizes_by_policy],peakReserved=torch.cuda.max_memory_reserved(),
                      blocks=[summary(x,len(x),sum(s['wave_wall'] for s in x))
                              for x in [stats[:len(stats)//2],stats[len(stats)//2:]]] if len(stats)>1 else [])
        save(args.out+'.json',result);print(json.dumps(result),flush=True)
        if sample_rows>=4096:
            from real_update import measure
            try:
                actual=measure(models[0],[s for b in sample_batches for s in b.segments()])
                save(args.out+'-update.json',actual)
            except Exception:
                traceback.print_exc()
        if args.tape:
            np.savez_compressed(args.out+'-tape.npz',actions=np.concatenate(tape),sizes=np.array(tapen),events=np.array(events),
                                warm=warm,workers=args.workers,games=args.games)
    finally:
        for c in conns:
            try:c.send('stop')
            except (BrokenPipeError,EOFError):pass
        for proc in procs:
            proc.join(4)
            if proc.is_alive():proc.terminate();proc.join()
        for c in conns:c.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(SIZES),required=True);p.add_argument('--workers',type=int,required=True)
    p.add_argument('--games',type=int,required=True);p.add_argument('--seconds',type=float,default=60);p.add_argument('--min-ticks',type=int,default=8)
    p.add_argument('--warm-ticks',type=int,default=16);p.add_argument('--warm-factor',type=float,default=1.)
    p.add_argument('--tape',action='store_true');p.add_argument('--random-parent',action='store_true');p.add_argument('--out',required=True)
    bench(p.parse_args())
