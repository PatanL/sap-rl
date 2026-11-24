"""Overlap fixed actor cohorts with GPU inference. Each cohort still receives a wide policy batch."""
import argparse,copy,json,multiprocessing as mp,time
from dataclasses import replace
import numpy as np
import torch
from common import SIZES,ParallelSelfPlayEnv,model_for,setup_gpu,save,SIMULATOR_REVISION
from collector import Page,worker,drain,serve
from pathlib import Path
clock=time.perf_counter
def intersect(spans,a,b):
    return sum(max(0.,min(e,b)-max(s,a)) for s,e in spans)
def run(a):
    ctx=mp.get_context('fork');torch.set_num_threads(1)
    sample=ParallelSelfPlayEnv().reset(seed=401,paired_tape=True)[0]
    h=SIZES[a.model][3]
    pages=[Page(ctx,a.games,sample,h) for _ in range(a.workers)]
    assert a.workers%a.cohorts==0
    groups=np.arange(a.workers).reshape(a.cohorts,-1).tolist()
    conns=[];procs=[]
    for w,page in enumerate(pages):
        local,remote=ctx.Pipe();pr=ctx.Process(target=worker,args=(w,a.games,h,page,remote,True));pr.start();remote.close()
        conns.append(local);procs.append(pr)
    try:
        setup_gpu()
        models=[model_for(a.model,seed=742+i,random_parent=a.random_parent or i!=0).eval().requires_grad_(False).cuda() for i in range(4)]
        if a.model=='parent' and not a.random_parent:
            models[1].load_state_dict(models[0].state_dict())
            for i,n in enumerate(('terminal-response','strategy'),2):
                ck=torch.load(Path('/residents')/(n+'.pt'),map_location='cpu',weights_only=True,mmap=True)
                models[i].load_state_dict(ck['model']);del ck
        torch.manual_seed(a.sampling_seed)
        bank=torch.zeros(a.workers*a.games*2,h,device='cuda')
        ready=[c.recv() for c in conns]
        tapes=[[] for _ in groups];nsizes=[[] for _ in groups];events=[[] for _ in groups]
        actual_batches=[];sample_rows=0;example=False
        def phase(warm):
            nonlocal sample_rows,example
            keys=('matches','battles','eligible','seat','accepted','forward','route','drain','coordwait',
                  'critengine','critobs','workengine','workobs','workerwait','workerwork','workerpack')
            total={k:0. for k in keys};batch_sizes=[];perpolicy=[[] for _ in models]
            stats0=sum(r[3] for r in ready);start=clock()
            def dispatch(c):
                ids=groups[c];ps=[pages[i] for i in ids];rs=[ready[i] for i in ids]
                sizes,forward,routing,actions=serve(ps,rs,models,bank,ids)
                batch_sizes.extend(sizes);total['forward']+=forward;total['route']+=routing
                owners=sorted(set(int(x) for page,row in zip(ps,rs) for x in page.req[:row[0],1]))
                for k,b in zip(owners,sizes):perpolicy[k].append(b)
                if a.tape:
                    tapes[c].append(actions.astype(np.int16));nsizes[c].append([r[0] for r in rs])
                for i in ids:conns[i].send('go')
            def receive(c):
                nonlocal sample_rows,example
                ids=groups[c];t=clock()
                for i in ids:ready[i]=conns[i].recv()
                end=clock();total['coordwait']+=end-t
                last=max((ready[i][4] for i in ids),key=lambda s:s['ready_at'])
                e=intersect(last['spans']['engine'],t,end);o=intersect(last['spans']['obs'],t,end)
                total['critengine']+=e-o;total['critobs']+=o
                m=[ready[i][4] for i in ids];sums=lambda key:sum(y[key] for y in m)
                keys={'matches':'matches','battles':'battles','eligible':'eligible','seat':'active','workengine':'engine','workobs':'observation',
                      'workerwait':'wait','workerwork':'work','workerpack':'request_pack'}
                for key,src in keys.items():total[key]+=sums(src)
                if a.tape:events[c].append([int(sums(k)) for k in ('active','eligible','matches','battles','scorehash')])
                t=clock()
                for i in ids:
                    _,segs,nrows,_,_=ready[i];total['accepted']+=nrows
                    owned=drain(pages[i],segs,nrows)
                    if owned is not None and not warm and sample_rows<4096:
                        actual_batches.append(owned);sample_rows+=nrows
                    if owned is not None and not example:
                        ss=owned.segments();assert sum(s.length for s in ss)==nrows
                        assert all(s.actions[-1,0].item()==0 and torch.isfinite(s.behavior_log_probs).all() for s in ss)
                        example=True
                total['drain']+=clock()-t
            for c in range(a.cohorts):dispatch(c)
            epochs=0
            while True:
                for c in range(a.cohorts):
                    receive(c);dispatch(c)
                epochs+=1
                if (warm and total['matches']>=a.workers*a.games*a.warm_factor and epochs>=a.warm_ticks) or (not warm and clock()-start>=a.seconds and epochs>=8):break
                if epochs>2500:raise RuntimeError('Phase exceeded horizon')
            for c in range(a.cohorts):receive(c)
            wall=clock()-start;epochs+=1
            assert total['eligible']-total['accepted']==sum(r[3] for r in ready)-stats0
            result=dict(seconds=wall,ticks=epochs,seat_s=total['seat']/wall,eligible_s=total['eligible']/wall,accepted_s=total['accepted']/wall,matches_s=total['matches']/wall,
                        critical_engine_s=total['critengine'],critical_obs_s=total['critobs'],forward_s=total['forward'],
                        critical_remaining_s=wall-total['critengine']-total['critobs']-total['forward'],
                        mean_policy_batch=total['seat']/len(batch_sizes),mean_items_per_wave=total['seat']/epochs/a.cohorts,
                        policyBatchMeans=[float(np.mean(v)) if v else None for v in perpolicy],batchPercentiles=np.percentile(batch_sizes,[10,50,90]).tolist(),
                        pendingStart=stats0,pendingEnd=sum(r[3] for r in ready),totals=total)
            return result
        warm=phase(True);print('warm',warm['ticks'],warm['totals']['matches'],flush=True)
        res=phase(False)
        res.update(model=a.model,random_parent=a.random_parent,workers=a.workers,gamesPerActor=a.games,cohorts=a.cohorts,warm_ticks=warm['ticks'],
                   sampling_seed=a.sampling_seed,parameters=sum(x.numel() for x in models[0].parameters()),peakReserved=torch.cuda.max_memory_reserved(),revision=SIMULATOR_REVISION)
        save(a.out+'.json',res);print(json.dumps(res),flush=True)
        if a.tape:
            steps=len(tapes[0]);assert all(len(x)==steps for x in tapes)
            np.savez_compressed(a.out+'-tape.npz',actions=np.concatenate([tapes[c][i] for i in range(steps) for c in range(len(groups))]),
                            sizes=np.concatenate([np.asarray(q) for q in nsizes],axis=1),
                            events=np.sum(events,axis=0),warm=warm['ticks'],workers=a.workers,games=a.games)
        if sample_rows>=4096:
            from real_update import measure
            segments=[s for batch in actual_batches for s in batch.segments()]
            save(a.out+'-update.json',measure(models[0],segments))
            if a.compare_learners:
                from real_update import native_default
                save(a.out+'-stock-update.json',native_default(models[0],segments))
                torch.cuda.empty_cache()
                results={}
                for size in SIZES:
                    env_model=model_for(size,seed=742,random_parent=True).eval().requires_grad_(False).cuda()
                    zero=torch.zeros(env_model.config.memory_dim)
                    common_turns=[replace(seg,initial_memory=zero) for seg in segments]
                    results[size]=measure(env_model,common_turns,loss_modes=(False,))[0]
                    save(a.out+'-same-turns.json',results)
                    del env_model
                    torch.cuda.empty_cache()
    finally:
        for c in conns:
            try:c.send('stop')
            except (BrokenPipeError,EOFError):pass
        for pr in procs:
            pr.join(3)
            if pr.is_alive():pr.terminate();pr.join()
        for c in conns:c.close()
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field,default in [('workers',8),('games',48),('cohorts',2),('warm-ticks',16),('sampling-seed',9019)]:
        p.add_argument('--'+field,type=int,default=default)
    p.add_argument('--warm-factor',type=float,default=1.);p.add_argument('--seconds',type=float,default=80)
    p.add_argument('--tape',action='store_true');p.add_argument('--model',choices=list(SIZES),required=True);p.add_argument('--out',required=True)
    p.add_argument('--random-parent',action='store_true');p.add_argument('--compare-learners',action='store_true')
    run(p.parse_args())
