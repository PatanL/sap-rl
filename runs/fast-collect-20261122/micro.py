import argparse,gc,time,statistics
from common import *
from sapai_gym1.rl.training import _learner_update_batch
from dataclasses import asdict

def timed(fn,reps,warm):
    for _ in range(warm):fn()
    torch.cuda.synchronize()
    samples=[]
    for _ in range(reps):
        start=time.perf_counter();fn();torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
    return samples

def row(ms,b):
    med=statistics.median(ms)
    return dict(ms=med,decisionsPerSecond=1000*b/med,rangeMs=[min(ms),max(ms)],samplesMs=ms)

def segments_for(b,turns,dim):
    length=min(b,8);eligible=[(x,a,z) for x,a,z in turns if len(a)>=length]
    segs=[]
    for j in range(b//length):
        x,a,z=eligible[j%len(eligible)]
        moves=[Transition(observation=o,action=np.asarray(d['primitive'],np.int64),end_freeze_mask=d['end_freeze_mask'],
                          behavior_log_prob=-1.,reward=0.,discount=1.) for o,d in zip(x[-length:],a[-length:])]
        moves[-1].reward=.1;moves[-1].discount=.99
        segs.append(TurnSegment(moves,torch.zeros(dim),z,False,j%3))
    return segs

def main():
    setup_gpu();turns=examples();obsrows=[o for x,a,z in turns for o in x];results=[]
    for name in SIZES:
        m=model_for(name).cuda()
        count=sum(p.numel() for p in m.parameters());m.eval()
        for precision in ('fp32','bf16'):
            for b in BATCHES:
                ix=np.random.default_rng(42+b).integers(len(obsrows),size=b)
                obs={k:torch.from_numpy(np.stack([obsrows[i][k] for i in ix])).cuda() for k in obsrows[0]}
                mem=m.initial_memory(b,'cuda')
                @torch.inference_mode()
                def infer():
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                        return m.act(obs,memory=mem,deterministic=False)
                try:
                    ms=timed(infer,9 if b<=64 else 5,3)
                    r=dict(kind='actor',model=name,param=count,batch=b,precision=precision,**row(ms,b))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache();r=dict(kind='actor',model=name,param=count,batch=b,precision=precision,oom=True)
                results.append(r);print(r['kind'],name,precision,b,r.get('ms','OOM'),flush=True);save('micro.json',results)
                del obs,mem
        m.train()
        # lr=0 and no saved weights. AdamW moment allocation happens in warmup.
        opt=torch.optim.AdamW(m.parameters(),lr=0.,eps=1e-5)
        check_before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        for precision in ('fp32','bf16'):
            for b in BATCHES:
                ss=segments_for(b,turns,m.config.memory_dim);group_size=16
                def update():
                    opt.zero_grad(set_to_none=True)
                    for i in range(0,len(ss),group_size):
                        sub=ss[i:i+group_size]
                        v=_learner_update_batch(m,opt,sub,'cuda',zero_grad=False,optimizer_step=False,
                            loss_scale=len(sub)/len(ss),use_bfloat16=precision=='bf16',lean_bootstrap=True)
                        if not np.isfinite(v.total_loss):raise ValueError('nonfinite loss')
                    torch.nn.utils.clip_grad_norm_(m.parameters(),40.,error_if_nonfinite=True);opt.step()
                try:
                    ms=timed(update,5 if b<=64 else 3,2)
                    r=dict(kind='update',model=name,param=count,batch=b,precision=precision,**row(ms,b))
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                    r=dict(kind='update',model=name,param=count,batch=b,precision=precision,oom=True)
                results.append(r);print(r['kind'],name,precision,b,r.get('ms','OOM'),flush=True);save('micro.json',results)
        assert all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in check_before.items())
        del m,opt,check_before; gc.collect();torch.cuda.empty_cache()
    save('micro-done.json',dict(rows=len(results),revision=SIMULATOR_REVISION,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),peakReserved=torch.cuda.max_memory_reserved()))
if __name__=='__main__':main()
