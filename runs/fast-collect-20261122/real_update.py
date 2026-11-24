"""Disposable zero-LR timing on the collector's own completed turn segments."""
import copy,math,time,statistics
from common import *
from sapai_gym1.rl.training import _learner_update_batch

def measure(model,segments,loss_modes=(False,True)):
    need=4096
    selected=[];total=0
    for s in segments:
        selected.append(s);total+=s.length
        if total>=need:break
    assert total>=need
    model=copy.deepcopy(model).train().requires_grad_(True)
    saved={k:v.cpu().clone() for k,v in model.state_dict().items()}
    groups={k:[] for k in ('base','opponent_history','opponent_adapter','end_freeze')}
    for k,p in model.named_parameters():
        owner='end_freeze' if k.startswith('end_freeze_head.') else ('opponent_adapter' if k.startswith('opponent_belief_adapter.') else ('opponent_history' if k.startswith('opponent_') else 'base'))
        groups[owner].append(p)
    opt=torch.optim.AdamW([dict(params=ps,lr=0.,name=k) for k,ps in groups.items()],eps=1e-5)
    buckets={}
    for s in selected:buckets.setdefault(1<<(s.length-1).bit_length(),[]).append(s)
    results=[]
    for fast in loss_modes:
        for ceiling in (None,128):
            batches=[]
            for width,rows in buckets.items():
                size=16 if ceiling is None else min(16,max(1,ceiling//width))
                batches.extend(rows[i:i+size] for i in range(0,len(rows),size))
            def step():
                opt.zero_grad(set_to_none=True)
                for group in batches:
                    count=sum(s.length for s in group)
                    m=_learner_update_batch(model,opt,group,'cuda',zero_grad=False,optimizer_step=False,loss_scale=count/total,
                        use_bfloat16=True,lean_bootstrap=True,batched_learner=fast,defer_metrics=fast)
                    if not fast:assert np.isfinite(m.total_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(),40.,error_if_nonfinite=True);opt.step()
            try:
                step();torch.cuda.synchronize();times=[]
                for _ in range(3):
                    t=time.perf_counter();step();torch.cuda.synchronize();times.append(time.perf_counter()-t)
                results.append(dict(batchedLoss=fast,rows=total,turns=len(selected),maxLen=max(s.length for s in selected),
                                microbatches=len(batches),paddedCeiling=ceiling,seconds=times,decisions_s=total/statistics.median(times)))
                break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                if ceiling:results.append(dict(batchedLoss=fast,oom=True))
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in saved.items())
    return results


def native_default(model,segments):
    """Time the stock learner_update grouping under the same allocator budget."""
    from sapai_gym1.rl.training import learner_update
    model=copy.deepcopy(model).train().requires_grad_(True)
    saved={k:v.cpu().clone() for k,v in model.state_dict().items()}
    names=('base','opponent_history','opponent_adapter','end_freeze')
    groups={k:[] for k in names}
    for name,p in model.named_parameters():
        group=next((k for k,prefix in [('end_freeze','end_freeze_head.'),('opponent_adapter','opponent_belief_adapter.'),('opponent_history','opponent_')] if name.startswith(prefix)),'base')
        groups[group].append(p)
    opt=torch.optim.AdamW([dict(params=p,lr=0.,name=k) for k,p in groups.items()],eps=1e-5)
    rows=0;selected=[]
    for s in segments:
        rows+=s.length;selected.append(s)
        if rows>=4096:break
    try:
        def step():
            metrics=learner_update(model,opt,selected,'cuda',use_bfloat16=True,lean_bootstrap=True)
            assert np.isfinite(metrics.total_loss)
            torch.cuda.synchronize()
        step();samples=[]
        for _ in range(3):
            t=time.perf_counter();step();samples.append(time.perf_counter()-t)
        result=dict(rows=rows,seconds=samples,decisions_s=rows/statistics.median(samples))
    except torch.cuda.OutOfMemoryError:
        result=dict(rows=rows,oom=True)
    assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in saved.items())
    return result
