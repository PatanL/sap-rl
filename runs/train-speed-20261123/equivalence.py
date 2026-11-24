"""Compare both objectives, accumulated gradients before clipping, and Adam parameter changes."""
from pathlib import Path
import argparse,gc,json,os,torch
from unittest.mock import patch
from fixture import PARENT,get_fixture
import learner as original
from learner_fast import update_fast
from runtime import verify_engine
TOLS={'fp32':dict(loss_abs=2e-5,gradient_relative_l2=3e-4,delta_relative_l2=1e-3,weight_max_abs=2e-6),
      'bf16':dict(loss_abs=.003,gradient_relative_l2=.035,delta_relative_l2=.035,weight_max_abs=8e-6)}
def settings(gpu):
    torch.set_num_threads(3 if not gpu else 1);torch.set_num_interop_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if gpu:
        _,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(float(os.environ.get('SAP_GPU_GIB',8))*1024**3/total)
def new(device,info):
    m,st,rates,sem,*_=original.prepare(PARENT,None,'parent',info,278);m.to(device)
    return m,original.optimizer_for(m,st,rates,1),sem
def rel(a,b):
    dif=ref=dot=bb=mx=0.
    for k in a:
        x=a[k].double();y=b[k].double();d=x-y
        dif+=float(d.square().sum());ref+=float(x.square().sum());bb+=float(y.square().sum());dot+=float((x*y).sum());mx=max(mx,float(d.abs().max()))
    return dict(relative_l2=(dif/ref)**.5,max_abs=mx,cosine=dot/(ref*bb)**.5)
def execute(fast,device,info,segments,prec,cap,compact=False):
    m,opt,sem=new(device,info);before={k:p.detach().cpu().clone() for k,p in m.named_parameters()};grads={}
    if fast and os.environ.get('SPEED_FUSED_LOSS')=='1':
        from learner_ops import enable_learner
        enable_learner(m)
    oldclip=torch.nn.utils.clip_grad_norm_
    def clip(params,*args,**kw):
        grads.update({k:p.grad.detach().cpu().clone() for k,p in m.named_parameters() if p.grad is not None})
        return oldclip(params,*args,**kw)
    with patch.object(torch.nn.utils,'clip_grad_norm_',clip):
        if fast:result=update_fast(m,opt,segments,sem,device,cap,prec=='bf16',compact=compact)
        else:
            orig=original._learner_update_batch
            def ref(*args,**kw):
                kw['use_bfloat16']=prec=='bf16'
                return orig(*args,**kw)
            with patch.object(original,'_learner_update_batch',ref):result=original.update(m,opt,segments,sem,device,128)
    delta={k:p.detach().cpu()-before[k] for k,p in m.named_parameters()}
    return result,grads,delta
def main():
    p=argparse.ArgumentParser();p.add_argument('--compact',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--prec',default='fp32');p.add_argument('--n',type=int,default=1536);p.add_argument('--cap',type=int,default=512);p.add_argument('--out',required=True)
    a=p.parse_args();settings(a.device=='cuda');info=verify_engine();seq=get_fixture(a.n)
    lhs,g1,d1=execute(False,a.device,info,seq,a.prec,a.cap);gc.collect()
    rhs,g2,d2=execute(True,a.device,info,seq,a.prec,a.cap,a.compact)
    assert g1.keys()==g2.keys()
    grad=rel(g1,g2);delta=rel(d1,d2)
    diff=dict(loss_abs=abs(rhs['loss']-lhs['loss']),gradient_relative_l2=grad['relative_l2'],delta_relative_l2=delta['relative_l2'],weight_max_abs=delta['max_abs'])
    ok=all(v<=TOLS[a.prec][k] for k,v in diff.items())
    heads={}
    for name,prefix in [('freeze','end_freeze_head.'),('forecast','opponent_'),('gru','memory_cell.'),('entity_transformer','entity_transformer.')]:
        a1={k:v for k,v in g1.items() if k.startswith(prefix)}
        if a1 and any(v.count_nonzero() for v in a1.values()):heads[name]=rel(a1,{k:g2[k] for k in a1})
    receipt=dict(gradient_subsets=heads,compact=a.compact,device=a.device,precision=a.prec,rows=sum(s.length for s in seq),cap=a.cap,reference=lhs,fast=rhs,gradient=grad,delta=delta,difference=diff,tolerance=TOLS[a.prec],passed=ok)
    Path(a.out).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
    if not ok:raise SystemExit(1)
if __name__=='__main__':main()
