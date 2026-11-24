"""Optional one-pass V-trace targets and one conditional END distribution for the teacher-forced learner."""
import torch
import triton
import triton.language as tl
from fast_actor import AsyncCheckedPolicy,enable
from sapai_gym1.rl.end_freeze import EndFreezeOutput
from sapai_gym1.rl.action_schema import ActionType,MAX_TEAM_SLOTS,MAX_SHOP_SLOTS
@triton.jit(do_not_specialize=['T','B'])
def _targets(LP,BP,V,R,D,BOOT,LEN,VT,ADV,T,B, BLOCK:tl.constexpr):
    j=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    length=tl.load(LEN+j,mask=j<B,other=0);boot=tl.load(BOOT+j,mask=j<B,other=0).to(tl.float32)
    next_v=boot;next_target=boot;correction=tl.full((BLOCK,),0,tl.float32)
    for offset in range(T):
        t=T-1-offset;valid=(j<B)&(t<length);i=t*B+j
        v=tl.load(V+i,mask=valid,other=0).to(tl.float32)
        rew=tl.load(R+i,mask=valid,other=0).to(tl.float32);disc=tl.load(D+i,mask=valid,other=0).to(tl.float32)
        logp=tl.load(LP+i,mask=valid,other=0).to(tl.float32);behavior=tl.load(BP+i,mask=valid,other=0).to(tl.float32)
        rho=tl.minimum(1.,tl.exp(logp-behavior))
        delta=rho*(rew+disc*next_v-v)
        correction=tl.where(valid,delta+disc*rho*correction,0.)
        vt=tl.where(valid,v+correction,0.);adv=tl.where(valid,rho*(rew+disc*next_target-v),0.)
        tl.store(VT+i,vt,mask=j<B);tl.store(ADV+i,adv,mask=j<B)
        next_v=tl.where(valid,v,boot);next_target=tl.where(valid,vt,boot)
def base_loss(p,out,boot,n):
    t,b=p.actions.shape[:2]
    targets=torch.empty((t,b),device=boot.device,dtype=torch.float32);adv=torch.empty_like(targets)
    with torch.no_grad():
        args=[v.contiguous() for v in (out.log_prob,p.behavior_log_probs,out.value,p.rewards,p.discounts,boot,p.lengths)]
        _targets[(triton.cdiv(b,32),)](*args,targets,adv,t,b,BLOCK=32,num_warps=1,enable_fp_fusion=False)
    valid=torch.arange(t,device=boot.device)[:,None]<p.lengths[None]
    return (-(out.log_prob*adv).sum()+.25*((targets-out.value).square()*valid).sum()-.01*(out.entropy*valid.float()).sum())/n

class LearnerPolicy(AsyncCheckedPolicy):
    def _end_freeze_decision(self,context,tokens,observation,type_dist,action_types,*,masks=None,deterministic=False):
        if masks is None:return super()._end_freeze_decision(context,tokens,observation,type_dist,action_types,masks=masks,deterministic=deterministic)
        head=self.end_freeze_head;start=1+MAX_TEAM_SLOTS
        z=head.scorer(torch.cat([context[:,None].expand(-1,MAX_SHOP_SLOTS,-1),tokens[:,start:start+MAX_SHOP_SLOTS]],dim=-1)).squeeze(-1)
        end=torch.full_like(action_types,int(ActionType.END_TURN))
        dist,allowed=head.distribution(z,observation,end)
        is_end=action_types==int(ActionType.END_TURN)
        ok=torch.where(is_end,allowed.gather(1,masks[:,None]).squeeze(1),masks==0)
        torch._assert_async(ok.all(),'Illegal teacher joint mask')
        lp=torch.where(is_end,dist.log_prob(masks),0.)
        return EndFreezeOutput(masks,lp,None,z),type_dist.probs[:,ActionType.END_TURN].float()*dist.entropy()
def enable_learner(m):
    enable(m);m.__class__=LearnerPolicy
