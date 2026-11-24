"""Coalesce reference groups for the network; preserve the original auxiliary-loss normalizations."""
from collections import defaultdict
from dataclasses import dataclass
import math,os
import torch
from torch.nn import functional as F
from learner import action_execution_metrics
from sapai_gym1.rl.vtrace import masked_impala_actor_critic_loss
from sapai_gym1.rl.learner_batch import PackedLearnerBatch

@dataclass
class Part:
    turn: object
    battle_weight: float
    pet_weight: float
    occupied_weight: float
    reference: int=0

def plan(turns, physical_rows, reference_rows=128):
    buckets=defaultdict(list)
    for s in turns:
        if s.length<1:raise ValueError('Empty turn')
        buckets[1<<(s.length-1).bit_length()].append(s)
    total=sum(s.length for s in turns)
    groups=[]
    for width,seqs in buckets.items():
        size=max(1,min(16,reference_rows//width))
        parts=[];virtual=[]
        for j in range(0,len(seqs),size):
            old=seqs[j:j+size]
            n=sum(s.length for s in old);scale=n/total
            battle_n=sum(s.length for s in old if s.battle_outcome is not None)
            bs=[s.bootstrap_observation for s in old]
            valid=[b for b in bs if 'opponent_history_board_mask' in b and bool(b['opponent_history_board_mask'][-1])]
            pet_n=sum(b['opponent_history_item_ids'][-1].numel() for b in valid)
            occ_n=sum(int(b['opponent_history_mask'][-1].sum()) for b in valid)
            weights=[scale/v if v else 0. for v in (battle_n,pet_n,occ_n)]
            block=[Part(s,*weights,reference=j) for s in old];parts.extend(block);virtual.append(block)
        geom=os.environ.get('SPEED_GEOMETRY','sorted')
        if geom=='legacy' or (geom=='guard16' and width==16) or (geom=='short' and width>8) or (geom=='medium' and width>16) or (geom=='long' and width<16):
            groups.extend(virtual);continue
        if geom.startswith('block-'):
            merged=[];longest=0
            for block in virtual:
                h=max(p.turn.length for p in block)
                if merged and max(longest,h)*(len(merged)+len(block))>physical_rows:
                    groups.append(merged);merged=[];longest=0
                longest=max(longest,h);merged+=block
            if merged:groups.append(merged)
            continue
        if geom=='aligned':
            families=defaultdict(list)
            for block in virtual:families[max(x.turn.length for x in block)].append(block)
            for t,blocks in families.items():
                merged=[]
                for block in blocks:
                    if merged and (len(merged)+len(block))*t>physical_rows:groups.append(merged);merged=[]
                    merged+=block
                if merged:groups.append(merged)
            continue
        # Closer lengths within this power-of-two bucket reduce padding. Reference weights are attached above.
        parts.sort(key=lambda r:r.turn.length)
        current=[]
        for p in parts:
            if current and (len(current)+1)*p.turn.length>physical_rows:
                groups.append(current);current=[]
            current.append(p)
        if current:groups.append(current)
    return groups

def pack(parts,device,compact=False):
    seq=[p.turn for p in parts];lens=torch.tensor([s.length for s in seq])
    ticks=torch.arange(int(lens.max()))[:,None]
    # Gather legal last observations/actions into padded positions just as the reference does.
    offsets=lens.cumsum(0)-lens
    index=(offsets+torch.minimum(ticks,lens[None]-1)).reshape(-1)
    shape=(len(ticks),len(seq))
    def gather(values):
        flat=torch.cat(values).index_select(0,index)
        return flat.view(*shape,*flat.shape[1:]).to(device)
    obs={k:gather([s.observations[k] for s in seq]) for k in seq[0].observations}
    boot={k:torch.stack([s.bootstrap_observation[k] for s in seq]).to(device) for k in seq[0].bootstrap_observation}
    a=gather([s.actions for s in seq])
    flat_masks=torch.cat([s.freeze_masks for s in seq])
    if bool((flat_masks<0).any() | (flat_masks>=1024).any()):raise ValueError('Invalid recorded joint action')
    mask=flat_masks.index_select(0,index).reshape(shape).to(device,dtype=torch.long)
    vals=gather([torch.stack((s.behavior_log_probs,s.rewards,s.discounts),-1) for s in seq])
    h=torch.stack([s.initial_memory for s in seq]).to(device)
    terminals=torch.tensor([s.terminal for s in seq],dtype=torch.bool,device=device)
    outcomes=torch.tensor([-1 if s.battle_outcome is None else s.battle_outcome for s in seq],device=device)
    w=torch.tensor([[p.battle_weight,p.pet_weight,p.occupied_weight] for p in parts],dtype=torch.float32,device=device)
    packed=PackedLearnerBatch(obs,a,vals[...,0],vals[...,1],vals[...,2],mask,lens.to(device),h,boot,terminals,outcomes)
    if compact:
        valid=ticks<lens[None];flat_ix=valid.flatten().nonzero().flatten()
        ordinal=valid.flatten().long().cumsum(0).view(shape)-1
        lookup=ordinal[torch.minimum(ticks,lens[None]-1),torch.arange(len(seq))[None,:]]
        # Verify constancy on CPU. A changed public history uses the fully general encoding branch.
        constant=all(torch.equal(s.observations[k],s.observations[k][:1].expand_as(s.observations[k]))
            for s in seq for k in obs if k.startswith('opponent_history_'))
        packed.compact_layout=(flat_ix.to(device),lookup.to(device),constant)
    return packed,w

def micro_loss(policy,parts,sem,device,bf16,total,compact=False):
    p,w=pack(parts,device,compact)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16 and device=='cuda'):
        geometry=os.environ.get('SPEED_GEOMETRY','sorted')
        if geometry.startswith('block-'):
            from block_shapes import sequence
            out=sequence(policy,p,parts,geometry)
        elif compact:
            from compact_sequence import evaluate
            out=evaluate(policy,p,p.compact_layout)
        else:out=policy.evaluate_sequence(p.observations,initial_memory=p.initial_memories,actions=p.actions,end_freeze_masks=p.freeze_masks,forecast_first_only=True)
        with torch.no_grad():
            h=out.memory_states[p.lengths-1,torch.arange(len(parts),device=device)]
            if geometry.startswith('block-'):
                from block_shapes import bootstrap
                boot=bootstrap(policy,p,parts,h).masked_fill(p.terminals,0)
            else:boot=policy.value_only(p.bootstrap_observations,memory=h).masked_fill(p.terminals,0)
    n=sum(q.turn.length for q in parts)
    if os.environ.get('SPEED_FUSED_LOSS')=='1' and device=='cuda':
        from learner_ops import base_loss
        value=base_loss(p,out,boot,n)
    else:
        value=masked_impala_actor_critic_loss(p.behavior_log_probs,out.log_prob,out.entropy,p.discounts,p.rewards,out.value,boot,p.lengths,validate_lengths=False).total
    valid=torch.arange(p.actions.shape[0],device=device)[:,None]<p.lengths[None]
    loss=value*(n/total)
    if sem['battle_aux_cost']:
        target=p.battle_outcomes.clamp_min(0).expand_as(valid)
        logits=out.battle_logits.float()
        ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='none').view_as(valid)
        loss=loss+sem['battle_aux_cost']*(ce*valid*(p.battle_outcomes>=0)*w[:,0]).sum()
    b=p.bootstrap_observations
    if sem['opponent_aux_cost'] and 'opponent_history_board_mask' in b:
        valid_board=b['opponent_history_board_mask'][:,-1].bool()
        occupied=b['opponent_history_mask'][:,-1].bool() & valid_board[:,None]
        num=b['opponent_history_numeric'][:,-1].float()
        def ce(logits,target):
            return F.cross_entropy(logits[0].float().transpose(1,2),target.long(),reduction='none')
        pets=ce(out.opponent_pet_logits,b['opponent_history_item_ids'][:,-1].masked_fill(~valid_board[:,None],0))
        status=ce(out.opponent_status_logits,b['opponent_history_status_ids'][:,-1].masked_fill(~occupied,0))
        num=num.masked_fill(~occupied[...,None],0.)
        level=ce(out.opponent_level_logits,(num[...,2]*3).round())
        stats=(out.opponent_stats[0].float()-num[...,:2]).square().mean(-1)
        temp=F.binary_cross_entropy_with_logits(out.opponent_temporary_logits[0].float(),num[...,5:7],reduction='none').mean(-1)
        aux=(pets*valid_board[:,None]*w[:,1,None]).sum()+((.25*status+.25*level+.5*stats+.1*temp)*occupied*w[:,2,None]).sum()
        loss=loss+sem['opponent_aux_cost']*aux
    return loss

def accumulate(policy,turns,sem,device,physical_rows,bf16,reference_rows,compact):
    groups=plan(turns,physical_rows,reference_rows);n=sum(s.length for s in turns)
    values=[];padded=0
    for group in groups:
        loss=micro_loss(policy,group,sem,device,bf16,n,compact)
        loss.backward();values.append(loss.detach())
        padded+=len(group)*max(x.turn.length for x in group)
    return float(torch.stack(values).sum()),padded,len(groups),n

def update_fast(policy,opt,turns,sem,device,physical_rows=512,bf16=True,reference_rows=128,compact=False,retry_oom=False):
    policy.train();fallbacks=0;longest=max(s.length for s in turns)
    while True:
        opt.zero_grad(set_to_none=True)
        try:
            loss,padded,groups,n=accumulate(policy,turns,sem,device,physical_rows,bf16,reference_rows,compact)
            break
        except torch.cuda.OutOfMemoryError:
            if not retry_oom or device!='cuda' or physical_rows<=longest:raise
        # The optimizer has not run. Drop all partial gradients and restart this very same batch.
        import gc
        opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        physical_rows=max(longest,physical_rows//2);fallbacks+=1
    if not math.isfinite(loss):raise ValueError('Nonfinite learner loss')
    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),40.,error_if_nonfinite=True)
    opt.step()  # Errors at/after this boundary must not be retried in place.
    return dict(decisions=n,turns=len(turns),microbatches=groups,padded_rows=padded,loss=loss,grad_norm=float(norm),
                effective_cap=physical_rows,oom_retries=fallbacks,primitive_actions=action_execution_metrics(turns)[0])
