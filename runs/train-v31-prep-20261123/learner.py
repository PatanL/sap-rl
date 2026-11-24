"""Prepare the checkpointed recurrent policy and use the pinned native V-trace learner."""
from collections import defaultdict
from dataclasses import asdict,replace
import hashlib,math
from pathlib import Path
import torch
from runtime import PARENT_SHA,SIMULATOR_REVISION,SOURCE_OF_TRUTH,OBSERVATION_REVISION,sha
from sapai_gym1.rl.model import EntityTransformerPolicy,EntityTransformerConfig
from sapai_gym1.rl.training import _learner_update_batch
from sapai_gym1.rl.causal_transport import checkpoint_context_metadata
from sapai_gym1.rl.end_freeze_checkpoint import END_FREEZE_SCHEMA
from scripts.train_entity_selfplay import action_execution_metrics,max_segment_lag

def read_checkpoint(path):
    return torch.load(path,map_location='cpu',weights_only=True,mmap=True)

def architecture(parent,arm):
    config=EntityTransformerConfig(**parent['config'])
    if arm=='fresh':
        config=replace(config,d_model=128,nhead=4,num_layers=4,dim_feedforward=512,memory_dim=192,opponent_history_dim=48)
    return config

def prepare(parent_path,resume_path,arm,info,seed):
    if sha(parent_path)!=PARENT_SHA:raise ValueError('Parent checkpoint SHA-256 mismatch')
    parent=read_checkpoint(parent_path)
    if parent['update']!=25772 or parent['observation_revision']!=OBSERVATION_REVISION or parent['training_semantics']['action_schema']!=END_FREEZE_SCHEMA:
        raise ValueError('Unexpected parent metadata')
    config=architecture(parent,arm)
    if resume_path is None:
        source=parent if arm=='parent' else None
    else:
        source=read_checkpoint(resume_path)
        if (source.get('simulator_revision')!=SIMULATOR_REVISION or source.get('engine_payload_sha256')!=info['payload_sha256']
            or source.get('wide_arm')!=arm or source.get('observation_revision')!=OBSERVATION_REVISION or source['config']!=asdict(config)):
            raise ValueError('Attempted resume from a different revision, observation schema or training arm')
    torch.manual_seed(seed)
    policy=EntityTransformerPolicy(config)
    if source is not None:policy.load_state_dict(source['model'],strict=True)
    step=int(source['update']) if source is not None else 0
    rows=int(source['transitions']) if source is not None else 0
    primitives=int(source.get('learner_primitive_actions',rows)) if source is not None else 0
    # For a new arm the baseline rates are explicit defaults, not transferred moments.
    rates={g['name']:float(g['lr']) for g in (source or parent)['optimizer']['param_groups']}
    if source is None:
        rates={name:3e-4 for name in rates}  # pinned trainer's from-scratch base/default head rate
    semantics=dict((source or parent)['training_semantics'])
    for group,field in [('opponent_history','opponent_module_learning_rate'),('opponent_adapter','opponent_adapter_learning_rate'),('end_freeze','end_freeze_learning_rate')]:
        semantics[field]=rates[group]
    if semantics['discount_mode']!='per_turn' or semantics['discount']!=.99 or semantics['reward_estimator']['samples']!=1:
        raise ValueError('Unsupported discount or reward semantics')
    return policy,source['optimizer'] if source is not None else None,rates,semantics,step,rows,primitives

def optimizer_for(policy,state,rates,scale):
    history=('opponent_history_','opponent_pet_head.','opponent_status_head.','opponent_level_head.','opponent_stats_head.','opponent_temporary_head.')
    groups={n:[] for n in ('base','opponent_history','opponent_adapter','end_freeze')}
    for name,p in policy.named_parameters():
        if name.startswith('end_freeze_head.'):key='end_freeze'
        elif name.startswith('opponent_belief_adapter.'):key='opponent_adapter'
        elif name.startswith(history):key='opponent_history'
        else:key='base'
        groups[key].append(p)
    opt=torch.optim.AdamW([{'params':ps,'name':name,'lr':rates[name]} for name,ps in groups.items()],eps=1e-5)
    if state is not None:
        if [g.get('name') for g in state['param_groups']]!=list(groups):raise ValueError('Optimizer parameter groups do not match source')
        opt.load_state_dict(state)
        for p,st in opt.state.items():
            if st and (st['exp_avg'].shape!=p.shape or st['exp_avg_sq'].shape!=p.shape):raise ValueError('Adam moment shape mismatch')
    for g in opt.param_groups:g['lr']=rates[g['name']]*scale
    return opt

def update(policy,opt,segments,semantics,device,max_padded_rows):
    count=sum(s.length for s in segments);buckets=defaultdict(list)
    for s in segments:buckets[1<<(s.length-1).bit_length()].append(s)
    opt.zero_grad(set_to_none=True);loss=0.;microbatches=0
    for width,seqs in buckets.items():
        group_size=max(1,min(16,max_padded_rows//width))
        for j in range(0,len(seqs),group_size):
            group=seqs[j:j+group_size];n=sum(s.length for s in group)
            metric=_learner_update_batch(policy,opt,group,device,zero_grad=False,optimizer_step=False,
                loss_scale=n/count,use_bfloat16=device=='cuda',lean_bootstrap=True,
                battle_aux_cost=semantics['battle_aux_cost'],opponent_aux_cost=semantics['opponent_aux_cost'])
            if metric.transitions!=n or not math.isfinite(metric.total_loss):raise RuntimeError('Invalid V-trace batch')
            loss+=metric.total_loss*n/count;microbatches+=1
    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),40.,error_if_nonfinite=True)
    opt.step()
    return {'decisions':count,'microbatches':microbatches,'turns':len(segments),'loss':loss,'grad_norm':float(norm),
            'primitive_actions':action_execution_metrics(segments)[0]}

def write_checkpoint(path,policy,opt,arm,info,semantics,version,rows,primitives):
    path=Path(path);temporary=path.with_suffix('.tmp')
    metadata=dict(semantics)
    for g in opt.param_groups:
        k={'opponent_history':'opponent_module_learning_rate','opponent_adapter':'opponent_adapter_learning_rate','end_freeze':'end_freeze_learning_rate'}.get(g['name'])
        if k:metadata[k]=g['lr']
    data=dict(model=policy.state_dict(),optimizer=opt.state_dict(),config=asdict(policy.config),update=version,
              transitions=rows,learner_primitive_actions=primitives,simulator_revision=SIMULATOR_REVISION,
              engine_payload_sha256=info['payload_sha256'],source_of_truth=SOURCE_OF_TRUTH,observation_revision=OBSERVATION_REVISION,
              causal_context=checkpoint_context_metadata(policy.config),training_semantics=metadata,wide_arm=arm,
              parent_checkpoint_sha256=PARENT_SHA if arm=='parent' else None,collector='wide-two-cohort-public-selfplay',
              actor_sampling='stochastic policy.act, no search',native_rust_scorer=False)
    torch.save(data,temporary);temporary.replace(path)

def weight_hash(policy):
    h=hashlib.sha256()
    for k,v in policy.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
