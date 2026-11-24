"""Coalesce expensive work while selectively retaining reference GEMM shapes."""
from itertools import groupby
from types import SimpleNamespace
import torch
from sapai_gym1.rl.model import EntityTransformerPolicy

def blocks(parts):
    start=0;out=[]
    for _,items in groupby(parts,key=lambda p:p.reference):
        s=list(items);out.append((start,start+len(s),max(p.turn.length for p in s)));start+=len(s)
    return out

class Encoders:
    def __init__(self,policy,entities=None,history=None):
        self.policy=policy;self.entities=entities;self.history=history
    def __getattr__(self,k):return getattr(self.policy,k)
    def encode_entities(self,obs):return self.policy.encode_entities(obs) if self.entities is None else self.entities
    def encode_opponent_history(self,obs):return self.policy.encode_opponent_history(obs) if self.history is None else self.history

def sequence(policy,p,parts,method):
    t,b=p.actions.shape[:2]
    spans=blocks(parts);flat={k:v.flatten(0,1) for k,v in p.observations.items()}
    def joined(fn):
        pieces=[]
        for l,r,h in spans:
            val=fn({k:v[:h,l:r].flatten(0,1) for k,v in p.observations.items()}).unflatten(0,(h,r-l))
            if h<t:val=torch.cat((val,val[-1:].expand(t-h,*val.shape[1:])))
            pieces.append(val)
        return torch.cat(pieces,1).flatten(0,1)
    def ev(proxy,ob,h,a,f):
        return EntityTransformerPolicy.evaluate_sequence(proxy,ob,initial_memory=h,actions=a,end_freeze_masks=f,forecast_first_only=True)
    if method not in ('block-head','block-head-history'):
        entities=joined(policy.encode_entities) if method in ('block-entity','block-both') else None
        history=joined(policy.encode_opponent_history) if method in ('block-history','block-both') else None
        return ev(Encoders(policy,entities,history),p.observations,p.initial_memories,p.actions,p.freeze_masks)
    entities=policy.encode_entities(flat).unflatten(0,(t,b));history=policy.encode_opponent_history(flat).unflatten(0,(t,b)) if method=='block-head' else None
    outs=[]
    for l,r,h in spans:
        z=Encoders(policy,entities[:h,l:r].flatten(0,1),history[:h,l:r].flatten(0,1) if history is not None else None)
        outs.append(ev(z,{k:v[:h,l:r] for k,v in p.observations.items()},p.initial_memories[l:r],p.actions[:h,l:r],p.freeze_masks[:h,l:r]))
    result={}
    for k in ('log_prob','entropy','value','memory_states','battle_logits','opponent_pet_logits','opponent_status_logits','opponent_level_logits','opponent_stats','opponent_temporary_logits'):
        vals=[getattr(o,k) for o in outs]
        if not k.startswith('opponent_'):
            vals=[torch.cat((v,v[-1:].expand(t-len(v),*v.shape[1:]))) if len(v)<t else v for v in vals]
        result[k]=torch.cat(vals,1)
    return SimpleNamespace(**result)
def bootstrap(policy,p,parts,h):
    return torch.cat([policy.value_only({k:v[l:r] for k,v in p.bootstrap_observations.items()},memory=h[l:r]) for l,r,_ in blocks(parts)])
