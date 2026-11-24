"""Frozen, target-free, deterministic full AR particle readout and owned cache."""
from collections import OrderedDict
import copy,hashlib,json,itertools
import numpy as np
import torch
from sapai_gym1.rl.opponent_context_data import PublicOpponentContext
from sapai_gym1.rl.opponent_forecast_data import collate_contexts,validate_pet

SCHEMA='ar-particle-belief-summary-v1-20260907'
UNIFORMS_SCHEMA='pcg64-2026090701-fp32-128x5x6-v1'
PAIRS=tuple(itertools.combinations(range(5),2))

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def model_digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def feature_names(species,perks):
    names=[]
    for slot in range(5):
        names += [f'slot{slot}/occupied']
        names += [f'slot{slot}/species/{s}' for s in species]
        names += [f'slot{slot}/perk/{s}' for s in (*perks,'EMPTY')]
        names += [f'slot{slot}/progress/{p}' for p in range(7)]
        names += [f'slot{slot}/{stat}/{moment}' for stat in ('attack','health') for moment in ('mean','variance')]
        names += [f'slot{slot}/temporary_attack',f'slot{slot}/temporary_health']
        names += [f'slot{slot}/entropy/{field}' for field in ('species','perk','progress')]
    names += [f'board/pet_count/{i}' for i in range(6)]
    names += [f'board/{stat}/{aggregate}/{moment}' for stat in ('attack','health') for aggregate in ('total','max') for moment in ('mean','variance')]
    names += ['board/total_attack_health_covariance']
    names += [f'board/occupied_pair/{a}-{b}' for a,b in PAIRS]
    names += [f'board/same_species_pair/{a}-{b}' for a,b in PAIRS]
    names += [f'board/duplicate_pair_count/{i}' for i in range(11)]
    names += ['particles/surprisal_mean','particles/surprisal_variance','particles/empirical_joint_entropy']
    return names

def _moments(x):
    mean=x.mean(1);return mean,(x.square().mean(1)-mean.square()).clamp_min(0)

def _entropy(p):return -(p*p.clamp_min(1e-30).log()).sum(-1)/np.log(p.shape[-1])

@torch.inference_mode()
def summarize(labels,logp,species_count,perk_count):
    """Only generated particles/logp. No encoder vector or true board input."""
    b,n,slots,fields=labels.shape
    assert slots==5 and fields==6 and logp.shape==(b,n) and n in (32,64,128)
    species=labels[...,0];occupied=species!=0
    ps=torch.nn.functional.one_hot(species,species_count).float().mean(1)
    perk=torch.where(occupied,labels[...,2],perk_count)
    pp=torch.nn.functional.one_hot(perk,perk_count+1).float().mean(1)
    progress=torch.where(occupied,labels[...,1],6)
    pl=torch.nn.functional.one_hot(progress,7).float().mean(1)
    atk=labels[...,3].float()*occupied/50
    hp=(labels[...,4].float()+1)*occupied/50
    ma,va=_moments(atk);mh,vh=_moments(hp)
    ta=(((labels[...,5]&1)!=0)&occupied).float().mean(1)
    th=(((labels[...,5]&2)!=0)&occupied).float().mean(1)
    per_slot=torch.cat([occupied.float().mean(1)[...,None],ps,pp,pl,
        torch.stack([ma,va,mh,vh,ta,th,_entropy(ps),_entropy(pp),_entropy(pl)],-1)],-1).flatten(1)
    pieces=[per_slot,torch.nn.functional.one_hot(occupied.sum(-1),6).float().mean(1)]
    for values in (atk,hp):
        for agg in (values.sum(-1)/5,values.max(-1).values):
            pieces.extend(x[:,None] for x in _moments(agg))
    total_a=atk.sum(-1)/5;total_h=hp.sum(-1)/5
    pieces.append(((total_a*total_h).mean(1)-total_a.mean(1)*total_h.mean(1))[:,None])
    occ_pairs=torch.stack([occupied[:,:,a]&occupied[:,:,c] for a,c in PAIRS],-1)
    same=torch.stack([(species[:,:,a]==species[:,:,c])&occupied[:,:,a]&occupied[:,:,c] for a,c in PAIRS],-1)
    pieces += [occ_pairs.float().mean(1),same.float().mean(1),torch.nn.functional.one_hot(same.sum(-1),11).float().mean(1)]
    pieces.extend(x[:,None] for x in _moments(-logp/100))
    flat=labels.flatten(2)
    frequencies=(flat[:,:,None,:]==flat[:,None,:,:]).all(-1).float().mean(-1)
    pieces.append((-frequencies.log().mean(1)/np.log(n))[:,None])
    result=torch.cat(pieces,-1)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite belief summary')
    return result

class BeliefReadout:
    """Owns a frozen model copy; replacement must use replace_model()."""
    def __init__(self,model,checkpoint_sha,uniforms_path,*,history_only,rules,capacity=512):
        if len(checkpoint_sha)!=64 or capacity<1:raise ValueError('Explicit model hash and positive cache capacity required')
        self.history_only=history_only;self.rules=rules;self.capacity=capacity
        table=np.load(uniforms_path,allow_pickle=False)
        if table.shape!=(128,5,6) or table.dtype!=np.float32 or not np.all((table>0)&(table<1)):
            raise ValueError('Invalid versioned uniform table')
        self.table_cpu=torch.from_numpy(table.copy());self.table_sha=sha(uniforms_path)
        self.cache=OrderedDict();self.hits=self.misses=self.decoder_calls=0
        self.replace_model(model,checkpoint_sha)

    def replace_model(self,model,checkpoint_sha):
        if len(checkpoint_sha)!=64:raise ValueError('Explicit checkpoint digest required')
        self._model=copy.deepcopy(model).eval().requires_grad_(False)
        self.checkpoint_sha=checkpoint_sha;self.weights_sha=model_digest(self._model)
        self.device=next(self._model.parameters()).device
        if next(self._model.parameters()).dtype!=torch.float32:raise ValueError('Readout v1 requires FP32')
        self.uniforms=self.table_cpu.to(self.device)
        self.names=feature_names(self._model.config.species,self._model.config.perks)
        self.feature_sha=hashlib.sha256(json.dumps(self.names,separators=(',',':')).encode()).hexdigest()
        self.cache.clear()

    def set_rules(self,rules):
        if not isinstance(rules,str) or not rules:raise ValueError('Rules revision required')
        if rules!=self.rules:self.rules=rules;self.cache.clear()

    def canonical_context(self,context):
        expected=set(PublicOpponentContext.__dataclass_fields__)|{'missing_history'}
        if set(context)!=expected or context['pack']!='Turtle-P1-vs-P1':raise ValueError('Only declared public context is accepted')
        boards=('opponent_last','opponent_previous','our_last')
        if context['missing_history'] != [context[k] is None for k in boards]:raise ValueError('Wrong missing-history flags')
        for k in ('turn','our_lives','opponent_lives','our_wins','opponent_wins'):
            if type(context[k]) is not int or context[k]<(1 if k=='turn' else 0):raise ValueError('Invalid public counters')
        for k in boards:
            board=context[k]
            if board is None:continue
            if len(board)!=5:raise ValueError('Invalid public board')
            for pet in board:
                if pet is not None:validate_pet(pet,self._model.species_ids,self._model.perk_ids)
        c=dict(context)
        if self.history_only:
            c.update(our_last=None,turn=1,our_lives=0,opponent_lives=0,our_wins=0,opponent_wins=0)
            c['missing_history']=[c['opponent_last'] is None,c['opponent_previous'] is None,True]
        return c

    def key(self,context,n):
        if n not in (32,64,128):raise ValueError('Only predeclared particle budgets')
        c=self.canonical_context(context)
        metadata=[SCHEMA,self.feature_sha,self.checkpoint_sha,self.weights_sha,self.rules,self.table_sha,
                  self.history_only,n,str(self.device),'fp32-highest-v1',c]
        return hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _draw(logits,u):
        lp=torch.log_softmax(logits.float(),-1)
        if not torch.isfinite(lp).all():raise ValueError('Nonfinite decoder probabilities')
        cdf=lp.exp().cumsum(-1);cdf[:,-1]=1
        choice=torch.searchsorted(cdf.contiguous(),u[:,None].contiguous()).squeeze(-1)
        return choice,lp.gather(1,choice[:,None]).squeeze(-1)

    @torch.inference_mode()
    def particles(self,contexts,n):
        if n not in (32,64,128):raise ValueError('Invalid particle budget')
        c=[self.canonical_context(x) for x in contexts]
        if not c:raise ValueError('Empty context batch')
        cpu=collate_contexts(c,self._model.species_ids,self._model.perk_ids)
        if self.history_only:cpu=cpu.history_only()
        with torch.autocast(self.device.type,enabled=False):
            hidden=self._model.encode_context(cpu).repeat_interleave(n,0)
            b=len(c);previous=self._model.bos[None].expand(b*n,-1)
            logp=hidden.new_zeros(b*n);slots=[]
            for j in range(5):
                hidden=self._model.slot_cell(previous+self._model.position.weight[j],hidden)
                species,lp=self._draw(self._model.species_head(hidden),self.uniforms[:n,j,0].repeat(b))
                logp+=lp;occupied=species!=0;state=hidden+self._model.species_embedding(species);fields=[species]
                for f,(head,emb) in enumerate(zip(self._model.field_heads,self._model.field_embeddings)):
                    choice,lp=self._draw(head(state),self.uniforms[:n,j,f+1].repeat(b))
                    logp+=torch.where(occupied,lp,0.);fields.append(torch.where(occupied,choice,0))
                    state=self._model.field_cell(emb(choice),state)
                previous=torch.where(occupied[:,None],state,self._model.species_embedding.weight[0][None])
                slots.append(torch.stack(fields,-1))
        self.decoder_calls+=1
        return torch.stack(slots,1).reshape(b,n,5,6),logp.reshape(b,n)

    @torch.inference_mode()
    def batch(self,contexts,n=64):
        labels,lp=self.particles(contexts,n)
        out=summarize(labels,lp,len(self._model.config.species),len(self._model.config.perks))
        assert out.shape==(len(contexts),len(self.names))
        return out

    def get(self,context,n=64):
        key=self.key(context,n)
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key].clone()
        self.misses+=1;value=self.batch([context],n)[0].cpu().clone();self.cache[key]=value
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return value.clone()
