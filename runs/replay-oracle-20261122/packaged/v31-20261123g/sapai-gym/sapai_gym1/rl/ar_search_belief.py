"""Immutable coherent particles replace only the synthetic belief branch."""
from collections import OrderedDict,Counter
from dataclasses import dataclass
import hashlib,json
import numpy as np
import torch
from sapai.turtle_pack import SIMULATOR_REVISION,TURTLE_PET_STATS,CURRENT_TURTLE_GENERATED_PETS
from .ar_decoder_readout import BeliefReadout
from .opponent_belief import OpponentBoardBelief
from .opponent_context_data import PublicOpponentContext
from .opponent_distribution import AutoregressiveOpponentBoard,OpponentDistributionConfig
from .opponent_forecast_data import PROGRESS,validate_pet
from .public_battle_board import public_records,from_public_records,to_state,from_state

PROVIDER_VERSION='history-ar-synthetic-particles-v1-20260907'
ALLOWED_SPECIES=frozenset(TURTLE_PET_STATS)|CURRENT_TURTLE_GENERATED_PETS

@dataclass(frozen=True)
class ParticlePool:
    context_key:str
    states:tuple[str|None,...]
    rejected_reasons:tuple[tuple[str,int],...]
    @property
    def retained_mass(self):return sum(x is not None for x in self.states)/len(self.states)

@dataclass(frozen=True)
class ParticleOpponentBelief(OpponentBoardBelief):
    pool:ParticlePool
    def sample_synthetic_team(self,rng):
        # A disjoint deterministic stream: do not consume an extra battle/shop
        # RNG draw merely because this provider has particles and baseline not.
        key=json.dumps(rng.bit_generator.state,sort_keys=True,separators=(',',':'))
        digest=hashlib.sha256((PROVIDER_VERSION+self.pool.context_key+key).encode()).digest()
        index=int.from_bytes(digest[:8],'little')%len(self.pool.states)
        value=self.pool.states[index]
        if value is None:
            team=self.modal_team();team._search_belief_source='belief-ar-invalid-fallback'
        else:
            team=from_state(json.loads(value));team._search_belief_source='belief-ar'
        team._search_particle_id=self.pool.context_key+':'+str(index)
        return team

class ARBeliefProvider:
    def __init__(self,readout,*,particles=64,capacity=512):
        if particles not in (32,64,128) or capacity<1:raise ValueError('Invalid frozen particle/cache budget')
        if readout.rules!=SIMULATOR_REVISION:raise ValueError('Forecast rules mismatch')
        self.readout=readout;self.particles=particles;self.capacity=capacity;self.cache=OrderedDict()
        self.calls=self.hits=self.misses=0;self.generated=self.invalid=0;self.reasons=Counter()
        self.minimum_retained_mass=1.

    @classmethod
    def load(cls,checkpoint,uniforms,*,device='cpu',particles=64,capacity=512):
        from .ar_decoder_readout import sha
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload.get('variant')!='history_autoregressive':raise ValueError('This screen fixes history-only AR')
        model=AutoregressiveOpponentBoard(OpponentDistributionConfig(**payload['config'])).to(device).eval()
        model.load_state_dict(payload['state_dict'])
        readout=BeliefReadout(model,sha(checkpoint),uniforms,history_only=True,rules=SIMULATOR_REVISION,capacity=capacity)
        return cls(readout,particles=particles,capacity=capacity)

    def _context(self,state):
        core=state.core
        return PublicOpponentContext(public_records(core.previous_opponent_state),
            public_records(core.previous_previous_opponent_state),public_records(core.previous_agent_state),
            int(core.agent.turn),int(core.agent.lives),int(core.opponent.lives),int(core.agent.wins),int(core.opponent.wins)).features()

    def _particle_state(self,row):
        board=[];config=self.readout._model.config
        for species,progress,perk,attack,health,temporary in row:
            if species==0:board.append(None);continue
            if not 0<species<len(config.species) or config.species[species] not in ALLOWED_SPECIES:
                raise ValueError('outside-current-Turtle-species')
            if not (0<=progress<len(PROGRESS) and 0<=perk<len(config.perks) and 0<=attack<=50 and 0<=health<50 and 0<=temporary<4):
                raise ValueError('invalid-fields')
            pet={'species':config.species[species],'level':PROGRESS[progress][0],'experience':PROGRESS[progress][1],
                 'perk':config.perks[perk],'attack':attack,'health':health+1,
                 'temporaryAttack':bool(temporary&1),'temporaryHealth':bool(temporary&2)}
            validate_pet(pet,self.readout._model.species_ids,self.readout._model.perk_ids);board.append(pet)
        return json.dumps(to_state(from_public_records(board)),separators=(',',':'),allow_nan=False)

    def __call__(self,state,base):
        self.calls+=1;budget=getattr(state,'work_budget',None)
        if budget is not None:budget.check()
        context=self._context(state);key=self.readout.key(context,self.particles)
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);pool=self.cache[key]
        else:
            self.misses+=1
            if budget is not None:budget.forecast_contexts+=1
            # Canonical B=1 FP32 path for every context; never regenerate a
            # cached world with a different context-batch kernel.
            labels,_=self.readout.particles([context],self.particles)
            values=[];reasons=Counter()
            for row in labels[0].cpu().tolist():
                try:values.append(self._particle_state(row))
                except ValueError as error:values.append(None);reasons[str(error)]+=1
            pool=ParticlePool(key,tuple(values),tuple(sorted(reasons.items())))
            self.generated+=len(values);self.invalid+=sum(reasons.values());self.reasons.update(reasons)
            self.minimum_retained_mass=min(self.minimum_retained_mass,pool.retained_mass)
            self.cache[key]=pool
            while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        if budget is not None:budget.check()
        # Preserve all legacy reservoir-score fields. Invalid particles retain
        # their probability and fall back to the baseline synthetic board;
        # they are NOT resampled/renormalized away or hidden in quality reports.
        return ParticleOpponentBelief(**{k:getattr(base,k) for k in OpponentBoardBelief.__dataclass_fields__},pool=pool)

    def metrics(self):
        return {'providerVersion':PROVIDER_VERSION,'calls':self.calls,'cacheHits':self.hits,'cacheMisses':self.misses,
            'generatedParticles':self.generated,'invalidParticles':self.invalid,'invalidReasons':dict(self.reasons),
            'minimumRetainedMass':self.minimum_retained_mass,'particleBudget':self.particles,
            'canonicalContextBatch':1,'temporaryAmountsInferred':False,
            'unknownShopReachabilityCertified':False,'invalidPolicy':'retain mass and baseline synthetic fallback'}
