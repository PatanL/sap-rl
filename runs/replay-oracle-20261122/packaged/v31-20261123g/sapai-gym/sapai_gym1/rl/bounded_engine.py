"""Budgeted public-only replanning; no production teaching authorization."""
from collections import Counter
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import re
import time

import torch

from sapai.turtle_pack import SIMULATOR_REVISION
from .action_schema import AutoregressiveAction,FlatActionCodec
from .joint_action import JointAction,PRIMITIVE_SCHEMA
from .decision_manifest import observation_hash
from .observation_contract import OBSERVATION_REVISION
from .fixed_interactive_evaluation import FixedBatchInteractiveEvaluator
from .public_opponent_model import (
    public_state_from_one_step_reservoir,PublicOpponentSupportMissing,ONE_STEP_PUBLIC_BELIEF_VERSION,
)
from .opponent_belief import OpponentBeliefMix
from .rollout_search import PublicSearchState
from .puct_search import PUCTConfig,StochasticPUCTSearch
from .search_work_budget import SearchWorkBudget,SearchBudgetExceeded

ENGINE_VERSION='bounded-public-replanning-v1-20260907'


@dataclass(frozen=True)
class EngineLimits:
    max_searches: int
    searches_per_turn: int
    total_decisions: int
    total_neural_states: int
    total_seconds: float
    decisions_per_search: int
    neural_states_per_search: int
    seconds_per_search: float
    minimum_start_seconds: float

    def __post_init__(self):
        for name in ('max_searches','searches_per_turn','total_decisions','total_neural_states',
                     'decisions_per_search','neural_states_per_search'):
            value=getattr(self,name)
            if type(value) is not int or value<0:raise ValueError(f'Invalid integer engine limit: {name}')
        for name in ('total_seconds','seconds_per_search','minimum_start_seconds'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError(f'Invalid finite engine time limit: {name}')
        if min(self.decisions_per_search,self.neural_states_per_search,self.seconds_per_search,self.minimum_start_seconds)<=0:
            raise ValueError('Per-call/minimum-start budgets must be positive; disable with max_searches=0')
        if self.minimum_start_seconds>self.seconds_per_search:
            raise ValueError('Minimum start time exceeds per-search allocation')


@dataclass(frozen=True)
class SearchTicket:
    owner: object
    turn: int
    is_root: bool
    budget: SearchWorkBudget
    allowed_decisions: int
    allowed_neural: int
    allowed_seconds: float


def action_record(action):
    if isinstance(action,JointAction):return action.to_dict()
    if isinstance(action,AutoregressiveAction):return JointAction(action).to_dict()
    raise ValueError('Planner returned an unsupported action object')


class BoundedPublicEngine:
    def __init__(self,evaluator,config,limits,*,checkpoint_sha256,memory_dim,reservoir,selector,
                 mixture=None,candidate_limit=32,opening_seed=71520000,clock=time.monotonic,
                 planner_factory=StochasticPUCTSearch):
        if not isinstance(config,PUCTConfig) or not isinstance(limits,EngineLimits):raise TypeError('Typed planner config and limits required')
        if not isinstance(checkpoint_sha256,str) or not re.fullmatch('[0-9a-f]{64}',checkpoint_sha256):raise ValueError('Exact model ownership hash required')
        if type(memory_dim) is not int or memory_dim<=0:raise ValueError('Positive memory dimension required')
        if selector not in ('gated','screened','tree'):raise ValueError('Explicit root selector required')
        if type(candidate_limit) is not int or candidate_limit<=0:raise ValueError('Positive retrieval candidate limit required')
        if type(opening_seed) is not int or opening_seed<0:raise ValueError('Explicit opening population seed required')
        self.evaluator,self.config,self.limits=evaluator,config,limits
        self.checkpoint_sha256,self.memory_dim=checkpoint_sha256,memory_dim
        self.reservoir,self.selector=reservoir,selector
        self.mixture=mixture or OpponentBeliefMix();self.candidate_limit=candidate_limit;self.opening_seed=opening_seed
        self.clock,self.planner_factory=clock,planner_factory
        self.config_hash=hashlib.sha256(json.dumps(asdict(config),sort_keys=True,allow_nan=False).encode()).hexdigest()
        self.counts=Counter();self.by_turn=Counter();self.traces=[];self._owner=object();self._active=None

    def begin(self,turn,*,is_root):
        if self._active is not None:raise RuntimeError('Overlapping searches in one engine budget')
        if type(turn) is not int or turn<1:raise ValueError('Invalid public turn')
        self.counts['ownDecisionsSeen']+=1
        limits=self.limits
        remaining_seconds=limits.total_seconds-self.counts['searchWallSeconds']
        reason=(
            'attempt_limit' if self.counts['searchAttempts']>=limits.max_searches else
            'turn_limit' if self.by_turn[turn]>=limits.searches_per_turn else
            'decision_limit' if self.counts['decisionTransitions']>=limits.total_decisions else
            'neural_limit' if self.counts['basePolicyNeuralStates']>=limits.total_neural_states else
            'time_admission' if min(limits.seconds_per_search,remaining_seconds)<limits.minimum_start_seconds else None)
        if reason:
            self.counts['skip_'+reason]+=1
            return None
        decisions=min(limits.decisions_per_search,limits.total_decisions-self.counts['decisionTransitions'])
        neural=min(limits.neural_states_per_search,limits.total_neural_states-self.counts['basePolicyNeuralStates'])
        seconds=min(limits.seconds_per_search,remaining_seconds)
        budget=SearchWorkBudget(max_decisions=decisions,max_neural_states=neural,seconds=seconds,clock=self.clock)
        ticket=SearchTicket(self._owner,turn,bool(is_root),budget,decisions,neural,seconds)
        self._active=ticket;self.counts['searchAttempts']+=1;self.by_turn[turn]+=1
        return ticket

    def _require_ticket(self,ticket):
        if ticket is not self._active or ticket.owner is not self._owner:raise RuntimeError('Foreign or already closed search ticket')

    def _finish(self,ticket,reason,**details):
        self._require_ticket(ticket);work=ticket.budget;elapsed=self.clock()-work.started
        if not (0<=work.decisions<=ticket.allowed_decisions and 0<=work.neural_states<=ticket.allowed_neural and elapsed>=0):
            raise RuntimeError('Planner violated its owned accounting limits')
        self.counts['decisionTransitions']+=work.decisions;self.counts['basePolicyNeuralStates']+=work.neural_states
        self.counts['searchWallSeconds']+=elapsed
        self.counts['maxDeadlineOvershootSeconds']=max(self.counts['maxDeadlineOvershootSeconds'],max(0.,elapsed-ticket.allowed_seconds))
        self.counts[reason]+=1
        self.traces.append({'turn':ticket.turn,'root':ticket.is_root,'reason':reason,'decisions':work.decisions,
            'basePolicyNeuralStates':work.neural_states,'wallSeconds':elapsed,'allocatedSeconds':ticket.allowed_seconds,**details})
        self._active=None

    def fallback(self,ticket,raw_action,next_memory,reason):
        self._finish(ticket,reason,executed=action_record(raw_action))
        return raw_action,next_memory

    def fail(self,ticket,error):
        self._finish(ticket,'fatal',error=repr(error))

    @staticmethod
    def _legal(action,public,raw):
        record=action_record(action);raw_record=action_record(raw)
        if record['schema']!=raw_record['schema']:raise ValueError('Planner action schema differs from live policy; never drop a mask')
        if isinstance(action,JointAction):action.expand(public.core,0)
        elif FlatActionCodec.encode(action) not in public.core._avail_actions(0):raise ValueError('Planner returned an illegal primitive')

    def plan(self,ticket,public,metadata,memory,raw_action,next_memory):
        self._require_ticket(ticket);seed=None
        try:
            ticket.budget.check()
            if not isinstance(public,PublicSearchState) or public.core.turn!=ticket.turn:
                raise ValueError('Engine requires the current sanitized public state, never a truth environment')
            if (metadata.get('version')!=ONE_STEP_PUBLIC_BELIEF_VERSION or metadata.get('retrievalHorizon')!=1
                    or metadata.get('candidateIds')!=list(public.opponent_trajectory_ids)
                    or metadata.get('targetTurn')!=ticket.turn or metadata.get('initialRetrievalInstalled') is not True
                    or type(metadata.get('candidateCount')) is not int or metadata['candidateCount']<1
                    or metadata['candidateCount']!=len(public.opponent_trajectory_ids)):
                raise ValueError('Initial one-step retrieval was not installed')
            if public.work_budget is not None:raise ValueError('Do not replace an existing external search budget')
            for value in (memory,next_memory):
                if not torch.is_tensor(value) or value.shape!=(self.memory_dim,) or not torch.isfinite(value).all():
                    raise ValueError('Invalid live recurrent memory')
            owned_memory=memory.detach().float().cpu().clone();work=public.clone();work.work_budget=ticket.budget
            work.enable_learned_opponent_belief(self.mixture)
            self._legal(raw_action,work,raw_action)
            digest=hashlib.sha256((ENGINE_VERSION+SIMULATOR_REVISION+OBSERVATION_REVISION+self.checkpoint_sha256+
                self.config_hash+observation_hash(public.observation())).encode()+owned_memory.numpy().tobytes()).digest()
            seed=int.from_bytes(digest[:8],'little')%(2**63-1)
            validation_seed=(seed+1)%(2**63-1)
            planner=self.planner_factory(self.evaluator,self.config)
            result=planner.search(work,owned_memory,seed=seed,validation_seed=validation_seed)
            ticket.budget.check()
            selected=getattr(result,{'gated':'selected_action','screened':'proposed_action','tree':'tree_proposed_action'}[self.selector])
            self._legal(selected,public,raw_action)
            omitted=float(result.omitted_end_mask_probability_mass)
            if not math.isfinite(omitted) or not 0<=omitted<=1:raise ValueError('Invalid omitted proposal mass')
            changed=action_record(selected)!=action_record(raw_action)
            self._finish(ticket,'changed' if changed else 'unchanged',seed=seed,selector=self.selector,
                executed=action_record(selected),raw=action_record(raw_action),
                omittedMaskMass=omitted,initialCandidateCount=metadata['candidateCount'])
            # Reuse the already-computed LIVE update. Speculative GRU updates
            # are never committed or imported from the planner.
            return selected,next_memory
        except SearchBudgetExceeded as error:
            return self.fallback(ticket,raw_action,next_memory,'budget_'+str(error))
        except BaseException as error:
            if self._active is ticket:self.fail(ticket,error)
            raise

    def report(self):
        return {'version':ENGINE_VERSION,'limits':asdict(self.limits),'selector':self.selector,
            'checkpointSHA256':self.checkpoint_sha256,'plannerConfigHash':self.config_hash,
            'counts':dict(self.counts),'traces':list(self.traces),
            'budgetScope':'All decision transitions/base-policy neural states across tree, screening and confirmation per actual future; wall includes initialization and is cooperative.',
            'hardRealTimeGuarantee':False,'trainingTargetsAuthorized':False}


class BoundedEngineEvaluator(FixedBatchInteractiveEvaluator):
    """Trusted truth boundary; the actual engine never receives record/seed/env."""
    def __init__(self,policy,*,engine_factory,**kwargs):
        super().__init__(policy,**kwargs);self.engine_factory=engine_factory;self.engines={};self._used_engines=set()

    def evaluate(self,environment,memories,acting_side,actions,seeds):
        actions=tuple(actions)
        if len(actions)!=1:raise ValueError('Full-engine evaluation needs one comparator root label; use the fixed evaluator for forced alternatives')
        self.engines={};self._used_engines=set()
        return super().evaluate(environment,memories,acting_side,actions,seeds)

    def _choose_with_engine(self,record,acting_side,observation,memory,side,action,next_memory,*,root):
        if side!=acting_side:return action,next_memory
        key=id(record)
        if key not in self.engines:
            engine=self.engine_factory()
            if not isinstance(engine,BoundedPublicEngine) or id(engine) in self._used_engines:
                raise ValueError('Every independent future requires a fresh bounded engine')
            self._used_engines.add(id(engine));self.engines[key]=engine
        engine=self.engines[key];ticket=engine.begin(int(record['environment'].core.turn),is_root=root)
        if ticket is None:return action,next_memory  # No public cloning after budget admission fails.
        core=record['environment'].core
        latest=core.previous_opponent_state if side==0 else core.previous_agent_state
        try:
            public,metadata=public_state_from_one_step_reservoir(core,side,engine.reservoir,
                limit=engine.candidate_limit,opening_seed=engine.opening_seed if latest is None else None)
        except PublicOpponentSupportMissing:
            return engine.fallback(ticket,action,next_memory,'missing_public_support')
        except BaseException as error:
            engine.fail(ticket,error);raise
        return engine.plan(ticket,public,metadata,memory,action,next_memory)

    def _root_execution_choice(self,record,acting_side,observation,memory,side,action,next_memory):
        # The requested root labels this comparison; the full engine starts
        # from the actual raw policy action and may replace it. Always record
        # the executed root, rather than pretending the label was forced.
        raw=record['policy_root_action']
        return self._choose_with_engine(record,acting_side,observation,memory,side,raw,next_memory,root=True)

    def _execution_choice(self,record,acting_side,observation,memory,side,action,next_memory):
        return self._choose_with_engine(record,acting_side,observation,memory,side,action,next_memory,root=False)

    def execution_reports(self):return [engine.report() for engine in self.engines.values()]
