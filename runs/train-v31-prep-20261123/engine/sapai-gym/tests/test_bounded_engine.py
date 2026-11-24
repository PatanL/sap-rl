"""Budget, actual execution and information-boundary CPU tests."""
from dataclasses import replace
from types import SimpleNamespace
import copy

import numpy as np
import pytest
import torch
from sapai import Team
from sapai_gym1.rl.bounded_engine import BoundedPublicEngine,BoundedEngineEvaluator,EngineLimits
from sapai_gym1.rl.fixed_interactive_evaluation import FixedBatchInteractiveEvaluator
from sapai_gym1.rl.puct_search import PUCTConfig
from sapai_gym1.rl.puct_search import StochasticPUCTSearch
from sapai_gym1.rl.rollout_search import NeuralSearchEvaluator
from sapai_gym1.rl.joint_action import JointAction
from sapai_gym1.rl.action_schema import AutoregressiveAction,ActionType
from sapai_gym1.rl.public_opponent_model import public_state_from_one_step_reservoir
from sapai_gym1.rl.public_battle_board import public_records
from test_initial_public_reservoir import RecordingReservoir,environment
from test_end_freeze_policy import policies,zero_gold_observations

RAW=JointAction(AutoregressiveAction(ActionType.END_TURN),0)
ALT=JointAction(AutoregressiveAction(ActionType.END_TURN),1)


class Clock:
    def __init__(self):self.value=0.
    def __call__(self):return self.value


def limits(**kwargs):
    value=EngineLimits(max_searches=3,searches_per_turn=3,total_decisions=15,total_neural_states=30,
        total_seconds=3.,decisions_per_search=10,neural_states_per_search=20,seconds_per_search=1.,minimum_start_seconds=.01)
    return replace(value,**kwargs)


class Planner:
    calls=[]
    def __init__(self,evaluator,config):pass
    def search(self,state,memory,*,seed,validation_seed):
        # The placeholder may contain the last PUBLIC board. It must not
        # contain the real opponent's hidden current team/shop or gold.
        assert len(state.core.opponent.shop)==0 and state.core.opponent.gold==0
        if state.core.previous_opponent_state is None:assert len(state.core.opponent.team)==0
        else:assert public_records(state.core.opponent.team)==public_records(state.core.previous_opponent_state)
        assert state.clone().work_budget is state.work_budget
        Planner.calls.append((seed,validation_seed,state.observation(),memory.clone()))
        # Explicitly represent work from all phases, not just tree nodes.
        for phase in ('tree','screen','confirmation'):
            state.work_budget.step();state.work_budget.step();state.work_budget.neural()
        state.core.agent.gold=30;memory.add_(100)  # Speculative mutation must not escape.
        return SimpleNamespace(selected_action=ALT,proposed_action=ALT,tree_proposed_action=ALT,omitted_end_mask_probability_mass=.1)


def engine(*,bound=None,clock=None,planner=Planner,reservoir=None,memory_dim=2):
    return BoundedPublicEngine(None,PUCTConfig(continuation_mode='tree_plan'),bound or limits(),
        checkpoint_sha256='a'*64,memory_dim=memory_dim,reservoir=reservoir or RecordingReservoir(),
        selector='screened',clock=clock or Clock(),planner_factory=planner)


def context(side=0):
    env=environment(side);public,meta=public_state_from_one_step_reservoir(env.core,side,RecordingReservoir())
    return env,public,meta


def test_owned_state_memory_and_all_phase_budgets():
    env,public,meta=context();before=public.observation();memory=torch.zeros(2);next_memory=torch.ones(2)
    e=engine();ticket=e.begin(2,is_root=True)
    chosen,returned=e.plan(ticket,public,meta,memory,RAW,next_memory)
    assert chosen==ALT and returned is next_memory and torch.equal(memory,torch.zeros(2))
    for key,value in before.items():np.testing.assert_array_equal(value,public.observation()[key])
    assert env.core.agent.gold==0
    assert e.counts['decisionTransitions']==6 and e.counts['basePolicyNeuralStates']==3
    assert e.traces[0]['root'] and e.traces[0]['initialCandidateCount']==1


def test_total_budget_is_not_reset_for_new_calls():
    _,public,meta=context();e=engine();memory=torch.zeros(2)
    for index in range(3):
        selected,_=e.plan(e.begin(2,is_root=index==0),public,meta,memory,RAW,memory)
        assert selected==(ALT if index<2 else RAW)
    assert e.counts['decisionTransitions']==15 and e.counts['searchAttempts']==3
    assert e.begin(3,is_root=False) is None
    assert any(t['reason'].startswith('budget_') for t in e.traces)


def test_preparation_and_cooperative_overrun_are_charged():
    _,public,meta=context();clock=Clock();e=engine(clock=clock,bound=limits(total_seconds=.5,seconds_per_search=.5))
    ticket=e.begin(2,is_root=True);clock.value=1.  # Time spent in public preparation already exceeds allowance.
    count=len(Planner.calls)
    chosen,_=e.plan(ticket,public,meta,torch.zeros(2),RAW,torch.zeros(2))
    assert chosen==RAW and len(Planner.calls)==count
    assert e.counts['searchWallSeconds']==1. and e.counts['maxDeadlineOvershootSeconds']==.5
    assert e.begin(3,is_root=False) is None


def test_foreign_and_double_closed_tickets_fail():
    a,b=engine(),engine();ticket=a.begin(2,is_root=True)
    with pytest.raises(RuntimeError):b.fallback(ticket,RAW,torch.zeros(2),'missing_public_support')
    a.fallback(ticket,RAW,torch.zeros(2),'missing_public_support')
    with pytest.raises(RuntimeError):a.fallback(ticket,RAW,torch.zeros(2),'missing_public_support')


@pytest.mark.parametrize('field,value',[('targetTurn',1),('initialRetrievalInstalled',False),('candidateCount',0),('candidateIds',[])])
def test_stale_or_uninstalled_initial_belief_is_fatal(field,value):
    _,public,meta=context();meta[field]=value;e=engine()
    with pytest.raises(ValueError,match='one-step retrieval'):
        e.plan(e.begin(2,is_root=True),public,meta,torch.zeros(2),RAW,torch.zeros(2))
    assert e.counts['fatal']==1


def test_illegal_or_malformed_planner_output_is_not_a_raw_fallback():
    class Bad(Planner):
        def search(self,*args,**kwargs):
            result=super().search(*args,**kwargs);result.proposed_action=object();return result
    _,public,meta=context();e=engine(planner=Bad)
    with pytest.raises(ValueError):e.plan(e.begin(2,is_root=True),public,meta,torch.zeros(2),RAW,torch.zeros(2))
    assert e.counts['fatal']==1 and e.counts['decisionTransitions']==6


@pytest.mark.parametrize('change',[{'max_searches':-1},{'max_searches':True},{'seconds_per_search':float('nan')},
    {'total_seconds':-1.},{'minimum_start_seconds':2.}])
def test_bad_limits_fail_closed(change):
    with pytest.raises(ValueError):limits(**change)


def test_turn_and_minimum_admission_limits_are_explicit():
    e=engine(bound=limits(searches_per_turn=1));ticket=e.begin(2,is_root=True)
    e.fallback(ticket,RAW,torch.zeros(2),'missing_public_support')
    assert e.begin(2,is_root=False) is None
    assert e.begin(3,is_root=False) is not None
    tiny=engine(bound=limits(total_seconds=.001))
    assert tiny.begin(2,is_root=True) is None and tiny.counts['searchAttempts']==0


@pytest.mark.parametrize('side',[0,1])
def test_truth_private_state_and_future_seed_never_reach_engine(side):
    env,_,_=context(side);before=env.observations()[side];Planner.calls=[]
    fake_policy=SimpleNamespace()
    outer=BoundedEngineEvaluator(fake_policy,engine_factory=lambda:engine(),device='cpu',batch_size=4)
    for seed in (17,29):
        clone=env.clone();clone.redeterminize(seed)
        hidden=clone.core.opponent if side==0 else clone.core.agent
        hidden.team=Team(['pet-dragon']);hidden.team[0].pet._attack=seed;hidden.gold=seed
        record={'environment':clone,'policy_root_action':RAW}
        result,_=outer._root_execution_choice(record,side,before,torch.zeros(2),side,RAW,torch.ones(2))
        assert result==ALT
    assert len(Planner.calls)==2 and Planner.calls[0][:2]==Planner.calls[1][:2]
    for key,value in Planner.calls[0][2].items():np.testing.assert_array_equal(value,Planner.calls[1][2][key])


@pytest.mark.parametrize('side',[0,1])
def test_real_evaluator_root_executes_joint_action_and_preserves_original(side):
    env,_,policy=policies();zero_gold_observations(env);env.core.MAX_TURN=2
    with torch.no_grad():policy.type_head.bias[ActionType.END_TURN]=100
    memories=[policy.initial_memory(1)[0] for _ in range(2)];before=env.observations()
    make=lambda:engine(bound=limits(max_searches=1),memory_dim=policy.config.memory_dim)
    outer=BoundedEngineEvaluator(policy,engine_factory=make,batch_size=4,horizon_battles=2,max_steps=100)
    results=outer.evaluate(env,memories,side,(RAW,),(17,19))
    assert len(results)==2 and all(r.terminated and r.match_score is not None for r in results)
    assert all(r.root_action==RAW and r.executed_root_action==ALT for r in results)
    assert len(outer.engines)==2 and all(e.counts['searchAttempts']==1 for e in outer.engines.values())
    for a,b in zip(before,env.observations()):
        for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert all(torch.equal(m,torch.zeros_like(m)) for m in memories)


def test_disabled_engine_has_exact_baseline_outcomes_and_no_public_cloning(monkeypatch):
    import sapai_gym1.rl.bounded_engine as module
    env,_,policy=policies();zero_gold_observations(env);env.core.MAX_TURN=2
    with torch.no_grad():policy.type_head.bias[ActionType.END_TURN]=100
    memories=[policy.initial_memory(1)[0] for _ in range(2)]
    baseline=FixedBatchInteractiveEvaluator(policy,batch_size=4,horizon_battles=2,max_steps=100)
    def forbidden(*args,**kwargs):raise AssertionError('Cloned public state with disabled budget')
    monkeypatch.setattr(module,'public_state_from_one_step_reservoir',forbidden)
    outer=BoundedEngineEvaluator(policy,engine_factory=lambda:engine(bound=limits(max_searches=0),memory_dim=policy.config.memory_dim),
        batch_size=4,horizon_battles=2,max_steps=100)
    assert baseline.evaluate(env,memories,0,(RAW,),(17,19))==outer.evaluate(env,memories,0,(RAW,),(17,19))


def test_missing_support_keeps_all_terminal_futures_as_raw_fallbacks():
    env,_,policy=policies();zero_gold_observations(env);env.core.MAX_TURN=2
    with torch.no_grad():policy.type_head.bias[ActionType.END_TURN]=100
    reservoir=RecordingReservoir();reservoir.opening_prior=lambda **kwargs:[]
    outer=BoundedEngineEvaluator(policy,engine_factory=lambda:engine(reservoir=reservoir,memory_dim=policy.config.memory_dim),
        batch_size=4,horizon_battles=2,max_steps=100)
    memories=[policy.initial_memory(1)[0] for _ in range(2)]
    results=outer.evaluate(env,memories,0,(RAW,),(17,19))
    assert len(results)==2 and all(r.terminated and r.executed_root_action==RAW for r in results)
    assert all(e.counts['missing_public_support']==1 for e in outer.engines.values())


def test_shared_engine_between_futures_is_rejected():
    env,_,policy=policies();zero_gold_observations(env);env.core.MAX_TURN=2
    with torch.no_grad():policy.type_head.bias[ActionType.END_TURN]=100
    shared=engine(bound=limits(max_searches=0),memory_dim=policy.config.memory_dim)
    outer=BoundedEngineEvaluator(policy,engine_factory=lambda:shared,batch_size=4,horizon_battles=2,max_steps=100)
    with pytest.raises(ValueError,match='fresh bounded engine'):
        outer.evaluate(env,[policy.initial_memory(1)[0] for _ in range(2)],0,(RAW,),(17,19))


@pytest.mark.parametrize('side',[0,1])
def test_actual_second_decision_replans_with_exact_live_memory(side):
    env,_,policy=policies();zero_gold_observations(env);env.core.MAX_TURN=2
    with torch.no_grad():policy.type_head.bias[ActionType.END_TURN]=100
    class FreezeThenEnd:
        def __init__(self,evaluator,config):pass
        def search(self,state,memory,**kwargs):
            state.work_budget.step();state.work_budget.neural()
            action=RAW if state.core.agent.shop[0].frozen else JointAction(AutoregressiveAction(ActionType.FREEZE,0),0)
            memory.add_(1000)  # A hypothetical update is not a live update.
            return SimpleNamespace(selected_action=action,proposed_action=action,tree_proposed_action=action,omitted_end_mask_probability_mass=0.)
    class Spy(BoundedEngineEvaluator):
        def _root_execution_choice(self,record,acting_side,observation,memory,side,action,next_memory):
            record['test_live_root_next']=next_memory.clone()
            return super()._root_execution_choice(record,acting_side,observation,memory,side,action,next_memory)
        def _execution_choice(self,record,acting_side,observation,memory,side,action,next_memory):
            if side==acting_side:
                torch.testing.assert_close(memory,record['test_live_root_next'],rtol=0,atol=0)
            return super()._execution_choice(record,acting_side,observation,memory,side,action,next_memory)
    outer=Spy(policy,engine_factory=lambda:engine(bound=limits(max_searches=2,searches_per_turn=2),
        memory_dim=policy.config.memory_dim,planner=FreezeThenEnd),batch_size=4,horizon_battles=2,max_steps=100)
    results=outer.evaluate(env,[policy.initial_memory(1)[0] for _ in range(2)],side,(RAW,),(17,19))
    assert all(r.terminated and r.executed_root_action.primitive.action_type==ActionType.FREEZE for r in results)
    assert all(e.counts['searchAttempts']==2 and e.counts['decisionTransitions']==2 for e in outer.engines.values())
    assert all([t['root'] for t in e.traces]==[True,False] for e in outer.engines.values())


def test_real_puct_uses_the_shared_budget_without_global_rng_or_memory_mutation():
    env,public,meta=context();_,_,policy=policies()
    memory=policy.initial_memory(1)[0]
    obs={k:torch.as_tensor(v)[None] for k,v in env.observations()[0].items()}
    with torch.no_grad():output=policy.act(obs,memory[None],deterministic=True)
    raw=JointAction(AutoregressiveAction(*output.actions[0].tolist()),int(output.end_freeze_mask[0]))
    cfg=PUCTConfig(max_simulations=2,max_simulator_transitions=10,horizon_battles=1,tree_leaf_rollout_steps=0,
        max_rollout_steps=30,root_screening_top_k=4,root_screening_min_survivors=2,root_screening_rollouts=2,
        root_screening_rounds=1,validation_rollouts=2,max_validation_rollouts=2,audit_rollouts=0,continuation_mode='tree_plan')
    e=BoundedPublicEngine(NeuralSearchEvaluator(policy,joint_mask_budget=2,joint_exploration_masks=0),cfg,
        limits(total_decisions=500,total_neural_states=500,total_seconds=15.,decisions_per_search=500,
               neural_states_per_search=500,seconds_per_search=15.),checkpoint_sha256='a'*64,
        memory_dim=policy.config.memory_dim,reservoir=RecordingReservoir(),selector='screened')
    before=torch.get_rng_state().clone();np_before=np.random.get_state()
    chosen,next_memory=e.plan(e.begin(2,is_root=True),public,meta,memory,raw,output.memory[0])
    assert e.counts['changed']+e.counts['unchanged']==1
    assert 0<e.counts['decisionTransitions']<=500 and 0<e.counts['basePolicyNeuralStates']<=500
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(memory,torch.zeros_like(memory))
    np_after=np.random.get_state()
    assert np_before[0]==np_after[0] and np_before[2:]==np_after[2:]
    np.testing.assert_array_equal(np_before[1],np_after[1])
    assert next_memory.data_ptr()==output.memory[0].data_ptr()
