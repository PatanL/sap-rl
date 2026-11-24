"""CPU algorithm-contract tests, not SAP strength evidence."""
from dataclasses import fields,is_dataclass,asdict
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from sapai_gym1.rl.puct_search import PUCTConfig,StochasticPUCTSearch
from sapai_gym1.rl.rollout_search import PolicyEvaluation,SearchTransition
from test_puct_search import ToyComboState,ToyComboEvaluator


def config(mode='legacy_plan_vs_raw'):
    return PUCTConfig(max_simulations=256,max_simulator_transitions=900,max_tree_depth=8,max_rollout_steps=8,
        horizon_battles=1,world_pool_size=8,min_root_visits=2,cpuct=3.,prior_floor=.4,leaf_value_weight=0.,
        validation_rollouts=16,max_validation_rollouts=16,audit_rollouts=16,confidence_z=0.,
        minimum_improvement=.1,continuation_mode=mode)


class RecordingSearch(StochasticPUCTSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.calls=[]
    def _screen_rollouts(self,state,memory,actions,seeds,*,use_tree):
        self.calls.append(('screen',tuple(actions),tuple(map(int,seeds)),use_tree))
        return super()._screen_rollouts(state,memory,actions,seeds,use_tree=use_tree)
    def _confirmation_rollouts(self,state,memory,actions,seeds,*,use_tree):
        self.calls.append(('confirmation',tuple(actions),tuple(map(int,seeds)),use_tree))
        return super()._confirmation_rollouts(state,memory,actions,seeds,use_tree=use_tree)


def test_student_mode_does_not_credit_an_unexecuted_followup():
    state=ToyComboState();memory=torch.zeros(2)
    legacy=RecordingSearch(ToyComboEvaluator(),config())
    old=legacy.search(state,memory,seed=77,validation_seed=301)
    student=RecordingSearch(ToyComboEvaluator(),config('student'))
    new=student.search(state,memory,seed=77,validation_seed=301)
    assert old.tree_proposed_action==old.proposed_action==state.shark and old.accepted
    assert new.tree_proposed_action==state.shark  # Same deeper tree; only the continuation contract changed.
    assert new.proposed_action==new.policy_action==state.monkey and not new.accepted
    assert new.validation_return_improvement==0.
    assert all(not use_tree for _,_,_,use_tree in student.calls)
    assert new.targets==() and new.diagnostic_targets_suppressed
    assert new.continuation_mode=='student'
    assert old.visit_counts==new.visit_counts and old.q_values==new.q_values


class BothPlansState(ToyComboState):
    """Both initial actions have a good follow-up; the raw first action is best."""
    def legal_actions(self):
        return [self.fly,self.end] if self.phase=='monkey' else super().legal_actions()
    def step(self,action,rng):
        if self.phase=='root':return super().step(action,rng)
        reward=(1. if action==self.fly else .2) if self.phase=='monkey' else (.9 if action==self.fly else 0.)
        self.terminated=True;self.battles=1
        return SearchTransition(reward,0.,True,0 if reward>.2 else 1)


class BothPlansEvaluator(ToyComboEvaluator):
    def evaluate(self,state,memory):
        if state.phase=='monkey':return PolicyEvaluation({state.fly:.05,state.end:.95},0.,torch.zeros(2))
        return super().evaluate(state,memory)


class KnownContinuationSearch(RecordingSearch):
    """Test-defined improved continuation, not a claim that PUCT discovered it.

    Isolate candidate-ranking semantics from whether a finite tree happens to
    explore both branches enough. Tree construction itself remains unchanged.
    """
    def tree_action(self,state,memory):
        if state.phase in ('monkey','shark'):
            return state.fly,torch.zeros(2)
        return None


def test_tree_plan_ranks_raw_root_with_its_good_followups_instead_of_zero():
    state=BothPlansState();memory=torch.zeros(2)
    legacy=KnownContinuationSearch(BothPlansEvaluator(),config());old=legacy.search(state,memory,seed=77,validation_seed=301)
    tree=KnownContinuationSearch(BothPlansEvaluator(),config('tree_plan'));new=tree.search(state,memory,seed=77,validation_seed=301)
    assert old.tree_proposed_action==new.tree_proposed_action==state.monkey
    assert old.proposed_action==state.shark  # Asymmetric ranking credits only the alternative's improved continuation.
    assert new.proposed_action==new.policy_action==state.monkey
    assert new.root_screening_scores[state.monkey]==pytest.approx(.8)
    assert new.root_screening_scores[state.shark]==pytest.approx(.7)
    assert new.accepted and new.validation_return_improvement==pytest.approx(.8)
    assert new.validation_tree_actions_mean>0
    assert new.screening_simulator_transitions>old.screening_simulator_transitions  # Extra raw-root planned work is counted.
    assert new.targets==() and new.diagnostic_targets_suppressed
    assert tree._training_targets()==()
    assert old.visit_counts==new.visit_counts and old.q_values==new.q_values
    planned=[actions for stage,actions,_,use_tree in tree.calls if stage=='screen' and use_tree]
    assert planned and all(state.monkey in actions for actions in planned)


@pytest.mark.parametrize('mode',['student','tree_plan'])
def test_modes_keep_matched_and_fresh_confirmation_worlds(mode):
    search=RecordingSearch(BothPlansEvaluator(),config(mode))
    search.search(BothPlansState(),torch.zeros(2),seed=77,validation_seed=301)
    screen=[c for c in search.calls if c[0]=='screen'];confirmation=[c for c in search.calls if c[0]=='confirmation']
    assert len(screen)%2==len(confirmation)%2==0
    for rows in (screen,confirmation):
        for a,b in zip(rows[::2],rows[1::2]):assert a[2]==b[2]
    screened={s for _,_,seeds,_ in screen for s in seeds}
    confirmed={s for _,_,seeds,_ in confirmation for s in seeds}
    assert screened.isdisjoint(confirmed)
    if len(confirmation)>2:assert set(confirmation[0][2]).isdisjoint(confirmation[-1][2])


@pytest.mark.parametrize('bad',[None,'raw',True,[]])
def test_unknown_contract_fails_closed(bad):
    with pytest.raises(ValueError,match='continuation contract'):
        StochasticPUCTSearch(ToyComboEvaluator(),PUCTConfig(continuation_mode=bad))


@pytest.mark.parametrize('mode',['student','tree_plan'])
def test_selection_only_cannot_mislabel_a_skipped_contract(mode):
    with pytest.raises(ValueError,match='selection-only'):
        StochasticPUCTSearch(ToyComboEvaluator(),PUCTConfig(continuation_mode=mode,selection_only=True))


def equal(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif is_dataclass(a):
        for f in fields(a):
            if not f.name.endswith('wall_seconds'):equal(getattr(a,f.name),getattr(b,f.name))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_default_legacy_is_exact_against_untouched_original_module():
    path=Path('/reference-puct.py')
    if not path.exists():pytest.skip('Explicit unchanged reference module mount required')
    spec=importlib.util.spec_from_file_location('sapai_gym1.rl._execution_reference_v14',path)
    reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
    args=asdict(config());args.pop('continuation_mode')
    for state,evaluator in ((ToyComboState(),ToyComboEvaluator()),(BothPlansState(),BothPlansEvaluator())):
        old=reference.StochasticPUCTSearch(evaluator,reference.PUCTConfig(**args)).search(state,torch.zeros(2),seed=77,validation_seed=301)
        new=StochasticPUCTSearch(evaluator,PUCTConfig(**args)).search(state,torch.zeros(2),seed=77,validation_seed=301)
        equal(old,new)  # Excludes only wall time and fields absent from the reference.
        assert new.continuation_mode=='legacy_plan_vs_raw' and not new.diagnostic_targets_suppressed
