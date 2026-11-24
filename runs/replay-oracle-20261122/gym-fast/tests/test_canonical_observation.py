"""Idempotent returned observations; explicit old-weights/new-input transfer."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from sapai_gym1.rl.action_schema import ActionType,FlatActionCodec
from sapai_gym1.rl.joint_action import JointAction,step_decisions
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.entity_observation import EntityObservationEncoder
from sapai_gym1.rl.decision_manifest import decision_record,restore_decision
from sapai_gym1.rl.observation_contract import (
    OBSERVATION_REVISION,LEGACY_OBSERVATION_REVISION,PREVIOUS_PUBLIC_OBSERVATION_REVISION,
    validate_checkpoint_observation,require_record_observation,
)


def equal(left,right):
    assert left.keys()==right.keys()
    for key in left:np.testing.assert_array_equal(left[key],right[key],err_msg=key)


def buy(env,side):
    return JointAction(next(a for a in map(FlatActionCodec.decode,env.core._avail_actions(side))
                           if a.action_type==ActionType.BUY_PET),0)


@pytest.mark.parametrize('side',[0,1])
def test_real_purchase_step_repeat_and_clone_agree(side):
    env=ParallelSelfPlayEnv();env.reset(seed=71692002)
    result=step_decisions(env,(buy(env,0),buy(env,1))).result
    assert not result.terminated and not result.battle_resolved
    first=result.observations[side]
    assert first['numeric'][0,7]==np.float32(2/3)
    equal(first,env.observations()[side]);equal(first,env.observations()[side])
    equal(first,env.clone().observations()[side])


def old_encoder():
    path=Path('/reference/sapai-gym/sapai_gym1/rl/entity_observation.py')
    assert path.is_file(),'Untouched frozen reference is required, not another candidate instance'
    spec=importlib.util.spec_from_file_location('sapai_gym1.rl._untouched_encoder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.EntityObservationEncoder()


@pytest.mark.parametrize('side',[0,1])
def test_only_intended_first_read_feature_changes_and_old_canonical_matches(side):
    env=ParallelSelfPlayEnv();env.reset(seed=71692002)
    action=buy(env,side)
    env.core.resolve_action(agent_idx=side,action=FlatActionCodec.encode(action.primitive))
    # Current order is not registered yet. Preserve that state exactly for
    # the untouched reference; calling env.clone would itself register it.
    reference=copy.deepcopy(env)
    old=old_encoder();first=old.encode(reference.core,side);canonical=old.encode(reference.core,side)
    current=env.observations()[side]
    equal(current,canonical)
    differences=[(key,tuple(map(int,index))) for key in first for index in np.argwhere(first[key]!=current[key])]
    assert differences==[('numeric',(0,7))]
    assert first['numeric'][0,7]==np.float32(1/3)


@pytest.mark.parametrize('seed',[71693001,71693002,71693003])
def test_repeated_outputs_on_real_random_action_sequences(seed):
    env=ParallelSelfPlayEnv();observations=env.reset(seed=seed);rng=np.random.default_rng(seed)
    for _ in range(60):
        decisions=[]
        for side,active in enumerate(env.active_players()):
            equal(observations[side],env.observations()[side])
            if not active:decisions.append(None);continue
            available=sorted(env.core._avail_actions(side))
            decisions.append(JointAction(FlatActionCodec.decode(int(rng.choice(available))),0))
        result=step_decisions(env,decisions).result
        for side in (0,1):equal(result.observations[side],env.observations()[side])
        if result.terminated:break
        observations=result.observations


@pytest.mark.parametrize('revision',[LEGACY_OBSERVATION_REVISION,PREVIOUS_PUBLIC_OBSERVATION_REVISION])
def test_old_weights_require_explicit_transfer_but_old_records_always_reject(revision):
    checkpoint={'observation_revision':revision}
    with pytest.raises(ValueError):validate_checkpoint_observation(checkpoint)
    assert validate_checkpoint_observation(checkpoint,allow_transfer=True)
    with pytest.raises(ValueError):require_record_observation({'observationRevision':revision})


@pytest.mark.parametrize('revision',['future-unknown',None,''])
def test_unknown_checkpoint_schema_rejected_even_with_transfer(revision):
    with pytest.raises(ValueError):validate_checkpoint_observation({'observation_revision':revision},allow_transfer=True)


def test_current_schema_and_missing_legacy_marker_contract():
    assert not validate_checkpoint_observation({'observation_revision':OBSERVATION_REVISION})
    with pytest.raises(ValueError):validate_checkpoint_observation({})
    assert validate_checkpoint_observation({},allow_transfer=True)
    require_record_observation({'observationRevision':OBSERVATION_REVISION})


@pytest.mark.parametrize('side',[0,1])
def test_new_record_replay_and_old_record_rejection(side):
    env=ParallelSelfPlayEnv();env.reset(seed=71692002);decisions=(buy(env,0),buy(env,1))
    result=step_decisions(env,decisions).result
    memories=[torch.zeros(512),torch.zeros(512)]
    record=decision_record(env,memories,side,('contract-only',),
        {'resetSeed':71692002,'actions':[[d.to_dict() for d in decisions]]},
        search_seed=1,validation_seed=2,truth_seeds=(3,4))
    restored,restored_memory,restored_side,_=restore_decision(record)
    assert restored_side==side and record['observationRevision']==OBSERVATION_REVISION
    equal(result.observations[side],restored.observations()[side])
    for a,b in zip(memories,restored_memory):assert torch.equal(a,b)
    record['observationRevision']=PREVIOUS_PUBLIC_OBSERVATION_REVISION
    with pytest.raises(ValueError):restore_decision(record)
