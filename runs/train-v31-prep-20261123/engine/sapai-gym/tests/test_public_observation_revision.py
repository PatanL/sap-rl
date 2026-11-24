"""Conservative observation migration, NOT evidence of live UI visibility.

Old panel actions are replayed only as consumed audit fixtures. No old GRU
memory, old outcome, or rewritten observation hash is used as fresh evidence.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from sapai import Pet, Team

from sapai_gym1.rl import ParallelSelfPlayEnv
from sapai_gym1.rl.observation_contract import (
    OBSERVATION_REVISION, LEGACY_OBSERVATION_REVISION,
    validate_checkpoint_observation, require_record_observation,
)
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.action_schema import AutoregressiveAction
from sapai_gym1.rl.rollout_search import PublicSearchState
from sapai_gym1.rl.decision_manifest import decision_record, restore_decision
from sapai_gym1.rl.joint_search_records import make_record, restore_record
from sapai_gym1.rl.training import stack_observations
from sapai_gym1.deployment.policy_adapter import _pet_from_view
from sapai_gym1.deployment.state_schema import PetView
from test_end_freeze_policy import policies

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_entity_selfplay import save_checkpoint
from evaluate_paired_policy import load_policy


def assert_observations_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def populated(side, level, perk):
    env = ParallelSelfPlayEnv(); env.reset(seed=412)
    c = env.core; c.turn = c.agent.turn = c.opponent.turn = 8
    enemy = Pet('ant'); enemy.level = level; enemy.status = perk
    enemy.experience = 2 if level == 2 else 0
    enemy._attack = 6; enemy._health = 9; enemy.sell_value = level + 3
    own = Pet('fish'); own.level = 2; own.sell_value = 5
    c.previous_agent_state = Team([own if side == 0 else enemy])
    c.previous_opponent_state = Team([enemy if side == 0 else own])
    c.previous_previous_agent_state = copy.deepcopy(c.previous_agent_state)
    c.previous_previous_opponent_state = copy.deepcopy(c.previous_opponent_state)
    player = c.agent if side == 0 else c.opponent
    player.team = Team([copy.deepcopy(own)])
    player._connect_objects()
    # Manual fixture surgery bypasses the environment's action path. Resolve
    # its ordinary legality bookkeeping before taking the comparison snapshot.
    c._agent_actions_cache = c._opponent_actions_cache = None
    c._avail_actions(0); c._avail_actions(1)
    return env


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('level', [1, 2, 3])
@pytest.mark.parametrize('perk', ['status-cake', 'status-garlic', 'none'])
def test_opponent_bonus_hidden_from_encoder_and_public_clone(side, level, perk):
    env = populated(side, level, perk); before = env.observations()[side]
    assert before['numeric'][16, 15] == np.float32(level/10)
    assert before['numeric'][1, 15] == np.float32(.5)  # owned exact sell value
    observed = env.core.previous_opponent_state if side == 0 else env.core.previous_agent_state
    older = env.core.previous_previous_opponent_state if side == 0 else env.core.previous_previous_agent_state
    observed[0].pet.sell_value = 99; older[0].pet.sell_value = 88
    assert_observations_equal(before, env.observations()[side])
    state = PublicSearchState.from_core(env.core, side)
    assert_observations_equal(before, state.observation())
    assert_observations_equal(before, state.clone().observation())


def test_owned_and_shop_sell_values_remain_exact():
    env = populated(0, 2, 'status-cake')
    before = env.observations()[0]
    env.core.agent.team[0].pet.sell_value = 7
    env.core.agent.shop[0].obj.sell_value = 6
    after = env.observations()[0]
    assert before['numeric'][1,15] == np.float32(.5)
    assert after['numeric'][1,15] == np.float32(.7)
    assert after['numeric'][6,15] == np.float32(.6)


@pytest.mark.parametrize('level', [1,2,3])
def test_live_fallback_uses_public_base_level_not_always_one(level):
    pet = _pet_from_view(PetView(item_id='pet-ant', attack=6, health=9, level=level))
    assert pet.sell_value == level
    explicit = _pet_from_view(PetView(item_id='pet-ant', attack=6, health=9, level=level, sell_value=8))
    assert explicit.sell_value == 8


def test_checkpoint_requires_explicit_transfer_but_never_unknown_revision():
    for old in ({}, {'observation_revision': LEGACY_OBSERVATION_REVISION}):
        with pytest.raises(ValueError, match='Observation contract mismatch'):
            validate_checkpoint_observation(old)
        assert validate_checkpoint_observation(old, allow_transfer=True)
    assert validate_checkpoint_observation({'observation_revision': OBSERVATION_REVISION}) is False
    for wrong in (None, 'unknown', ''):
        with pytest.raises(ValueError):
            validate_checkpoint_observation({'observation_revision': wrong}, allow_transfer=True)


def test_actual_checkpoint_writer_evaluator_load_and_resume(tmp_path):
    env, _, model = policies(); opt = torch.optim.AdamW(model.parameters(), lr=.001)
    batch = stack_observations(env.observations(), 'cpu')
    model.value_only(batch).square().sum().backward(); opt.step()
    path = tmp_path/'current.pt'
    save_checkpoint(path, model, opt, model.config, 4, 100, discount_mode='per_turn', discount=.99,
        battle_aux_cost=.05, opponent_aux_cost=.1, opponent_module_learning_rate=.001, opponent_adapter_learning_rate=.001)
    restored, payload = load_policy(path, torch.device('cpu'))
    assert payload['observation_revision'] == OBSERVATION_REVISION
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.9); opt2.load_state_dict(payload['optimizer'])
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key, value in opt.state.get(p, {}).items():
            torch.testing.assert_close(value, opt2.state[q][key], rtol=0, atol=0)
            assert value.data_ptr() != opt2.state[q][key].data_ptr()
    with torch.inference_mode():
        rng = torch.get_rng_state(); a = model.act(batch); rng_a = torch.get_rng_state()
        torch.set_rng_state(rng); b = restored.act(batch)
    for field in vars(a): torch.testing.assert_close(getattr(a,field), getattr(b,field), rtol=0, atol=0)
    assert torch.equal(rng_a, torch.get_rng_state())
    payload.pop('observation_revision'); torch.save(payload, tmp_path/'legacy.pt')
    with pytest.raises(ValueError): load_policy(tmp_path/'legacy.pt', torch.device('cpu'))
    loaded, _ = load_policy(tmp_path/'legacy.pt', torch.device('cpu'), allow_observation_transfer=True)
    for p,q in zip(model.parameters(), loaded.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)


@pytest.mark.parametrize('side', [0,1])
@pytest.mark.parametrize('joint', [False,True])
@pytest.mark.parametrize('greedy', [False,True])
def test_private_counter_perturbation_does_not_change_action_memory_or_rng(side,joint,greedy):
    _, primitive, joint_policy = policies(); model = joint_policy if joint else primitive
    env = populated(side, 2, 'status-cake'); a = env.observations()[side]
    previous = env.core.previous_opponent_state if side==0 else env.core.previous_agent_state
    previous[0].pet.sell_value = 91; b = env.observations()[side]
    memory = model.initial_memory(1)
    with torch.inference_mode():
        rng = torch.get_rng_state()
        first = model(stack_observations([a],'cpu'),memory=memory,deterministic=greedy,allow_end_freeze=joint)
        after = torch.get_rng_state(); torch.set_rng_state(rng)
        second = model(stack_observations([b],'cpu'),memory=memory,deterministic=greedy,allow_end_freeze=joint)
    for field in ('actions','log_prob','memory','value','type_logits'):
        torch.testing.assert_close(getattr(first,field),getattr(second,field),rtol=0,atol=0)
    assert torch.equal(after, torch.get_rng_state())


def test_manifests_records_and_old_memory_fail_closed():
    env, _, model = policies(); memory = [model.initial_memory(1)[0] for _ in range(2)]
    record = decision_record(env,memory,0,(),{'resetSeed':291,'actions':[]},search_seed=1,validation_seed=2,truth_seeds=[3])
    assert record['observationRevision']==OBSERVATION_REVISION
    restore_decision(record)
    old = copy.deepcopy(record); old.pop('observationRevision')
    with pytest.raises(ValueError,match='revision mismatch'): restore_decision(old)
    action = JointAction(AutoregressiveAction(0,10,5),0)
    row = make_record(env.observations()[0],memory[0],[action],[1.],checkpoint_sha256='a'*64)
    restore_record(row,checkpoint_sha256='a'*64)
    row.pop('observationRevision')
    with pytest.raises(ValueError,match='revision mismatch'): restore_record(row,checkpoint_sha256='a'*64)
    with pytest.raises(ValueError): require_record_observation({'observationRevision':'unknown'})


def test_consumed_state17_action_prefix_now_has_public_input_parity():
    path = Path('/audit-panel/state-0017.json')
    if not path.exists(): pytest.skip('Explicit consumed audit panel mount required')
    record = json.loads(path.read_text())
    with pytest.raises(ValueError,match='revision mismatch'): restore_decision(record)
    env = ParallelSelfPlayEnv(); env.reset(seed=record['replay']['resetSeed'])
    for pair in record['replay']['actions']:
        step_decisions(env,[None if a is None else JointAction.from_dict(a) for a in pair])
    # Deliberately do NOT load recorded recurrent memory or relabel this as a
    # corrected-policy state. This only replays simulator actions for an audit.
    side = record['side']; previous = env.core.previous_opponent_state if side==0 else env.core.previous_agent_state
    assert previous[0].pet.name=='pet-ant' and previous[0].pet.level==2 and previous[0].pet.sell_value==3
    assert env.observations()[side]['numeric'][16,15] == np.float32(.2)
    assert_observations_equal(env.observations()[side],PublicSearchState.from_core(env.core,side).observation())
