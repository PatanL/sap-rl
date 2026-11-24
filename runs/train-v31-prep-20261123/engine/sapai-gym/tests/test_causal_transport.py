from dataclasses import replace
import copy
import json
import multiprocessing as mp
from pathlib import Path
import sys
import threading

import numpy as np
import pytest
import torch
from sapai import Team

from sapai_gym1.rl.causal_transport import (CONTEXT_SCHEMA, PREFIX, LAYOUT, capture_context, refresh_context,
    validate_context, validate_checkpoint_context, checkpoint_context_metadata, initialize_causal_from_parent)
from sapai_gym1.rl.model import EntityTransformerPolicy
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.distributed import SharedActorBuffers, SharedTrajectoryRing, DynamicInferenceBatcher, _write_observation, DistributedSelfPlayRuntime
from sapai_gym1.rl.training import Transition, TurnSegment, learner_update
from sapai_gym1.rl.learner_batch import pack_learner_batch
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.action_schema import AutoregressiveAction
from test_end_freeze_policy import policies


def setup(mode='residual'):
    _, primitive, base = policies()
    config = replace(base.config, causal_context_schema=CONTEXT_SCHEMA, causal_adapter_mode=mode, causal_context_width=32)
    new = EntityTransformerPolicy(config).eval()
    initialize_causal_from_parent(new, base.state_dict())
    env = ParallelSelfPlayEnv(causal_context_schema=CONTEXT_SCHEMA); env.reset(seed=181)
    return env, base, new, primitive


def assert_context_equal(a, b):
    for key in LAYOUT:
        np.testing.assert_array_equal(a[key], b[key])


def test_public_context_both_perspectives_cache_private_perturbation_and_reset():
    env, _, _, _ = setup()
    initial = env.observations()
    assert all(validate_context(o, required=True) for o in initial)
    assert env.encoder.observation_space().contains(initial[0])
    assert initial[0][PREFIX+'globals'][5:].tolist() == [1., 1., 1.]
    env.core.previous_opponent_state = Team(['pet-fish']); env.core.previous_previous_opponent_state = Team(['pet-ant'])
    env.core.previous_agent_state = Team(['pet-pig']); env.core.agent.lives = 4; env.core.opponent.lives = 3
    refresh_context(env.core)
    before = env.observations()
    assert before[0][PREFIX+'globals'][1:3].tolist() == pytest.approx([4/6, 3/6])
    assert before[1][PREFIX+'globals'][1:3].tolist() == pytest.approx([3/6, 4/6])
    assert before[0][PREFIX+'species'][10] == before[1][PREFIX+'species'][0]
    assert before[0][PREFIX+'species'][0] == before[1][PREFIX+'species'][10]
    env.core.agent.team = Team(['pet-dragon']); env.core.opponent.team = Team(['pet-cat'])
    env.core.agent.shop.roll(); env.core.opponent.shop.roll(); env.core.agent.gold = 2
    after = env.observations()
    for a, b in zip(before, after): assert_context_equal(a, b)
    # Returned arrays are owned observations, not mutable aliases to cache.
    after[0][PREFIX+'species'][:] = 0
    assert_context_equal(before[0], env.observations()[0])
    env.reset(seed=181)
    for a, b in zip(initial, env.observations()): assert_context_equal(a, b)


@pytest.mark.parametrize('bad', ['partial', 'version', 'vocab', 'dtype', 'shape', 'nan', 'missing_flags'])
def test_invalid_context_fails_at_model_and_actor_request_boundary(bad):
    env, _, policy, _ = setup(); observation = env.observations()[0]
    shared = SharedActorBuffers.create(observation, policy.config.memory_dim, mp.get_context('spawn'))
    broken = copy.deepcopy(observation)
    if bad == 'partial': broken.pop(PREFIX+'globals')
    if bad == 'version': broken[PREFIX+'version'][:] = 2
    if bad == 'vocab': broken[PREFIX+'vocab'][0] ^= 1
    if bad == 'dtype': broken[PREFIX+'numbers'] = broken[PREFIX+'numbers'].astype(np.float64)
    if bad == 'shape': broken[PREFIX+'numbers'] = broken[PREFIX+'numbers'][:10]
    if bad == 'nan': broken[PREFIX+'numbers'][0, 0] = np.nan
    if bad == 'missing_flags': broken[PREFIX+'globals'][5] = 0
    with pytest.raises(ValueError): _write_observation(shared, 0, broken)
    with pytest.raises(ValueError):
        policy.act({k: torch.as_tensor(v)[None] for k, v in broken.items()})


@pytest.mark.parametrize('terminal', [False, True])
def test_public_reveal_bootstrap_and_terminal_context(terminal):
    env, _, policy, _ = setup()
    if terminal: env.core.MAX_TURN = 2
    before = env.observations()[0]
    for _ in range(100):
        observations = env.observations(); decisions = []
        with torch.inference_mode():
            for side, active in enumerate(env.active_players()):
                if not active: decisions.append(None); continue
                assert_context_equal(before, observations[0])
                output = policy.act({k: torch.as_tensor(v)[None] for k, v in observations[side].items()}, deterministic=True)
                decisions.append(JointAction(AutoregressiveAction(*output.actions[0].tolist()), int(output.end_freeze_mask[0])))
        result = step_decisions(env, decisions).result
        if result.battle_resolved: break
    else: raise AssertionError('Fixture did not finish a shop')
    assert result.terminated is terminal
    for side, observation in enumerate(result.observations):
        assert validate_context(observation, required=True)
        assert observation[PREFIX+'globals'][5:].tolist() == [0., 1., 0.]
        assert_context_equal(observation, {**observation, **capture_context(env.core, side)})


@pytest.mark.parametrize('resident', [False, True])
@pytest.mark.parametrize('gold', [0, 10])
@pytest.mark.parametrize('entropy_mode', ['legacy_path', 'exact_joint'])
def test_real_actor_request_inference_neutrality_and_rng(resident, gold, entropy_mode):
    env, base, policy, primitive = setup(); env.core.agent.gold = env.core.opponent.gold = gold
    base.config.entropy_mode = policy.config.entropy_mode = entropy_mode
    observations = env.observations(); memory_dim = policy.config.memory_dim
    shared = SharedActorBuffers.create(observations[0], memory_dim, mp.get_context('spawn'))
    for side in (0, 1): _write_observation(shared, side, observations[side])
    shared.active[:] = 1; shared.policy_ids[:] = torch.tensor([0, 1 if resident else 0])
    stop = threading.Event(); batcher = DynamicInferenceBatcher(policy, [shared], 'cpu', threading.RLock(), stop,
        resident_policies={1: primitive} if resident else {}, max_batch_delay_ms=0, lean_policy_execution=True)
    torch.manual_seed(812); rng = torch.get_rng_state()
    with torch.inference_mode():
        if resident:
            expected = [p.act({k:torch.as_tensor(v)[None] for k,v in obs.items()}, memory=p.initial_memory(1))
                        for p, obs in zip((base, primitive), observations)]
        else:
            expected = [base.act({k:torch.as_tensor(np.stack([o[k] for o in observations])) for k in observations[0]}, memory=base.initial_memory(2))]
    expected_rng = torch.get_rng_state(); torch.set_rng_state(rng)
    batcher.start(); shared.request_event.set()
    try:
        assert shared.response_event.wait(10)
        assert torch.equal(expected_rng, torch.get_rng_state())
        for key, field in [('actions', 'actions'), ('log_probs', 'log_prob'), ('next_memories', 'memory')]:
            wanted = torch.cat([getattr(o, field) for o in expected])
            torch.testing.assert_close(getattr(shared, key), wanted, atol=0, rtol=0)
        wanted_masks = torch.cat([torch.full((len(o.actions),), -1, dtype=torch.long) if o.end_freeze_mask is None else o.end_freeze_mask for o in expected])
        torch.testing.assert_close(shared.end_freeze_masks.long(), wanted_masks, atol=0, rtol=0)
    finally: stop.set(); batcher.stop()


@pytest.mark.parametrize('mode', ['control', 'residual'])
@pytest.mark.parametrize('entropy_mode', ['legacy_path', 'exact_joint'])
def test_actual_spawned_actor_dense_ring_learner_and_overwrite_ownership(mode, entropy_mode):
    env, base, policy, primitive = setup(mode)
    base.config.entropy_mode = policy.config.entropy_mode = entropy_mode
    collector = DistributedSelfPlayRuntime(policy, num_actors=1, sample_observation=env.observations()[0], device='cpu', games_per_actor=1,
        queue_size=8, historical_policy=primitive, use_historical_opponent=True, learn_from_resident_sides=False,
        discount_mode='per_turn', lean_policy_execution=True)
    collector.start()
    try:
        segments = collector.collect_segments(2, timeout=30, dense=True)
    finally: collector.stop()
    assert all((s.behavior_versions >= 0).all() for s in segments)
    packed = pack_learner_batch(segments, 'cpu', joint_actions=True)
    stripped = {k:v for k,v in packed.observations.items() if not k.startswith(PREFIX)}
    initial = packed.initial_memories
    with torch.inference_mode():
        rng = torch.get_rng_state()
        old = base.evaluate_sequence(stripped, initial, packed.actions, end_freeze_masks=packed.freeze_masks)
        expected_rng = torch.get_rng_state(); torch.set_rng_state(rng)
        new = policy.evaluate_sequence(packed.observations, initial, packed.actions, end_freeze_masks=packed.freeze_masks)
        assert torch.equal(expected_rng, torch.get_rng_state())
    for key in vars(old):
        torch.testing.assert_close(getattr(old,key), getattr(new,key), atol=0, rtol=0)
    for index, segment in enumerate(segments):
        torch.testing.assert_close(new.log_prob[:segment.length,index], segment.behavior_log_probs, atol=3e-6, rtol=0)
    # A+B-owned storage survives release and explicit overwrite of every ring field.
    saved = {k:v.clone() for k,v in segments[0].observations.items()}
    for tensor in collector.trajectory_ring.observations.values(): tensor.zero_()
    for k in saved: torch.testing.assert_close(saved[k], segments[0].observations[k], atol=0, rtol=0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0001)
    result = learner_update(policy, optimizer, segments, 'cpu', use_bfloat16=False, lean_bootstrap=True, batched_learner=False)
    assert np.isfinite(result.total_loss)
    with pytest.raises(ValueError, match='Package C'):
        learner_update(policy, optimizer, segments, 'cpu', batched_learner=True)


def test_ring_rejects_bad_context_before_taking_slot_and_releases_failed_reads():
    env, _, policy, _ = setup(); observation = env.observations()[0]
    with torch.inference_mode(): output = policy.act({k:torch.as_tensor(v)[None] for k,v in observation.items()}, deterministic=True)
    row = Transition(observation, output.actions[0].numpy(), float(output.log_prob[0]), end_freeze_mask=int(output.end_freeze_mask[0]))
    segment = TurnSegment([row], torch.zeros(policy.config.memory_dim), observation, True)
    context = mp.get_context('spawn'); ring = SharedTrajectoryRing.create(observation, policy.config.memory_dim, context, capacity=1)
    try:
        bad = copy.deepcopy(segment); bad.transitions[0].observation[PREFIX+'version'][:] = 999
        with pytest.raises(ValueError): ring.write(bad, context.Event())
        assert ring.stats()['ready'] == 0
        assert ring.write(segment, context.Event())
        # Corrupt the ready slot after producer validation to exercise the
        # owned-copy read gate and its finally-release path.
        ring.observations[PREFIX+'version'].fill_(999)
        with pytest.raises(ValueError): ring.read_dense(1, 5)
        assert ring.write(segment, context.Event())  # no leaked lease/deadlock
        assert ring.read_dense(1, 5).lengths == (1,)
    finally: ring.close()
