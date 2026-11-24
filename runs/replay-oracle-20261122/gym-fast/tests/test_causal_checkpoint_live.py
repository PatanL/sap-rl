from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_entity_selfplay import save_checkpoint
from evaluate_paired_policy import load_policy, run_batched_checkpoint_evaluation
from sapai_gym1.rl.causal_transport import CONTEXT_SCHEMA, PREFIX, validate_checkpoint_context, refresh_context
from sapai_gym1.deployment.state_schema import LiveGameState
from sapai_gym1.deployment.policy_adapter import LivePolicyEngine, environment_from_live_state, live_state_from_environment
from test_causal_transport import setup, assert_context_equal


def save(path, model, optimizer=None):
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=.001)
    save_checkpoint(path, model, optimizer, model.config, 4, 100, discount_mode='per_turn', discount=.99,
                    battle_aux_cost=.05, opponent_aux_cost=.1, opponent_module_learning_rate=.001, opponent_adapter_learning_rate=.001)
    return optimizer


def test_actual_checkpoint_save_resume_metadata_weights_optimizer_and_old_compatibility(tmp_path):
    env, base, new, _ = setup(); observations = env.observations()[0]
    batch = {k: torch.as_tensor(v)[None] for k,v in observations.items()}
    optimizer = torch.optim.AdamW(new.parameters(), lr=.001)
    optimizer.zero_grad(); new.value_only(batch).square().sum().backward(); optimizer.step()
    path = tmp_path/'new.pt'; save(path, new, optimizer)
    restored, payload = load_policy(path, torch.device('cpu'))
    assert payload['causal_context']['schema'] == CONTEXT_SCHEMA
    assert payload['causal_context']['adapterMode'] == 'residual'
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.1)
    restored_optimizer.load_state_dict(payload['optimizer'])
    for (name, first), (_, second) in zip(new.named_parameters(), restored.named_parameters()):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        a, b = optimizer.state.get(first, {}), restored_optimizer.state.get(second, {})
        assert a.keys() == b.keys()
        for key in a:
            if torch.is_tensor(a[key]): torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    with torch.inference_mode():
        torch.manual_seed(331); rng = torch.get_rng_state(); before = new.act(batch)
        torch.set_rng_state(rng); after = restored.act(batch)
    for key in vars(before): torch.testing.assert_close(getattr(before,key), getattr(after,key), atol=0, rtol=0)
    old_path = tmp_path/'old.pt'; save(old_path, base)
    old = torch.load(old_path, weights_only=False); old.pop('causal_context')
    for key in list(old['config']):
        if key.startswith('causal_'): old['config'].pop(key)
    torch.save(old, old_path)
    old_model, _ = load_policy(old_path, torch.device('cpu'))
    assert old_model.config.causal_context_schema is None and old_model.causal_forecast is None
    for k, v in base.state_dict().items(): torch.testing.assert_close(v, old_model.state_dict()[k], atol=0, rtol=0)
    for corruption in ('missing', 'version', 'mode', 'vocab'):
        bad = copy.deepcopy(payload)
        if corruption == 'missing': bad.pop('causal_context')
        if corruption == 'version': bad['causal_context']['schema'] = 'wrong'
        if corruption == 'mode': bad['causal_context']['adapterMode'] = 'control'
        if corruption == 'vocab': bad['causal_context']['vocabularySHA256'] = '0'*64
        with pytest.raises(ValueError): validate_checkpoint_context(bad)


def test_live_v2_public_schema_matches_actor_encoding_and_fails_closed(tmp_path):
    from sapai import Team
    env, base, new, _ = setup()
    env.core.previous_agent_state = Team(['pet-fish']); env.core.previous_opponent_state = Team(['pet-ant'])
    env.core.previous_previous_opponent_state = Team(['pet-pig']); refresh_context(env.core)
    state = live_state_from_environment(env.core)
    assert state.schema_version == 2 and state.causal_context_schema == CONTEXT_SCHEMA
    restored_state = LiveGameState.from_dict(json.loads(json.dumps(state.to_dict())))
    live_env = environment_from_live_state(restored_state)
    assert_context_equal(env.observations()[0], env.encoder.encode(live_env, 0))
    path = tmp_path/'live.pt'; save(path, new)
    engine = LivePolicyEngine(path, threads=1)
    recommendation = engine.recommend(restored_state)
    assert recommendation.decision is not None
    legacy_env, _, _, _ = setup(); legacy_env.core.causal_context_schema = None
    legacy_state = live_state_from_environment(legacy_env.core)
    assert legacy_state.schema_version == 1 and 'causal_context_schema' not in legacy_state.to_dict()
    with pytest.raises(ValueError, match='declare'):
        engine.recommend(legacy_state)
    for field, value in [('causal_context_schema', 'unknown'), ('opponent_current_shop', [])]:
        bad = state.to_dict(); bad[field] = value
        with pytest.raises(ValueError): LiveGameState.from_dict(bad)


def test_paired_evaluator_supports_context_policy_against_legacy_resident():
    _, _, new, primitive = setup()
    # One complete seed pair is a CPU execution test, not a strength panel.
    records = []
    outcomes, seats, scores = run_batched_checkpoint_evaluation(new, primitive, np.array([713]), torch.device('cpu'),
        batch_size=2, progress_every=0, game_records=records)
    assert sum(outcomes.values()) == 2 and len(records) == 2 and len(scores) == 1
