#!/usr/bin/env python3
"""CPU-only production-weight actor -> owned ring -> learner neutrality gate."""
import argparse
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import threading

import numpy as np
import torch
from gpu_actor_equivalence_v2 import load_production_package
from sapai_gym1.rl.model import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.causal_transport import CONTEXT_SCHEMA, PREFIX, initialize_causal_from_parent
from sapai_gym1.rl.distributed import SharedActorBuffers, SharedTrajectoryRing, DynamicInferenceBatcher, _write_observation
from sapai_gym1.rl.training import Transition, TurnSegment
from sapai_gym1.rl.learner_batch import pack_learner_batch
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.action_schema import AutoregressiveAction


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--reference-rl', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Use a new report')
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    reference, _ = load_production_package(args.reference_rl)
    old = reference.EntityTransformerPolicy(reference.EntityTransformerConfig(**payload['config'])).eval()
    old.load_state_dict(payload['model'])
    config = replace(EntityTransformerConfig(**payload['config']), causal_context_schema=CONTEXT_SCHEMA,
                     causal_adapter_mode='residual')
    new = EntityTransformerPolicy(config).eval(); initialize_causal_from_parent(new, payload['model'])
    env = ParallelSelfPlayEnv(causal_context_schema=CONTEXT_SCHEMA); env.reset(seed=609207)
    env.core.agent.gold = env.core.opponent.gold = 0
    obs = env.observations(); batch = {k:torch.as_tensor(np.stack([o[k] for o in obs])) for k in obs[0]}
    context = mp.get_context('spawn'); shared = SharedActorBuffers.create(obs[0], config.memory_dim, context)
    for side in (0,1): _write_observation(shared, side, obs[side])
    shared.active[:] = 1
    stop = threading.Event(); server = DynamicInferenceBatcher(new, [shared], 'cpu', threading.RLock(), stop,
                                                             max_batch_delay_ms=0, lean_policy_execution=True)
    torch.manual_seed(609208); rng = torch.get_rng_state()
    with torch.inference_mode(): expected = old.act(batch, old.initial_memory(2))
    expected_rng = torch.get_rng_state(); torch.set_rng_state(rng)
    server.start(); shared.request_event.set()
    try:
        if not shared.response_event.wait(20): raise RuntimeError('Inference did not finish')
        assert torch.equal(expected_rng, torch.get_rng_state())
        for actual, wanted in [(shared.actions, expected.actions), (shared.log_probs, expected.log_prob),
                               (shared.next_memories, expected.memory), (shared.end_freeze_masks.long(), expected.end_freeze_mask)]:
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
    finally: stop.set(); server.stop()
    decisions = [JointAction(AutoregressiveAction(*shared.actions[s].tolist()), int(shared.end_freeze_masks[s])) for s in (0,1)]
    transition = step_decisions(env, decisions).result
    ring = SharedTrajectoryRing.create(obs[0], config.memory_dim, context, capacity=2)
    try:
        for side in (0,1):
            row = Transition(observation=obs[side], action=shared.actions[side].numpy().copy(),
                behavior_log_prob=float(shared.log_probs[side]), reward=transition.rewards[side],
                behavior_version=int(shared.policy_versions[side]),
                discount=0. if transition.terminated else .99 if transition.battle_resolved else 1.,
                end_freeze_mask=int(shared.end_freeze_masks[side]))
            segment = TurnSegment([row], torch.zeros(config.memory_dim), transition.observations[side], transition.terminated)
            ring.write(segment, context.Event())
        dense = ring.read_dense(2, 5).segments()
        if len(dense) < 2: dense.extend(ring.read_dense(2-len(dense), 5).segments())
        packed = pack_learner_batch(dense, 'cpu', joint_actions=True)
        before = {k:v.clone() for k,v in packed.observations.items()}
        for value in ring.observations.values(): value.zero_()
        with torch.inference_mode():
            baseline = old.evaluate_sequence({k:v for k,v in packed.observations.items() if not k.startswith(PREFIX)},
                packed.initial_memories, packed.actions, end_freeze_masks=packed.freeze_masks)
            current = new.evaluate_sequence(packed.observations, packed.initial_memories, packed.actions,
                                            end_freeze_masks=packed.freeze_masks)
        for key in vars(baseline): torch.testing.assert_close(getattr(baseline,key), getattr(current,key), atol=0, rtol=0)
        for key in before: torch.testing.assert_close(before[key], packed.observations[key], atol=0, rtol=0)
    finally: ring.close()
    report = {'passed':True,'device':'cpu-fp32','actorRows':2,'learnerSegments':2,
        'checkpointSHA256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'referenceModelSHA256':hashlib.sha256((args.reference_rl/'model.py').read_bytes()).hexdigest(),
        'schema':CONTEXT_SCHEMA,'actorOutputsAndRngBitExact':True,'learnerPolicyValueMemoryBitExact':True,
        'denseOwnershipVerified':True,'packageC':False,'gpuUsed':False,'optimizerSteps':0,'productionTrainingEnabled':False}
    args.output.write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report))


if __name__ == '__main__': main()
