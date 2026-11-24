#!/usr/bin/env python3
"""Interleaved CPU inference overhead check; not a training throughput claim."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch

from sapai_gym1.rl import EntityTransformerConfig, EntityTransformerPolicy, ParallelSelfPlayEnv
from sapai_gym1.rl.end_freeze_checkpoint import initialize_end_freeze_from_primitive
from sapai_gym1.rl.training import stack_observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = EntityTransformerConfig(**checkpoint['config'])
    parent = EntityTransformerPolicy(config).eval()
    parent.load_state_dict(checkpoint['model'])
    joint = EntityTransformerPolicy(replace(config, end_freeze=True)).eval()
    initialize_end_freeze_from_primitive(joint, checkpoint['model'])
    env = ParallelSelfPlayEnv()
    env.reset(seed=4400701)
    rows = []
    with torch.inference_mode():
        for gold in (10, 0):
            env.core.agent.gold = gold
            observation = env.observations()[0]
            for size in (1, 12, 64):
                batch = stack_observations([observation] * size, 'cpu')
                memory = parent.initial_memory(size)
                policies = [(parent, {}), (joint, {'allow_end_freeze': True})]
                for policy, options in policies:
                    for _ in range(2):
                        policy(batch, memory=memory, **options)
                samples = [[], []]
                for repeat in range(args.repetitions):
                    for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                        policy, options = policies[index]
                        start = time.perf_counter()
                        policy(batch, memory=memory, **options)
                        samples[index].append(1000 * (time.perf_counter() - start))
                medians = [statistics.median(values) for values in samples]
                rows.append({'gold': gold, 'batchSize': size, 'primitiveMs': medians[0],
                             'jointMs': medians[1], 'jointToPrimitiveLatency': medians[1] / medians[0],
                             'samplesMs': samples})
    report = {'checkpoint': str(args.checkpoint), 'cpuThreads': 1, 'repetitions': args.repetitions,
              'rows': rows, 'concurrentGpuTraining': True,
              'note': 'Interleaved synthetic-state CPU inference microbenchmark. Not GPU or end-to-end training speed.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({**report, 'rows': [{k: v for k, v in row.items() if k != 'samplesMs'} for row in rows]}, indent=2))


if __name__ == '__main__':
    main()
