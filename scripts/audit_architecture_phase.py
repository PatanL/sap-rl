#!/usr/bin/env python3
"""Read-only source/data/evaluation checks; emits a separate review attestation.

Run in the pinned CPU environment. Never schedules, stops or alters training.
Historical reports remain unchanged, including gaps in their provenance.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local_path(project, path):
    if path.startswith('/workspace/'):
        return project / path.removeprefix('/workspace/')
    # Host absolute paths are mapped into /workspace in the review container.
    if path.startswith('/home/dzliu/pzliu/sap-rl/'):
        return project / path.removeprefix('/home/dzliu/pzliu/sap-rl/')
    return Path(path)


def audit_evaluation(path, original, project, *, expected_candidate_hash, expected_seed):
    report = json.loads(path.read_text())
    assert sha(path) == sha(original), 'Copied evaluation differs from original'
    count = report['pairs']
    assert count == 100 and report['games'] == count*2
    assert report['seed'] == expected_seed
    seeds = np.random.default_rng(expected_seed).integers(0, 2**31-1, size=count).tolist()
    assert seeds == report['pairSeeds'], 'Seed panel differs from declared generator'
    games = report['gameResults']
    assert len(games) == 2*count
    indexed = {(g['pairIndex'], g['candidateSide']): g for g in games}
    assert set(indexed) == {(i, side) for i in range(count) for side in (0, 1)}
    points = {'win': 1., 'draw': .5, 'loss': 0.}
    scores = []
    for i, seed in enumerate(seeds):
        pair = [indexed[i, side] for side in (0, 1)]
        assert all(g['seed'] == seed and g['terminationReason'] in ('lives', 'turn_limit') for g in pair)
        assert all(g['turnLimit'] == 25 and g['rearrangementLimit'] == 2 for g in pair)
        scores.append(sum(points[g['candidateResult']] for g in pair)/2)
    assert scores == report['pairScores']
    assert abs(statistics.mean(scores)-report['pairedMeanScore']) < 1e-12
    assert dict(Counter(g['candidateResult'] for g in games)) == report['outcomes']
    assert dict(Counter(g['terminationReason'] for g in games)) == report['terminationCounts']
    assert report['runtimeSimulatorRevision'] == 'turtle-current-frozen-shop-cap-v28-20260905'
    assert report['evaluationRevision'] == 'paired-seat-with-termination-v1-20260905'
    candidate = local_path(project, report['checkpoint'])
    opponent = local_path(project, report['opponentCheckpoint'])
    assert sha(candidate) == expected_candidate_hash
    return {'report': str(path), 'reportSHA256': sha(path), 'candidateSHA256': sha(candidate),
            'opponentSHA256': sha(opponent), 'seedPanelSHA256': hashlib.sha256(json.dumps(seeds).encode()).hexdigest(),
            'completePairedGames': len(games), 'recomputedMeanScore': statistics.mean(scores),
            'reportedPaired95CI': report['pairedScore95CI'], 'runtimeRevision': report['runtimeSimulatorRevision'],
            'candidateSchema': report['candidateActionSchema'], 'opponentSchema': report['opponentActionSchema'],
            'checksPassed': True, 'repeatedEvaluation': False,
            'provenanceLimitation': 'Original reports did not capture opponent content hashes before execution. '
                'Current contents are attested now; this does not retroactively prove pre-run opponent hashes.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('/workspace'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.project/'runs/performance-review-20260906/ab-recovery-9050-v4'
    prior = root.parent/'abc-early-9050-v3'
    spec = json.loads((root/'campaign-spec.json').read_text())
    source = local_path(args.project, spec['source'])
    manifest_path = local_path(args.project, spec['sourceManifest'])
    manifest = json.loads(manifest_path.read_text())
    report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'checksPassed': False,
              'trainingModified': False, 'gpuUsed': False, 'errors': []}
    try:
        assert sha(manifest_path) == spec['sourceManifestSHA256']
        for name, expected in manifest['files'].items():
            assert sha(source/name) == expected, f'Frozen source changed: {name}'
        parent = local_path(args.project, spec['parent'])
        assert sha(parent) == spec['parentSHA256']
        report.update(sourceManifestSHA256=sha(manifest_path), sourceInventorySHA256=manifest['sha256'],
                      parentSHA256=sha(parent), runtimeFlags=spec['runtimeFlags'], sourceFilesVerified=len(manifest['files']))
        gate = json.loads((root/'fixed-exact-joint-bf16.json').read_text())
        assert gate['gatePassed'] and gate['variant'] == 'ab' and gate['entropyMode'] == 'exact_joint'
        assert gate['device'] == 'cuda' and gate['precision'] == 'bf16'
        assert gate['decisions'] >= 8192
        assert gate['hashes']['candidateSourceManifest'] == sha(manifest_path)
        assert gate['hashes']['checkpoint'] == sha(local_path(args.project, gate['checkpoint']))
        assert gate['hashes']['fixedBatch'] == sha(local_path(args.project, gate['data']))
        assert gate['hashes']['runner'] == sha(source/'sapai-gym/scripts/benchmark_abc_fixed_v2.py')
        reference = args.project/'runs/turtle-league-v45-corrected-from-v44-head-3250/source-batched-v1/sapai-gym/sapai_gym1/rl'
        assert gate['hashes']['referenceModel'] == sha(reference/'model.py')
        assert gate['hashes']['referenceLearner'] == sha(reference/'training.py')
        report['exactEntropyGate'] = {'reportSHA256': sha(root/'fixed-exact-joint-bf16.json'),
            'variant': 'ab', 'entropyMode': 'exact_joint', 'precision': 'bf16', 'decisions': gate['decisions'],
            'ownerCheckpointSHA256': gate['hashes']['checkpoint'], 'fixedBatchSHA256': gate['hashes']['fixedBatch'],
            'checksPassed': True,
            'qualification': 'Under exact entropy, reference uses the new exact-capable model with the frozen per-segment learner. '
                'It is not a claim that the old production model supports exact entropy. '
                'This report compares AB versus legacy execution, not exact-versus-legacy entropy cost. '
                'Exact-mode peak memory was not recorded by this gate.'}
        report['reusedEvaluations'] = [audit_evaluation(root/'evaluations'/f'{name}.json',
            prior/'evaluations'/f'{name}.json', args.project, expected_candidate_hash=spec['parentSHA256'],
            expected_seed=61909050) for name in ('latest-vs-6250', 'latest-vs-v31')]
        report['checksPassed'] = True
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    with args.output.open('x') as output:
        json.dump(report, output, indent=2); output.write('\n')
    print(json.dumps(report, indent=2))
    if not report['checksPassed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
