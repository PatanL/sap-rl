#!/usr/bin/env python3
"""Create an immutable post-entropy forecast plan. Starts no job/service."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path('/home/dzliu/pzliu/sap-rl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); project = args.project
    v45 = project/'runs/turtle-league-v45-corrected-from-v44-head-3250'
    populations = {'v45-9050': v45/'update-009050.pt', 'v45-6250': v45/'update-006250.pt',
        'v45-8800': v45/'update-008800.pt',
        'v31-1000': project/'runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt'}
    pairs = [('v45-9050', 'v45-9050'), ('v45-6250', 'v45-6250'), ('v45-9050', 'v45-6250'),
             ('v45-8800', 'v45-9050'), ('v45-8800', 'v45-6250'), ('v31-1000', 'v45-9050'), ('v31-1000', 'v45-6250')]
    plan = {'schema': 'next-forecast-quality-plan-v1-20260906', 'status': 'prepared-not-queued',
        'startsJobs': False, 'policyArchitectureActivated': False,
        'requires': ['entropy arms and predeclared evaluation complete',
                     'reviewed entropy-selection record naming one immutable common parent/mode',
                     'separate source snapshot and explicit resource boundary',
                     'dataset generation complete and frozen splits validated'],
        'entropySelectionPolicy': 'No automatic promotion. Convincing exact improvement may select exact; unresolved prefers cheaper incumbent; regression selects legacy. Freeze same choice for both later policy arms.',
        'image': 'sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1',
        'populationProvenance': {name: {'path': str(path), 'sha256': sha(path)} for name, path in populations.items()},
        'collection': {'gamesPerPair': 40, 'totalGames': 280, 'bothPerspectives': True, 'completeMatchesOnly': True,
            'terminalTurnsRetained': True, 'futureSurvivalFiltering': False, 'maxCpuCores': 1,
            'neuralCollectionDuringEntropyTiming': False,
            'generator': 'sapai-gym/scripts/generate_opponent_one_step_data.py',
            'pairs': [{'index': i, 'policies': pair, 'games': 40, 'seed': 60910000+i*10000,
                       'populationHeldOut': 'v31-1000' in pair,
                       'allowHistoricalTrainingRevisionMismatch': 'v31-1000' in pair} for i, pair in enumerate(pairs)]},
        'splits': {'unit': 'whole match, both perspectives and all turns', 'salt': 'forecast-production-panel-20260906-v1',
            'heldoutPopulationHashes': [sha(populations['v31-1000'])], 'validationFraction': .2, 'randomTestFraction': 0.,
            'testPopulationGames': 80, 'archiveAndPilotRowsInFinalTest': False,
            'minimumCompleteMatchesForQualityRunner': 200},
        'quality': {'variants': ['history_marginal', 'causal_marginal', 'causal_autoregressive'],
            'width': 128, 'species': 93, 'perks': 13, 'epochs': 30, 'batchSize': 128, 'learningRate': .0003,
            'seed': 609071, 'samplesPerContext': 128,
            'runner': 'sapai-gym/scripts/run_opponent_forecast_quality.py',
            'checkpointSelection': 'validation NLL only, identical epoch budget/permutations',
            'finalTest': 'separate one-shot --stage test after all checkpoints fixed',
            'primary': 'exact held-out board NLL; paired whole-match bootstrap intervals for B-A and C-B',
            'secondary': ['event Brier/calibration', 'species/perk/level marginal diagnostics', 'turn/population NLL',
                          'syntax validity; explicitly unknown exact hidden-shop reachability', 'parameters, wall cost, CUDA allocation peak'],
            'qualitySelectionAutomatic': False},
        'policyIntegration': {'stage1': 'standalone forecast training, old policy/history/adapter untouched',
            'stage2': 'separate common-parent control-vs-zero-initialized-residual strength experiment after reviewed forecast gain',
            'neutralityRequired': ['policy logits', 'joint log probability', 'value', 'memory', 'RNG parity in matched execution mode'],
            'actorLearnerCausalContextTransportReady': False},
        'superhumanClaim': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(plan, output, indent=2); output.write('\n')
    print(json.dumps({'output': str(args.output), 'gamesPlanned': 280, 'queued': False}))


if __name__ == '__main__':
    main()
