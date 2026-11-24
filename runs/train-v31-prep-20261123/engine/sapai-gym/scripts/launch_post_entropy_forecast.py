#!/usr/bin/env python3
"""Fail-closed post-entropy forecast launcher. PREPARED ONLY; no auto policy run.

Default is validation only. --execute must be explicitly invoked after a
reviewed selection and clean completion/resource checks. Tests inject a fake
command runner and never touch real containers/services.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

IMAGE = 'sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1'
AB_FLAGS = ['--lean-policy-execution', '--lean-bootstrap', '--dense-trajectories']
DE_FLAGS = ['inference_staging', 'inference_pinned_staging', 'inference_validated_decoder', 'inference_early_release', 'inference_cuda_stream']
SELECTION_SCHEMA = 'reviewed-post-entropy-forecast-selection-v1'
PLAN_SCHEMA = 'next-forecast-quality-plan-v1-20260906'
DATASET_SCHEMA = 'causal-opponent-one-step-v1-20260906'


class LaunchBlocked(ValueError):
    pass


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition: raise LaunchBlocked(message)


def checked(command, run=subprocess.run):
    result = run(command, capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, 'Command/query failed: '+command[0])
    return result.stdout.strip()


def verify_frozen_inputs(selection):
    source = Path(selection['source']).resolve()
    for manifest_key, hash_key, nested in [('sourceManifest','sourceManifestSHA256',True),('assetsManifest','assetsManifestSHA256',False)]:
        path = Path(selection[manifest_key])
        require(sha(path) == selection[hash_key], 'Frozen input manifest changed')
        payload = json.loads(path.read_text()); files = payload['files'] if nested else payload
        for name, digest in files.items():
            target = (source/name).resolve()
            require(target.is_relative_to(source), 'Source entry escapes snapshot')
            require(sha(target) == digest, 'Frozen source or asset changed: '+name)
    require(sha(selection['checkpoint']) == selection['checkpointSHA256'], 'Selected checkpoint changed')
    require(sha(selection['forecastPlan']) == selection['forecastPlanSHA256'], 'Reviewed forecast plan changed')


def owned(path, project):
    path = Path(path).resolve()
    require(path.is_relative_to((project/'runs').resolve()), 'All selected artifacts must be under project runs')
    return path


def verify_evaluation(report, expected_seed, pairs=100):
    require(report.get('pairs') == pairs and report.get('games') == 2*pairs and report.get('seed') == expected_seed,
            'Wrong/incomplete entropy evaluation budget')
    seeds, records = report['pairSeeds'], report['gameResults']
    require(len(seeds) == pairs and len(set(seeds)) == pairs and len(records) == pairs*2, 'Incomplete paired evaluation')
    actual = {(r['pairIndex'], r['seed'], r['candidateSide']) for r in records}
    require(actual == {(i, s, side) for i,s in enumerate(seeds) for side in (0,1)}, 'Missing/duplicate paired seats')
    counts = Counter(r['candidateResult'] for r in records)
    require(not set(counts)-{'win','draw','loss'} and dict(counts) == report['outcomes'], 'Invalid evaluation outcomes')
    require(abs((counts['win']+.5*counts['draw'])/(pairs*2)-report['pairedMeanScore']) < 1e-12, 'Inconsistent evaluation score')
    interval = report.get('pairedScore95CI', [])
    require(len(interval) == 2 and all(math.isfinite(x) and 0 <= x <= 1 for x in interval)
            and interval[0] <= report['pairedMeanScore'] <= interval[1], 'Invalid paired uncertainty')
    indexed = {(r['pairIndex'],r['candidateSide']):r for r in records}
    points = {'win':1.,'draw':.5,'loss':0.}
    require(report['pairScores'] == [sum(points[indexed[i,s]['candidateResult']] for s in (0,1))/2 for i in range(pairs)],
            'Paired scores do not match completed games')
    require(report.get('runtimeSimulatorRevision') == 'turtle-current-frozen-shop-cap-v28-20260905', 'Evaluation rules mismatch')
    require(report.get('candidateActionSchema') == report.get('opponentActionSchema') == 'primitive-plus-end-freeze-v1-20260905',
            'Evaluation action schema mismatch')
    require(all(r.get('terminationReason') in ('lives','turn_limit') and r.get('turnLimit') == 25 and r.get('rearrangementLimit') == 2 for r in records),
            'Incomplete/changed evaluation termination rules')


def resolve_report_checkpoint(path, project):
    return (project/'runs'/path.removeprefix('/workspace/runs/')).resolve() if path.startswith('/workspace/runs/') else Path(path).resolve()


def resources_clean(run=subprocess.run):
    raw = checked(['systemctl','--user','show','sap-architecture-campaign.service','-p','LoadState','-p','ActiveState',
                   '-p','SubState','-p','MainPID','-p','Result','-p','ExecMainCode','-p','ExecMainStatus'], run)
    values = dict(line.split('=',1) for line in raw.splitlines() if '=' in line)
    require(values.get('LoadState') == 'loaded' and values.get('ActiveState') == 'inactive'
            and values.get('SubState') == 'dead' and values.get('MainPID') == '0'
            and values.get('Result') == 'success' and values.get('ExecMainCode') == '1'
            and values.get('ExecMainStatus') == '0', 'Entropy service has not completed cleanly')
    require(not checked(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], run), 'GPU has another owner')
    running = checked(['docker','ps','--format','{{.Names}}'], run).splitlines()
    if 'sap-one-step-cpu-corpus' in running:
        paused = checked(['docker','inspect','-f','{{.State.Paused}}','sap-one-step-cpu-corpus'], run)
        require(paused == 'true', 'Older CPU neural collector is still running; wait, do not evict it')
    require('sap-architecture-job' not in running, 'Entropy container still exists/runs')
    require(checked(['docker','image','inspect','--format','{{.Id}}', IMAGE], run) == IMAGE, 'Pinned image mismatch')


def validate_selection(path, project, *, run=subprocess.run):
    selection = json.loads(Path(path).read_text())
    require(selection.get('schema') == SELECTION_SCHEMA and selection.get('reviewed') is True, 'Explicit reviewed entropy selection required')
    require(selection.get('entropyMode') in ('legacy_path','exact_joint'), 'Unknown selected entropy mode')
    runtime = selection.get('runtime', {})
    require(runtime.get('flags') == AB_FLAGS and runtime.get('packageC') is False and runtime.get('nativeSimulation') is False,
            'Selected runtime must preserve A+B with C/native OFF')
    require(runtime.get('DEFlags') == {k:False for k in DE_FLAGS}
            and all(v is False for v in runtime.get('DEFlags', {}).values()), 'Every D/E flag must be explicitly OFF')
    require(selection.get('runtimeSHA256') == object_hash(runtime), 'Runtime hash mismatch')
    require(selection.get('image') == IMAGE, 'Only the pinned reviewed image is supported')
    source = owned(selection['source'], project)
    manifest_path = owned(selection['sourceManifest'], project)
    require(sha(manifest_path) == selection['sourceManifestSHA256'], 'Source manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe source manifest path')
        require(sha(source/name) == digest, 'Frozen source changed: '+name)
    assets_path = owned(selection['assetsManifest'], project)
    require(sha(assets_path) == selection['assetsManifestSHA256'], 'Assets manifest changed')
    assets = json.loads(assets_path.read_text())
    for name, digest in assets.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe asset path')
        require(sha(source/name) == digest, 'Required source asset changed')
    bundle = hashlib.sha256(json.dumps({'code': manifest['sha256'], 'assets': assets}, sort_keys=True).encode()).hexdigest()
    require(selection['sourceBundleSHA256'] == bundle, 'Source bundle hash mismatch')
    parent = owned(selection['checkpoint'], project)
    require(sha(parent) == selection['checkpointSHA256'], 'Selected parent hash mismatch')
    campaign = owned(selection['entropyCampaign'], project)
    state = json.loads((campaign/'campaign-state.json').read_text())
    require(state.get('phase') == 'entropy_results_ready', 'Entropy training/evaluation is not complete')
    checkpoints = {Path(v).resolve() for v in state['checkpoints'].values()} | {Path(state['parent']).resolve()}
    require(parent in checkpoints, 'Do not select benchmark weights or an unrelated parent')
    require(set(state['checkpoints']) == {'legacy_path','exact_joint'}, 'Both entropy arms must complete')
    for mode, checkpoint in state['checkpoints'].items():
        require(owned(checkpoint, project).is_file() and Path(checkpoint).name == 'update-010050.pt', 'Missing final entropy checkpoint')
        log = (campaign/f'train-{mode}.log').read_text()
        require('finished updates=10050 ' in log, 'Missing natural training-completion evidence')
        require((campaign/f'train-{mode}-wall.json').is_file(), 'Missing completed training wall record')
    evidence = selection.get('evaluations', {})
    for name, seed in [('exact-vs-legacy',62909050),('legacy-vs-parent',62919050),('exact-vs-parent',62919050)]:
        report_path = campaign/'evaluations'/f'{name}.json'
        require(evidence.get(name) == sha(report_path), 'Reviewed evaluation hash mismatch: '+name)
        report = json.loads(report_path.read_text())
        verify_evaluation(report, seed)
        expected_candidate = state['checkpoints']['exact_joint' if name.startswith('exact') else 'legacy_path']
        expected_opponent = state['checkpoints']['legacy_path'] if name == 'exact-vs-legacy' else state['parent']
        require(resolve_report_checkpoint(report['checkpoint'], project) == Path(expected_candidate).resolve()
                and resolve_report_checkpoint(report['opponentCheckpoint'], project) == Path(expected_opponent).resolve(),
                'Evaluation candidate/opponent identity mismatch')
    direct = json.loads((campaign/'evaluations/exact-vs-legacy.json').read_text())
    if direct['pairedMeanScore'] >= .52 and direct['pairedScore95CI'][0] <= .5:
        path = campaign/'evaluations/exact-vs-legacy-confirmation.json'
        require(evidence.get('exact-vs-legacy-confirmation') == sha(path), 'Triggered confirmation is not reviewed/complete')
        confirmation = json.loads(path.read_text())
        verify_evaluation(confirmation, 62929050, pairs=200)
        require(resolve_report_checkpoint(confirmation['checkpoint'], project) == Path(state['checkpoints']['exact_joint']).resolve()
                and resolve_report_checkpoint(confirmation['opponentCheckpoint'], project) == Path(state['checkpoints']['legacy_path']).resolve(),
                'Confirmation candidate/opponent identity mismatch')
    plan_path = owned(selection['forecastPlan'], project)
    require(sha(plan_path) == selection['forecastPlanSHA256'], 'Forecast plan hash mismatch')
    plan = json.loads(plan_path.read_text())
    require(selection.get('forecastPlanVersion') == plan.get('schema') == PLAN_SCHEMA
            and selection.get('forecastDatasetVersion') == DATASET_SCHEMA, 'Forecast plan/dataset version mismatch')
    collection = plan['collection']; pairs = collection['pairs']
    require(collection['totalGames'] == 280 and len(pairs) == 7 and sum(p['games'] for p in pairs) == 280,
            'Fresh collection must use the reviewed 280-match plan')
    require(collection['maxCpuCores'] == 1 and collection['bothPerspectives'] is True and collection['completeMatchesOnly'] is True
            and collection['terminalTurnsRetained'] is True and collection['futureSurvivalFiltering'] is False,
            'Forecast collection semantics changed')
    require(sorted(p['index'] for p in pairs) == list(range(7)) and all(p['games'] == 40 for p in pairs),
            'Duplicate/missing collection pair IDs or changed per-pair budget')
    split = plan['splits']; quality = plan['quality']
    population_hashes = {p['sha256'] for p in plan['populationProvenance'].values()}
    heldout = set(split['heldoutPopulationHashes'])
    require(bool(heldout) and heldout <= population_hashes and split['archiveAndPilotRowsInFinalTest'] is False,
            'A fresh independent population holdout is required')
    require(0 <= split['validationFraction'] < 1 and 0 <= split['randomTestFraction'] < 1
            and split['validationFraction']+split['randomTestFraction'] < 1, 'Invalid split fractions')
    seeds = []
    for pair in pairs:
        require(len(pair['policies']) == 2 and all(p in plan['populationProvenance'] for p in pair['policies']), 'Invalid collection policies')
        require(type(pair['seed']) is int and 0 <= pair['seed'] < 2**32-40, 'Invalid reset seed')
        seeds.extend(range(pair['seed'], pair['seed']+40))
        is_heldout = any(plan['populationProvenance'][p]['sha256'] in heldout for p in pair['policies'])
        require(pair['populationHeldOut'] is is_heldout and type(pair['allowHistoricalTrainingRevisionMismatch']) is bool,
                'Population holdout/revision declaration mismatch')
    require(len(set(seeds)) == 280, 'Collection reset-seed ranges overlap')
    require(sum(p['games'] for p in pairs if p['populationHeldOut']) == 80 and split['testPopulationGames'] == 80,
            'Require 200 train/validation-eligible matches and 80 population-holdout matches')
    require(quality['variants'] == ['history_marginal','causal_marginal','causal_autoregressive']
            and quality['width'] == 128 and quality['epochs'] > 0 and quality['batchSize'] > 0
            and quality['samplesPerContext'] >= 2 and quality['learningRate'] > 0,
            'Invalid three-way forecasting comparison')
    for population in plan['populationProvenance'].values():
        require(sha(owned(population['path'], project)) == population['sha256'], 'Population checkpoint hash mismatch')
    resources_clean(run)
    verify_frozen_inputs(selection)
    return selection, plan


def docker_job(selection, project, output, label, arguments, *, gpu=False):
    resources_clean()  # Recheck before EVERY job; never stop somebody else's work.
    verify_frozen_inputs(selection)
    command = ['docker','run','--rm','--name','sap-post-entropy-forecast','--user','1000:1000','--cpus','1','--memory','8g',
        '--network','none','-e','OMP_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','PYTHONPATH=/workspace/sapai:/workspace/sapai-gym',
        '-v',selection['source']+'/sapai:/workspace/sapai:ro','-v',selection['source']+'/sapai-gym:/workspace/sapai-gym:ro',
        '-v',str(project/'runs')+':/workspace/runs:ro','-v',str(output)+':/output','-w','/workspace/sapai-gym']
    if gpu: command += ['--gpus','all']
    command += [IMAGE,'python',*arguments]
    with (output/f'{label}.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def execute(selection, plan, project, output, *, forecast_device):
    output = owned(output, project)
    require(output != (project/'runs').resolve(), 'Output must be a specific new experiment directory')
    output.mkdir(parents=True, exist_ok=False)
    (output/'reviewed-selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    (output/'frozen-collection-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    cpath = lambda p: '/workspace/runs/'+str(Path(p).resolve().relative_to((project/'runs').resolve()))
    # Inspect actual checkpoint metadata in an isolated CPU process, not by
    # trusting its filename or resetting the selected policy's entropy mode.
    code = "import torch,json; d=torch.load("+repr(cpath(selection['checkpoint']))+",map_location='cpu',weights_only=False); assert d['config'].get('entropy_mode','legacy_path')=="+repr(selection['entropyMode'])+"; print('parent_metadata_verified')"
    docker_job(selection, project, output, 'parent-metadata', ['-c', code])
    shards = []
    for pair in plan['collection']['pairs']:
        paths = [plan['populationProvenance'][p]['path'] for p in pair['policies']]
        shard = f"/output/pair-{pair['index']:02d}.json"; shards.append(shard)
        args = ['scripts/generate_opponent_one_step_data.py',cpath(paths[0]),'--opponent-checkpoint',cpath(paths[1]),
                '--games',str(pair['games']),'--seed',str(pair['seed']),'--output',shard]
        if pair['allowHistoricalTrainingRevisionMismatch']: args += ['--allow-checkpoint-training-revision-mismatch']
        docker_job(selection, project, output, f"collect-{pair['index']:02d}", args)
    split = plan['splits']
    args = ['scripts/prepare_opponent_forecast_dataset.py',*shards,'--output','/output/dataset.json',
            '--salt',split['salt'],'--validation-fraction',str(split['validationFraction']),'--test-fraction',str(split['randomTestFraction'])]
    for population in split['heldoutPopulationHashes']: args += ['--heldout-population',population]
    docker_job(selection, project, output, 'freeze-splits', args)
    quality = plan['quality']
    common = ['scripts/run_opponent_forecast_quality.py','/output/dataset.json','--output-directory','/output/quality',
        '--device',forecast_device,'--epochs',str(quality['epochs']),'--batch-size',str(quality['batchSize']),
        '--learning-rate',str(quality['learningRate']),'--samples',str(quality['samplesPerContext']),'--seed',str(quality['seed'])]
    if forecast_device == 'cuda': common += ['--gpu-boundary-approved']
    docker_job(selection, project, output, 'forecast-train', [*common,'--stage','train'], gpu=forecast_device=='cuda')
    docker_job(selection, project, output, 'forecast-final-test', [*common,'--stage','test'], gpu=forecast_device=='cuda')
    (output/'state.json').write_text(json.dumps({'phase':'awaiting_reviewed_forecast_improvement',
        'entropyMode':selection['entropyMode'],'parent':selection['checkpoint'],'policyResidualLaunchAuthorized':False,
        'qualityResult':str(output/'quality/quality-comparison.json'),'automaticPolicyLaunch':False}, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path('/home/dzliu/pzliu/sap-rl'))
    parser.add_argument('--output', type=Path); parser.add_argument('--execute', action='store_true')
    parser.add_argument('--forecast-device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    try:
        selection, plan = validate_selection(args.selection, args.project.resolve())
        if args.execute:
            require(args.output is not None, 'An explicit new output directory is required')
            execute(selection, plan, args.project.resolve(), args.output, forecast_device=args.forecast_device)
        else:
            print(json.dumps({'validated':True,'executed':False,'queued':False}))
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({'launchBlocked':str(error),'policyArchitectureLaunched':False}))
        raise SystemExit(2)


if __name__ == '__main__': main()
