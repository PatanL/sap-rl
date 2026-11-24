#!/usr/bin/env python3
"""Guarded GPU chain, true-wall runtime screen and reverse confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time

from benchmark_guards import verify_launch
from early_abc_handoff import write_json, PROJECT, IMAGE


FLAGS = ['--lean-policy-execution','--lean-bootstrap','--dense-trajectories','--batched-learner']


def container_path(path):
    return '/workspace/runs/' + str(Path(path).relative_to(PROJECT/'runs'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def gpu_sample(stop, path):
    with path.open('x') as output:
        while not stop.is_set():
            stamp = time.perf_counter_ns()
            result = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,power.draw','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=15)
            output.write(json.dumps({'monotonicNs':stamp,'utcNs':time.time_ns(),
                                     'returnCode':result.returncode,'reading':result.stdout.strip()})+'\n')
            output.flush()
            stop.wait(1)


def run_gpu(request, proof, label, arguments):
    root = Path(request['directory'])
    verify_launch(request, proof)
    source = Path(request['source'])
    subprocess.run(['python3', str(PROJECT/'scripts/v44_preflight.py'), 'source', str(source),
                    str(root/'source-manifest.json')], check=True)
    cmd = ['docker','run','--rm','--name','sap-abc-v2-job','--gpus','all','--ipc','host','--network','none',
           '-e','OMP_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1',
           '-v',f'{source}/sapai:/workspace/sapai:ro','-v',f'{source}/sapai-gym:/workspace/sapai-gym:ro',
           '-v',f'{PROJECT}/runs:/workspace/runs','-w','/workspace/sapai-gym',IMAGE,'python',*arguments]
    write_json(root/'state.json', {'phase':label,'productionCheckpoint':request['checkpoint'], 'deployed':False})
    stop = threading.Event()
    sampler = threading.Thread(target=gpu_sample,args=(stop,root/f'{label}-gpu.jsonl'),daemon=True)
    sampler.start()
    started = time.perf_counter_ns()
    try:
        with (root/f'{label}.log').open('x') as log:
            result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{label} failed with exit {result.returncode}; see its report/log')
    finally:
        stop.set(); sampler.join(timeout=20)
        write_json(root/f'{label}-process-window.json', {'startMonotonicNs':started,
            'endMonotonicNs':time.perf_counter_ns(),'includesStartupWarmupAndShutdown':True})


def runtime_summary(root, label):
    report = json.loads((root/label/'measurement-window.json').read_text())
    if report['acceptedCounterEnd']-report['acceptedCounterStart'] != report['acceptedDecisions']:
        raise ValueError('Window decision counter mismatch')
    if sum(r['decisions'] for r in report['rows']) != report['acceptedDecisions']:
        raise ValueError('Accepted-decision numerator mismatch')
    import re
    rows = [dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in (root/f'{label}.log').read_text().splitlines()
            if line.startswith('update=')]
    first, last = report['rows'][0]['update'], report['rows'][-1]['update']
    rows = [row for row in rows if first <= int(row['update']) <= last]
    if len(rows) != len(report['rows']):
        raise ValueError('Missing measured update logs')
    if sum(int(r['learner_primitives'])-int(r['end_toggles']) for r in rows) != report['acceptedDecisions']:
        raise ValueError('Primitive-toggle count disagrees with actual learner counter delta')
    report['maximumLag'] = max(int(r['lag'].split('/')[1]) for r in rows)
    report['staleSegments'] = sum(int(r['lag_drop'].split('/')[0]) for r in rows)
    report['staleDecisions'] = sum(int(r['lag_drop'].split('/')[1]) for r in rows)
    report['staleSegmentRate'] = report['staleSegments']/(sum(int(r['segments']) for r in rows)+report['staleSegments'])
    report['maximumReadyQueue'] = max(max(map(int,r['ring'].split('/'))) for r in rows)
    readings = []
    for line in (root/f'{label}-gpu.jsonl').read_text().splitlines():
        sample = json.loads(line)
        if report['startMonotonicNs'] <= sample['monotonicNs'] <= report['endMonotonicNs'] and sample['returnCode'] == 0:
            readings.append(float(sample['reading'].split(',')[0]))
    report['meanGpuUtilizationInWindowDiagnostic'] = statistics.mean(readings) if readings else None
    report['gpuUtilizationWindowSamples'] = len(readings)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--completion',type=Path,required=True)
    args = parser.parse_args()
    request, proof = json.loads(args.request.read_text()), json.loads(args.completion.read_text())
    root = Path(request['directory'])
    package = request.get('candidatePackages','abc')
    selected_flags = FLAGS if package=='abc' else FLAGS[:3] if package=='ab' else FLAGS[:2]
    reusable = Path(request['reuseCorrectnessRoot']) if request.get('reuseCorrectnessRoot') else None
    def reuse(label):
        if reusable is None or not (reusable/f'{label}.json').exists():
            return False
        previous=json.loads((reusable/'decision.json').read_text())
        if previous['hashes']['candidateManifest']!=digest(root/'source-manifest.json') or previous['hashes']['productionParent']!=proof['checkpointSHA256']:
            raise ValueError('Cannot reuse correctness evidence from different source/parent')
        report=json.loads((reusable/f'{label}.json').read_text())
        if not report.get('passed',report.get('gatePassed',False)):
            return False
        shutil.copyfile(reusable/f'{label}.json',root/f'{label}.json')
        return True
    decision = {'version':'abc-explicit-go-no-go-v2','rolloutAccepted':False,'benchmarkPassed':False,
                'gpuEquivalencePassed':False,'gpuActorPassed':False,'legacyRuntimeValidated':False,
                'rolloutAuthorized':bool(request.get('rolloutAuthorized')), 'deployed':False,
              'benchmarkWeightsMayEnterProduction':False,'productionParent':request['checkpoint']}
    try:
        verify_launch(request,proof)
        prod = PROJECT/'runs/turtle-league-v45-corrected-from-v44-head-3250/source-batched-v1/sapai-gym/sapai_gym1/rl'
        fixed_parent = PROJECT/'runs/turtle-league-v45-corrected-from-v44-head-3250/update-004800.pt'
        batch = PROJECT/'runs/performance/v46-fixed-batch.pt'
        decision['hashes'] = {'fixedBatch':digest(batch),'fixedOwningCheckpoint':digest(fixed_parent),
            'productionParent':digest(request['checkpoint']),'referenceModel':digest(prod/'model.py'),
            'referenceLearner':digest(prod/'training.py'),'candidateManifest':digest(root/'source-manifest.json'),
            'runner':digest(__file__)}
        if not (root/'initial-league.json').exists():
            shutil.copyfile(Path(request['checkpoint']).parent/'league.json',root/'initial-league.json')
        decision['hashes']['initialLeague'] = digest(root/'initial-league.json')
        if not reuse('actor-gate'):
            run_gpu(request,proof,'actor-gate',['scripts/gpu_actor_equivalence_v2.py','--checkpoint',container_path(request['checkpoint']),
                '/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt',
                '--production-rl',container_path(prod),'--output',container_path(root/'actor-gate.json')])
        decision['gpuActorPassed'] = bool(json.loads((root/'actor-gate.json').read_text()).get('passed'))
        for variant in ('legacy',package):
            for precision in ('fp32','bf16'):
                label = f'fixed-{variant}-{precision}'
                if reuse(label):
                    continue
                run_gpu(request,proof,label,['scripts/benchmark_abc_fixed_v2.py','--checkpoint',container_path(fixed_parent),
                    '--data',container_path(batch),'--reference',container_path(prod/'training.py'),
                    '--production-rl',container_path(prod),'--candidate-manifest',container_path(root/'source-manifest.json'),
                    '--variant',variant,'--precision',precision,'--warmup','1','--repeats','2',
                    '--output',container_path(root/f'{label}.json')])
            if variant == 'legacy':
                decision['legacyRuntimeValidated'] = True
        decision['gpuEquivalencePassed'] = True
        summaries = {}
        for label, optimized, measured in [('screen-baseline',False,30),('screen-abc',True,30),
                                             ('confirm-abc',True,100),('confirm-baseline',False,100)]:
            folder = root/label
            folder.mkdir(exist_ok=False)
            shutil.copyfile(root/'initial-league.json',folder/'league.json')
            target = request['targetUpdate']+10+measured
            params = ['scripts/train_entity_selfplay.py','--distributed','--device','cuda','--updates',str(target),
                '--resume-checkpoint',container_path(request['checkpoint']),'--resume-learning-rate','2e-6',
                '--end-freeze-learning-rate','5e-5','--opponent-module-learning-rate','1e-4','--opponent-adapter-learning-rate','1e-5',
                '--num-envs','32','--games-per-actor','3','--segments-per-update','64','--transitions-per-update','8192',
                '--trajectory-queue-size','1024','--inference-batch-delay-ms','5','--max-behavior-policy-lag','1',
                '--checkpoint-every','100','--checkpoint-dir',container_path(folder),'--discount-mode','per_turn','--discount','.99',
                '--allow-simulator-mismatch',
                '--battle-aux-cost','.05','--opponent-aux-cost','.1','--resident-checkpoint',
                '/workspace/runs/turtle-v44-end-freeze-ablation/control/update-003250.pt','--max-historical-residents','4',
                '--seed',str(60900000+request['targetUpdate']+(100000 if label.startswith('confirm') else 0)),
                '--inference-telemetry','--telemetry-warmup-updates','10','--benchmark-window']
            run_gpu(request,proof,label,params+(selected_flags if optimized else []))
            summaries[label] = runtime_summary(root,label)
            write_json(root/'runtime-results.json',summaries)
            if label == 'screen-abc' and summaries[label]['wallDecisionsPerSecond'] < .98*summaries['screen-baseline']['wallDecisionsPerSecond']:
                decision['reason'] = 'Combined A-C screen regressed; retain runtime and isolate packages separately.'
                break
        if 'confirm-baseline' in summaries:
            ratios = [summaries[f'{stage}-abc']['wallDecisionsPerSecond']/summaries[f'{stage}-baseline']['wallDecisionsPerSecond']
                      for stage in ('screen','confirm')]
            loop_ratios = [summaries[f'{stage}-abc']['timedLoopDecisionsPerSecond']/summaries[f'{stage}-baseline']['timedLoopDecisionsPerSecond']
                          for stage in ('screen','confirm')]
            healthy = all(s['maximumLag'] <= 1 and s['staleSegmentRate'] <= .001 for s in summaries.values())
            tails_ok = all(summaries[f'{stage}-abc']['inference']['actorEnqueueToConsumeMilliseconds']['p95'] <=
                           1.5*summaries[f'{stage}-baseline']['inference']['actorEnqueueToConsumeMilliseconds']['p95'] for stage in ('screen','confirm'))
            memory_ok = all(s['memory']['allocatedEnd']-s['memory']['allocatedStart'] < 2*1024**3 for s in summaries.values())
            decision.update(benchmarkPassed=True,wallSpeedups=ratios,timedLoopSpeedups=loop_ratios,
                            lagDropsHealthy=healthy,responseTailsAcceptable=tails_ok,memoryGrowthAcceptable=memory_ok)
            decision['rolloutAccepted'] = min(ratios) >= 1.03 and statistics.mean(ratios) >= 1.05 and healthy and tails_ok and memory_ok
            decision['acceptedPackages'] = (['A','B','C'] if package=='abc' else ['A','B'] if package=='ab' else ['A']) if decision['rolloutAccepted'] else []
            decision['source'] = request['source']
            decision['flags'] = selected_flags if decision['rolloutAccepted'] else []
            decision['reason'] = 'Predeclared repeatable 5% mean/3% each screen-confirm gate plus health, memory and tails; no benchmark weights selected.'
    except BaseException as error:
        decision['firstFailure'] = str(error)
        raise
    finally:
        write_json(root/'decision.json',decision)
        print(json.dumps(decision,indent=2),flush=True)


if __name__ == '__main__':
    main()
