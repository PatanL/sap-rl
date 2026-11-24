#!/usr/bin/env python3
"""Sparse strength check, immutable common parent, then matched entropy arms."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from early_abc_handoff import PROJECT, IMAGE, write_json


def cpath(path):
    return '/workspace/runs/' + str(Path(path).relative_to(PROJECT/'runs'))


def job(spec, label, arguments):
    root = Path(spec['directory'])
    # Intentional ownership follows the completed handoff; never evict others.
    while True:
        result = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True)
        if result.returncode == 0 and not result.stdout.strip():
            break
        time.sleep(20)
    subprocess.run(['python3',str(PROJECT/'scripts/v44_preflight.py'),'source',spec['source'],spec['sourceManifest']],check=True)
    write_json(root/'campaign-state.json',{'phase':label,'parent':spec['parent'],'source':spec['source'],'runtimeFlags':spec['runtimeFlags']})
    cmd=['docker','run','--rm','--name','sap-architecture-job','--gpus','all','--ipc','host','--network','none',
         '-e','OMP_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1',
         '-v',f'{spec["source"]}/sapai:/workspace/sapai:ro','-v',f'{spec["source"]}/sapai-gym:/workspace/sapai-gym:ro',
         '-v',f'{PROJECT}/runs:/workspace/runs','-w','/workspace/sapai-gym',IMAGE,'python',*arguments]
    started=time.perf_counter_ns()
    with (root/f'{label}.log').open('a') as output:
        subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,check=True)
    write_json(root/f'{label}-wall.json',{'startMonotonicNs':started,'endMonotonicNs':time.perf_counter_ns(),
                                      'scope':'Whole process including loading, checkpoints and shutdown'})


def evaluate(spec, candidate, opponent, label, seed, pairs=100):
    path=Path(spec['directory'])/'evaluations'/f'{label}.json'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return
    job(spec,label,['scripts/evaluate_paired_policy.py',cpath(candidate),'--opponent','checkpoint',
        '--opponent-checkpoint',cpath(opponent),'--device','cuda','--batch-size','64','--pairs',str(pairs),
        '--seed',str(seed),'--progress-every','100','--output',cpath(path)])


def train(spec, mode, folder, target):
    folder.mkdir(exist_ok=True)
    league=folder/'league.json'
    if not league.exists():
        shutil.copyfile(Path(spec['directory'])/'initial-league.json',league)
    saved=sorted(folder.glob('update-*.pt'))
    parent=saved[-1] if saved else Path(spec['parent'])
    if int(parent.stem.split('-')[-1]) >= target:
        return
    arguments=['scripts/train_entity_selfplay.py','--distributed','--device','cuda','--updates',str(target),
        '--resume-checkpoint',cpath(parent),'--resume-learning-rate','2e-6','--end-freeze-learning-rate','5e-5',
        '--opponent-module-learning-rate','1e-4','--opponent-adapter-learning-rate','1e-5',
        '--num-envs','32','--games-per-actor','3','--segments-per-update','64','--transitions-per-update','8192',
        '--trajectory-queue-size','1024','--inference-batch-delay-ms','5','--max-behavior-policy-lag','1',
        '--checkpoint-every','100','--checkpoint-dir',cpath(folder),'--discount-mode','per_turn','--discount','.99',
        '--allow-simulator-mismatch',
        '--battle-aux-cost','.05','--opponent-aux-cost','.1','--resident-checkpoint',
        '/workspace/runs/turtle-v44-end-freeze-ablation/control/update-003250.pt','--max-historical-residents','4',
        '--seed','60919050',*spec['runtimeFlags']]
    if spec['newLegacyValidated']:
        arguments += ['--entropy-mode',mode]
    job(spec,f'train-{mode}',arguments)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('spec',type=Path);args=parser.parse_args()
    spec=json.loads(args.spec.read_text());root=Path(spec['directory']);parent=Path(spec['parent'])
    with parent.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=spec['parentSHA256']:
            raise RuntimeError('Immutable common parent changed')
    v45=PROJECT/'runs/turtle-league-v45-corrected-from-v44-head-3250'
    anchor=PROJECT/'runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt'
    if spec['sparseAuthorized']:
        evaluate(spec,parent,v45/'update-006250.pt','latest-vs-6250',61909050)
        evaluate(spec,parent,anchor,'latest-vs-v31',61909050)
    if not spec['entropyAuthorized']:
        # Keep useful self-play running when a new-interface correctness gate failed.
        train(spec,'legacy_path',v45,int(parent.stem.split('-')[-1])+1000)
        return
    production_rl=v45/'source-batched-v1/sapai-gym/sapai_gym1/rl'
    gate=root/'fixed-exact-joint-bf16.json'
    if not gate.exists():
        job(spec,'exact-gpu-gate',['scripts/benchmark_abc_fixed_v2.py','--checkpoint',cpath(v45/'update-004800.pt'),
            '--data','/workspace/runs/performance/v46-fixed-batch.pt','--reference',cpath(production_rl/'training.py'),
            '--production-rl',cpath(production_rl),'--candidate-manifest',cpath(spec['sourceManifest']),
            '--variant',('abc' if '--batched-learner' in spec['runtimeFlags'] else 'ab' if '--dense-trajectories' in spec['runtimeFlags'] else 'a' if '--lean-bootstrap' in spec['runtimeFlags'] else 'legacy'),
            '--precision','bf16','--entropy-mode','exact_joint','--warmup','1','--repeats','2','--output',cpath(gate)])
    if not json.loads(gate.read_text()).get('gatePassed'):
        raise RuntimeError('Exact entropy GPU gate failed; do not start strength experiment')
    target=int(parent.stem.split('-')[-1])+spec['updatesPerEntropyArm']
    checkpoints={}
    for mode in ('legacy_path','exact_joint'):
        folder=root/f'entropy-{mode}'
        train(spec,mode,folder,target)
        checkpoints[mode]=folder/f'update-{target:06d}.pt'
    evaluate(spec,checkpoints['exact_joint'],checkpoints['legacy_path'],'exact-vs-legacy',62909050)
    evaluate(spec,checkpoints['legacy_path'],parent,'legacy-vs-parent',62919050)
    evaluate(spec,checkpoints['exact_joint'],parent,'exact-vs-parent',62919050)
    direct=json.loads((root/'evaluations/exact-vs-legacy.json').read_text())
    if direct['pairedMeanScore']>=.52 and direct['pairedScore95CI'][0]<=.5:
        evaluate(spec,checkpoints['exact_joint'],checkpoints['legacy_path'],'exact-vs-legacy-confirmation',62929050,pairs=200)
    write_json(root/'campaign-state.json',{'phase':'entropy_results_ready','parent':str(parent),
        'checkpoints':{k:str(v) for k,v in checkpoints.items()},'promotionAutomatic':False,
        'wallCostReports':[str(root/f'train-{mode}-wall.json') for mode in checkpoints]})
    # The optional CPU corpus was paused during controlled timing. Release it
    # only after both entropy timing windows too, so it cannot favor one arm.
    paused=subprocess.run(['docker','inspect','-f','{{.State.Paused}}','sap-one-step-cpu-corpus'],
                          capture_output=True,text=True)
    if paused.returncode==0 and paused.stdout.strip()=='true':
        subprocess.run(['docker','unpause','sap-one-step-cpu-corpus'],check=True)


if __name__ == '__main__':
    main()
