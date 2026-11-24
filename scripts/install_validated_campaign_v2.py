#!/usr/bin/env python3
"""Separate rollout authorization from benchmark acceptance and parent choice."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from benchmark_guards import verify_launch, query_unit, LaunchBlocked
from early_abc_handoff import PROJECT, write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request',type=Path)
    parser.add_argument('completion',type=Path)
    parser.add_argument('decision',type=Path)
    args = parser.parse_args()
    request, proof, decision = [json.loads(p.read_text()) for p in (args.request,args.completion,args.decision)]
    controller='sap-ab-validation.service' if 'ab-recovery' in request['directory'] else 'sap-abc-early.service'
    if query_unit(controller)['ActiveState'] != 'active':
        raise LaunchBlocked('Controller is stopping/inactive; do not turn cancellation into a production resume')
    verify_launch(request,proof)
    root = args.request.parent
    legacy_ok = all((root/f'fixed-legacy-{p}.json').exists() and json.loads((root/f'fixed-legacy-{p}.json').read_text()).get('gatePassed') for p in ('fp32','bf16'))
    accepted = bool(decision.get('rolloutAccepted') and request.get('rolloutAuthorized'))
    original = PROJECT/'runs/turtle-league-v45-corrected-from-v44-head-3250'
    source = Path(request['source']) if accepted or legacy_ok else original/'source-batched-v1'
    manifest = root/'source-manifest.json' if accepted or legacy_ok else original/'source-manifest-batched-v1.json'
    spec = {'request':str(args.request),'completion':str(args.completion), 'directory':str(root),
            'parent':request['checkpoint'],'parentSHA256':proof['checkpointSHA256'],
            'source':str(source),'sourceManifest':str(manifest), 'sourceManifestSHA256':digest(manifest),
            'runtimeFlags':decision.get('flags',[]) if accepted else [],
            'packagesAccepted':decision.get('acceptedPackages',[]) if accepted else [],
            'newLegacyValidated':legacy_ok,'entropyAuthorized':bool(request.get('entropyAblationAuthorized') and legacy_ok),
            'sparseAuthorized':bool(request.get('sparseEvaluationAuthorized')), 'updatesPerEntropyArm':1000,
            'benchmarkWeightsUsed':False}
    write_json(root/'campaign-spec.json',spec)
    if not (root/'initial-league.json').exists():
        import shutil
        shutil.copyfile(original/'league.json',root/'initial-league.json')
    state_path = PROJECT/'runs/CAMPAIGN_STATE.json'
    old_manifest=json.loads((original/'source-manifest-batched-v1.json').read_text())['sha256']
    model=json.loads((original/'config.json').read_text())['model']
    common={'simulatorRevision':proof['metadata']['simulator'],'entropyMode':'legacy_path',
            'modelSchema':model,'actionSchema':proof['metadata']['semantics'].get('action_schema')}
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'schemaVersion':2, 'selectedStrength':{**common,'checkpoint':str(original/'update-006250.pt'),
        'runtimeOptimizationFlags':[],'sourceManifestHash':old_manifest,
        'selectionStatus':'provisional evaluated reference, not a permanent 65-percent promotion',
        'reason':'Evaluated reference; latest does not mean champion.',
        'evaluations':[str(original/'evaluations'/f'paired-6250-vs-{p}.json') for p in ('parent','control','v31')]}}
    state.update(operationalParent={**common,'checkpoint':spec['parent'],'checkpointSHA256':spec['parentSHA256'],
                                   'trainingSourceManifestHash':old_manifest},
        experimentCommonParent={**common,'checkpoint':spec['parent'],'checkpointSHA256':spec['parentSHA256'],
            'runtimeOptimizationFlags':spec['runtimeFlags'],'experimentSourceManifestHash':json.loads(manifest.read_text())['sha256'],
            'reason':'Immutable real production parent for both arms; not benchmark weights.'},
        currentExperimentSpec=str(root/'campaign-spec.json'))
    write_json(state_path,state)
    unit = Path('/home/dzliu/.config/systemd/user/sap-architecture-campaign.service')
    unit.write_text('[Unit]\nDescription=Sparse v45 strength panel and isolated entropy common-parent experiment\n\n'
        '[Service]\nType=simple\nWorkingDirectory=/home/dzliu/pzliu/sap-rl\n'
        f'ExecStart=/usr/bin/python3 {PROJECT}/scripts/architecture_campaign_v2.py {root}/campaign-spec.json\n'
        'ExecStop=-/usr/bin/docker stop --timeout 60 sap-architecture-job\nTimeoutStopSec=120\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start','sap-architecture-campaign.service'],check=True)
    write_json(root/'production-handoff.json', {**spec,'service':'sap-architecture-campaign.service',
        'rolloutAuthorized':request.get('rolloutAuthorized',False), 'runtimeDeployed':accepted,
        'rollbackSource':str(original/'source-batched-v1'),'rollbackCheckpoint':request['checkpoint']})


if __name__ == '__main__':
    main()
