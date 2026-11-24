#!/usr/bin/env python3
"""Invocation-bound natural-chunk handoff. Never signals the active learner."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import time

from benchmark_guards import LaunchBlocked, query_unit, require_completion
from launcher_hold import LauncherHold

PROJECT = Path('/home/dzliu/pzliu/sap-rl')
IMAGE = 'sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1'


def command(args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=kwargs.pop('timeout', 30), **kwargs).stdout


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def docker_record(identifier):
    fmt = '{"id":"{{.Id}}","state":{{json .State}},"command":{{json .Config.Cmd}},"name":"{{.Name}}"}'
    return json.loads(command(['docker', 'inspect', '-f', fmt, identifier]))


def checkpoint_proof(path):
    container_path = '/workspace/runs/' + str(path.relative_to(PROJECT/'runs'))
    code = ("import torch,json; d=torch.load(" + repr(container_path) + ",map_location='cpu',weights_only=False);"
            "print(json.dumps({'update':d['update'],'simulator':d['simulator_revision'],"
            "'semantics':d.get('training_semantics',{}),'transitions':d['transitions']}))")
    output = command(['docker', 'run', '--rm', '--name', 'sap-abc-checkpoint-proof', '--network', 'none',
        '--cpus', '1', '-e', 'OMP_NUM_THREADS=1', '-v', f'{PROJECT}/runs:/workspace/runs:ro',
        IMAGE, 'python', '-c', code], timeout=120)
    return json.loads(next(line for line in reversed(output.splitlines()) if line.startswith('{')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    root = args.request.parent
    unit = query_unit(request['service'])
    if (unit['ActiveState'] != 'active' or unit['SubState'] != 'running'
            or unit['InvocationID'] != request['invocationId'] or int(unit['MainPID']) != request['launcherPid']):
        raise LaunchBlocked('Requested launcher invocation is no longer running')
    initial = docker_record(request['containerId'])
    if not initial['state']['Running'] or initial['id'] != request['containerId']:
        raise LaunchBlocked('Requested learner is no longer active')
    index = initial['command'].index('--updates')
    if int(initial['command'][index+1]) != request['targetUpdate']:
        raise LaunchBlocked('Wrong active training target')
    checkpoint = Path(request['checkpoint'])
    if checkpoint.exists():
        raise LaunchBlocked('Target checkpoint already exists; refusing stale completion evidence')
    hold = LauncherHold(request['launcherPid'], 'continue_v45_overnight.sh')
    stopped_service = False
    cancelled = False
    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('Handoff cancelled')
    signal.signal(signal.SIGTERM, cancel)
    try:
        hold.acquire()
        write_json(root/'state.json', {'phase': 'waiting_natural_chunk', 'target': request['targetUpdate'],
            'learnerContinues': True, 'launcherOnlyHeld': True, 'requestId': request['requestId']})
        deadline = time.monotonic() + 10800
        while True:
            current_unit = query_unit(request['service'])
            if (current_unit['ActiveState'] != 'active' or current_unit['InvocationID'] != request['invocationId']
                    or not hold.is_held()):
                raise LaunchBlocked('Service was manually changed or launcher ownership was lost')
            record = docker_record(request['containerId'])
            if not record['state']['Running']:
                break
            if time.monotonic() > deadline:
                raise LaunchBlocked('Natural checkpoint boundary timed out; leave normal training in control')
            time.sleep(5)
        if record['state']['ExitCode'] != 0:
            raise LaunchBlocked('Learner failed or was interrupted; no benchmark launch')
        # Readiness is a separate validation gate. Never idle the GPU for
        # unfinished implementation work: hand control back if it is absent.
        if not (root/'READY.json').is_file():
            write_json(root/'state.json', {'phase': 'deferred_not_ready', 'target': request['targetUpdate']})
            return
        logs = command(['docker', 'logs', '--tail', '60', request['containerId']])
        matches = [line for line in logs.splitlines() if re.fullmatch(
            rf'finished updates={request["targetUpdate"]} transitions=\d+ parameters=[\d,]+ device=cuda', line)]
        if len(matches) != 1 or not checkpoint.is_file():
            raise LaunchBlocked('Missing affirmative finished record or target checkpoint')
        metadata = checkpoint_proof(checkpoint)
        with checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        proof = {k: request[k] for k in ('requestId','invocationId','containerId','targetUpdate','checkpoint')}
        proof.update(kind='natural-training-chunk-v2', containerExitCode=0, containerRunning=False,
            containerFinishedAt=record['state']['FinishedAt'], checkpointAbsentAtRequest=True,
            launcherHeldAtCompletion=hold.is_held(), checkpointUpdate=metadata['update'],
            checkpointSHA256=digest, finishedRecord=matches[0], shutdownInitiator=request['requestId'],
            metadata=metadata, serviceBeforeAuthorizedStop=query_unit(request['service']))
        before = proof['serviceBeforeAuthorizedStop']
        if before['ActiveState'] != 'active' or before['InvocationID'] != request['invocationId'] or not hold.is_held():
            raise LaunchBlocked('Service changed before the authorized stop; completion is ambiguous')
        require_completion(request, proof, file_hash=digest)
        # The actual learner has exited 0 and saved its final checkpoint.
        # systemd sends TERM then CONT to terminate this held launcher cleanly.
        command(['systemctl', '--user', 'stop', request['service']], timeout=120)
        stopped_service = True
        proof['serviceAfterAuthorizedStop'] = query_unit(request['service'])
        write_json(root/'completion.json', proof)
        write_json(root/'state.json', {'phase': 'gpu_validation', 'target': request['targetUpdate']})
        result = subprocess.run(['python3', str(PROJECT/'scripts/abc_validation_runner_v2.py'),
            '--request', str(args.request), '--completion', str(root/'completion.json')])
        decision = json.loads((root/'decision.json').read_text()) if (root/'decision.json').exists() else {}
        if result.returncode != 0:
            decision = {**decision, 'rolloutAccepted': False, 'error': 'GPU validation did not complete/pass'}
        # Rollout authorization is not implied by a successful benchmark.
        if query_unit('sap-abc-early.service')['ActiveState'] != 'active':
            cancelled = True
            raise KeyboardInterrupt('Controller is stopping; do not resume training automatically')
        if request.get('sparseEvaluationAuthorized') or (decision.get('rolloutAccepted') and request.get('rolloutAuthorized')):
            command(['python3', str(PROJECT/'scripts/install_validated_campaign_v2.py'), str(args.request),
                     str(root/'completion.json'), str(root/'decision.json')], timeout=60)
        else:
            write_json(root/'state.json', {'phase': 'retained_original_runtime', 'decision': decision,
                                         'productionParent': str(checkpoint)})
            command(['systemctl', '--user', 'start', request['service']])
            stopped_service = False
    except BaseException as error:
        write_json(root/'error.json', {'error': str(error), 'cancelled': cancelled, 'target': request['targetUpdate']})
        raise
    finally:
        # Never leave the scheduling launcher suspended on an exception.
        hold.release()
        if stopped_service and not cancelled:
            # Installer marks a durable handoff before starting a replacement.
            if not (root/'production-handoff.json').exists() and query_unit('sap-abc-early.service')['ActiveState'] == 'active':
                subprocess.run(['systemctl', '--user', 'start', request['service']], timeout=30)


if __name__ == '__main__':
    main()
