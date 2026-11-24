"""Fail-closed, read-only GPU launch guards. No time/idle-utilization shortcut."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


class LaunchBlocked(RuntimeError):
    pass


UNIT_FIELDS = ('LoadState', 'ActiveState', 'SubState', 'Result', 'ExecMainStatus', 'InvocationID', 'MainPID')


def query_unit(name, run=subprocess.run):
    result = run(['systemctl', '--user', 'show', name, *sum((['-p', k] for k in UNIT_FIELDS), [])],
                 capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise LaunchBlocked(f'Cannot establish service state: {name}')
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if set(UNIT_FIELDS) - fields.keys() or fields['LoadState'] != 'loaded':
        raise LaunchBlocked(f'Incomplete/unknown service state: {name}')
    return fields


def require_inactive_success(name, fields):
    if fields['ActiveState'] != 'inactive' or fields['SubState'] != 'dead':
        raise LaunchBlocked(f'{name} is not inactive/dead: {fields["ActiveState"]}/{fields["SubState"]}')
    if fields['Result'] != 'success' or fields['MainPID'] != '0':
        raise LaunchBlocked(f'{name} has no clean completed service result')
    if fields['ExecMainStatus'] not in ('0', '15', '143'):
        raise LaunchBlocked(f'{name} has an unexpected launcher exit status')


def require_completion(request, proof, *, file_hash=None):
    if not proof:
        raise LaunchBlocked('Affirmative invocation-bound completion evidence is missing')
    for key in ('requestId', 'invocationId', 'containerId', 'targetUpdate', 'checkpoint'):
        if proof.get(key) != request.get(key):
            raise LaunchBlocked(f'Completion evidence belongs to another request: {key}')
    if not re.fullmatch('[0-9a-f]{32}', str(proof.get('invocationId', ''))):
        raise LaunchBlocked('Missing systemd invocation provenance')
    if proof.get('kind') != 'natural-training-chunk-v2' or proof.get('containerExitCode') != 0:
        raise LaunchBlocked('Training chunk did not finish normally')
    if proof.get('containerRunning') is not False or not proof.get('containerFinishedAt'):
        raise LaunchBlocked('Container completion is not established')
    if not proof.get('checkpointAbsentAtRequest') or not proof.get('launcherHeldAtCompletion'):
        raise LaunchBlocked('Completion could be stale or a manual stop')
    if proof.get('checkpointUpdate') != request.get('targetUpdate'):
        raise LaunchBlocked('Saved checkpoint does not match the completed target')
    expected = rf'finished updates={request["targetUpdate"]} transitions=\d+ parameters=[\d,]+ device=cuda'
    if not re.fullmatch(expected, str(proof.get('finishedRecord', ''))):
        raise LaunchBlocked('Missing exact normal-finish record')
    digest = proof.get('checkpointSHA256', '')
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise LaunchBlocked('Checkpoint content hash missing')
    if file_hash is not None and digest != file_hash:
        raise LaunchBlocked('Checkpoint changed after completion certification')
    if proof.get('shutdownInitiator') != request['requestId']:
        raise LaunchBlocked('Service stop was not the authorized post-completion handoff')


def verify_launch(request, proof, *, run=subprocess.run, check_files=True):
    for name in request['guardServices']:
        require_inactive_success(name, query_unit(name, run))
    digest = None
    if check_files:
        path = Path(request['checkpoint'])
        if not path.is_file():
            raise LaunchBlocked('Completed checkpoint is absent')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require_completion(request, proof, file_hash=digest)
    result = run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                 capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise LaunchBlocked('Cannot establish GPU ownership')
    if result.stdout.strip():
        raise LaunchBlocked('GPU has an owner; do not evict it')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    parser.add_argument('completion', type=Path)
    args = parser.parse_args()
    verify_launch(json.loads(args.request.read_text()), json.loads(args.completion.read_text()))


if __name__ == '__main__':
    main()
