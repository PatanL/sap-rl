#!/usr/bin/env python3
"""Record v39 health without interrupting learning or starting evaluations."""
import datetime
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

RUN = Path('/home/dzliu/pzliu/sap-rl/runs/turtle-league-v39-unrestricted-from-v38-250')


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    return result.stdout + result.stderr


def parse_service_fields(text):
    return dict(line.split('=', 1) for line in text.splitlines() if '=' in line)


def service_warnings(service, training_state, evaluation_running):
    """Report authoritative failures, not a manual pause or a missing probe."""
    fields = parse_service_fields(service)
    warnings = []
    if fields.get('ActiveState') == 'failed':
        warnings.append(f'Training service failed: {fields.get("Result", "unknown")}')
    if (fields.get('ActiveState') == 'active' and not evaluation_running
            and training_state and training_state.get('Status') in ('dead', 'exited')):
        warnings.append(f'Training container exited with code {training_state.get("ExitCode")}')
    return warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--service', default='sap-v39-unrestricted.service')
    parser.add_argument('--training-container', default='sap-rl-v39-unrestricted')
    parser.add_argument('--evaluation-container', default='sap-rl-v39-evaluation')
    args = parser.parse_args()
    now = datetime.datetime.now(datetime.timezone.utc)
    logs = command('docker', 'logs', '--timestamps', '--tail', '200', args.training_container)
    updates = []
    for line in logs.splitlines():
        if ' update=' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        fields['timestamp'] = line.split()[0]
        updates.append(fields)
    last = updates[-1] if updates else {}
    warnings = []
    for row in updates[-20:]:
        for key in ('loss', 'value', 'grad', 'entropy'):
            if key in row and not math.isfinite(float(row[key])):
                warnings.append(f'Nonfinite {key} at update {row["update"]}')
    service = command('systemctl', '--user', 'show', args.service,
                      '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result').strip()
    evaluation_running = command('docker', 'inspect', '-f', '{{.State.Running}}',
                                 args.evaluation_container).strip() == 'true'
    training_probe = command('docker', 'inspect', '-f', '{{json .State}}', args.training_container)
    try:
        training_state = json.loads(training_probe)
    except json.JSONDecodeError:
        training_state = None  # A failed observation is not proof of termination.
    warnings.extend(service_warnings(service, training_state, evaluation_running))
    if (last and not evaluation_running
            and parse_service_fields(service).get('ActiveState') in ('active', 'activating')):
        stamp = datetime.datetime.fromisoformat(last['timestamp'].replace('Z', '+00:00'))
        if (now - stamp).total_seconds() > 600:
            warnings.append('No completed update for more than 10 minutes')
    speeds = [float(row['update_fps']) for row in updates[-20:] if 'update_fps' in row]
    payload = {
        'checkedAt': now.isoformat(), 'service': service,
        'evaluationRunning': evaluation_running,
        'trainingContainerState': training_state,
        'trainingContainerProbeError': training_probe if training_state is None else None,
        'lastUpdate': last,
        'meanRecentTransitionsPerSecond': statistics.mean(speeds) if speeds else None,
        'latestCheckpoint': max((p.name for p in args.run.glob('update-*.pt')), default=None),
        'completedEvaluations': sorted(p.name for p in (args.run / 'evaluations').glob('paired-*.json')),
        'warnings': sorted(set(warnings)),
    }
    temp = args.run / 'health.json.tmp'
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(args.run / 'health.json')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
