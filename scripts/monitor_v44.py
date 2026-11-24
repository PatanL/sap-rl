#!/usr/bin/env python3
"""Observe the queued experiment; never start, stop or promote anything."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from monitor_v39 import command, parse_service_fields


CAMPAIGN = Path('/home/dzliu/pzliu/sap-rl/runs/turtle-v44-end-freeze-ablation')
STAGES = ('smoke-control', 'smoke-end-freeze', 'control', 'end-freeze', 'evaluation')


def phase_name(running, service_state, parent_state):
    for stage in STAGES:
        if f'sap-v44-{stage}' in running:
            return stage
    if service_state not in ('active', 'activating', 'deactivating', 'inactive', 'failed'):
        return 'unknown'
    if service_state == 'failed':
        return 'failed'
    if service_state != 'active':
        return 'stopped_or_complete'
    return 'waiting_for_parent' if parent_state in ('active', 'activating') else 'waiting_or_transitioning'


def main():
    now = datetime.now(timezone.utc)
    service = command('systemctl', '--user', 'show', 'sap-v44-ablation.service',
                      '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result')
    parent = command('systemctl', '--user', 'show', 'sap-v39-unrestricted.service', '-p', 'ActiveState')
    running = set(command('docker', 'ps', '--format', '{{.Names}}').splitlines())
    fields = parse_service_fields(service)
    phase = phase_name(running, fields.get('ActiveState'), parse_service_fields(parent).get('ActiveState'))
    warnings = []
    if fields.get('ActiveState') == 'failed':
        warnings.append('v44 coordinator failed; inspect journal before resuming')
    last = None
    if phase in STAGES and phase != 'evaluation':
        log = command('docker', 'logs', '--timestamps', '--tail', '100', f'sap-v44-{phase}')
        updates = []
        for line in log.splitlines():
            if ' update=' not in line:
                continue
            row = dict(re.findall(r'(\w+)=([^\s]+)', line))
            row['timestamp'] = line.split()[0]
            updates.append(row)
        for row in updates:
            for key in ('loss', 'value', 'grad', 'entropy'):
                if not math.isfinite(float(row[key])):
                    warnings.append(f'Nonfinite {key} in {phase} update {row["update"]}')
        if updates:
            last = updates[-1]
            stamp = datetime.fromisoformat(last['timestamp'].replace('Z', '+00:00'))
            if (now - stamp).total_seconds() > 600:
                warnings.append(f'No completed {phase} update for more than ten minutes')
    payload = {
        'checkedAt': now.isoformat(), 'service': fields, 'phase': phase,
        'runningExperimentContainers': sorted(running.intersection({f'sap-v44-{s}' for s in STAGES})),
        'lastUpdate': last,
        'checkpoints': {stage: max((p.name for p in (CAMPAIGN / stage).glob('update-*.pt')), default=None)
                        for stage in STAGES if stage != 'evaluation'},
        'completedEvaluations': sorted(p.name for p in (CAMPAIGN / 'evaluations').glob('*.json')),
        'warnings': sorted(set(warnings)),
    }
    temporary = CAMPAIGN / 'health.json.tmp'
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(CAMPAIGN / 'health.json')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
