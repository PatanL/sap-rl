#!/usr/bin/env python3
"""Read-only completion checks for the queued v44 experiment."""
import argparse
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
import re
import statistics


SIMULATOR = 'turtle-current-live-chocolate-maxlevel-v27-20260904'


def source_inventory(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.json', '.toml', '.cfg'):
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError('Empty frozen-source inventory')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'version': 'v44-frozen-code-data-v1', 'sha256': digest, 'files': files}


def check_source(root, manifest):
    actual = source_inventory(root)
    expected = json.loads(manifest.read_text())
    if actual != expected:
        raise ValueError('Frozen training source changed; refusing a mixed-source experiment')


def validate_evaluation(report, update, opponent_suffix):
    if report.get('checkpointUpdate') != update or report.get('games') != 200 or report.get('pairs') != 100:
        raise ValueError('Unexpected evaluation checkpoint or sample size')
    if report.get('runtimeSimulatorRevision') != SIMULATOR or report.get('checkpointSimulatorRevision') != SIMULATOR:
        raise ValueError('Mixed simulator revisions in prerequisite evaluation')
    if not report.get('opponentCheckpoint', '').endswith(opponent_suffix):
        raise ValueError('Unexpected prerequisite opponent')
    seeds = report['pairSeeds']
    records = report['gameResults']
    if len(seeds) != 100 or len(set(seeds)) != 100 or len(records) != 200:
        raise ValueError('Incomplete evaluation records')
    expected = {(index, seed, seat) for index, seed in enumerate(seeds) for seat in (0, 1)}
    actual = {(r['pairIndex'], r['seed'], r['candidateSide']) for r in records}
    if actual != expected:
        raise ValueError('Missing or duplicated paired seats')
    outcomes = Counter(r['candidateResult'] for r in records)
    if set(outcomes) - {'win', 'draw', 'loss'} or dict(outcomes) != report['outcomes']:
        raise ValueError('Evaluation outcome counts do not match records')
    score = (outcomes['win'] + .5 * outcomes['draw']) / 200
    if not math.isclose(score, report['pairedMeanScore'], abs_tol=1e-12):
        raise ValueError('Evaluation score does not match records')
    if dict(Counter(r['terminationReason'] for r in records)) != report['terminationCounts']:
        raise ValueError('Evaluation termination counts do not match records')
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in report['pairedScore95CI']):
        raise ValueError('Invalid evaluation interval')


def parent_ready(run):
    required = [run / 'update-002250.pt', run / 'league.json']
    reports = [run / 'evaluations' / f'paired-2250-vs-{panel}.json' for panel in ('parent', 'v31')]
    if not all(path.is_file() for path in required + reports):
        return False
    suffixes = [
        'turtle-league-v38-chocolate-legality-recalibration-from-v36-100/update-000250.pt',
        'turtle-league-v31-search-integration-from-v25-300/update-001000.pt',
    ]
    for path, suffix in zip(reports, suffixes):
        validate_evaluation(json.loads(path.read_text()), 2250, suffix)
    members = json.loads(required[1].read_text())['members']
    if not any(m['member_id'] == 'historical-002250' for m in members):
        raise ValueError('Final checkpoint is missing its durable league entry')
    return True


def validate_smoke(log, expected_update, *, joint):
    updates = []
    for line in log.splitlines():
        if line.startswith('update='):
            fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
            for key in ('loss', 'policy', 'value', 'entropy', 'grad'):
                if not math.isfinite(float(fields[key])):
                    raise ValueError(f'Nonfinite {key} in GPU smoke')
            updates.append(fields)
    if not updates or int(updates[-1]['update']) != expected_update:
        raise ValueError('GPU smoke did not finish its declared training steps')
    if f'finished updates={expected_update} ' not in log:
        raise ValueError('GPU smoke lacks a successful completion record')
    if joint:
        if not any(int(row.get('end_toggles', '0')) > 0 for row in updates):
            raise ValueError('GPU smoke did not exercise any end-freeze toggles')
        if 'action_schema=primitive-plus-end-freeze-v1-20260905' not in log:
            raise ValueError('GPU smoke did not use the joint action schema')


def smoke_speed_ratio(control_log, joint_log):
    def speed(log):
        rows = {}
        for line in log.splitlines():
            if line.startswith('update='):
                fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
                rows[int(fields['update'])] = float(fields['update_fps'])
        recent = [rows[update] for update in sorted(rows)[-2:]]
        if len(recent) != 2 or any(not math.isfinite(x) or x <= 0 for x in recent):
            raise ValueError('Insufficient finite GPU smoke throughput measurements')
        return statistics.median(recent)
    return speed(joint_log) / speed(control_log)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    parent = commands.add_parser('parent')
    parent.add_argument('run', type=Path)
    smoke = commands.add_parser('smoke')
    smoke.add_argument('log', type=Path)
    smoke.add_argument('--update', type=int, required=True)
    smoke.add_argument('--joint', action='store_true')
    source = commands.add_parser('source')
    source.add_argument('root', type=Path)
    source.add_argument('manifest', type=Path)
    source.add_argument('--record', action='store_true')
    compare = commands.add_parser('compare-smokes')
    compare.add_argument('control', type=Path)
    compare.add_argument('joint', type=Path)
    args = parser.parse_args()
    if args.command == 'parent':
        if not parent_ready(args.run):
            raise SystemExit(3)  # Not ready, not a failure; caller keeps waiting.
    elif args.command == 'smoke':
        validate_smoke(args.log.read_text(), args.update, joint=args.joint)
    elif args.command == 'compare-smokes':
        ratio = smoke_speed_ratio(args.control.read_text(), args.joint.read_text())
        print(f'joint_to_control_smoke_throughput={ratio:.4f}', flush=True)
        if ratio < .5:
            raise SystemExit('Joint action path is over twice as slow; profile before full training')
    elif args.record:
        with args.manifest.open('x') as stream:
            json.dump(source_inventory(args.root), stream, indent=2, sort_keys=True)
    else:
        check_source(args.root, args.manifest)


if __name__ == '__main__':
    main()
