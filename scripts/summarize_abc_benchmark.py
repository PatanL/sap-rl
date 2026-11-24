#!/usr/bin/env python3
"""Accepted-decision throughput, not primitive-click or update-count speed."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def summarize(folder):
    rows = [dict(re.findall(r'(\w+)=([^\s]+)', line)) for line in (folder/'training.log').read_text().splitlines()
            if line.startswith('update=')]
    if len(rows) != 40 or int(rows[-1]['update']) != 6290:
        raise ValueError('Incomplete runtime screen')
    rows = rows[10:]
    decisions = sum(int(r['learner_primitives']) - int(r['end_toggles']) for r in rows)
    seconds = sum(float(r['collect_s']) + float(r['learn_s']) for r in rows)
    for row in rows:
        for key in ('loss', 'policy', 'value', 'entropy', 'grad'):
            if not math.isfinite(float(row[key])):
                raise ValueError('Nonfinite benchmark learner')
    stale_segments = sum(int(row['lag_drop'].split('/')[0]) for row in rows)
    stale_decisions = sum(int(row['lag_drop'].split('/')[1]) for row in rows)
    maximum_lag = max(int(row['lag'].split('/')[1]) for row in rows)
    if maximum_lag > 1:
        raise ValueError('Lag bound violated')
    utilization = []
    for line in (folder/'gpu.csv').read_text().splitlines()[1:]:
        try:
            utilization.append(float(line.split(',')[1].replace('%', '').strip()))
        except (ValueError, IndexError):
            pass
    return {'measuredUpdates': len(rows), 'acceptedDecisions': decisions, 'timedLoopSeconds': seconds,
            'acceptedDecisionsPerSecond': decisions/seconds,
            'meanCollectSeconds': statistics.mean(float(r['collect_s']) for r in rows),
            'meanLearnPublishSeconds': statistics.mean(float(r['learn_s']) for r in rows),
            'staleSegments': stale_segments, 'staleDecisions': stale_decisions,
            'staleSegmentRate': stale_segments/(sum(int(r['segments']) for r in rows)+stale_segments),
            'maximumLag': maximum_lag, 'meanGpuUtilizationDiagnostic': statistics.mean(utilization) if utilization else None,
            'inference': json.loads((folder/'inference-telemetry.json').read_text())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    result = {variant: summarize(args.directory / f'runtime-{variant}') for variant in ('baseline', 'abc')}
    result['speedup'] = result['abc']['acceptedDecisionsPerSecond']/result['baseline']['acceptedDecisionsPerSecond']
    result['deployed'] = False
    result['benchmarkWeightsUsedForTraining'] = False
    result['interpretation'] = 'Single ordered A/B screen; repeat/reverse order and cover resident turnover before a deployment decision.'
    (args.directory/'runtime-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
