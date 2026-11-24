#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
run=/workspace/runs/turtle-league-v38-chocolate-legality-recalibration-from-v36-100
run_host="${runs_host}/turtle-league-v38-chocolate-legality-recalibration-from-v36-100"
parent=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
v31=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
final_candidate=${run}/update-000300.pt
candidate=${run}/selected.pt

while [[ ! -f "${runs_host}${final_candidate#/workspace/runs}" ]]; do
  sleep 20
done
mkdir -p "${run_host}/evaluations"

paired_eval() {
  local checkpoint=$1 opponent=$2 output=$3 seed=$4 pairs=${5:-200}
  if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then
    return
  fi
  docker run --rm --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/evaluate_paired_policy.py \
      "${checkpoint}" --opponent checkpoint \
      --opponent-checkpoint "${opponent}" \
      --device cuda --batch-size 64 --pairs "${pairs}" \
      --progress-every 100 --seed "${seed}" --output "${output}"
}

selection_reports=()
for update in 50 100 150 200 250 300; do
  update_name="$(printf 'update-%06d.pt' "${update}")"
  output="${run}/evaluations/curve-${update_name%.pt}-vs-v36-selection.json"
  selection_reports+=("${run_host}/evaluations/${output##*/}")
  paired_eval "${run}/${update_name}" "${parent}" "${output}" 3804500 100
done

selected_name="$(python3 - "${run_host}/selection.json" "${selection_reports[@]}" <<'PY'
import json, math, sys
from pathlib import Path

output = Path(sys.argv[1])
rows = [json.load(open(path)) for path in sys.argv[2:]]
candidate_revisions = {row.get('checkpointSimulatorRevision') for row in rows}
runtime_revisions = {row.get('runtimeSimulatorRevision') for row in rows}
if len(candidate_revisions) != 1 or candidate_revisions != runtime_revisions:
    raise SystemExit(
        'refusing mixed-revision checkpoint selection: '
        f'candidate={candidate_revisions} runtime={runtime_revisions}'
    )
rows.sort(key=lambda row: row['checkpointUpdate'])
baseline_scores = rows[0].get('pairScores')
baseline_seeds = rows[0].get('pairSeeds')
if baseline_scores is None or baseline_seeds is None:
    raise SystemExit('selection reports must retain pair scores and seeds')
if any(row.get('pairSeeds') != baseline_seeds for row in rows):
    raise SystemExit('selection reports do not share exact pair seeds')

def paired_delta(row):
    values = [
        float(left) - float(right)
        for left, right in zip(row['pairScores'], baseline_scores)
    ]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return mean, [mean - margin, mean + margin]

best = max(rows, key=lambda row: (row['pairedMeanScore'], row['checkpointUpdate']))
payload = {
    'method': 'common-100-pair-screen-then-fresh-200-pair-confirmation',
    'selectionSeed': 3804500,
    'candidates': [],
    'selectedCheckpoint': best['checkpoint'],
    'selectedUpdate': best['checkpointUpdate'],
}
for row in rows:
    delta, delta_ci = paired_delta(row)
    payload['candidates'].append({
        'checkpoint': row['checkpoint'],
        'update': row['checkpointUpdate'],
        'score': row['pairedMeanScore'],
        'ci95': row['pairedScore95CI'],
        'pairedDeltaVsUpdate50': delta,
        'pairedDeltaVsUpdate50CI95': delta_ci,
    })
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(Path(best['checkpoint']).name)
PY
)"
ln -sfn "${selected_name}" "${run_host}/selected.pt"

paired_eval "${candidate}" "${parent}" \
  "${run}/evaluations/paired-selected-vs-v36-corrected-runtime.json" \
  3804501 200
paired_eval "${candidate}" "${v31}" \
  "${run}/evaluations/paired-selected-vs-v31-corrected-runtime.json" \
  3804502 200

python3 - "${run_host}/selection.json" \
  "${run_host}/evaluations/paired-selected-vs-v36-corrected-runtime.json" \
  "${run_host}/evaluations/paired-selected-vs-v31-corrected-runtime.json" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
selection = json.loads(path.read_text())
parent = json.load(open(sys.argv[2]))
v31 = json.load(open(sys.argv[3]))
selection['freshConfirmation'] = {
    'pairs': parent['pairs'],
    'versusParentSeed': parent['seed'],
    'versusParentScore': parent['pairedMeanScore'],
    'versusParentCI95': parent['pairedScore95CI'],
    'versusV31Seed': v31['seed'],
    'versusV31Score': v31['pairedMeanScore'],
    'versusV31CI95': v31['pairedScore95CI'],
}
path.write_text(json.dumps(selection, indent=2, sort_keys=True) + '\n')
print(json.dumps(selection['freshConfirmation'], sort_keys=True))
PY
