#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
run=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000
run_host="${runs_host}/turtle-league-v36-summons-tier1-recalibration-from-v31-1000"
final_candidate=${run}/update-000500.pt
candidate=${run}/selected.pt
parent=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
permanent=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
reservoir=/workspace/runs/search/v36-selected-opponent-beliefs.json
critic_report=/workspace/runs/search/v36-selected-critic-calibration.json

while [[ ! -f "${runs_host}${final_candidate#/workspace/runs}" ]]; do
  sleep 20
done
mkdir -p "${run_host}/evaluations"

paired_eval() {
  local checkpoint=$1
  local opponent=$2
  local output=$3
  local seed=$4
  local pairs=${5:-200}
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
      --device cuda --batch-size 64 --pairs "${pairs}" --progress-every 100 \
      --seed "${seed}" --output "${output}"
}

# Select rather than assume the last recalibration checkpoint is strongest.
# The five candidates share one paired-seat seed panel; the selected winner is
# then confirmed on a fresh, larger panel to avoid selecting evaluation noise.
selection_reports=()
for update in 100 200 300 400 500; do
  update_name="$(printf 'update-%06d.pt' "${update}")"
  update_checkpoint="${run}/${update_name}"
  output="${run}/evaluations/curve-${update_name%.pt}-vs-v31-selection.json"
  selection_reports+=("${run_host}/evaluations/${output##*/}")
  paired_eval "${update_checkpoint}" "${parent}" "${output}" 3604500 100
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
if baseline_scores is None:
    raise SystemExit('checkpoint selection reports must retain per-pair scores')
baseline_seeds = rows[0].get('pairSeeds')
if baseline_seeds is None or any(
    row.get('pairSeeds') != baseline_seeds for row in rows
):
    raise SystemExit('checkpoint selection reports do not share exact pair seeds')

def paired_delta(row):
    differences = [
        float(left) - float(right)
        for left, right in zip(row['pairScores'], baseline_scores)
    ]
    mean = sum(differences) / len(differences)
    if len(differences) < 2:
        return mean, [mean, mean]
    variance = sum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    margin = 1.96 * math.sqrt(variance / len(differences))
    return mean, [mean - margin, mean + margin]

best = max(
    rows, key=lambda row: (row['pairedMeanScore'], row['checkpointUpdate'])
)
payload = {
    'method': 'common-100-pair-screen-then-fresh-200-pair-confirmation',
    'selectionSeed': 3604500,
    'candidates': [],
    'selectedCheckpoint': best['checkpoint'],
    'selectedUpdate': best['checkpointUpdate'],
}
for row in rows:
    delta, delta_ci = paired_delta(row)
    payload['candidates'].append(
        {
            'checkpoint': row['checkpoint'],
            'update': row['checkpointUpdate'],
            'score': row['pairedMeanScore'],
            'ci95': row['pairedScore95CI'],
            'pairedDeltaVsUpdate100': delta,
            'pairedDeltaVsUpdate100CI95': delta_ci,
        }
    )
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(Path(best['checkpoint']).name)
PY
)"
ln -sfn "${selected_name}" "${run_host}/selected.pt"

# Both old checkpoints are evaluated under the corrected runtime. They remain
# valid frozen behavior baselines, but their own revision fields make clear
# that they were trained under pre-fix generated-pet mechanics.
paired_eval "${candidate}" "${parent}" \
  "${run}/evaluations/paired-selected-vs-v31-corrected-runtime.json" \
  3604501 200
paired_eval "${candidate}" "${permanent}" \
  "${run}/evaluations/paired-selected-vs-v25-corrected-runtime.json" \
  3604502 200

python3 - "${run_host}/selection.json" \
  "${run_host}/evaluations/paired-selected-vs-v31-corrected-runtime.json" \
  "${run_host}/evaluations/paired-selected-vs-v25-corrected-runtime.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
selection = json.loads(path.read_text())
parent = json.load(open(sys.argv[2]))
permanent = json.load(open(sys.argv[3]))
selection['freshConfirmation'] = {
    'pairs': parent['pairs'],
    'versusParentSeed': parent['seed'],
    'versusParentScore': parent['pairedMeanScore'],
    'versusParentCI95': parent['pairedScore95CI'],
    'versusPermanentSeed': permanent['seed'],
    'versusPermanentScore': permanent['pairedMeanScore'],
    'versusPermanentCI95': permanent['pairedScore95CI'],
}
path.write_text(json.dumps(selection, indent=2, sort_keys=True) + '\n')
PY

if [[ ! -f "${runs_host}${reservoir#/workspace/runs}" ]]; then
  reservoir_checkpoints=(
    "${candidate}"
    "${parent}"
    "${permanent}"
    /workspace/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt
  )
  reservoir_games=(256 128 64 64)
  reservoir_parts=()
  for index in 0 1 2 3; do
    component="/workspace/runs/search/v36-selected-opponent-beliefs-component-${index}.json"
    reservoir_parts+=("${component}")
    if [[ -f "${runs_host}${component#/workspace/runs}" ]]; then
      continue
    fi
    mismatch_args=()
    if (( index > 0 )); then
      mismatch_args+=(--allow-checkpoint-training-revision-mismatch)
    fi
    docker run --rm --name "sap-v36-opponent-reservoir-${index}" \
      --gpus all --ipc host --network none \
      -v "${project_root}/sapai:/workspace/sapai" \
      -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
      -v "${runs_host}:/workspace/runs" \
      -w /workspace/sapai-gym "${image}" \
      python scripts/generate_opponent_trajectory_reservoir.py \
        "${reservoir_checkpoints[index]}" --output "${component}" \
        --games "${reservoir_games[index]}" --batch-size 128 \
        --device cuda --seed "$((3604601 + index))" \
        "${mismatch_args[@]}"
  done
  docker run --rm --network none \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/merge_opponent_trajectory_reservoirs.py \
      "${reservoir_parts[@]}" --output "${reservoir}"
fi

if [[ ! -f "${runs_host}${critic_report#/workspace/runs}" ]]; then
  parts=()
  containers=()
  for shard in $(seq 0 7); do
    shard_id="$(printf '%02d' "${shard}")"
    part="/workspace/runs/search/v36-selected-critic-calibration-part-${shard_id}.json"
    parts+=("${part}")
    if [[ -f "${runs_host}${part#/workspace/runs}" ]]; then
      continue
    fi
    container="sap-v36-critic-${shard_id}"
    docker rm -f "${container}" >/dev/null 2>&1 || true
    docker run -d --name "${container}" \
      --gpus all --ipc host --network none \
      -v "${project_root}/sapai:/workspace/sapai" \
      -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
      -v "${runs_host}:/workspace/runs" \
      -w /workspace/sapai-gym "${image}" \
      python scripts/evaluate_critic_calibration.py \
        "${candidate}" --opponent-reservoir "${reservoir}" \
        --output "${part}" --states 64 --rollouts 32 \
        --horizon-battles 10 --max-continuation-steps 500 \
        --opponent-candidates 32 --state-probability 0.15 \
        --device cuda --batch-size 128 --seed "$((3604700 + shard))" \
        >/dev/null
    containers+=("${container}")
  done
  for container in "${containers[@]}"; do
    exit_code="$(docker wait "${container}")"
    docker logs "${container}"
    docker rm "${container}" >/dev/null
    if [[ "${exit_code}" != 0 ]]; then
      echo "${container} failed with exit ${exit_code}" >&2
      exit 1
    fi
  done
  docker run --rm --network none \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/merge_critic_calibrations.py \
      "${parts[@]}" --output "${critic_report}"
fi

python3 - "${run_host}/evaluations/paired-selected-vs-v31-corrected-runtime.json" \
  "${run_host}/evaluations/paired-selected-vs-v25-corrected-runtime.json" \
  "${runs_host}${critic_report#/workspace/runs}" <<'PY'
import json, sys
for path in sys.argv[1:3]:
    result = json.load(open(path))
    print(
        f"{path}: score={result['pairedMeanScore']:.4f} "
        f"ci={result['pairedScore95CI']}"
    )
critic = json.load(open(sys.argv[3]))
overall = critic['overall']
print(
    "critic "
    f"states={overall['states']} correlation={overall['correlation']:.4f} "
    f"mae={overall['mae']:.4f} bias={overall['bias']:.4f} "
    f"leaf_weight={critic['recommendedLeafValueWeight']}"
)
PY
