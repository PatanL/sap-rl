#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
run=/workspace/runs/turtle-league-v38-chocolate-legality-recalibration-from-v36-100
candidate=${run}/selected.pt
reservoir=/workspace/runs/search/v38-selected-opponent-beliefs.json
critic_report=/workspace/runs/search/v38-selected-critic-calibration.json
parent_eval_host="${runs_host}/turtle-league-v38-chocolate-legality-recalibration-from-v36-100/evaluations/paired-selected-vs-v36-corrected-runtime.json"

while [[ ! -f "${runs_host}${candidate#/workspace/runs}" ]]; do
  sleep 20
done
while [[ ! -f "${parent_eval_host}" ]]; do sleep 20; done

# Do not spend the search budget on a clearly collapsed transfer. Statistical
# parity is allowed because the purpose of this short branch is revision
# correction, not a champion claim.
python3 - "${parent_eval_host}" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
if result['pairedMeanScore'] < 0.45 or result['pairedScore95CI'][1] <= 0.50:
    raise SystemExit(
        'v38 is clearly weaker than its parent; reservoir/search blocked'
    )
PY

if [[ ! -f "${runs_host}${reservoir#/workspace/runs}" ]]; then
  checkpoints=(
    "${candidate}"
    /workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
    /workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
    /workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
  )
  games=(256 128 64 64)
  parts=()
  for index in 0 1 2 3; do
    component="/workspace/runs/search/v38-selected-opponent-beliefs-component-${index}.json"
    parts+=("${component}")
    if [[ -f "${runs_host}${component#/workspace/runs}" ]]; then
      continue
    fi
    mismatch_args=()
    if (( index > 0 )); then
      mismatch_args+=(--allow-checkpoint-training-revision-mismatch)
    fi
    docker run --rm --name "sap-v38-opponent-reservoir-${index}" \
      --gpus all --ipc host --network none \
      -v "${project_root}/sapai:/workspace/sapai" \
      -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
      -v "${runs_host}:/workspace/runs" \
      -w /workspace/sapai-gym "${image}" \
      python scripts/generate_opponent_trajectory_reservoir.py \
        "${checkpoints[index]}" --output "${component}" \
        --games "${games[index]}" --batch-size 128 \
        --device cuda --seed "$((3804601 + index))" \
        "${mismatch_args[@]}"
  done
  docker run --rm --network none \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/merge_opponent_trajectory_reservoirs.py \
      "${parts[@]}" --output "${reservoir}"
fi

if [[ ! -f "${runs_host}${critic_report#/workspace/runs}" ]]; then
  parts=()
  containers=()
  for shard in $(seq 0 7); do
    shard_id="$(printf '%02d' "${shard}")"
    part="/workspace/runs/search/v38-selected-critic-calibration-part-${shard_id}.json"
    parts+=("${part}")
    if [[ -f "${runs_host}${part#/workspace/runs}" ]]; then
      continue
    fi
    container="sap-v38-critic-${shard_id}"
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
        --device cuda --batch-size 128 --seed "$((3804700 + shard))" \
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

python3 - "${runs_host}${critic_report#/workspace/runs}" <<'PY'
import json, sys
critic = json.load(open(sys.argv[1]))
print("reservoir games=512 trajectories=1024 components=4")
overall = critic['overall']
print(
    f"critic states={overall['states']} "
    f"correlation={overall['correlation']:.4f} "
    f"mae={overall['mae']:.4f} bias={overall['bias']:.4f} "
    f"leaf_weight={critic['recommendedLeafValueWeight']}"
)
PY
