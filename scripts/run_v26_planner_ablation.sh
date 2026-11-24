#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v25-frozen:20260903-planner6
candidate_host="${runs_host}/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt"
candidate=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
reservoir=/workspace/runs/search/v25-update300-opponent-beliefs-v2.json
critic_host="${runs_host}/search/v25-update300-critic-calibration-v2.json"
output=/workspace/runs/search/v25-planner-depth-ablation-recal300-v6.json

if [[ ! -f "${candidate_host}" ]]; then
  echo "Missing recalibrated checkpoint: ${candidate_host}" >&2
  exit 1
fi

while [[ ! -f "${critic_host}" ]]; do
  sleep 30
done

leaf_value_weight="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["recommendedLeafValueWeight"])' \
  "${critic_host}")"

if [[ ! -f "${runs_host}${output#/workspace/runs}" ]]; then
  part_outputs=()
  containers=()
  for shard in $(seq 0 7); do
    shard_id="$(printf '%02d' "${shard}")"
    part="/workspace/runs/search/v25-planner-depth-ablation-recal300-v6-part-${shard_id}.json"
    part_outputs+=("${part}")
    if [[ -f "${runs_host}${part#/workspace/runs}" ]]; then
      continue
    fi
    container_name="sap-rl-v26-planner-ablation-${shard_id}"
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    docker run -d --name "${container_name}" \
      --gpus all --ipc host --network none \
      -v "${runs_host}:/workspace/runs" \
      "${image}" \
      python scripts/benchmark_planner_depth.py \
        "${candidate}" \
        --opponent-reservoir "${reservoir}" \
        --output "${part}" \
        --states 2 \
        --horizon-battles 5 \
        --rollouts-per-action 4 \
        --confirmation-rollouts 16 \
        --validation-rollouts 32 \
        --leaf-value-weight "${leaf_value_weight}" \
        --opponent-candidates 32 \
        --device cuda \
        --batch-size 128 \
        --seed "$((2601001 + shard))" >/dev/null
    containers+=("${container_name}")
  done
  for container_name in "${containers[@]}"; do
    exit_code="$(docker wait "${container_name}")"
    docker logs "${container_name}"
    docker rm "${container_name}" >/dev/null
    if [[ "${exit_code}" != "0" ]]; then
      echo "${container_name} failed with exit ${exit_code}" >&2
      exit 1
    fi
  done
  docker run --rm --network none \
    -v "${runs_host}:/workspace/runs" \
    -v "${project_root}/sapai-gym/scripts/merge_planner_ablations.py:/tmp/merge_planner_ablations.py:ro" \
    "${image}" \
    python /tmp/merge_planner_ablations.py \
      "${part_outputs[@]}" \
      --output "${output}"
fi
