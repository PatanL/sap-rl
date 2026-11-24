#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
checkpoint=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
reservoir=/workspace/runs/search/v35-v31-opponent-beliefs.json
report=/workspace/runs/search/v35-v31-critic-calibration.json

if [[ -f "${runs_host}${report#/workspace/runs}" ]]; then
  exit 0
fi

parts=()
containers=()
for shard in $(seq 0 7); do
  shard_id="$(printf '%02d' "${shard}")"
  part="/workspace/runs/search/v35-v31-critic-calibration-part-${shard_id}.json"
  parts+=("${part}")
  if [[ -f "${runs_host}${part#/workspace/runs}" ]]; then
    continue
  fi
  container="sap-v35-critic-${shard_id}"
  docker rm -f "${container}" >/dev/null 2>&1 || true
  docker run -d --name "${container}" \
    --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/evaluate_critic_calibration.py \
      "${checkpoint}" \
      --opponent-reservoir "${reservoir}" \
      --output "${part}" \
      --states 64 \
      --rollouts 32 \
      --horizon-battles 10 \
      --max-continuation-steps 500 \
      --opponent-candidates 32 \
      --state-probability 0.15 \
      --device cuda \
      --batch-size 128 \
      --seed "$((2853700 + shard))" >/dev/null
  containers+=("${container}")
done

for container in "${containers[@]}"; do
  exit_code="$(docker wait "${container}")"
  docker logs "${container}"
  docker rm "${container}" >/dev/null
  if [[ "${exit_code}" != "0" ]]; then
    echo "${container} failed with exit ${exit_code}" >&2
    exit 1
  fi
done

docker run --rm --network none \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym "${image}" \
  python scripts/merge_critic_calibrations.py \
    "${parts[@]}" --output "${report}"
