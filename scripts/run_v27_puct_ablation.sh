#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v27-puct:20260903e
checkpoint=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
reservoir=/workspace/runs/search/v25-update300-opponent-beliefs-v2.json
natural_report="${runs_host}/search/v25-planner-depth-ablation-recal300-v6.json"
output=/workspace/runs/search/v27-puct-equal-compute-vs-root.json

while [[ ! -f "${natural_report}" ]]; do
  sleep 30
done

if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then
  exit 0
fi

parts=()
containers=()
for shard in $(seq 0 7); do
  shard_id="$(printf '%02d' "${shard}")"
  part="/workspace/runs/search/v27-puct-equal-compute-vs-root-part-${shard_id}.json"
  parts+=("${part}")
  if [[ -f "${runs_host}${part#/workspace/runs}" ]]; then
    continue
  fi
  container_name="sap-rl-v27-puct-ablation-${shard_id}"
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
  docker run -d --name "${container_name}" \
    --gpus all --ipc host --network none \
    -v "${runs_host}:/workspace/runs" \
    "${image}" \
    python scripts/benchmark_puct_equal_compute.py \
      "${checkpoint}" \
      --opponent-reservoir "${reservoir}" \
      --output "${part}" \
      --states 1 \
      --validation-rollouts 32 \
      --horizon-battles 5 \
      --leaf-value-weight 0.5 \
      --opponent-candidates 32 \
      --device cuda \
      --batch-size 128 \
      --seed "$((2711001 + shard))" >/dev/null
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
  "${image}" \
  python scripts/merge_puct_equal_compute.py \
    "${parts[@]}" \
    --output "${output}"
