#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
checkpoint=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
reservoir=/workspace/runs/search/v36-selected-opponent-beliefs.json
critic_host="${runs_host}/search/v36-selected-critic-calibration.json"
output=/workspace/runs/search/v36-puct-v10-equal-compute-vs-root.json

while [[ ! -f "${critic_host}" ]]; do
  sleep 20
done
if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then
  exit 0
fi
leaf_value_weight="$(python3 - "${critic_host}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['recommendedLeafValueWeight'])
PY
)"

parts=()
containers=()
for shard in $(seq 0 7); do
  shard_id="$(printf '%02d' "${shard}")"
  part="/workspace/runs/search/v36-puct-v10-equal-compute-vs-root-part-${shard_id}.json"
  parts+=("${part}")
  if [[ -f "${runs_host}${part#/workspace/runs}" ]]; then
    continue
  fi
  container="sap-v36-puct-equal-${shard_id}"
  docker rm -f "${container}" >/dev/null 2>&1 || true
  docker run -d --name "${container}" \
    --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/benchmark_puct_equal_compute.py \
      "${checkpoint}" --opponent-reservoir "${reservoir}" \
      --output "${part}" --states 4 \
      --validation-rollouts 32 --comparison-rollouts 64 \
      --horizon-battles 5 --leaf-value-weight "${leaf_value_weight}" \
      --puct-cpuct 2 --puct-prior-floor 0.10 \
      --puct-min-root-visits 2 \
      --opponent-candidates 32 --device cuda --batch-size 128 \
      --seed "$((3604900 + shard))" >/dev/null
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
  python scripts/merge_puct_equal_compute.py \
    "${parts[@]}" --output "${output}"
