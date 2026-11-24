#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
checkpoint=/workspace/runs/turtle-league-v38-chocolate-legality-recalibration-from-v36-100/selected.pt
reservoir=/workspace/runs/search/v38-selected-opponent-beliefs.json
critic_host="${runs_host}/search/v38-selected-critic-calibration.json"
output=/workspace/runs/search/v38-puct-v10-targeted-budgeted-hybrid-vs-root.json

if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then exit 0; fi
if [[ ! -f "${critic_host}" ]]; then
  echo "missing critic calibration: ${critic_host}" >&2
  exit 1
fi
leaf_value_weight="$(python3 - "${critic_host}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['recommendedLeafValueWeight'])
PY
)"

minimum_turns=(2 4 5 7 9 10 11 13)
maximum_turns=(3 4 6 8 9 10 12 15)
parts=()
containers=()
for shard in $(seq 0 7); do
  shard_id="$(printf '%02d' "${shard}")"
  part="/workspace/runs/search/v38-puct-v10-targeted-budgeted-hybrid-vs-root-part-${shard_id}.json"
  parts+=("${part}")
  if [[ -f "${runs_host}${part#/workspace/runs}" ]]; then continue; fi
  container="sap-v38-puct-hybrid-${shard_id}"
  docker rm -f "${container}" >/dev/null 2>&1 || true
  docker run -d --name "${container}" --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/benchmark_puct_equal_compute.py \
      "${checkpoint}" --opponent-reservoir "${reservoir}" \
      --output "${part}" --states 4 --state-mode opportunity \
      --minimum-turn "${minimum_turns[shard]}" \
      --maximum-turn "${maximum_turns[shard]}" \
      --validation-rollouts 32 --comparison-rollouts 64 \
      --horizon-battles 5 --leaf-value-weight "${leaf_value_weight}" \
      --puct-cpuct 2 --puct-prior-floor 0.10 --puct-min-root-visits 2 \
      --puct-transition-budget-scale 0.8 \
      --puct-external-confirmation-only \
      --puct-root-screening-top-k 4 \
      --puct-root-screening-rollouts 2 \
      --puct-root-screening-rounds 2 \
      --puct-root-screening-min-survivors 2 \
      --opponent-candidates 32 --device cuda --batch-size 128 \
      --seed "$((3805300 + shard))" >/dev/null
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
