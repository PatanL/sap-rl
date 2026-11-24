#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
raw=/workspace/runs/search/v38-puct-v10-targeted-equal-compute-vs-root.json
calibrated=/workspace/runs/search/v38-puct-v10-targeted-calibrated-equal-compute-vs-root.json
output=/workspace/runs/search/v38-puct-v10-targeted-critic-calibration-comparison.json

if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then exit 0; fi
for input in "${raw}" "${calibrated}"; do
  if [[ ! -f "${runs_host}${input#/workspace/runs}" ]]; then
    echo "missing required benchmark: ${input}" >&2
    exit 1
  fi
done

docker run --rm --network none \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym "${image}" \
  python scripts/compare_puct_calibration.py \
    "${raw}" "${calibrated}" --output "${output}"
