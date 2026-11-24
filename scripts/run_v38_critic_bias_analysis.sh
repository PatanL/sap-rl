#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
input="${project_root}/runs/search/v38-selected-critic-calibration.json"
output=/workspace/runs/search/v38-selected-critic-bias-calibration.json
while [[ ! -f "${input}" ]]; do sleep 20; done
if [[ -f "${project_root}/runs${output#/workspace/runs}" ]]; then exit 0; fi
docker run --rm --network none \
  -v "${project_root}/sapai:/workspace/sapai" \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${project_root}/runs:/workspace/runs" \
  -w /workspace/sapai-gym sap-rl-v28-puct:20260903h \
  python scripts/analyze_critic_bias_calibration.py \
    /workspace/runs/search/v38-selected-critic-calibration.json \
    --output "${output}" --ridge 1.0 --minimum-mae-improvement 0.02
