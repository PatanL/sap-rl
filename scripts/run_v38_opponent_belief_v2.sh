#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
checkpoint=/workspace/runs/turtle-league-v38-chocolate-legality-recalibration-from-v36-100/selected.pt
reservoir=/workspace/runs/search/v38-selected-opponent-beliefs.json
output=/workspace/runs/search/v38-selected-opponent-belief-evaluation-v2.json

if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then exit 0; fi
docker run --rm --name sap-v38-opponent-belief-v2 \
  --gpus all --ipc host --network none \
  -v "${project_root}/sapai:/workspace/sapai" \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym "${image}" \
  python scripts/evaluate_opponent_belief.py "${checkpoint}" \
    --opponent-reservoir "${reservoir}" --output "${output}" \
    --states 512 --candidate-limit 32 --device cuda --batch-size 256 \
    --seed 3805101
