#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h

while [[ ! -f "${runs_host}/search/v30-multifidelity-puct-pilot.summary.json" ]]; do
  sleep 20
done
if [[ -f "${runs_host}/search/v29-parent-anchor-holdout.summary.json" ]]; then
  exit 0
fi

docker run --rm --name sap-rl-v29-parent-anchor-holdout \
  --gpus all --ipc host --network none \
  -v "${project_root}/sapai:/workspace/sapai" \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/generate_policy_anchor_demonstrations.py \
    /workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt \
    --output /workspace/runs/search/v29-parent-anchor-holdout.jsonl \
    --summary-output /workspace/runs/search/v29-parent-anchor-holdout.summary.json \
    --games 8 \
    --max-sequences 128 \
    --teacher-weight 1 \
    --device cuda \
    --seed 2792002
