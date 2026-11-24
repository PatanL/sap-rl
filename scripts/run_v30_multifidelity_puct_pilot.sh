#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
prefix=/workspace/runs/search/v30-multifidelity-puct-pilot

while [[ ! -f "${runs_host}/search/v29-parent-anchor.summary.json" \
      || ! -f "${runs_host}/search/v29-puct-v7-freeze-check.summary.json" ]]; do
  sleep 20
done
python3 -c '
import json, sys
report = json.load(open(sys.argv[1]))
if report.get("accepted", 0) < 1:
    raise SystemExit("PUCT v7 did not recover the known Chocolate action")
' "${runs_host}/search/v29-puct-v7-freeze-check.summary.json"
if [[ -f "${runs_host}/search/v30-multifidelity-puct-pilot.summary.json" ]]; then
  exit 0
fi

train_parts=()
holdout_parts=()
audit_parts=()
summary_parts=()
containers=()
for shard in $(seq 0 7); do
  shard_id="$(printf '%02d' "${shard}")"
  train_part="${prefix}-part-${shard_id}.jsonl"
  holdout_part="${prefix}-holdout-part-${shard_id}.jsonl"
  audit_part="${prefix}-audit-part-${shard_id}.jsonl"
  summary_part="${prefix}-summary-part-${shard_id}.json"
  train_parts+=("${train_part}")
  holdout_parts+=("${holdout_part}")
  audit_parts+=("${audit_part}")
  summary_parts+=("${summary_part}")
  if [[ -f "${runs_host}${summary_part#/workspace/runs}" ]]; then
    continue
  fi
  container_name="sap-rl-v30-multifidelity-puct-${shard_id}"
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
  docker run -d --name "${container_name}" \
    --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/generate_puct_demonstrations.py \
      /workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt \
      --opponent-reservoir /workspace/runs/search/v25-update300-opponent-beliefs-v2.json \
      --output "${train_part}" \
      --holdout-output "${holdout_part}" \
      --audit-output "${audit_part}" \
      --summary-output "${summary_part}" \
      --games 64 \
      --max-search-attempts 8 \
      --max-searches-per-player-turn 1 \
      --opportunity-probability 1 \
      --general-probability 0.05 \
      --minimum-turn 2 \
      --maximum-turn 20 \
      --holdout-fraction 0.20 \
      --transition-budget 7500 \
      --max-simulations 4096 \
      --min-root-visits 2 \
      --horizon-battles 5 \
      --validation-rollouts 32 \
      --max-validation-rollouts 128 \
      --audit-rollouts 64 \
      --cpuct 2 \
      --prior-floor 0.10 \
      --leaf-value-weight 0.5 \
      --internal-target-weight 0.25 \
      --minimum-target-probability 0.02 \
      --opponent-candidates 32 \
      --screening-transition-budget 1000 \
      --screening-max-simulations 512 \
      --screening-validation-rollouts 8 \
      --screening-max-validation-rollouts 16 \
      --screening-bypass-probability 0.05 \
      --device cuda \
      --batch-size 128 \
      --seed "$((2800001 + shard))" >/dev/null
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
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/merge_puct_demonstrations.py \
    --train "${train_parts[@]}" \
    --holdout "${holdout_parts[@]}" \
    --audit "${audit_parts[@]}" \
    --summaries "${summary_parts[@]}" \
    --output "${prefix}.jsonl" \
    --holdout-output "${prefix}-holdout.jsonl" \
    --audit-output "${prefix}.audit.jsonl" \
    --summary-output "${prefix}.summary.json"
