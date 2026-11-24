#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903b
checkpoint=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
reservoir=/workspace/runs/search/v25-update300-opponent-beliefs-v2.json
prefix=v28b-puct-teacher-pilot
merged_train="/workspace/runs/search/${prefix}.jsonl"
merged_holdout="/workspace/runs/search/${prefix}-holdout.jsonl"
merged_audit="/workspace/runs/search/${prefix}.audit.jsonl"
merged_summary="/workspace/runs/search/${prefix}.summary.json"

if [[ -f "${runs_host}/search/${prefix}.summary.json" ]]; then
  exit 0
fi

train_parts=()
holdout_parts=()
audit_parts=()
summary_parts=()
containers=()
for shard in $(seq 0 7); do
  shard_id="$(printf '%02d' "${shard}")"
  train_part="/workspace/runs/search/${prefix}-part-${shard_id}.jsonl"
  holdout_part="/workspace/runs/search/${prefix}-holdout-part-${shard_id}.jsonl"
  audit_part="/workspace/runs/search/${prefix}-audit-part-${shard_id}.jsonl"
  summary_part="/workspace/runs/search/${prefix}-summary-part-${shard_id}.json"
  train_parts+=("${train_part}")
  holdout_parts+=("${holdout_part}")
  audit_parts+=("${audit_part}")
  summary_parts+=("${summary_part}")
  if [[ -f "${runs_host}${summary_part#/workspace/runs}" ]]; then
    continue
  fi
  target_turns=(2 3 4 5 6 7 9 11)
  target_turn="${target_turns[shard]}"
  container_name="sap-rl-v28-puct-pilot-${shard_id}"
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
  docker run -d --name "${container_name}" \
    --gpus all --ipc host --network none \
    -v "${runs_host}:/workspace/runs" \
    "${image}" \
    python scripts/generate_puct_demonstrations.py \
      "${checkpoint}" \
      --output "${train_part}" \
      --holdout-output "${holdout_part}" \
      --audit-output "${audit_part}" \
      --summary-output "${summary_part}" \
      --opponent-reservoir "${reservoir}" \
      --games 64 \
      --max-search-attempts 8 \
      --max-searches-per-player-turn 1 \
      --opportunity-probability 0.70 \
      --general-probability 0.20 \
      --minimum-turn "${target_turn}" \
      --maximum-turn "${target_turn}" \
      --holdout-fraction 0.20 \
      --transition-budget 7500 \
      --max-simulations 4096 \
      --horizon-battles 5 \
      --validation-rollouts 32 \
      --max-validation-rollouts 128 \
      --audit-rollouts 64 \
      --cpuct 2.0 \
      --prior-floor 0.10 \
      --leaf-value-weight 0.5 \
      --internal-target-weight 0.25 \
      --minimum-target-probability 0.02 \
      --opponent-candidates 32 \
      --device cuda \
      --batch-size 128 \
      --seed "$((2751001 + shard))" >/dev/null
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
  python scripts/merge_puct_demonstrations.py \
    --train "${train_parts[@]}" \
    --holdout "${holdout_parts[@]}" \
    --audit "${audit_parts[@]}" \
    --summaries "${summary_parts[@]}" \
    --output "${merged_train}" \
    --holdout-output "${merged_holdout}" \
    --audit-output "${merged_audit}" \
    --summary-output "${merged_summary}"

python3 -c '
import json, sys
report = json.load(open(sys.argv[1]))
print(json.dumps({
    "attempted": report["attempted"],
    "accepted": report["accepted"],
    "trainExamples": report["trainExamples"],
    "holdoutExamples": report["holdoutExamples"],
    "outerAudit": report["outerAudit"],
}, sort_keys=True))
' "${runs_host}/search/${prefix}.summary.json"
