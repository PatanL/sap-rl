#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
sapai_host="${project_root}/sapai"
gym_host="${project_root}/sapai-gym"
image=sap-rl-v28-puct:20260903h
prefix=/workspace/runs/search/v29-validated-candidates
candidates=/workspace/runs/search/v29-conditional-failure-seeds.json
checkpoint=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
reservoir=/workspace/runs/search/v25-update300-opponent-beliefs-v2.json

while [[ ! -f "${runs_host}/search/v29-conditional-puct.summary.json" ]]; do
  sleep 20
done
if [[ -f "${runs_host}/search/v29-validated-candidates.summary.json" ]]; then
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
  container_name="sap-rl-v29-candidate-validator-${shard_id}"
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
  docker run -d --name "${container_name}" \
    --gpus all --ipc host --network none \
    -v "${sapai_host}:/workspace/sapai" \
    -v "${gym_host}:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/generate_validated_candidate_demonstrations.py \
      "${checkpoint}" \
      "${candidates}" \
      --opponent-reservoir "${reservoir}" \
      --output "${train_part}" \
      --holdout-output "${holdout_part}" \
      --audit-output "${audit_part}" \
      --summary-output "${summary_part}" \
      --initial-rollouts 128 \
      --max-rollouts 512 \
      --audit-rollouts 256 \
      --horizon-battles 5 \
      --opponent-candidates 32 \
      --confidence-z 1.96 \
      --minimum-improvement 0 \
      --holdout-fraction 0.20 \
      --candidate-source expert-diagnostic \
      --device cuda \
      --batch-size 128 \
      --seed "$((2795001 + shard))" \
      --shard-index "${shard}" \
      --num-shards 8 >/dev/null
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
  -v "${sapai_host}:/workspace/sapai" \
  -v "${gym_host}:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/merge_validated_candidate_demonstrations.py \
    --train "${train_parts[@]}" \
    --holdout "${holdout_parts[@]}" \
    --audit "${audit_parts[@]}" \
    --summaries "${summary_parts[@]}" \
    --output "${prefix}.jsonl" \
    --holdout-output "${prefix}-holdout.jsonl" \
    --audit-output "${prefix}.audit.jsonl" \
    --summary-output "${prefix}.summary.json"
