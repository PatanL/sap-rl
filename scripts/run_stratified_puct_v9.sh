#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 CHECKPOINT_UNDER_RUNS OUTPUT_STEM BASE_SEED [HOLDOUT_FRACTION]" >&2
  exit 2
fi

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
checkpoint=$1
stem=$2
base_seed=$3
holdout_fraction=${4:-0.20}
prefix="/workspace/runs/search/${stem}"

if [[ "${checkpoint}" != /workspace/runs/* ]]; then
  echo "checkpoint must be an absolute /workspace/runs path" >&2
  exit 2
fi
if [[ ! -f "${runs_host}${checkpoint#/workspace/runs}" ]]; then
  echo "checkpoint does not exist: ${checkpoint}" >&2
  exit 2
fi
if [[ -f "${runs_host}/search/${stem}.summary.json" ]]; then
  exit 0
fi

# Narrow, staggered windows prevent each shard from exhausting its attempt
# budget at the left edge of one broad band. The previous pilot spent all 64
# searches on turns 2--5 and never inspected ordinary late-game decisions.
minimum_turns=(2 4 5 7 9 10 11 13)
maximum_turns=(3 4 6 8 9 10 12 15)

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
  container_name="sap-rl-stratified-puct-${shard_id}"
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
  docker run -d --name "${container_name}" \
    --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/generate_puct_demonstrations.py \
      "${checkpoint}" \
      --opponent-reservoir /workspace/runs/search/v25-update300-opponent-beliefs-v2.json \
      --output "${train_part}" \
      --holdout-output "${holdout_part}" \
      --audit-output "${audit_part}" \
      --summary-output "${summary_part}" \
      --games 64 \
      --max-search-attempts 8 \
      --max-searches-per-player-turn 2 \
      --reserve-zero-gold-search-slot \
      --opportunity-probability 1 \
      --general-probability 0.05 \
      --minimum-turn "${minimum_turns[shard]}" \
      --maximum-turn "${maximum_turns[shard]}" \
      --holdout-fraction "${holdout_fraction}" \
      --transition-budget 7500 \
      --max-simulations 4096 \
      --min-root-visits 2 \
      --horizon-battles 5 \
      --validation-rollouts 32 \
      --max-validation-rollouts 512 \
      --audit-rollouts 256 \
      --root-screening-top-k 8 \
      --root-screening-rollouts 8 \
      --root-screening-rounds 3 \
      --root-screening-min-survivors 4 \
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
      --screening-minimum-improvement 0.02 \
      --screening-bypass-probability 0.10 \
      --opportunity-screening-bypass-probability 0.50 \
      --general-screening-bypass-probability 0.10 \
      --device cuda \
      --batch-size 128 \
      --seed "$((base_seed + shard))" >/dev/null
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

docker run --rm --network none \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/analyze_puct_exploration.py \
    --audit "${prefix}.audit.jsonl" \
    --output "${prefix}.exploration.json" \
    --minimum-states-per-band 8 \
    --minimum-full-searches-per-band 4 \
    --minimum-strict-roots 4 \
    --minimum-strict-bands 2 \
    --minimum-strict-train-roots 3 \
    --minimum-strict-holdout-roots 1

filter_root_targets() {
  local input=$1
  local output=$2
  local summary=$3
  local host_input="${runs_host}${input#/workspace/runs}"
  if ! python3 - "${host_input}" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    teacher = json.loads(line).get("teacher", {})
    if (
        teacher.get("name") == "stochastic-puct"
        and teacher.get("rootAccepted")
        and teacher.get("targetIsRoot")
        and float(teacher.get("outerAuditReturnImprovement") or 0) > 0
        and float(teacher.get("outerAuditReturnLowerBound") or 0) > 0
    ):
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    return
  fi
  docker run --rm --network none \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/filter_puct_demonstrations.py --root-only \
      --minimum-outer-lower-bound 0 \
      --input "${input}" \
      --output "${output}" \
      --summary-output "${summary}"
}

filter_root_targets \
  "${prefix}.jsonl" \
  "${prefix}-strict-root-only.jsonl" \
  "${prefix}-strict-root-only.summary.json"
filter_root_targets \
  "${prefix}-holdout.jsonl" \
  "${prefix}-strict-root-only-holdout.jsonl" \
  "${prefix}-strict-root-only-holdout.summary.json"

docker run --rm --network none \
  -v "${project_root}/sapai:/workspace/sapai" \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/build_search_expert_review.py \
    --puct-audit "${prefix}.audit.jsonl" \
    --output "/workspace/runs/search-expert-review-${stem}.html" \
    --limit 100
