#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image="${PUCT_IMAGE:-sap-rl-v28-puct:20260903h}"
checkpoint="${PUCT_CHECKPOINT:-/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt}"
reservoir="${PUCT_RESERVOIR:-/workspace/runs/search/v35-v31-opponent-beliefs.json}"
stem="${PUCT_STEM:-v35-large-stratified-puct-v9}"
prefix="/workspace/runs/search/${stem}"
leaf_value_weight="${PUCT_LEAF_VALUE_WEIGHT:-0.5}"
seed_base="${PUCT_SEED_BASE:-2853501}"
container_prefix="${PUCT_CONTAINER_PREFIX:-sap-v35-puct}"
review_output="${PUCT_REVIEW_OUTPUT:-/workspace/runs/search-expert-review-v35-large-puct-v9.html}"

if [[ ! -f "${runs_host}${checkpoint#/workspace/runs}" ]]; then
  echo "checkpoint does not exist: ${checkpoint}" >&2
  exit 2
fi
while [[ ! -f "${runs_host}${reservoir#/workspace/runs}" ]]; do
  sleep 10
done
chunks_per_window="${PUCT_CHUNKS_PER_WINDOW:-20}"
expected_chunks=$((8 * chunks_per_window))
completed_chunks="$(find "${runs_host}/search" -maxdepth 1 -type f \
  -name "${stem}-window-*-chunk-*.summary.json" | wc -l)"
if [[ -f "${runs_host}/search/${stem}.exploration.json" \
      && "${completed_chunks}" -ge "${expected_chunks}" ]]; then
  exit 0
fi

# Two turn-window shards per broad turn band give 1,280 searched states total.
# Eight workers outperform sixteen on this unified-memory device: a measured
# 16-worker pilot slowed cheap screens from ~26 s to ~83 s each, reducing
# aggregate throughput despite higher apparent GPU utilization.
# The prior corpus contained only 128 states and 13 independently confirmed
# train roots, which was too small to distill without memorization. The
# 512-state audit projected 73 strict roots at 1,024 states, slightly below the
# predefined 75-root floor, so four resumable chunks per window were added.
minimum_turns=(2 4 5 7 9 10 11 13)
maximum_turns=(3 4 6 8 9 10 12 15)

train_parts=()
holdout_parts=()
audit_parts=()
summary_parts=()
attempts_per_chunk=8

# Each completed eight-state chunk has its own durable summary.  After a host,
# service, or container restart, completed chunks are skipped and at most one
# small in-flight chunk per worker is repeated.
for window in $(seq 0 7); do
  window_id="$(printf '%02d' "${window}")"
  for chunk in $(seq 0 $((chunks_per_window - 1))); do
    chunk_id="$(printf '%02d' "${chunk}")"
    part="${prefix}-window-${window_id}-chunk-${chunk_id}"
    train_parts+=("${part}.jsonl")
    holdout_parts+=("${part}-holdout.jsonl")
    audit_parts+=("${part}.audit.jsonl")
    summary_parts+=("${part}.summary.json")
  done
done

run_worker() {
  local worker=$1
  for chunk in $(seq 0 $((chunks_per_window - 1))); do
    # Rotate every worker through every turn window.  The modulo assignment
    # gives each worker one task per chunk wave and a balanced mix of early,
    # mid, late, and endgame work, avoiding a long slow-band tail.
    for window in $(seq 0 7); do
      if (( (window + chunk) % 8 != worker )); then
        continue
      fi
    local window_id chunk_id part summary_part container_name
    window_id="$(printf '%02d' "${window}")"
    chunk_id="$(printf '%02d' "${chunk}")"
    part="${prefix}-window-${window_id}-chunk-${chunk_id}"
    summary_part="${part}.summary.json"
    if [[ -f "${runs_host}${summary_part#/workspace/runs}" ]]; then
      continue
    fi
    container_name="${container_prefix}-${window_id}-${chunk_id}"
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    docker run --rm --name "${container_name}" \
      --gpus all --ipc host --network none \
      -v "${project_root}/sapai:/workspace/sapai" \
      -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
      -v "${runs_host}:/workspace/runs" \
      -w /workspace/sapai-gym \
      "${image}" \
      python scripts/generate_puct_demonstrations.py \
        "${checkpoint}" \
        --opponent-reservoir "${reservoir}" \
        --output "${part}.jsonl" \
        --holdout-output "${part}-holdout.jsonl" \
        --audit-output "${part}.audit.jsonl" \
        --summary-output "${summary_part}" \
        --games 128 \
        --max-search-attempts "${attempts_per_chunk}" \
        --max-searches-per-player-turn 2 \
        --reserve-zero-gold-search-slot \
        --opportunity-probability 1 \
        --general-probability 0.05 \
        --minimum-turn "${minimum_turns[window]}" \
        --maximum-turn "${maximum_turns[window]}" \
        --holdout-fraction 0.20 \
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
        --exhaustive-root-probability 0.05 \
        --leaf-value-weight "${leaf_value_weight}" \
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
        --seed "$((seed_base + window * 1000 + chunk))"
    done
  done
}

worker_pids=()
for worker in $(seq 0 7); do
  run_worker "${worker}" &
  worker_pids+=("$!")
done
failed=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "${worker_pid}"; then
    failed=1
  fi
done
if (( failed )); then
  echo "one or more v35 PUCT workers failed" >&2
  exit 1
fi

docker run --rm --network none \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym "${image}" \
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
  -w /workspace/sapai-gym "${image}" \
  python scripts/analyze_puct_exploration.py \
    --audit "${prefix}.audit.jsonl" \
    --output "${prefix}.exploration.json" \
    --minimum-states-per-band 128 \
    --minimum-full-searches-per-band 64 \
    --minimum-strict-roots 75 \
    --minimum-strict-bands 4 \
    --minimum-strict-train-roots 60 \
    --minimum-strict-holdout-roots 15 \
    --minimum-exhaustive-root-audits 20 \
    --minimum-opponent-sources 4

docker run --rm --network none \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym "${image}" \
  python scripts/filter_puct_demonstrations.py --root-only \
    --minimum-outer-lower-bound 0 \
    --input "${prefix}.jsonl" \
    --output "${prefix}-strict-root-only.jsonl" \
    --summary-output "${prefix}-strict-root-only.summary.json"

docker run --rm --network none \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym "${image}" \
  python scripts/filter_puct_demonstrations.py --root-only \
    --minimum-outer-lower-bound 0 \
    --input "${prefix}-holdout.jsonl" \
    --output "${prefix}-strict-root-only-holdout.jsonl" \
    --summary-output "${prefix}-strict-root-only-holdout.summary.json"

docker run --rm --network none \
  -v "${project_root}/sapai:/workspace/sapai" \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym "${image}" \
  python scripts/build_search_expert_review.py \
    --puct-audit "${prefix}.audit.jsonl" \
    --output "${review_output}" \
    --limit 200
