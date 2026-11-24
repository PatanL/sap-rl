#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image="${RARE_PUCT_IMAGE:-sap-rl-v28-puct:20260903h}"
checkpoint="${RARE_PUCT_CHECKPOINT:-/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt}"
reservoir="${RARE_PUCT_RESERVOIR:-/workspace/runs/search/v35-v31-opponent-beliefs.json}"
critic_host="${RARE_PUCT_CRITIC_HOST:-${runs_host}/search/v35-v31-critic-calibration.json}"
manifest="${RARE_PUCT_MANIFEST:-/workspace/runs/search/v35-rare-puct-manifest.json}"
prefix="${RARE_PUCT_PREFIX:-/workspace/runs/search/v35-rare-puct}"
num_shards="${RARE_PUCT_NUM_SHARDS:-39}"
manifest_seed="${RARE_PUCT_MANIFEST_SEED:-2853800}"
search_seed="${RARE_PUCT_SEARCH_SEED:-2853900}"
container_prefix="${RARE_PUCT_CONTAINER_PREFIX:-sap-v35-rare}"
review_output="${RARE_PUCT_REVIEW_OUTPUT:-/workspace/runs/search-expert-review-v35-rare-puct.html}"
prefix_host="${runs_host}${prefix#/workspace/runs}"
ready_host="${prefix_host}.ready.json"

while [[ ! -f "${critic_host}" ]]; do
  sleep 30
done
if [[ -f "${ready_host}" ]]; then
  exit 0
fi

leaf_value_weight="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["recommendedLeafValueWeight"])' \
  "${critic_host}")"

if [[ ! -f "${runs_host}${manifest#/workspace/runs}" ]]; then
  docker run --rm --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/generate_puct_scenario_manifest.py \
      --output "${manifest}" --seeds-per-scenario 12 --seed "${manifest_seed}"
fi

train_parts=()
holdout_parts=()
audit_parts=()
summary_parts=()
for shard in $(seq 0 $((num_shards - 1))); do
  shard_id="$(printf '%02d' "${shard}")"
  train_parts+=("${prefix}-part-${shard_id}.jsonl")
  holdout_parts+=("${prefix}-holdout-part-${shard_id}.jsonl")
  audit_parts+=("${prefix}-audit-part-${shard_id}.jsonl")
  summary_parts+=("${prefix}-summary-part-${shard_id}.json")
done

run_worker() {
  local worker=$1
  for ((shard=worker; shard<num_shards; shard+=8)); do
    local shard_id summary container
    shard_id="$(printf '%02d' "${shard}")"
    summary="${prefix}-summary-part-${shard_id}.json"
    if [[ -f "${runs_host}${summary#/workspace/runs}" ]]; then
      continue
    fi
    container="${container_prefix}-${shard_id}"
    docker rm -f "${container}" >/dev/null 2>&1 || true
    docker run --rm --name "${container}" \
      --gpus all --ipc host --network none \
      -v "${project_root}/sapai:/workspace/sapai" \
      -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
      -v "${runs_host}:/workspace/runs" \
      -w /workspace/sapai-gym "${image}" \
      python scripts/generate_puct_scenario_demonstrations.py \
        "${checkpoint}" "${manifest}" \
        --opponent-reservoir "${reservoir}" \
        --output "${prefix}-part-${shard_id}.jsonl" \
        --holdout-output "${prefix}-holdout-part-${shard_id}.jsonl" \
        --audit-output "${prefix}-audit-part-${shard_id}.jsonl" \
        --summary-output "${summary}" \
        --holdout-fraction 0.20 \
        --transition-budget 7500 --max-simulations 4096 \
        --min-root-visits 2 --horizon-battles 5 \
        --validation-rollouts 32 --max-validation-rollouts 512 \
        --audit-rollouts 256 \
        --leaf-value-weight "${leaf_value_weight}" \
        --shard-index "${shard}" --num-shards "${num_shards}" \
        --device cuda --batch-size 128 --seed "$((search_seed + shard))"
  done
}

pids=()
for worker in $(seq 0 7); do
  run_worker "${worker}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed )); then
  echo "one or more rare-scenario PUCT workers failed" >&2
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

python3 - "${prefix_host}-strict-root-only.summary.json" \
  "${prefix_host}-strict-root-only-holdout.summary.json" \
  "${prefix_host}.audit.jsonl" "${ready_host}" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
train = json.load(open(sys.argv[1]))
holdout = json.load(open(sys.argv[2]))
train_roots = int(train['rootTargetNodes'])
holdout_roots = int(holdout['rootTargetNodes'])
strict_scenarios = Counter()
with open(sys.argv[3]) as stream:
    for line in stream:
        row = json.loads(line)
        if (
            row.get('accepted')
            and float(row.get('outerAuditReturnLowerBound') or 0.0) > 0.0
        ):
            strict_scenarios[row.get('scenario', 'unknown')] += 1
print(
    "rare_strict_roots "
    f"train={train_roots} holdout={holdout_roots} "
    f"train_sequences={train['keptExamples']} "
    f"holdout_sequences={holdout['keptExamples']} "
    f"scenarios={len(strict_scenarios)}"
)
if train_roots < 12 or holdout_roots < 3 or len(strict_scenarios) < 8:
    raise SystemExit(
        "rare PUCT corpus did not meet the 12/3 root and 8-scenario "
        "readiness floor"
    )
Path(sys.argv[4]).write_text(json.dumps({
    "readyForDistillation": True,
    "trainRoots": train_roots,
    "holdoutRoots": holdout_roots,
    "strictScenarioCount": len(strict_scenarios),
    "strictScenarios": dict(sorted(strict_scenarios.items())),
}, indent=2, sort_keys=True) + "\n")
PY
