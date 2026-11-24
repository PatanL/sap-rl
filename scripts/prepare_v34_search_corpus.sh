#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
first=/workspace/runs/search/v33-stratified-puct-v9
second=/workspace/runs/search/v33b-stratified-puct-v9
combined=/workspace/runs/search/v34-combined-stratified-puct-v9

while [[ ! -f "${runs_host}/search/v33-stratified-puct-v9.summary.json" \
      || ! -f "${runs_host}/search/v33b-stratified-puct-v9.summary.json" ]]; do
  sleep 30
done
if [[ -f "${runs_host}/search/v34-combined-stratified-puct-v9.exploration.json" ]]; then
  exit 0
fi

docker run --rm --network none \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/merge_puct_demonstrations.py \
    --train "${first}.jsonl" "${second}.jsonl" \
    --holdout "${first}-holdout.jsonl" "${second}-holdout.jsonl" \
    --audit "${first}.audit.jsonl" "${second}.audit.jsonl" \
    --summaries "${first}.summary.json" "${second}.summary.json" \
    --output "${combined}.jsonl" \
    --holdout-output "${combined}-holdout.jsonl" \
    --audit-output "${combined}.audit.jsonl" \
    --summary-output "${combined}.summary.json"

docker run --rm --network none \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/analyze_puct_exploration.py \
    --audit "${combined}.audit.jsonl" \
    --output "${combined}.exploration.json" \
    --minimum-states-per-band 16 \
    --minimum-full-searches-per-band 8 \
    --minimum-strict-roots 8 \
    --minimum-strict-bands 2 \
    --minimum-strict-train-roots 5 \
    --minimum-strict-holdout-roots 2

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
  "${combined}.jsonl" \
  "${combined}-strict-root-only.jsonl" \
  "${combined}-strict-root-only.summary.json"
filter_root_targets \
  "${combined}-holdout.jsonl" \
  "${combined}-strict-root-only-holdout.jsonl" \
  "${combined}-strict-root-only-holdout.summary.json"

docker run --rm --network none \
  -v "${project_root}/sapai:/workspace/sapai" \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/build_search_expert_review.py \
    --puct-audit "${combined}.audit.jsonl" \
    --output /workspace/runs/search-expert-review-v34-combined-puct-v9.html \
    --limit 100

python3 - "${runs_host}/search/v34-combined-stratified-puct-v9.exploration.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
print(json.dumps(report["readiness"], sort_keys=True))
PY
