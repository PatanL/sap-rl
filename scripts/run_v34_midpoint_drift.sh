#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
run=/workspace/runs/turtle-league-v34-stratified-puct-integration-from-v31-1000
run_host="${runs_host}/turtle-league-v34-stratified-puct-integration-from-v31-1000"
evaluation_host="${run_host}/evaluations"
checkpoint="${run}/update-000500.pt"

while [[ ! -f "${run_host}/update-000500.pt" ]]; do
  sleep 30
done

evaluate() {
  local data=$1
  local output=$2
  if [[ -f "${evaluation_host}/${output}" ]]; then
    return
  fi
  docker run --rm --cpus 2 --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/evaluate_curriculum.py \
      "${checkpoint}" "${data}" --threads 2 --batch-size 64 \
      --output "${run}/evaluations/${output}" >/dev/null
}

evaluate /workspace/runs/search/v34-combined-stratified-puct-v9-strict-root-only.jsonl \
  strict-update500.json
evaluate /workspace/runs/search/v34-combined-stratified-puct-v9-strict-root-only-holdout.jsonl \
  strict-holdout-update500.json
evaluate /workspace/runs/search/v34-parent-anchor-holdout.jsonl \
  anchor-holdout-update500.json
evaluate /workspace/runs/curriculum/turtle-p1-scenarios-v6-20260902.jsonl \
  habits-update500.json

python3 - "${evaluation_host}" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
load = lambda name: json.loads((directory / name).read_text())["overall"]
strict_before = load("strict-before.json")
strict_500 = load("strict-update500.json")
holdout_before = load("strict-holdout-before.json")
holdout_500 = load("strict-holdout-update500.json")
anchor_before = load("anchor-holdout-before.json")
anchor_500 = load("anchor-holdout-update500.json")
habits_500 = load("habits-update500.json")
checks = {
    "strictCrossEntropyBetterThanParent": (
        strict_500["weightedCrossEntropy"]
        < strict_before["weightedCrossEntropy"]
    ),
    "strictHoldoutCrossEntropyBetterThanParent": (
        holdout_500["weightedCrossEntropy"]
        < holdout_before["weightedCrossEntropy"]
    ),
    "habitAccuracyAtLeast85Percent": (
        habits_500["fullActionAccuracy"] >= 0.85
    ),
    "anchorCrossEntropyWithin025": (
        anchor_500["weightedCrossEntropy"]
        <= anchor_before["weightedCrossEntropy"] + 0.25
    ),
}
payload = {
    "checkpoint": "update-000500.pt",
    "checks": checks,
    "healthy": all(checks.values()),
    "strictCrossEntropyBefore": strict_before["weightedCrossEntropy"],
    "strictCrossEntropy500": strict_500["weightedCrossEntropy"],
    "strictHoldoutCrossEntropyBefore": holdout_before["weightedCrossEntropy"],
    "strictHoldoutCrossEntropy500": holdout_500["weightedCrossEntropy"],
    "habitAccuracy500": habits_500["fullActionAccuracy"],
    "anchorCrossEntropyBefore": anchor_before["weightedCrossEntropy"],
    "anchorCrossEntropy500": anchor_500["weightedCrossEntropy"],
}
(directory / "midpoint-drift-summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, sort_keys=True))
PY
