#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
run_host="${project_root}/runs/turtle-league-v31-search-integration-from-v25-300"
eval_host="${run_host}/evaluations"
selection_host="${run_host}/v33-source-selection.json"
parent=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
candidate=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
image=sap-rl-v28-puct:20260903h

while [[ ! -f "${eval_host}/paired-update1000-vs-parent.json" \
      || ! -f "${eval_host}/habits-update1000.json" \
      || ! -f "${eval_host}/anchor-holdout-update1000.json" ]]; do
  sleep 30
done

if [[ ! -f "${eval_host}/root-only-update1000.json" ]]; then
  docker run --rm --cpus 2 --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${project_root}/runs:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/evaluate_curriculum.py \
      "${candidate}" \
      /workspace/runs/search/v31-strict-puct-root-only.jsonl \
      --threads 2 --batch-size 64 \
      --output \
        /workspace/runs/turtle-league-v31-search-integration-from-v25-300/evaluations/root-only-update1000.json \
      >/dev/null
fi
if [[ ! -f "${eval_host}/root-only-holdout-update1000.json" ]]; then
  docker run --rm --cpus 2 --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${project_root}/runs:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/evaluate_curriculum.py \
      "${candidate}" \
      /workspace/runs/search/v31-strict-puct-root-only-holdout.jsonl \
      --threads 2 --batch-size 64 \
      --output \
        /workspace/runs/turtle-league-v31-search-integration-from-v25-300/evaluations/root-only-holdout-update1000.json \
      >/dev/null
fi

selected="$(python3 - "${eval_host}" "${selection_host}" "${parent}" "${candidate}" <<'PY'
import json
import sys
from pathlib import Path

evaluation_dir = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
parent, candidate = sys.argv[3:5]
paired = json.loads(
    (evaluation_dir / "paired-update1000-vs-parent.json").read_text()
)
habits = json.loads(
    (evaluation_dir / "habits-update1000.json").read_text()
)["overall"]
anchor_before = json.loads(
    (evaluation_dir / "anchor-holdout-before.json").read_text()
)["overall"]
anchor_after = json.loads(
    (evaluation_dir / "anchor-holdout-update1000.json").read_text()
)["overall"]
root_integration = json.loads(
    (evaluation_dir / "root-only-update150.json").read_text()
)["overall"]
root_final = json.loads(
    (evaluation_dir / "root-only-update1000.json").read_text()
)["overall"]
root_holdout_final = json.loads(
    (evaluation_dir / "root-only-holdout-update1000.json").read_text()
)["overall"]
root_holdout_before = json.loads(
    (evaluation_dir / "root-only-holdout-before.json").read_text()
)["overall"]

checks = {
    "pairedPointEstimateAtLeastEven": paired["pairedMeanScore"] >= 0.50,
    "pairedUpperBoundAboveEven": paired["pairedScore95CI"][1] > 0.50,
    "habitAccuracyAtLeast85Percent": habits["fullActionAccuracy"] >= 0.85,
    "anchorCrossEntropyWithin025": (
        anchor_after["weightedCrossEntropy"]
        <= anchor_before["weightedCrossEntropy"] + 0.25
    ),
    "rootProbabilityAtLeast80PercentOfIntegration": (
        root_final["meanTargetProbability"]
        >= 0.8 * root_integration["meanTargetProbability"]
    ),
    "rootHoldoutCrossEntropyBetterThanParent": (
        root_holdout_final["weightedCrossEntropy"]
        < root_holdout_before["weightedCrossEntropy"]
    ),
}
selected = candidate if all(checks.values()) else parent
payload = {
    "selectedCheckpoint": selected,
    "candidateCheckpoint": candidate,
    "fallbackCheckpoint": parent,
    "checks": checks,
    "pairedMeanScore": paired["pairedMeanScore"],
    "pairedScore95CI": paired["pairedScore95CI"],
    "habitFullActionAccuracy": habits["fullActionAccuracy"],
    "anchorCrossEntropyBefore": anchor_before["weightedCrossEntropy"],
    "anchorCrossEntropyAfter": anchor_after["weightedCrossEntropy"],
    "rootProbabilityIntegration": root_integration["meanTargetProbability"],
    "rootProbabilityFinal": root_final["meanTargetProbability"],
    "rootHoldoutProbabilityFinal": root_holdout_final[
        "meanTargetProbability"
    ],
    "rootHoldoutCrossEntropyBefore": root_holdout_before[
        "weightedCrossEntropy"
    ],
    "rootHoldoutCrossEntropyFinal": root_holdout_final[
        "weightedCrossEntropy"
    ],
    "selectionRule": "non-regressive exploratory source; not permanent promotion",
}
selection_path.parent.mkdir(parents=True, exist_ok=True)
selection_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(selected)
PY
)"

exec "${project_root}/scripts/run_stratified_puct_v9.sh" \
  "${selected}" v33-stratified-puct-v9 2830001
