#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
candidate_host="${runs_host}/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt"
reservoir_host="${runs_host}/search/v36-selected-opponent-beliefs.json"
critic_host="${runs_host}/search/v36-selected-critic-calibration.json"
eval_host="${runs_host}/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/evaluations/paired-selected-vs-v31-corrected-runtime.json"

for required in "${candidate_host}" "${reservoir_host}" "${critic_host}" "${eval_host}"; do
  while [[ ! -f "${required}" ]]; do
    sleep 20
  done
done

# Do not spend a large search budget on a clearly collapsed transfer branch.
# This is deliberately a weak safety gate, not a champion-promotion claim.
python3 - "${eval_host}" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
if result['pairedMeanScore'] < 0.45 or result['pairedScore95CI'][1] <= 0.50:
    raise SystemExit(
        'v36 recalibration is clearly weaker than v31; corrected PUCT blocked'
    )
PY

leaf_value_weight="$(python3 - "${critic_host}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['recommendedLeafValueWeight'])
PY
)"

export PUCT_CHECKPOINT=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
export PUCT_RESERVOIR=/workspace/runs/search/v36-selected-opponent-beliefs.json
export PUCT_STEM=v36-large-stratified-puct-v10
export PUCT_LEAF_VALUE_WEIGHT="${leaf_value_weight}"
export PUCT_SEED_BASE=3605001
export PUCT_CONTAINER_PREFIX=sap-v36-puct
export PUCT_REVIEW_OUTPUT=/workspace/runs/search-expert-review-v36-large-puct-v10.html
exec "${project_root}/scripts/run_v35_large_puct.sh"
