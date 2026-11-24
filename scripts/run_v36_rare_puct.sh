#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
export RARE_PUCT_CHECKPOINT=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
export RARE_PUCT_RESERVOIR=/workspace/runs/search/v36-selected-opponent-beliefs.json
export RARE_PUCT_CRITIC_HOST="${project_root}/runs/search/v36-selected-critic-calibration.json"
export RARE_PUCT_MANIFEST=/workspace/runs/search/v36-rare-puct-manifest.json
export RARE_PUCT_PREFIX=/workspace/runs/search/v36-rare-puct
export RARE_PUCT_NUM_SHARDS=84
export RARE_PUCT_MANIFEST_SEED=3605800
export RARE_PUCT_SEARCH_SEED=3605900
export RARE_PUCT_CONTAINER_PREFIX=sap-v36-rare
export RARE_PUCT_REVIEW_OUTPUT=/workspace/runs/search-expert-review-v36-rare-puct.html
exec "${project_root}/scripts/run_v35_rare_puct.sh"
