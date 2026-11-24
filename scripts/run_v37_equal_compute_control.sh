#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
export CONTROL_PARENT=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
export CONTROL_PERMANENT=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
export CONTROL_SEARCH_RUN=/workspace/runs/turtle-league-v37-corrected-puct-integration-from-v36-selected
export CONTROL_RUN=/workspace/runs/turtle-league-v37-pure-selfplay-control-from-v36-selected
export CONTROL_CONTAINER_PREFIX=sap-v37
export CONTROL_SEED_BASE=3707001
export CONTROL_PARENT_EVAL_TAG=v36
export CONTROL_PERMANENT_EVAL_TAG=v25
export CONTROL_ALLOW_OPPONENT_REVISION_MISMATCH=1
exec "${project_root}/scripts/run_v35_equal_compute_control.sh"
