#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
export SEARCH_PARENT=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
export SEARCH_PERMANENT=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
export SEARCH_STEM=v36-large-stratified-puct-v10
export SEARCH_RARE_PREFIX=v36-rare-puct
export SEARCH_CRITIC_REPORT_HOST="${project_root}/runs/search/v36-selected-critic-calibration.json"
export SEARCH_RUN=/workspace/runs/turtle-league-v37-corrected-puct-integration-from-v36-selected
export SEARCH_CURRICULUM=/workspace/runs/curriculum/turtle-p1-scenarios-v7-20260904.jsonl
export SEARCH_CONTAINER_PREFIX=sap-v37
export SEARCH_SEED_BASE=3706001
export SEARCH_PARENT_EVAL_TAG=v36
export SEARCH_PERMANENT_EVAL_TAG=v25
export SEARCH_ALLOW_OPPONENT_REVISION_MISMATCH=1
exec "${project_root}/scripts/run_v35_search_integration.sh"
