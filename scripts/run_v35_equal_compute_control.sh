#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image="${CONTROL_IMAGE:-sap-rl-v28-puct:20260903h}"
parent="${CONTROL_PARENT:-/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt}"
permanent="${CONTROL_PERMANENT:-/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt}"
search_run="${CONTROL_SEARCH_RUN:-/workspace/runs/turtle-league-v35-large-puct-integration-from-v31-1000}"
control="${CONTROL_RUN:-/workspace/runs/turtle-league-v35-pure-selfplay-control-from-v31-1000}"
control_host="${runs_host}${control#/workspace/runs}"
search_run_host="${runs_host}${search_run#/workspace/runs}"
container_prefix="${CONTROL_CONTAINER_PREFIX:-sap-v35}"
seed_base="${CONTROL_SEED_BASE:-2854001}"
parent_eval_tag="${CONTROL_PARENT_EVAL_TAG:-v31}"
permanent_eval_tag="${CONTROL_PERMANENT_EVAL_TAG:-v25}"
allow_opponent_revision_mismatch="${CONTROL_ALLOW_OPPONENT_REVISION_MISMATCH:-0}"

mkdir -p "${control_host}/evaluations"

common_training_args=(
  --distributed
  --num-envs 32
  --games-per-actor 3
  --segments-per-update 64
  --transitions-per-update 8192
  --trajectory-queue-size 1024
  --inference-batch-delay-ms 5
  --max-behavior-policy-lag 1
  --checkpoint-dir "${control}"
  --discount-mode per_turn
  --discount 0.99
  --battle-aux-cost 0.05
  --opponent-aux-cost 0.1
  --opponent-module-learning-rate 1e-4
  --opponent-adapter-learning-rate 1e-5
  --resident-checkpoint "${permanent}"
  --resident-checkpoint /workspace/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt
  --resident-checkpoint /workspace/runs/turtle-league-v21-live-sync-transfer-v19-600-to5000/update-010800.pt
  --allow-simulator-mismatch
  --max-historical-residents 4
)

latest_checkpoint_at_most() {
  local target=$1
  local fallback=$2
  local latest="${fallback}"
  local filename number
  while IFS= read -r filename; do
    number="${filename#update-}"
    number="${number%.pt}"
    if (( 10#${number} <= target )); then
      latest="${control}/${filename}"
    fi
  done < <(
    find "${control_host}" -maxdepth 1 -type f -name 'update-*.pt' \
      -printf '%f\n' | sort -V
  )
  echo "${latest}"
}

train_to() {
  local target=$1
  local fallback=$2
  local seed=$3
  local anchor_mode=${4:-unanchored}
  local checkpoint_name
  checkpoint_name="$(printf 'update-%06d.pt' "${target}")"
  if [[ -f "${control_host}/${checkpoint_name}" ]]; then
    return
  fi
  local resume
  resume="$(latest_checkpoint_at_most $((target - 1)) "${fallback}")"
  local init_args
  if [[ "${resume}" == "${parent}" ]]; then
    init_args=(--init-checkpoint "${parent}" --learning-rate 1e-6)
  else
    init_args=(--resume-checkpoint "${resume}" --resume-learning-rate 1e-6)
  fi
  local anchor_args=(--disable-expert-anchor)
  if [[ "${anchor_mode}" == anchored ]]; then
    anchor_args=(--expert-anchor-checkpoint "${parent}")
  fi
  docker run --rm --name "${container_prefix}-pure-control" \
    --gpus all --ipc host \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/train_entity_selfplay.py \
      --updates "${target}" "${init_args[@]}" --checkpoint-every 200 \
      "${common_training_args[@]}" "${anchor_args[@]}" --seed "${seed}"
}

paired_eval() {
  local checkpoint=$1 opponent=$2 output=$3 seed=$4
  if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then
    return
  fi
  docker run --rm --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/evaluate_paired_policy.py \
      "${checkpoint}" --opponent checkpoint \
      --opponent-checkpoint "${opponent}" --device cuda --batch-size 64 \
      --pairs 200 --progress-every 100 --seed "${seed}" \
      --output "${output}"
}

# Same number of ordinary RL updates as the search-guided branch. This isolates
# the effect of the search warm start while holding subsequent RL compute fixed.
# Match the search branch's bounded parent-retention schedule: retain the
# initialization policy through update 1,000, then remove it from the resident
# pool for unrestricted league learning.
train_to 1000 "${parent}" "${seed_base}" anchored
control_1000="${control}/update-001000.pt"
train_to 5000 "${control_1000}" "$((seed_base + 1))" unanchored
control_5000="${control}/update-005000.pt"
paired_eval "${control_5000}" "${parent}" \
  "${control}/evaluations/paired-update5000-vs-${parent_eval_tag}.json" "$((seed_base + 2))"
paired_eval "${control_5000}" "${permanent}" \
  "${control}/evaluations/paired-update5000-vs-${permanent_eval_tag}.json" "$((seed_base + 3))"
if [[ -f "${search_run_host}/update-005000.pt" ]]; then
  paired_eval "${control_5000}" "${search_run}/update-005000.pt" \
    "${control}/evaluations/paired-control5000-vs-search5000.json" "$((seed_base + 4))"
fi

# Approximately match the extra GPU time spent generating/calibrating search.
# Both the 5k and 10k control checkpoints are retained, allowing the final
# report to show same-RL-update and approximate equal-total-compute results.
train_to 10000 "${control_5000}" "$((seed_base + 5))" unanchored
control_10000="${control}/update-010000.pt"
paired_eval "${control_10000}" "${parent}" \
  "${control}/evaluations/paired-update10000-vs-${parent_eval_tag}.json" "$((seed_base + 6))"
paired_eval "${control_10000}" "${permanent}" \
  "${control}/evaluations/paired-update10000-vs-${permanent_eval_tag}.json" "$((seed_base + 7))"
if [[ -f "${search_run_host}/update-005000.pt" ]]; then
  paired_eval "${control_10000}" "${search_run}/update-005000.pt" \
    "${control}/evaluations/paired-control10000-vs-search5000.json" "$((seed_base + 8))"
fi

promotion_revision_args=()
if [[ "${allow_opponent_revision_mismatch}" == 1 ]]; then
  promotion_revision_args+=(--allow-opponent-training-revision-mismatch)
fi
if ! python3 - "${control_host}/evaluations/paired-update10000-vs-${parent_eval_tag}.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
raise SystemExit(
    0
    if result['pairedMeanScore'] >= 0.50
    and result['pairedScore95CI'][1] > 0.50
    else 1
)
PY
then
  echo "pure-self-play control did not preserve parent strength; not promoted"
elif docker run --rm --network none \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/promote_league_checkpoint.py \
      "${control}/evaluations/paired-update10000-vs-${permanent_eval_tag}.json" \
      --league "${control}/league.json" \
      --minimum-score 0.65 --minimum-pairs 100 --minimum-ci-low 0.50 \
      "${promotion_revision_args[@]}"; then
  echo "pure-self-play control update 10000 promoted over permanent champion"
else
  echo "pure-self-play control update 10000 did not clear promotion gate"
fi
