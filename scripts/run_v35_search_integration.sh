#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image="${SEARCH_IMAGE:-sap-rl-v28-puct:20260903h}"
parent="${SEARCH_PARENT:-/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt}"
permanent="${SEARCH_PERMANENT:-/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt}"
stem="${SEARCH_STEM:-v35-large-stratified-puct-v9}"
strict=/workspace/runs/search/${stem}-strict-root-only.jsonl
strict_holdout=/workspace/runs/search/${stem}-strict-root-only-holdout.jsonl
rare_prefix="${SEARCH_RARE_PREFIX:-v35-rare-puct}"
rare_strict=/workspace/runs/search/${rare_prefix}-strict-root-only.jsonl
rare_strict_holdout=/workspace/runs/search/${rare_prefix}-strict-root-only-holdout.jsonl
exploration_host="${runs_host}/search/${stem}.exploration.json"
critic_report_host="${SEARCH_CRITIC_REPORT_HOST:-${runs_host}/search/v35-v31-critic-calibration.json}"
run="${SEARCH_RUN:-/workspace/runs/turtle-league-v35-large-puct-integration-from-v31-1000}"
run_host="${runs_host}${run#/workspace/runs}"
curriculum="${SEARCH_CURRICULUM:-/workspace/runs/curriculum/turtle-p1-scenarios-v6-20260902.jsonl}"
container_prefix="${SEARCH_CONTAINER_PREFIX:-sap-v35}"
seed_base="${SEARCH_SEED_BASE:-2853601}"
parent_eval_tag="${SEARCH_PARENT_EVAL_TAG:-v31}"
permanent_eval_tag="${SEARCH_PERMANENT_EVAL_TAG:-v25}"
allow_opponent_revision_mismatch="${SEARCH_ALLOW_OPPONENT_REVISION_MISMATCH:-0}"

while [[ ! -f "${exploration_host}" ]]; do
  sleep 30
done
while [[ ! -f "${critic_report_host}" ]]; do
  sleep 30
done
python3 - "${critic_report_host}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
overall = report["overall"]
print(
    "parent_critic_calibration "
    f"states={overall['states']} correlation={overall['correlation']:.4f} "
    f"mae={overall['mae']:.4f} bias={overall['bias']:.4f} "
    f"recommended_leaf_weight={report['recommendedLeafValueWeight']}"
)
PY
if ! python3 -c '
import json, sys
report=json.load(open(sys.argv[1]))
raise SystemExit(0 if report["readiness"]["readyForDistillation"] else 1)
' "${exploration_host}"; then
  echo "search corpus did not clear coverage/readiness; integration not started"
  exit 0
fi
while [[ ! -f "${runs_host}${strict#/workspace/runs}" \
      || ! -f "${runs_host}${strict_holdout#/workspace/runs}" \
      || ! -f "${runs_host}${rare_strict#/workspace/runs}" \
      || ! -f "${runs_host}${rare_strict_holdout#/workspace/runs}" ]]; do
  sleep 10
done

mkdir -p "${run_host}/evaluations"

evaluate_data() {
  local checkpoint=$1
  local data=$2
  local output=$3
  if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then
    return
  fi
  docker run --rm --cpus 4 --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/evaluate_curriculum.py \
      "${checkpoint}" "${data}" --threads 4 --batch-size 64 \
      --output "${output}"
}

paired_eval() {
  local checkpoint=$1
  local opponent=$2
  local output=$3
  local seed=$4
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

evaluate_data "${parent}" "${strict}" "${run}/evaluations/strict-before.json"
evaluate_data "${parent}" "${strict_holdout}" \
  "${run}/evaluations/strict-holdout-before.json"
evaluate_data "${parent}" "${rare_strict}" \
  "${run}/evaluations/rare-strict-before.json"
evaluate_data "${parent}" "${rare_strict_holdout}" \
  "${run}/evaluations/rare-strict-holdout-before.json"

# Give every confirmed search sequence approximately three supervised
# exposures during integration, independent of corpus size.  v34 replayed 13
# roots far too often; this schedule cannot silently turn a small corpus into
# thousands of nearly identical optimizer examples.
general_demo_sequences="$(grep -cve '^[[:space:]]*$' "${runs_host}${strict#/workspace/runs}")"
rare_demo_sequences="$(grep -cve '^[[:space:]]*$' "${runs_host}${rare_strict#/workspace/runs}")"
demo_sequences=$((general_demo_sequences + rare_demo_sequences))
demo_batch_size=64
demo_steps=$(( (3 * demo_sequences + demo_batch_size - 1) / demo_batch_size ))
(( demo_steps > 0 )) || { echo "empty strict corpus" >&2; exit 1; }
demo_every=$(( 300 / demo_steps ))
(( demo_every >= 1 )) || demo_every=1
echo "general_demo_sequences=${general_demo_sequences} rare_demo_sequences=${rare_demo_sequences} demo_sequences=${demo_sequences} target_exposures=3 demo_steps=${demo_steps} demo_every=${demo_every}"

common_training_args=(
  --distributed
  --num-envs 32
  --games-per-actor 3
  --segments-per-update 64
  --transitions-per-update 8192
  --trajectory-queue-size 1024
  --inference-batch-delay-ms 5
  --max-behavior-policy-lag 1
  --checkpoint-dir "${run}"
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

if [[ ! -f "${run_host}/update-000300.pt" ]]; then
  docker run --rm --name "${container_prefix}-integration" \
    --gpus all --ipc host \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/train_entity_selfplay.py \
      --updates 300 --init-checkpoint "${parent}" \
      --learning-rate 1e-6 --checkpoint-every 100 \
      "${common_training_args[@]}" \
      --demo-data "${strict}" "${rare_strict}" \
      --demo-batch-size "${demo_batch_size}" \
      --demo-until-update 300 --demo-every "${demo_every}" \
      --demo-learning-rate-scale 0.05 \
      --expert-anchor-checkpoint "${parent}" \
      --seed "${seed_base}"
fi

phase_a="${run}/update-000300.pt"
evaluate_data "${phase_a}" "${strict}" "${run}/evaluations/strict-update300.json"
evaluate_data "${phase_a}" "${strict_holdout}" \
  "${run}/evaluations/strict-holdout-update300.json"
evaluate_data "${phase_a}" "${rare_strict}" \
  "${run}/evaluations/rare-strict-update300.json"
evaluate_data "${phase_a}" "${rare_strict_holdout}" \
  "${run}/evaluations/rare-strict-holdout-update300.json"
evaluate_data "${phase_a}" \
  "${curriculum}" \
  "${run}/evaluations/habits-update300.json"
paired_eval "${phase_a}" "${parent}" \
  "${run}/evaluations/paired-update300-vs-${parent_eval_tag}.json" "$((seed_base + 1))"

# Human-authored habits and search-target cross entropy are diagnostics.  Only
# direct paired strength controls whether the unrestricted recovery phase is
# worth running.
if ! python3 -c '
import json, sys
p=json.load(open(sys.argv[1]))
raise SystemExit(0 if p["pairedMeanScore"] >= .45 and p["pairedScore95CI"][1] > .50 else 1)
' "${run_host}/evaluations/paired-update300-vs-${parent_eval_tag}.json"; then
  echo "search integration is clearly weaker than its parent; recovery phase not started"
  exit 0
fi

# Stop replaying search data.  This phase tests whether ordinary diverse
# league self-play retains useful distilled behavior and converts it into
# robust strength rather than memorizing the finite corpus.
latest_checkpoint_at_most() {
  local target=$1
  local fallback=$2
  local latest="${fallback}"
  local filename number
  while IFS= read -r filename; do
    number="${filename#update-}"
    number="${number%.pt}"
    if (( 10#${number} <= target )); then
      latest="${run}/${filename}"
    fi
  done < <(
    find "${run_host}" -maxdepth 1 -type f -name 'update-*.pt' \
      -printf '%f\n' | sort -V
  )
  echo "${latest}"
}

if [[ ! -f "${run_host}/update-001000.pt" ]]; then
  phase_b_resume="$(latest_checkpoint_at_most 999 "${phase_a}")"
  docker run --rm --name "${container_prefix}-unrestricted" \
    --gpus all --ipc host \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/train_entity_selfplay.py \
      --updates 1000 --resume-checkpoint "${phase_b_resume}" \
      --resume-learning-rate 1e-6 --checkpoint-every 100 \
      "${common_training_args[@]}" \
      --expert-anchor-checkpoint "${parent}" \
      --seed "$((seed_base + 2))"
fi

candidate_1000="${run}/update-001000.pt"
evaluate_data "${candidate_1000}" "${strict}" "${run}/evaluations/strict-update1000.json"
evaluate_data "${candidate_1000}" "${strict_holdout}" \
  "${run}/evaluations/strict-holdout-update1000.json"
evaluate_data "${candidate_1000}" "${rare_strict}" \
  "${run}/evaluations/rare-strict-update1000.json"
evaluate_data "${candidate_1000}" "${rare_strict_holdout}" \
  "${run}/evaluations/rare-strict-holdout-update1000.json"
evaluate_data "${candidate_1000}" \
  "${curriculum}" \
  "${run}/evaluations/habits-update1000.json"
paired_eval "${candidate_1000}" "${parent}" \
  "${run}/evaluations/paired-update1000-vs-${parent_eval_tag}.json" "$((seed_base + 3))"

if ! python3 -c '
import json, sys
p=json.load(open(sys.argv[1]))
raise SystemExit(0 if p["pairedMeanScore"] >= .45 and p["pairedScore95CI"][1] > .50 else 1)
' "${run_host}/evaluations/paired-update1000-vs-${parent_eval_tag}.json"; then
  echo "search update 1000 is clearly weaker than its parent; long self-play phase not started"
  exit 0
fi

# Ordinary diverse-league self-play is the primary learner. Search replay
# ended at update 300; updates 301--5000 are unrestricted. A midpoint strength
# check prevents spending the full matched-compute budget on a collapsed run.
if [[ ! -f "${run_host}/update-003000.pt" ]]; then
  phase_c_resume="$(latest_checkpoint_at_most 2999 "${candidate_1000}")"
  docker run --rm --name "${container_prefix}-long-selfplay-a" \
    --gpus all --ipc host \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/train_entity_selfplay.py \
      --updates 3000 --resume-checkpoint "${phase_c_resume}" \
      --resume-learning-rate 1e-6 --checkpoint-every 200 \
      "${common_training_args[@]}" --disable-expert-anchor \
      --seed "$((seed_base + 4))"
fi

candidate_3000="${run}/update-003000.pt"
paired_eval "${candidate_3000}" "${parent}" \
  "${run}/evaluations/paired-update3000-vs-${parent_eval_tag}.json" "$((seed_base + 5))"
if ! python3 -c '
import json, sys
p=json.load(open(sys.argv[1]))
raise SystemExit(0 if p["pairedMeanScore"] >= .45 and p["pairedScore95CI"][1] > .50 else 1)
' "${run_host}/evaluations/paired-update3000-vs-${parent_eval_tag}.json"; then
  echo "search update 3000 is clearly weaker than its parent; final self-play phase not started"
  exit 0
fi

if [[ ! -f "${run_host}/update-005000.pt" ]]; then
  phase_d_resume="$(latest_checkpoint_at_most 4999 "${candidate_3000}")"
  docker run --rm --name "${container_prefix}-long-selfplay-b" \
    --gpus all --ipc host \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/train_entity_selfplay.py \
      --updates 5000 --resume-checkpoint "${phase_d_resume}" \
      --resume-learning-rate 1e-6 --checkpoint-every 200 \
      "${common_training_args[@]}" --disable-expert-anchor \
      --seed "$((seed_base + 6))"
fi

candidate="${run}/update-005000.pt"
evaluate_data "${candidate}" "${strict}" "${run}/evaluations/strict-update5000.json"
evaluate_data "${candidate}" "${strict_holdout}" \
  "${run}/evaluations/strict-holdout-update5000.json"
evaluate_data "${candidate}" "${rare_strict}" \
  "${run}/evaluations/rare-strict-update5000.json"
evaluate_data "${candidate}" "${rare_strict_holdout}" \
  "${run}/evaluations/rare-strict-holdout-update5000.json"
evaluate_data "${candidate}" \
  "${curriculum}" \
  "${run}/evaluations/habits-update5000.json"
paired_eval "${candidate}" "${parent}" \
  "${run}/evaluations/paired-update5000-vs-${parent_eval_tag}.json" "$((seed_base + 7))"
paired_eval "${candidate}" "${permanent}" \
  "${run}/evaluations/paired-update5000-vs-${permanent_eval_tag}.json" "$((seed_base + 8))"

promotion_revision_args=()
if [[ "${allow_opponent_revision_mismatch}" == 1 ]]; then
  promotion_revision_args+=(--allow-opponent-training-revision-mismatch)
fi
if ! python3 - "${run_host}/evaluations/paired-update5000-vs-${parent_eval_tag}.json" <<'PY'
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
  echo "search update 5000 did not preserve parent strength; not promoted"
elif docker run --rm --network none \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/promote_league_checkpoint.py \
      "${run}/evaluations/paired-update5000-vs-${permanent_eval_tag}.json" \
      --league "${run}/league.json" \
      --minimum-score 0.65 --minimum-pairs 100 --minimum-ci-low 0.50 \
      "${promotion_revision_args[@]}"; then
  echo "search update 5000 promoted over the permanent champion"
else
  echo "search update 5000 did not clear the 65% permanent promotion gate"
fi
