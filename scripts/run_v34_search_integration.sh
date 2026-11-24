#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
parent=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
run=/workspace/runs/turtle-league-v34-stratified-puct-integration-from-v31-1000
run_host="${runs_host}/turtle-league-v34-stratified-puct-integration-from-v31-1000"
strict=/workspace/runs/search/v34-combined-stratified-puct-v9-strict-root-only.jsonl
strict_holdout=/workspace/runs/search/v34-combined-stratified-puct-v9-strict-root-only-holdout.jsonl
exploration_host="${runs_host}/search/v34-combined-stratified-puct-v9.exploration.json"
anchor=/workspace/runs/search/v34-parent-anchor.jsonl
anchor_holdout=/workspace/runs/search/v34-parent-anchor-holdout.jsonl
weak_anchor=/workspace/runs/search/v34-parent-anchor-weak.jsonl

while [[ ! -f "${exploration_host}" ]]; do
  sleep 30
done
if ! python3 -c '
import json, sys
report=json.load(open(sys.argv[1]))
raise SystemExit(0 if report["readiness"]["readyForDistillation"] else 1)
' "${exploration_host}"; then
  echo "v34 combined search corpus did not clear readiness; integration not started"
  exit 0
fi
while [[ ! -f "${runs_host}${strict#/workspace/runs}" \
      || ! -f "${runs_host}${strict_holdout#/workspace/runs}" ]]; do
  sleep 10
done

generate_anchor() {
  local output=$1
  local summary=$2
  local games=$3
  local sequences=$4
  local seed=$5
  if [[ -f "${runs_host}${summary#/workspace/runs}" ]]; then
    return
  fi
  docker run --rm --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/generate_policy_anchor_demonstrations.py \
      "${parent}" --output "${output}" --summary-output "${summary}" \
      --games "${games}" --max-sequences "${sequences}" \
      --teacher-weight 0.25 --device cuda --seed "${seed}"
}

generate_anchor "${anchor}" \
  /workspace/runs/search/v34-parent-anchor.summary.json 16 256 2851001
generate_anchor "${anchor_holdout}" \
  /workspace/runs/search/v34-parent-anchor-holdout.summary.json 8 128 2851002

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
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/evaluate_curriculum.py \
      "${checkpoint}" "${data}" --threads 4 --batch-size 64 \
      --output "${output}"
}

evaluate_data "${parent}" "${strict}" "${run}/evaluations/strict-before.json"
evaluate_data "${parent}" "${strict_holdout}" "${run}/evaluations/strict-holdout-before.json"
evaluate_data "${parent}" "${anchor_holdout}" "${run}/evaluations/anchor-holdout-before.json"

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
  --expert-anchor-checkpoint "${parent}"
  --resident-checkpoint /workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
  --resident-checkpoint /workspace/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt
  --resident-checkpoint /workspace/runs/turtle-league-v21-live-sync-transfer-v19-600-to5000/update-010800.pt
  --allow-simulator-mismatch
  --max-historical-residents 4
)

if [[ ! -f "${run_host}/update-000150.pt" ]]; then
  phase_a_init=(--init-checkpoint "${parent}" --learning-rate 1e-6)
  mapfile -t phase_a_checkpoints < <(
    find "${run_host}" -maxdepth 1 -type f \
      -name 'update-*.pt' -printf '%f\n' 2>/dev/null | sort -V
  )
  if (( ${#phase_a_checkpoints[@]} )); then
    latest_checkpoint="${phase_a_checkpoints[-1]}"
    phase_a_init=(
      --resume-checkpoint "${run}/${latest_checkpoint}"
      --resume-learning-rate 1e-6
    )
  fi
  docker run --rm --name sap-rl-v34-integration \
    --gpus all --ipc host \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/train_entity_selfplay.py \
      --updates 150 "${phase_a_init[@]}" --checkpoint-every 50 \
      "${common_training_args[@]}" \
      --demo-data "${strict}" "${anchor}" \
      --demo-batch-size 64 --demo-until-update 150 --demo-every 2 \
      --demo-learning-rate-scale 0.10 --seed 2852001
fi

integration="${run}/update-000150.pt"
evaluate_data "${integration}" "${strict}" "${run}/evaluations/strict-update150.json"
evaluate_data "${integration}" "${strict_holdout}" "${run}/evaluations/strict-holdout-update150.json"
evaluate_data "${integration}" "${anchor_holdout}" "${run}/evaluations/anchor-holdout-update150.json"
evaluate_data "${integration}" \
  /workspace/runs/curriculum/turtle-p1-scenarios-v6-20260902.jsonl \
  "${run}/evaluations/habits-update150.json"

integration_paired="${run}/evaluations/paired-update150-vs-parent.json"
if [[ ! -f "${runs_host}${integration_paired#/workspace/runs}" ]]; then
  docker run --rm --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/evaluate_paired_policy.py \
      "${integration}" --opponent checkpoint \
      --opponent-checkpoint "${parent}" --device cuda --batch-size 64 \
      --pairs 100 --progress-every 100 --seed 2852002 \
      --output "${integration_paired}"
fi

if ! python3 -c '
import json, sys
strict_before=json.load(open(sys.argv[1]))["overall"]
strict_after=json.load(open(sys.argv[2]))["overall"]
holdout_before=json.load(open(sys.argv[3]))["overall"]
holdout_after=json.load(open(sys.argv[4]))["overall"]
anchor_before=json.load(open(sys.argv[5]))["overall"]
anchor_after=json.load(open(sys.argv[6]))["overall"]
paired=json.load(open(sys.argv[8]))
fail=[]
if strict_after["weightedCrossEntropy"] >= strict_before["weightedCrossEntropy"]:
    fail.append("strict target cross-entropy did not improve")
if holdout_after["weightedCrossEntropy"] >= holdout_before["weightedCrossEntropy"]:
    fail.append("strict holdout did not improve")
if anchor_after["weightedCrossEntropy"] > anchor_before["weightedCrossEntropy"] + .25:
    fail.append("parent retention regressed")
# Expert-authored behavior probes are diagnostic only.  A stronger policy is
# allowed to violate them; paired strength and simulator correctness are the
# integration gates.
if paired["pairedMeanScore"] < .45 or paired["pairedScore95CI"][1] <= .50:
    fail.append("paired integration safety gate failed")
if fail:
    raise SystemExit("; ".join(fail))
' \
  "${run_host}/evaluations/strict-before.json" \
  "${run_host}/evaluations/strict-update150.json" \
  "${run_host}/evaluations/strict-holdout-before.json" \
  "${run_host}/evaluations/strict-holdout-update150.json" \
  "${run_host}/evaluations/anchor-holdout-before.json" \
  "${run_host}/evaluations/anchor-holdout-update150.json" \
  "${run_host}/evaluations/habits-update150.json" \
  "${run_host}/evaluations/paired-update150-vs-parent.json"; then
  echo "v34 integration gate failed; unrestricted phase not started"
  exit 0
fi

if [[ ! -f "${run_host}/update-001000.pt" ]]; then
  phase_b_resume="${integration}"
  mapfile -t phase_b_checkpoints < <(
    find "${run_host}" -maxdepth 1 -type f \
      -name 'update-*.pt' -printf '%f\n' | sort -V
  )
  if (( ${#phase_b_checkpoints[@]} )); then
    latest_checkpoint="${phase_b_checkpoints[-1]}"
    phase_b_resume="${run}/${latest_checkpoint}"
  fi
  docker run --rm --name sap-rl-v34-unrestricted \
    --gpus all --ipc host \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/train_entity_selfplay.py \
      --updates 1000 --resume-checkpoint "${phase_b_resume}" \
      --resume-learning-rate 1e-6 --checkpoint-every 100 \
      "${common_training_args[@]}" \
      --demo-data "${strict}" "${weak_anchor}" --demo-batch-size 64 \
      --demo-until-update 1000 --demo-every 16 \
      --demo-learning-rate-scale 0.10 --seed 2852003
fi

candidate="${run}/update-001000.pt"
evaluate_data "${candidate}" "${strict}" "${run}/evaluations/strict-update1000.json"
evaluate_data "${candidate}" "${strict_holdout}" "${run}/evaluations/strict-holdout-update1000.json"
evaluate_data "${candidate}" "${anchor_holdout}" "${run}/evaluations/anchor-holdout-update1000.json"
evaluate_data "${candidate}" \
  /workspace/runs/curriculum/turtle-p1-scenarios-v6-20260902.jsonl \
  "${run}/evaluations/habits-update1000.json"

paired="${run}/evaluations/paired-update1000-vs-parent.json"
if [[ ! -f "${runs_host}${paired#/workspace/runs}" ]]; then
  docker run --rm --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/evaluate_paired_policy.py \
      "${candidate}" --opponent checkpoint \
      --opponent-checkpoint "${parent}" --device cuda --batch-size 64 \
      --pairs 200 --progress-every 100 --seed 2852004 \
      --output "${paired}"
fi

if docker run --rm --network none \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/promote_league_checkpoint.py \
      "${paired}" --league "${run}/league.json" \
      --minimum-score 0.65 --minimum-pairs 100 --minimum-ci-low 0.50; then
  echo "v34 update 1000 promoted to permanent champion"
else
  echo "v34 update 1000 did not clear the permanent paired-strength gate"
fi
