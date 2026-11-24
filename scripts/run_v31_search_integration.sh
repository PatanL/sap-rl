#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
parent=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
run=/workspace/runs/turtle-league-v31-search-integration-from-v25-300
run_host="${runs_host}/turtle-league-v31-search-integration-from-v25-300"
strict=/workspace/runs/search/v31-strict-puct.jsonl
strict_holdout=/workspace/runs/search/v31-strict-puct-holdout.jsonl
candidates=/workspace/runs/search/v29-validated-candidates.jsonl
candidate_holdout=/workspace/runs/search/v29-validated-candidates-holdout.jsonl
anchor=/workspace/runs/search/v29-parent-anchor.jsonl
anchor_holdout=/workspace/runs/search/v29-parent-anchor-holdout.jsonl

while [[ ! -f "${runs_host}/search/v30-multifidelity-puct-pilot.summary.json" \
      || ! -f "${runs_host}/search/v32-multifidelity-puct-v9.summary.json" \
      || ! -f "${runs_host}/search/v29-validated-candidates.summary.json" \
      || ! -f "${runs_host}/search/v29-parent-anchor.summary.json" \
      || ! -f "${runs_host}/search/v29-parent-anchor-holdout.summary.json" ]]; do
  sleep 20
done

python3 -c '
import json, sys
report=json.load(open(sys.argv[1]))
outer=report["outerAudit"]
if report["accepted"] < 5 or outer["lowerBound95"] is None or outer["lowerBound95"] <= 0:
    raise SystemExit("strict candidate corpus failed its aggregate outer gate")
' "${runs_host}/search/v29-validated-candidates.summary.json"

if [[ ! -f "${runs_host}/search/v31-strict-puct.summary.json" ]]; then
  docker run --rm --network none \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/filter_puct_demonstrations.py \
      --input \
        /workspace/runs/search/v28-puct-teacher-combined.jsonl \
        /workspace/runs/search/v29-conditional-puct.jsonl \
        /workspace/runs/search/v29-puct-v7-freeze-check.jsonl \
        /workspace/runs/search/v30-multifidelity-puct-pilot.jsonl \
        /workspace/runs/search/v32-multifidelity-puct-v9.jsonl \
      --output "${strict}" \
      --summary-output /workspace/runs/search/v31-strict-puct.summary.json
  docker run --rm --network none \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/filter_puct_demonstrations.py \
      --input \
        /workspace/runs/search/v28-puct-teacher-combined-holdout.jsonl \
        /workspace/runs/search/v29-conditional-puct-holdout.jsonl \
        /workspace/runs/search/v30-multifidelity-puct-pilot-holdout.jsonl \
        /workspace/runs/search/v32-multifidelity-puct-v9-holdout.jsonl \
      --output "${strict_holdout}" \
      --summary-output /workspace/runs/search/v31-strict-puct-holdout.summary.json
fi

mkdir -p "${run_host}/evaluations"
evaluate_data() {
  local checkpoint=$1
  local data=$2
  local output=$3
  if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then
    return
  fi
  docker run --rm --network none \
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
evaluate_data "${parent}" "${candidates}" "${run}/evaluations/candidates-before.json"
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
  --resident-checkpoint /workspace/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt
  --resident-checkpoint /workspace/runs/turtle-league-v21-live-sync-transfer-v19-600-to5000/update-006500.pt
  --resident-checkpoint /workspace/runs/turtle-league-v21-live-sync-transfer-v19-600-to5000/update-010800.pt
  --allow-simulator-mismatch
  --max-historical-residents 4
)

if [[ ! -f "${run_host}/update-000150.pt" ]]; then
  docker run --rm --name sap-rl-v31-integration \
    --gpus all --ipc host \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/train_entity_selfplay.py \
      --updates 150 \
      --init-checkpoint "${parent}" \
      --learning-rate 1e-6 \
      --checkpoint-every 50 \
      "${common_training_args[@]}" \
      --demo-data "${strict}" "${candidates}" "${anchor}" \
      --demo-batch-size 64 \
      --demo-until-update 150 \
      --demo-every 2 \
      --demo-learning-rate-scale 0.10 \
      --seed 2810001
fi

integration="${run}/update-000150.pt"
evaluate_data "${integration}" "${strict}" "${run}/evaluations/strict-update150.json"
evaluate_data "${integration}" "${strict_holdout}" "${run}/evaluations/strict-holdout-update150.json"
evaluate_data "${integration}" "${candidates}" "${run}/evaluations/candidates-update150.json"
evaluate_data "${integration}" "${candidate_holdout}" "${run}/evaluations/candidates-holdout-update150.json"
evaluate_data "${integration}" "${anchor_holdout}" "${run}/evaluations/anchor-holdout-update150.json"
evaluate_data "${integration}" /workspace/runs/curriculum/turtle-p1-scenarios-v6-20260902.jsonl "${run}/evaluations/habits-update150.json"

integration_paired="${run}/evaluations/paired-update150-vs-parent.json"
if [[ ! -f "${runs_host}${integration_paired#/workspace/runs}" ]]; then
  docker run --rm --gpus all --ipc host --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
    "${image}" \
    python scripts/evaluate_paired_policy.py \
      "${integration}" \
      --opponent checkpoint --opponent-checkpoint "${parent}" \
      --device cuda --batch-size 64 --pairs 100 \
      --progress-every 100 --seed 2810002 \
      --output "${integration_paired}"
fi

if ! python3 -c '
import json, sys
before=json.load(open(sys.argv[1]))["overall"]
after=json.load(open(sys.argv[2]))["overall"]
habits=json.load(open(sys.argv[3]))["overall"]
paired=json.load(open(sys.argv[4]))
anchor_before=json.load(open(sys.argv[5]))["overall"]
anchor_after=json.load(open(sys.argv[6]))["overall"]
fail=[]
if after["meanTargetProbability"] <= before["meanTargetProbability"]:
    fail.append("strict targets did not improve")
if habits["fullActionAccuracy"] < .85:
    fail.append("habit retention fell below 85%")
if paired["pairedMeanScore"] < .45 or paired["pairedScore95CI"][1] <= .50:
    fail.append("paired integration safety gate failed")
if anchor_after["weightedCrossEntropy"] > anchor_before["weightedCrossEntropy"] + .25:
    fail.append("champion-policy holdout retention regressed")
if fail:
    raise SystemExit("; ".join(fail))
' \
  "${run_host}/evaluations/strict-before.json" \
  "${run_host}/evaluations/strict-update150.json" \
  "${run_host}/evaluations/habits-update150.json" \
  "${run_host}/evaluations/paired-update150-vs-parent.json" \
  "${run_host}/evaluations/anchor-holdout-before.json" \
  "${run_host}/evaluations/anchor-holdout-update150.json"; then
  echo "v31 integration gate failed; unrestricted phase not started"
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
  echo "v31_phase_b_resume=${phase_b_resume}"
  docker run --rm --name sap-rl-v31-unrestricted \
    --gpus all --ipc host \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym \
      "${image}" \
    python scripts/train_entity_selfplay.py \
      --updates 1000 \
      --resume-checkpoint "${phase_b_resume}" \
      --resume-learning-rate 2e-6 \
      --checkpoint-every 100 \
      "${common_training_args[@]}" \
      --demo-data "${strict}" "${candidates}" \
      --demo-batch-size 64 \
      --demo-until-update 1000 \
      --demo-every 16 \
      --demo-learning-rate-scale 0.10 \
      --seed 2810003
fi

candidate="${run}/update-001000.pt"
evaluate_data "${candidate}" "${strict}" "${run}/evaluations/strict-update1000.json"
evaluate_data "${candidate}" "${strict_holdout}" "${run}/evaluations/strict-holdout-update1000.json"
evaluate_data "${candidate}" "${anchor_holdout}" "${run}/evaluations/anchor-holdout-update1000.json"
evaluate_data "${candidate}" /workspace/runs/curriculum/turtle-p1-scenarios-v6-20260902.jsonl "${run}/evaluations/habits-update1000.json"

paired="${run}/evaluations/paired-update1000-vs-parent.json"
docker run --rm --gpus all --ipc host --network none \
  -v "${project_root}/sapai:/workspace/sapai" \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/evaluate_paired_policy.py \
    "${candidate}" \
    --opponent checkpoint --opponent-checkpoint "${parent}" \
    --device cuda --batch-size 64 --pairs 200 \
    --progress-every 100 --seed 2810004 --output "${paired}"

promotion_behavior_gate=0
if python3 -c '
import json, sys
habits=json.load(open(sys.argv[1]))["overall"]
anchor_before=json.load(open(sys.argv[2]))["overall"]
anchor_after=json.load(open(sys.argv[3]))["overall"]
raise SystemExit(0 if (
    habits["fullActionAccuracy"] >= .85
    and anchor_after["weightedCrossEntropy"]
        <= anchor_before["weightedCrossEntropy"] + .25
) else 1)
' \
  "${run_host}/evaluations/habits-update1000.json" \
  "${run_host}/evaluations/anchor-holdout-before.json" \
  "${run_host}/evaluations/anchor-holdout-update1000.json"; then
  promotion_behavior_gate=1
fi

if [[ "${promotion_behavior_gate}" == 1 ]] && docker run --rm --network none \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym \
  "${image}" \
  python scripts/promote_league_checkpoint.py \
    "${paired}" --league "${run}/league.json" \
    --minimum-score 0.65 --minimum-pairs 100 --minimum-ci-low 0.50; then
  echo "v31 update 1000 promoted to permanent champion"
else
  echo "v31 update 1000 did not clear the permanent strength/behavior gate"
fi
