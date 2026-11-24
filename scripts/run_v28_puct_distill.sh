#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
natural_prefix=/workspace/runs/search/v28b-puct-teacher-pilot
scenario_prefix=/workspace/runs/search/v28d-puct-scenario-failures
combined_prefix=/workspace/runs/search/v28-puct-teacher-combined
checkpoint=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
run=/workspace/runs/turtle-league-v28-puct-distilled-from-v25-300

while [[ ! -f "${runs_host}/search/v28b-puct-teacher-pilot.summary.json" \
      || ! -f "${runs_host}/search/v28d-puct-scenario-failures.summary.json" ]]; do
  sleep 30
done

if [[ ! -f "${runs_host}/search/v28-puct-teacher-combined.summary.json" ]]; then
  docker run --rm --network none \
    -v "${runs_host}:/workspace/runs" \
    "${image}" \
    python scripts/merge_puct_demonstrations.py \
      --train "${scenario_prefix}.jsonl" \
      --holdout "${scenario_prefix}-holdout.jsonl" \
      --audit "${scenario_prefix}.audit.jsonl" \
      --summaries "${scenario_prefix}.summary.json" \
      --output "${combined_prefix}.jsonl" \
      --holdout-output "${combined_prefix}-holdout.jsonl" \
      --audit-output "${combined_prefix}.audit.jsonl" \
      --summary-output "${combined_prefix}.summary.json"
fi

python3 -c '
import json, sys
report = json.load(open(sys.argv[1]))
outer = report["outerAudit"]
if report["candidateEligible"] < 5:
    raise SystemExit("fewer than five independently positive PUCT candidates")
if report["trainExamples"] < 10:
    raise SystemExit("fewer than ten weighted PUCT training examples")
if outer["lowerBound95"] is None or outer["lowerBound95"] <= 0:
    raise SystemExit("PUCT outer-audit 95% lower bound did not exceed zero")
print(json.dumps({
    "candidateEligible": report["candidateEligible"],
    "deployedAccepted": report["deployedAccepted"],
    "trainExamples": report["trainExamples"],
    "holdoutExamples": report["holdoutExamples"],
    "outerAudit": outer,
}, sort_keys=True))
' "${runs_host}/search/v28-puct-teacher-combined.summary.json"

mkdir -p "${runs_host}/turtle-league-v28-puct-distilled-from-v25-300/evaluations"

if [[ ! -f "${runs_host}/turtle-league-v28-puct-distilled-from-v25-300/evaluations/targets-before.json" ]]; then
  docker run --rm --network none \
    -v "${runs_host}:/workspace/runs" \
    "${image}" \
    python scripts/evaluate_curriculum.py \
      "${checkpoint}" \
      "${combined_prefix}.jsonl" \
      --threads 4 \
      --batch-size 64 \
      --output "${run}/evaluations/targets-before.json"
fi

if [[ ! -f "${runs_host}/turtle-league-v28-puct-distilled-from-v25-300/update-001000.pt" ]]; then
  initialization_args=(
    --init-checkpoint "${checkpoint}"
    --learning-rate 2e-6
  )
  latest_checkpoint="$({
    find "${runs_host}/turtle-league-v28-puct-distilled-from-v25-300" \
      -maxdepth 1 -type f -name 'update-*.pt' -print 2>/dev/null || true
  } | sort | tail -n 1)"
  if [[ -n "${latest_checkpoint}" ]]; then
    initialization_args=(
      --resume-checkpoint "/workspace/runs/turtle-league-v28-puct-distilled-from-v25-300/${latest_checkpoint##*/}"
      --resume-learning-rate 2e-6
    )
  fi
  docker run --rm --name sap-rl-v28-puct-train \
    --gpus all --ipc host \
    -v "${runs_host}:/workspace/runs" \
    "${image}" \
    python scripts/train_entity_selfplay.py \
      --distributed \
      --updates 1000 \
      --num-envs 32 \
      --games-per-actor 3 \
      --segments-per-update 64 \
      --transitions-per-update 8192 \
      --trajectory-queue-size 1024 \
      --inference-batch-delay-ms 5 \
      --max-behavior-policy-lag 1 \
      --checkpoint-every 100 \
      --checkpoint-dir "${run}" \
      "${initialization_args[@]}" \
      --discount-mode per_turn \
      --discount 0.99 \
      --battle-aux-cost 0.05 \
      --opponent-aux-cost 0.1 \
      --opponent-module-learning-rate 1e-4 \
      --opponent-adapter-learning-rate 1e-5 \
      --demo-data "${combined_prefix}.jsonl" \
      --demo-batch-size 64 \
      --demo-until-update 1000 \
      --demo-every 4 \
      --demo-learning-rate-scale 0.1 \
      --expert-anchor-checkpoint "${checkpoint}" \
      --resident-checkpoint /workspace/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt \
      --resident-checkpoint /workspace/runs/turtle-league-v21-live-sync-transfer-v19-600-to5000/update-006500.pt \
      --resident-checkpoint /workspace/runs/turtle-league-v21-live-sync-transfer-v19-600-to5000/update-010800.pt \
      --allow-simulator-mismatch \
      --max-historical-residents 4 \
      --seed 2780001
fi

candidate="${run}/update-001000.pt"
docker run --rm --network none \
  -v "${runs_host}:/workspace/runs" \
  "${image}" \
  python scripts/evaluate_curriculum.py \
    "${candidate}" \
    "${combined_prefix}.jsonl" \
    --threads 4 \
    --batch-size 64 \
    --output "${run}/evaluations/targets-after.json"

docker run --rm --network none \
  -v "${runs_host}:/workspace/runs" \
  "${image}" \
  python scripts/evaluate_curriculum.py \
    "${candidate}" \
    /workspace/runs/curriculum/turtle-p1-scenarios-v6-20260902.jsonl \
    --threads 4 \
    --batch-size 64 \
    --output "${run}/evaluations/habits-update1000.json"

paired_report="${run}/evaluations/paired-update1000-vs-v25-300.json"
docker run --rm --gpus all --ipc host --network none \
  -v "${runs_host}:/workspace/runs" \
  "${image}" \
  python scripts/evaluate_paired_policy.py \
    "${candidate}" \
    --opponent checkpoint \
    --opponent-checkpoint "${checkpoint}" \
    --device cuda \
    --batch-size 64 \
    --pairs 200 \
    --progress-every 100 \
    --seed 2780002 \
    --output "${paired_report}"

if docker run --rm --network none \
  -v "${runs_host}:/workspace/runs" \
  "${image}" \
  python scripts/promote_league_checkpoint.py \
    "${paired_report}" \
    --league "${run}/league.json" \
    --minimum-score 0.65 \
    --minimum-pairs 100 \
    --minimum-ci-low 0.50; then
  echo "PUCT-distilled update 1000 promoted to permanent champion"
else
  echo "PUCT-distilled update 1000 did not clear the permanent gate"
fi
