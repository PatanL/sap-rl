#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v25-frozen:20260903-planner6
recalibration_container=sap-rl-v25-recalibration
recalibration_dir=/workspace/runs/turtle-league-v25-recalibration-from-v22-2000
recalibration_checkpoint="${recalibration_dir}/update-000300.pt"
reservoir=/workspace/runs/search/v25-update300-opponent-beliefs-v2.json
critic_report=/workspace/runs/search/v25-update300-critic-calibration-v2.json
search_data=/workspace/runs/search/v25-update300-search-v6.jsonl
search_holdout=/workspace/runs/search/v25-update300-search-v6-holdout.jsonl
search_audit=/workspace/runs/search/v25-update300-search-v6.audit.jsonl
search_summary=/workspace/runs/search/v25-update300-search-v6.summary.json
search_run=/workspace/runs/turtle-league-v25-search-guided-from-recal300

while [[ "$(docker inspect -f '{{.State.Running}}' "${recalibration_container}" 2>/dev/null || true)" == "true" ]]; do
  sleep 30
done

if [[ ! -f "${runs_host}/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt" ]]; then
  echo "Recalibration did not produce update-000300.pt" >&2
  exit 1
fi

if [[ ! -f "${runs_host}/search/v25-update300-opponent-beliefs-v2.json" ]]; then
  docker run --rm --name sap-rl-v25-opponent-reservoir \
    --gpus all --ipc host --network none \
    -v "${runs_host}:/workspace/runs" \
    "${image}" \
    python scripts/generate_opponent_trajectory_reservoir.py \
      "${recalibration_checkpoint}" \
      --output "${reservoir}" \
      --games 256 \
      --batch-size 128 \
      --device cuda \
      --seed 2503001
fi

if [[ ! -f "${runs_host}/search/v25-update300-critic-calibration-v2.json" ]]; then
  calibration_parts=()
  calibration_containers=()
  for shard in $(seq 0 7); do
    part_name="v25-update300-critic-calibration-v2-part-$(printf '%02d' "${shard}").json"
    part_host="${runs_host}/search/${part_name}"
    calibration_parts+=("/workspace/runs/search/${part_name}")
    if [[ -f "${part_host}" ]]; then
      continue
    fi
    container_name="sap-rl-v25-critic-calibration-${shard}"
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    docker run -d --name "${container_name}" \
      --gpus all --ipc host --network none \
      -v "${runs_host}:/workspace/runs" \
      "${image}" \
      python scripts/evaluate_critic_calibration.py \
        "${recalibration_checkpoint}" \
        --opponent-reservoir "${reservoir}" \
        --output "/workspace/runs/search/${part_name}" \
        --states 64 \
        --rollouts 32 \
        --horizon-battles 10 \
        --max-continuation-steps 500 \
        --opponent-candidates 32 \
        --state-probability 0.15 \
        --device cuda \
        --batch-size 128 \
        --seed "$((2503005 + shard))" >/dev/null
    calibration_containers+=("${container_name}")
  done
  for container_name in "${calibration_containers[@]}"; do
    exit_code="$(docker wait "${container_name}")"
    # The calibration containers write their result files as root through the
    # bind mount, so the search directory may not be host-user writable. Keep
    # shard logs in journald instead of redirecting them from the host shell.
    # A failed redirect here previously aborted the wrapper after every shard
    # itself had completed successfully.
    docker logs "${container_name}"
    docker rm "${container_name}" >/dev/null
    if [[ "${exit_code}" != "0" ]]; then
      echo "${container_name} failed with exit ${exit_code}" >&2
      exit 1
    fi
  done
  docker run --rm --network none \
    -v "${runs_host}:/workspace/runs" \
    "${image}" \
    python scripts/merge_critic_calibrations.py \
      "${calibration_parts[@]}" \
      --output "${critic_report}"
fi

leaf_value_weight="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["recommendedLeafValueWeight"])' \
  "${runs_host}/search/v25-update300-critic-calibration-v2.json")"

if [[ ! -f "${runs_host}/search/v25-update300-search-v6.jsonl" ]]; then
  search_train_parts=()
  search_holdout_parts=()
  search_audit_parts=()
  search_summary_parts=()
  search_containers=()
  for shard in $(seq 0 7); do
    shard_id="$(printf '%02d' "${shard}")"
    train_part="/workspace/runs/search/v25-update300-search-v6-part-${shard_id}.jsonl"
    holdout_part="/workspace/runs/search/v25-update300-search-v6-holdout-part-${shard_id}.jsonl"
    audit_part="/workspace/runs/search/v25-update300-search-v6-audit-part-${shard_id}.jsonl"
    summary_part="/workspace/runs/search/v25-update300-search-v6-summary-part-${shard_id}.json"
    search_train_parts+=("${train_part}")
    search_holdout_parts+=("${holdout_part}")
    search_audit_parts+=("${audit_part}")
    search_summary_parts+=("${summary_part}")
    if [[ -f "${runs_host}/search/${summary_part##*/}" ]]; then
      continue
    fi
    shard_attempts=62
    if (( shard < 4 )); then
      shard_attempts=63
    fi
    container_name="sap-rl-v25-search-generate-${shard_id}"
    docker rm -f "${container_name}" >/dev/null 2>&1 || true
    docker run -d --name "${container_name}" \
      --gpus all --ipc host --network none \
      -v "${runs_host}:/workspace/runs" \
      "${image}" \
      python scripts/generate_search_demonstrations.py \
        "${recalibration_checkpoint}" \
        --output "${train_part}" \
        --holdout-output "${holdout_part}" \
        --holdout-fraction 0.20 \
        --audit-output "${audit_part}" \
        --summary-output "${summary_part}" \
        --compact \
        --games 32 \
        --max-search-attempts "${shard_attempts}" \
        --max-searches-per-player-turn 1 \
        --rollouts-per-action 4 \
        --confirmation-rollouts 16 \
        --confirmation-top-k 4 \
        --validation-rollouts 32 \
        --audit-rollouts 32 \
        --horizon-battles 5 \
        --max-continuation-steps 150 \
        --minimum-improvement 0.10 \
        --confidence-z 1.96 \
        --exhaustive-root-probability 0.10 \
        --leaf-value-weight "${leaf_value_weight}" \
        --target-temperature 0.10 \
        --target-top-k 4 \
        --minimum-target-probability 0.02 \
        --opportunity-probability 0.35 \
        --general-probability 0.03 \
        --opponent-reservoir "${reservoir}" \
        --opponent-candidates 32 \
        --require-opponent-reservoir \
        --prune-affordable-freezes \
        --device cuda \
        --search-batch-size 128 \
        --seed "$((2503100 + shard))" >/dev/null
    search_containers+=("${container_name}")
  done
  for container_name in "${search_containers[@]}"; do
    exit_code="$(docker wait "${container_name}")"
    docker logs "${container_name}"
    docker rm "${container_name}" >/dev/null
    if [[ "${exit_code}" != "0" ]]; then
      echo "${container_name} failed with exit ${exit_code}" >&2
      exit 1
    fi
  done
  docker run --rm --network none \
    -v "${runs_host}:/workspace/runs" \
    -v "${project_root}/sapai-gym/scripts/merge_search_shards.py:/tmp/merge_search_shards.py:ro" \
    "${image}" \
    python /tmp/merge_search_shards.py \
      --train "${search_train_parts[@]}" \
      --holdout "${search_holdout_parts[@]}" \
      --audit "${search_audit_parts[@]}" \
      --summaries "${search_summary_parts[@]}" \
      --output "${search_data}" \
      --holdout-output "${search_holdout}" \
      --audit-output "${search_audit}" \
      --summary-output "${search_summary}"
fi

python3 -c '
import json, sys
report = json.load(open(sys.argv[1]))
outer = report.get("outerAudit", {})
lower = outer.get("lowerBound95")
if lower is None or lower <= 0.0:
    raise SystemExit(
        "untouched outer-audit 95% lower bound must exceed zero "
        "before search distillation"
    )
' "${runs_host}${search_summary#/workspace/runs}"

search_examples="$(wc -l < "${runs_host}${search_data#/workspace/runs}")"
if (( search_examples < 50 )); then
  echo "Only ${search_examples} weighted search targets; refusing to train" >&2
  exit 1
fi
holdout_examples="$(wc -l < "${runs_host}${search_holdout#/workspace/runs}")"
if (( holdout_examples < 10 )); then
  echo "Only ${holdout_examples} held-out search targets; refusing to train" >&2
  exit 1
fi

mkdir -p "${runs_host}/turtle-league-v25-search-guided-from-recal300/evaluations"
if [[ ! -f "${runs_host}/turtle-league-v25-search-guided-from-recal300/evaluations/search-targets-before.json" ]]; then
  docker run --rm --network none \
    -v "${runs_host}:/workspace/runs" \
    "${image}" \
    python scripts/evaluate_curriculum.py \
      "${recalibration_checkpoint}" \
      "${search_data}" \
      --threads 4 \
      --batch-size 64 \
      --output "${search_run}/evaluations/search-targets-before.json"
fi

if [[ ! -f "${runs_host}/turtle-league-v25-search-guided-from-recal300/evaluations/search-holdout-before.json" ]]; then
  docker run --rm --network none \
    -v "${runs_host}:/workspace/runs" \
    "${image}" \
    python scripts/evaluate_curriculum.py \
      "${recalibration_checkpoint}" \
      "${search_holdout}" \
      --threads 4 \
      --batch-size 64 \
      --output "${search_run}/evaluations/search-holdout-before.json"
fi

if [[ ! -f "${runs_host}/turtle-league-v25-search-guided-from-recal300/update-001000.pt" ]]; then
  initialization_args=(
    --init-checkpoint "${recalibration_checkpoint}"
    --learning-rate 2e-6
  )
  latest_checkpoint="$({
    find "${runs_host}/turtle-league-v25-search-guided-from-recal300" \
      -maxdepth 1 -type f -name 'update-*.pt' -print 2>/dev/null || true
  } | sort | tail -n 1)"
  if [[ -n "${latest_checkpoint}" ]]; then
    initialization_args=(
      --resume-checkpoint "/workspace/runs/turtle-league-v25-search-guided-from-recal300/${latest_checkpoint##*/}"
      --resume-learning-rate 2e-6
    )
  fi
  docker run --rm --name sap-rl-v25-search-train \
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
      --checkpoint-dir "${search_run}" \
      "${initialization_args[@]}" \
      --discount-mode per_turn \
      --discount 0.99 \
      --battle-aux-cost 0.05 \
      --opponent-aux-cost 0.1 \
      --opponent-module-learning-rate 1e-4 \
      --opponent-adapter-learning-rate 1e-5 \
      --demo-data "${search_data}" \
      --demo-batch-size 64 \
      --demo-until-update 1000 \
      --demo-every 4 \
      --demo-learning-rate-scale 0.1 \
      --expert-anchor-checkpoint "${recalibration_checkpoint}" \
      --resident-checkpoint /workspace/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt \
      --resident-checkpoint /workspace/runs/turtle-league-v21-live-sync-transfer-v19-600-to5000/update-006500.pt \
      --resident-checkpoint /workspace/runs/turtle-league-v21-live-sync-transfer-v19-600-to5000/update-010800.pt \
      --allow-simulator-mismatch \
      --max-historical-residents 4 \
      --seed 2503003
fi

candidate="${search_run}/update-001000.pt"
docker run --rm --network none \
  -v "${runs_host}:/workspace/runs" \
  "${image}" \
  python scripts/evaluate_curriculum.py \
    "${candidate}" \
    "${search_data}" \
    --threads 4 \
    --batch-size 64 \
    --output "${search_run}/evaluations/search-targets-after.json"

docker run --rm --network none \
  -v "${runs_host}:/workspace/runs" \
  "${image}" \
  python scripts/evaluate_curriculum.py \
    "${candidate}" \
    "${search_holdout}" \
    --threads 4 \
    --batch-size 64 \
    --output "${search_run}/evaluations/search-holdout-after.json"

docker run --rm --network none \
  -v "${runs_host}:/workspace/runs" \
  "${image}" \
  python scripts/evaluate_curriculum.py \
    "${candidate}" \
    /workspace/runs/curriculum/turtle-p1-scenarios-v6-20260902.jsonl \
    --threads 4 \
    --batch-size 64 \
    --output "${search_run}/evaluations/habits-update1000.json"

paired_report="${search_run}/evaluations/paired-update1000-vs-recal300.json"
docker run --rm --gpus all --ipc host --network none \
  -v "${runs_host}:/workspace/runs" \
  "${image}" \
  python scripts/evaluate_paired_policy.py \
    "${candidate}" \
    --opponent checkpoint \
    --opponent-checkpoint "${recalibration_checkpoint}" \
    --device cuda \
    --batch-size 64 \
    --pairs 200 \
    --progress-every 100 \
    --seed 2503004 \
    --output "${paired_report}"

if docker run --rm --network none \
  -v "${runs_host}:/workspace/runs" \
  "${image}" \
  python scripts/promote_league_checkpoint.py \
    "${paired_report}" \
    --league "${search_run}/league.json" \
    --minimum-score 0.65 \
    --minimum-pairs 100 \
    --minimum-ci-low 0.50; then
  echo "Search-guided update 1000 promoted to permanent champion"
else
  echo "Search-guided update 1000 did not clear the permanent gate"
fi
