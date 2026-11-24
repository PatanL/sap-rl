#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
run=/workspace/runs/turtle-league-v38-chocolate-legality-recalibration-from-v36-100
run_host="${runs_host}/turtle-league-v38-chocolate-legality-recalibration-from-v36-100"
parent=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
target_updates=300

mkdir -p "${run_host}"
latest_checkpoint="$({
  find "${run_host}" -maxdepth 1 -type f -name 'update-*.pt' -printf '%f\n'
} | sort -V | tail -1)"

initialization=(--init-checkpoint "${parent}" --learning-rate 1e-6)
run_seed=3804001
if [[ -n "${latest_checkpoint}" ]]; then
  latest_update="${latest_checkpoint#update-}"
  latest_update="${latest_update%.pt}"
  if (( 10#${latest_update} >= target_updates )); then
    exit 0
  fi
  initialization=(
    --resume-checkpoint "${run}/${latest_checkpoint}"
    --resume-learning-rate 1e-6
  )
  run_seed=$((3804001 + 10#${latest_update}))
fi

docker rm -f sap-v38-chocolate-recalibration >/dev/null 2>&1 || true
exec docker run --rm --name sap-v38-chocolate-recalibration \
  --gpus all --ipc host --network none \
  -v "${project_root}/sapai:/workspace/sapai" \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym "${image}" \
  python scripts/train_entity_selfplay.py \
    --distributed --updates "${target_updates}" \
    --num-envs 32 --games-per-actor 3 \
    --segments-per-update 64 --transitions-per-update 8192 \
    --trajectory-queue-size 1024 --inference-batch-delay-ms 5 \
    --max-behavior-policy-lag 1 --checkpoint-every 50 \
    --checkpoint-dir "${run}" "${initialization[@]}" \
    --allow-simulator-mismatch \
    --discount-mode per_turn --discount 0.99 \
    --battle-aux-cost 0.05 --opponent-aux-cost 0.1 \
    --opponent-module-learning-rate 1e-4 \
    --opponent-adapter-learning-rate 1e-5 \
    --expert-anchor-checkpoint "${parent}" \
    --resident-checkpoint \
      /workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt \
    --resident-checkpoint \
      /workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt \
    --resident-checkpoint \
      /workspace/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt \
    --max-historical-residents 4 --seed "${run_seed}"
