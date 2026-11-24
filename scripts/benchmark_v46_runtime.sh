#!/usr/bin/env bash
set -euo pipefail
project=/home/dzliu/pzliu/sap-rl
main_host="${project}/runs/turtle-league-v45-corrected-from-v44-head-3250"
parent=/workspace/runs/turtle-league-v45-corrected-from-v44-head-3250/update-004900.pt
variant=$1
[[ "${variant}" == baseline || "${variant}" == batched ]]
source_root="${main_host}/source"
[[ "${variant}" == batched ]] && source_root="${main_host}/source-batched-v1"
host_run="${project}/runs/performance/v46-runtime-${variant}"
run="/workspace/runs/performance/v46-runtime-${variant}"
[[ ! -d "${host_run}" ]]  # Explicit new benchmark, never overwrite a prior trial.
mkdir -p "${host_run}"
cp "${main_host}/league.json" "${host_run}/league.json"
docker run --rm --name "sap-v46-runtime-${variant}" --gpus all --ipc host --network none \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -v "${source_root}/sapai:/workspace/sapai:ro" \
  -v "${source_root}/sapai-gym:/workspace/sapai-gym:ro" \
  -v "${project}/runs:/workspace/runs" -w /workspace/sapai-gym \
  sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1 \
  python scripts/train_entity_selfplay.py --distributed --device cuda --updates 4907 \
    --resume-checkpoint "${parent}" --resume-learning-rate 2e-6 --end-freeze-learning-rate 5e-5 \
    --opponent-module-learning-rate 1e-4 --opponent-adapter-learning-rate 1e-5 \
    --num-envs 32 --games-per-actor 3 --segments-per-update 64 --transitions-per-update 8192 \
    --trajectory-queue-size 1024 --inference-batch-delay-ms 5 --max-behavior-policy-lag 1 \
    --checkpoint-every 100 --checkpoint-dir "${run}" --discount-mode per_turn --discount .99 \
    --battle-aux-cost .05 --opponent-aux-cost .1 --max-historical-residents 4 \
    --allow-simulator-mismatch --seed 4604900 2>&1 | tee "${host_run}/training.log"
