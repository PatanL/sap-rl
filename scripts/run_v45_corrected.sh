#!/usr/bin/env bash
set -euo pipefail
project=/home/dzliu/pzliu/sap-rl
name=turtle-league-v45-corrected-from-v44-head-3250
host_run="${project}/runs/${name}"
run="/workspace/runs/${name}"
source_root="${host_run}/source-batched-v1"
image=sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1
parent=/workspace/runs/turtle-v44-end-freeze-ablation/end-freeze/update-003250.pt
control=/workspace/runs/turtle-v44-end-freeze-ablation/control/update-003250.pt
reference=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
mkdir -p "${host_run}/evaluations"

run_job() {
  local container=$1
  shift
  while true; do
    if pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
      [[ -z "${pids//[[:space:]]/}" ]] && break
    fi
    sleep 30
  done
  python3 "${project}/scripts/v44_preflight.py" source "${source_root}" "${host_run}/source-manifest-batched-v1.json"
  if docker inspect "${container}" >/dev/null 2>&1; then
    [[ $(docker inspect -f '{{.State.Running}}' "${container}") == false ]]
    docker rm "${container}" >/dev/null
  fi
  docker run --name "${container}" --gpus all --ipc host --network none \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
    -v "${source_root}/sapai:/workspace/sapai:ro" \
    -v "${source_root}/sapai-gym:/workspace/sapai-gym:ro" \
    -v "${project}/runs:/workspace/runs" -w /workspace/sapai-gym "${image}" "$@"
}

train() {
  local directory=$1 target=$2 container=$3
  local folder="${host_run}${directory}" destination="${run}${directory}"
  local latest resume="${parent}" previous=3250 digits
  mkdir -p "${folder}"
  if [[ ! -f "${folder}/league.json" ]]; then
    cp "${project}/runs/turtle-v44-end-freeze-ablation/end-freeze/league.json" "${folder}/league.json"
  fi
  latest=$(find "${folder}" -maxdepth 1 -type f -name 'update-*.pt' -printf '%f\n' | sort -V | tail -1)
  if [[ -n "${latest}" ]]; then
    digits="${latest#update-}"; digits="${digits%.pt}"
    previous=$((10#${digits}))
    resume="${destination}/${latest}"
  fi
  (( previous >= target )) && return
  run_job "${container}" python scripts/train_entity_selfplay.py \
    --distributed --device cuda --updates "${target}" --resume-checkpoint "${resume}" \
    --resume-learning-rate 2e-6 --end-freeze-learning-rate 5e-5 \
    --opponent-module-learning-rate 1e-4 --opponent-adapter-learning-rate 1e-5 \
    --num-envs 32 --games-per-actor 3 --segments-per-update 64 \
    --transitions-per-update 8192 --trajectory-queue-size 1024 \
    --inference-batch-delay-ms 5 --max-behavior-policy-lag 1 \
    --checkpoint-every 100 --checkpoint-dir "${destination}" \
    --discount-mode per_turn --discount .99 --battle-aux-cost .05 --opponent-aux-cost .1 \
    --resident-checkpoint "${control}" --max-historical-residents 4 \
    --allow-simulator-mismatch --seed "$((4500001 + previous))" \
    2>&1 | tee -a "${folder}/training.log"
}

# Smoke weights are diagnostic only; main starts from the untouched parent.
train /smoke 3252 sap-v45-gpu-smoke
python3 "${project}/scripts/v44_preflight.py" smoke "${host_run}/smoke/training.log" --update 3252 --joint
for target in 4250 5250 6250; do
  train '' "${target}" sap-rl-v45-corrected
  for panel in parent control v31; do
    opponent="${parent}"
    [[ "${panel}" == control ]] && opponent="${control}"
    [[ "${panel}" == v31 ]] && opponent="${reference}"
    report="paired-${target}-vs-${panel}"
    [[ -f "${host_run}/evaluations/${report}.json" ]] && continue
    run_job sap-rl-v45-evaluation python scripts/evaluate_paired_policy.py \
      "${run}/$(printf 'update-%06d.pt' "${target}")" \
      --opponent checkpoint --opponent-checkpoint "${opponent}" \
      --device cuda --batch-size 64 --pairs 100 --seed 4506001 \
      --progress-every 100 --output "${run}/evaluations/${report}.json" \
      2>&1 | tee -a "${host_run}/evaluations/${report}.log"
  done
done
