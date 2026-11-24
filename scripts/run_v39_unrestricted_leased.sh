#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
run_name=turtle-league-v39-unrestricted-from-v38-250
run_host="${project_root}/runs/${run_name}"
run="/workspace/runs/${run_name}"
source_root="${run_host}/source-v2"
image=sap-rl-v28-puct:20260903h
parent=/workspace/runs/turtle-league-v38-chocolate-legality-recalibration-from-v36-100/update-000250.pt
container=sap-rl-v39-unrestricted

[[ -f "${source_root}/sapai/sapai/turtle_pack.py" ]]
[[ -f "${source_root}/sapai-gym/scripts/train_entity_selfplay.py" ]]
mkdir -p "${run_host}/evaluations"

latest="$(find "${run_host}" -maxdepth 1 -type f -name 'update-*.pt' -printf '%f\n' | sort -V | tail -1)"
resume="${parent}"
last_update=250
if [[ -n "${latest}" ]]; then
  resume="${run}/${latest}"
  digits="${latest#update-}"
  digits="${digits%.pt}"
  last_update=$((10#${digits}))
fi

# A full 1,000 additional updates before the first scheduled evaluation.
for target in 1250 2250; do
  if (( last_update < target )); then
    if docker inspect "${container}" >/dev/null 2>&1; then
      if [[ "$(docker inspect -f '{{.State.Running}}' "${container}")" == true ]]; then
        echo "Refusing to replace the active v39 container" >&2
        exit 1
      fi
      docker rm "${container}" >/dev/null
    fi
    docker run --name "${container}" --gpus all --ipc host --network none \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -v "${source_root}/sapai:/workspace/sapai:ro" \
      -v "${source_root}/sapai-gym:/workspace/sapai-gym:ro" \
      -v "${project_root}/runs:/workspace/runs" \
      -w /workspace/sapai-gym "${image}" \
      python scripts/train_entity_selfplay.py \
        --distributed --updates "${target}" \
        --resume-checkpoint "${resume}" --resume-learning-rate 2e-6 \
        --num-envs 32 --games-per-actor 3 \
        --segments-per-update 64 --transitions-per-update 8192 \
        --trajectory-queue-size 1024 --inference-batch-delay-ms 5 \
        --max-behavior-policy-lag 1 --checkpoint-every 100 \
        --checkpoint-dir "${run}" \
        --discount-mode per_turn --discount 0.99 \
        --battle-aux-cost 0.05 --opponent-aux-cost 0.1 \
        --opponent-module-learning-rate 1e-4 \
        --opponent-adapter-learning-rate 1e-5 \
        --expert-anchor-checkpoint "${parent}" \
        --resident-checkpoint /workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt \
        --resident-checkpoint /workspace/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt \
        --resident-checkpoint /workspace/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt \
        --max-historical-residents 4 --allow-simulator-mismatch \
        --seed "$((3905001 + last_update))"
    last_update="${target}"
    resume="${run}/$(printf 'update-%06d.pt' "${target}")"
  fi

  for panel in parent v31; do
    opponent="${parent}"
    if [[ "${panel}" == v31 ]]; then
      opponent=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
    fi
    report="${run}/evaluations/paired-${target}-vs-${panel}.json"
    if [[ -f "${project_root}/runs/${run_name}/evaluations/paired-${target}-vs-${panel}.json" ]]; then
      continue
    fi
    # Same seed family and both seats at both checkpoints. These are learning
    # curve measurements, not permanent-promotion selection seeds.
    docker run --rm --name sap-rl-v39-evaluation --gpus all --ipc host --network none \
      -e PYTHONDONTWRITEBYTECODE=1 \
      -v "${source_root}/sapai:/workspace/sapai:ro" \
      -v "${source_root}/sapai-gym:/workspace/sapai-gym:ro" \
      -v "${project_root}/runs:/workspace/runs" \
      -w /workspace/sapai-gym "${image}" \
      python scripts/evaluate_paired_policy.py \
        "${run}/$(printf 'update-%06d.pt' "${target}")" \
        --opponent checkpoint --opponent-checkpoint "${opponent}" \
        --device cuda --batch-size 64 --pairs 100 --seed 3906001 \
        --progress-every 100 --output "${report}"
  done
done
