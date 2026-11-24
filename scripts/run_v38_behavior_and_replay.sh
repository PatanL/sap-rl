#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
run=/workspace/runs/turtle-league-v38-chocolate-legality-recalibration-from-v36-100
run_host="${runs_host}/turtle-league-v38-chocolate-legality-recalibration-from-v36-100"
candidate=${run}/selected.pt
parent=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
curriculum=/workspace/runs/curriculum/turtle-p1-scenarios-v8-20260904.jsonl
seed=3804801

while [[ ! -f "${runs_host}${candidate#/workspace/runs}" ]]; do sleep 20; done
mkdir -p "${run_host}/evaluations"

for spec in \
  "${candidate}|${run}/evaluations/habits-selected-v8.json" \
  "${parent}|${run}/evaluations/habits-parent-v36-current-v8.json"; do
  checkpoint="${spec%%|*}"
  output="${spec#*|}"
  if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then continue; fi
  docker run --rm --cpus 4 --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/evaluate_curriculum.py \
      "${checkpoint}" "${curriculum}" --threads 4 --batch-size 64 \
      --output "${output}"
done

for perspective in p0 p1; do
  output="/workspace/runs/game-inspector-v38-selected-vs-v36-seed${seed}-${perspective}.html"
  trace="/workspace/runs/game-inspector-v38-selected-vs-v36-seed${seed}-${perspective}.json"
  if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then continue; fi
  docker run --rm --cpus 4 --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/visualize_entity_game.py \
      "${candidate}" --opponent-checkpoint "${parent}" \
      --seed "${seed}" --perspective "${perspective}" --threads 4 \
      --output "${output}" --trace-json "${trace}"
done
