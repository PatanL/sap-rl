#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
candidate=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/selected.pt
opponent=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
seed=3604801

while [[ ! -f "${runs_host}${candidate#/workspace/runs}" ]]; do
  sleep 20
done
mkdir -p "${runs_host}/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/evaluations"

habits=/workspace/runs/turtle-league-v36-summons-tier1-recalibration-from-v31-1000/evaluations/habits-selected-v7.json
if [[ ! -f "${runs_host}${habits#/workspace/runs}" ]]; then
  docker run --rm --cpus 4 --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/evaluate_curriculum.py \
      "${candidate}" \
      /workspace/runs/curriculum/turtle-p1-scenarios-v7-20260904.jsonl \
      --threads 4 --batch-size 64 --output "${habits}"
fi

for perspective in p0 p1; do
  output="/workspace/runs/game-inspector-v36-500-vs-v31-seed${seed}-${perspective}.html"
  trace="/workspace/runs/game-inspector-v36-500-vs-v31-seed${seed}-${perspective}.json"
  if [[ -f "${runs_host}${output#/workspace/runs}" ]]; then
    continue
  fi
  docker run --rm --cpus 4 --network none \
    -v "${project_root}/sapai:/workspace/sapai" \
    -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
    -v "${runs_host}:/workspace/runs" \
    -w /workspace/sapai-gym "${image}" \
    python scripts/visualize_entity_game.py \
      "${candidate}" --opponent-checkpoint "${opponent}" \
      --seed "${seed}" --perspective "${perspective}" --threads 4 \
      --output "${output}" --trace-json "${trace}"
done
