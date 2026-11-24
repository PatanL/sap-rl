#!/usr/bin/env bash
set -euo pipefail

candidate=/workspace/runs/turtle-league-v39-unrestricted-from-v38-250/update-001250.pt
reference=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
for sample_matchup in forward reverse; do
  sample_p0="${candidate}"
  sample_p1="${reference}"
  sample_label0='P0 v39 update 1250'
  sample_label1='P1 v31 update 1000'
  if [[ "${sample_matchup}" == reverse ]]; then
    sample_p0="${reference}"
    sample_p1="${candidate}"
    sample_label0='P0 v31 update 1000'
    sample_label1='P1 v39 update 1250'
  fi
  for sample_perspective in p0 p1; do
    sample_name="game-v39-1250-v31-1000-seed344327552-${sample_matchup}-${sample_perspective}"
    python scripts/visualize_entity_game.py "${sample_p0}" \
      --opponent-checkpoint "${sample_p1}" --seed 344327552 --threads 1 \
      --perspective "${sample_perspective}" \
      --player-label "${sample_label0}" --opponent-label "${sample_label1}" \
      --output "/workspace/runs/${sample_name}.html" \
      --trace-json "/workspace/runs/${sample_name}.json"
  done
done
