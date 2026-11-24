#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != *-* ]]; then
  echo "usage: $0 WINDOW-CHUNK" >&2
  exit 2
fi

project_root=/home/dzliu/pzliu/sap-rl
runs_host="${project_root}/runs"
image=sap-rl-v28-puct:20260903h
checkpoint=/workspace/runs/turtle-league-v31-search-integration-from-v25-300/update-001000.pt
reservoir=/workspace/runs/search/v35-v31-opponent-beliefs.json
prefix=/workspace/runs/search/v35-large-stratified-puct-v9
minimum_turns=(2 4 5 7 9 10 11 13)
maximum_turns=(3 4 6 8 9 10 12 15)

key=$1
window_text="${key%%-*}"
chunk_text="${key##*-}"
window=$((10#${window_text}))
chunk=$((10#${chunk_text}))
if (( window < 0 || window > 7 || chunk < 0 || chunk > 19 )); then
  echo "invalid v35 PUCT chunk: ${key}" >&2
  exit 2
fi
window_id="$(printf '%02d' "${window}")"
chunk_id="$(printf '%02d' "${chunk}")"
part="${prefix}-window-${window_id}-chunk-${chunk_id}"
summary="${part}.summary.json"
if [[ -f "${runs_host}${summary#/workspace/runs}" ]]; then
  exit 0
fi

container="sap-v35-puct-fill-${window_id}-${chunk_id}"
docker rm -f "${container}" >/dev/null 2>&1 || true
docker run --rm --name "${container}" \
  --gpus all --ipc host --network none \
  -v "${project_root}/sapai:/workspace/sapai" \
  -v "${project_root}/sapai-gym:/workspace/sapai-gym" \
  -v "${runs_host}:/workspace/runs" \
  -w /workspace/sapai-gym "${image}" \
  python scripts/generate_puct_demonstrations.py \
    "${checkpoint}" --opponent-reservoir "${reservoir}" \
    --output "${part}.jsonl" \
    --holdout-output "${part}-holdout.jsonl" \
    --audit-output "${part}.audit.jsonl" \
    --summary-output "${summary}" \
    --games 128 --max-search-attempts 8 \
    --max-searches-per-player-turn 2 --reserve-zero-gold-search-slot \
    --opportunity-probability 1 --general-probability 0.05 \
    --minimum-turn "${minimum_turns[window]}" \
    --maximum-turn "${maximum_turns[window]}" \
    --holdout-fraction 0.20 \
    --transition-budget 7500 --max-simulations 4096 \
    --min-root-visits 2 --horizon-battles 5 \
    --validation-rollouts 32 --max-validation-rollouts 512 \
    --audit-rollouts 256 \
    --root-screening-top-k 8 --root-screening-rollouts 8 \
    --root-screening-rounds 3 --root-screening-min-survivors 4 \
    --cpuct 2 --prior-floor 0.10 --leaf-value-weight 0.5 \
    --internal-target-weight 0.25 --minimum-target-probability 0.02 \
    --opponent-candidates 32 \
    --screening-transition-budget 1000 \
    --screening-max-simulations 512 \
    --screening-validation-rollouts 8 \
    --screening-max-validation-rollouts 16 \
    --screening-minimum-improvement 0.02 \
    --screening-bypass-probability 0.10 \
    --opportunity-screening-bypass-probability 0.50 \
    --general-screening-bypass-probability 0.10 \
    --device cuda --batch-size 128 \
    --seed "$((2853501 + window * 1000 + chunk))"
