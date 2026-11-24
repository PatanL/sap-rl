#!/usr/bin/env bash
# A queued continuation of the validated frozen campaign, not an architecture ablation.
set -euo pipefail
project=/home/dzliu/pzliu/sap-rl
name=turtle-league-v45-corrected-from-v44-head-3250
folder="${project}/runs/${name}"
destination="/workspace/runs/${name}"
source_root="${folder}/source-batched-v1"
image=sha256:3ed8782e02a5d46e400bc39a5a77654c26e8a86b1d210c160e5965b765955ce1
# At least twelve hours after the user's request; finish the last 400-update
# chunk naturally, preserving a final checkpoint rather than killing a learner.
deadline=$(date -u -d '2026-09-06 18:15:00 UTC' +%s)
container=sap-rl-v45-overnight

while systemctl --user is-active --quiet sap-v45-corrected.service; do sleep 20; done
if systemctl --user is-failed --quiet sap-v45-corrected.service; then
  echo 'Predecessor failed; refusing to hide a campaign failure.' >&2
  exit 1
fi
for panel in parent control v31; do
  test -s "${folder}/evaluations/paired-6250-vs-${panel}.json"
done

while (( $(date -u +%s) < deadline )); do
  # Never evict an unrelated GPU process. Select the checkpoint AFTER waiting.
  while true; do
    if pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
      [[ -z "${pids//[[:space:]]/}" ]] && break
    fi
    sleep 20
  done
  (( $(date -u +%s) >= deadline )) && break
  python3 "${project}/scripts/v44_preflight.py" source "${source_root}" "${folder}/source-manifest-batched-v1.json"
  latest=$(find "${folder}" -maxdepth 1 -type f -name 'update-*.pt' -printf '%f\n' | sort -V | tail -1)
  test -n "${latest}"
  digits="${latest#update-}"; digits="${digits%.pt}"
  previous=$((10#${digits})); target=$((previous + 400))
  (( previous >= 6250 ))
  if docker inspect "${container}" >/dev/null 2>&1; then
    [[ $(docker inspect -f '{{.State.Running}}' "${container}") == false ]]
    docker rm "${container}" >/dev/null
  fi
  echo "Overnight continuation: ${previous} -> ${target}; finish current chunk after deadline."
  docker run --name "${container}" --gpus all --ipc host --network none \
    -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
    -v "${source_root}/sapai:/workspace/sapai:ro" \
    -v "${source_root}/sapai-gym:/workspace/sapai-gym:ro" \
    -v "${project}/runs:/workspace/runs" -w /workspace/sapai-gym "${image}" \
    python scripts/train_entity_selfplay.py --distributed --device cuda \
    --updates "${target}" --resume-checkpoint "${destination}/${latest}" \
    --resume-learning-rate 2e-6 --end-freeze-learning-rate 5e-5 \
    --opponent-module-learning-rate 1e-4 --opponent-adapter-learning-rate 1e-5 \
    --num-envs 32 --games-per-actor 3 --segments-per-update 64 \
    --transitions-per-update 8192 --trajectory-queue-size 1024 \
    --inference-batch-delay-ms 5 --max-behavior-policy-lag 1 \
    --checkpoint-every 100 --checkpoint-dir "${destination}" \
    --discount-mode per_turn --discount .99 --battle-aux-cost .05 --opponent-aux-cost .1 \
    --resident-checkpoint /workspace/runs/turtle-v44-end-freeze-ablation/control/update-003250.pt \
    --max-historical-residents 4 --allow-simulator-mismatch \
    --seed "$((4500001 + previous))" 2>&1 | tee -a "${folder}/training.log"
done
echo 'Overnight window complete; final chunk and checkpoint finished normally.'
