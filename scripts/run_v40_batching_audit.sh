#!/usr/bin/env bash
set -euo pipefail

# Queue behind this exact, currently running probe rather than adding another
# concurrent CPU workload. A wait timeout is never interpreted as completion.
code="$(docker wait sap-v40-cached-plan-numerical-cpu)"
if [[ "${code}" != 0 ]]; then
  echo 'Numerical probe failed; not starting its follow-up batching audit' >&2
  exit 1
fi

docker run --name sap-v40-batching-audit-cpu --entrypoint python \
  --network none --cpus 1 --cpu-shares 64 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/sapai:/workspace/sapai-gym \
  -v /home/dzliu/pzliu/sap-rl/sapai:/workspace/sapai:ro \
  -v /home/dzliu/pzliu/sap-rl/sapai-gym:/workspace/sapai-gym:ro \
  -v /home/dzliu/pzliu/sap-rl/runs:/workspace/runs \
  -w /workspace/sapai-gym sap-rl-v28-puct:20260903h \
  -u scripts/audit_interactive_batching.py \
  /workspace/runs/turtle-league-v38-chocolate-legality-recalibration-from-v36-100/update-000250.pt \
  --record /workspace/runs/search/v40-fixed-decisions/state-0003.json \
  --comparison-row /workspace/runs/search/v40-interactive-states1-15.parts/state-0003.json \
  --output /workspace/runs/search/v40-interactive-batching-state3.json
