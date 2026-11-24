#!/usr/bin/env bash
set -euo pipefail

root=/home/dzliu/pzliu/sap-rl
name=sap-rl-v21-live-sync-transfer-v19-600-to5000
image=sap-rl-v21-frozen:20260901

if docker inspect "${name}" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "${name}")" == "true" ]]; then
    echo "${name} already running"
    exit 0
  fi
  docker rm "${name}" >/dev/null
fi

docker run -d \
  --name "${name}" \
  --restart=unless-stopped \
  --gpus all \
  --ipc=host \
  -e V21_TARGET_UPDATES="${V21_TARGET_UPDATES:-5000}" \
  -v "${root}/runs":/workspace/runs \
  "${image}" \
  bash scripts/run_v21_live_sync_transfer_v19_600.sh
