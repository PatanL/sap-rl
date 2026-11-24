#!/usr/bin/env bash
set -euo pipefail

project_root=/home/dzliu/pzliu/sap-rl
container_name=sap-live-arena-agent
image=sap-live-agent:20260901
default_checkpoint="${project_root}/runs/turtle-league-v19-opponent-transfer-v18-5000/update-000600.pt"
checkpoint="${SAP_LIVE_CHECKPOINT:-${default_checkpoint}}"
dashboard="${SAP_LIVE_DASHBOARD:-http://100.97.32.64:8770}"
mode="${1:-resume}"
if [[ $# -gt 0 ]]; then
  shift
fi
extra_args=("$@")

case "${mode}" in
  resume)
    navigation_args=(--navigate-from-main)
    ;;
  resume-shop)
    # Browser already displays the journaled in-progress shop.
    navigation_args=()
    ;;
  new)
    navigation_args=(--navigate-from-main --new-arena)
    ;;
  fresh-shop)
    # The browser is already on a never-played turn-1 shop. Visually bootstrap
    # it without replaying main-menu navigation or loading an old journal.
    navigation_args=(--new-arena)
    ;;
  *)
    echo "Usage: $0 [resume|resume-shop|new|fresh-shop]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing checkpoint: ${checkpoint}" >&2
  exit 1
fi

if ! curl -fsS --max-time 5 "${dashboard}/status.json" >/dev/null; then
  echo "Live dashboard is not reachable at ${dashboard}" >&2
  echo "Start sap-live-browser.service or the recording browser first" >&2
  exit 1
fi

docker rm -f "${container_name}" >/dev/null 2>&1 || true

exec docker run --rm \
  --name "${container_name}" \
  --network host \
  --cpus 8 \
  -e OMP_NUM_THREADS=8 \
  -e MKL_NUM_THREADS=8 \
  -v "${project_root}":/workspace \
  "${image}" \
  python sapai-gym/scripts/live_arena_agent.py \
    "/workspace/${checkpoint#${project_root}/}" \
    --dashboard "${dashboard}" \
    --threads 8 \
    --loop \
    --armed \
    --state-output /workspace/runs/live-agent/current-arena-state.json \
    "${navigation_args[@]}" \
    "${extra_args[@]}"
