#!/usr/bin/env bash
set -euo pipefail

host_root=/home/dzliu/pzliu/sap-rl
runs_dir="${host_root}/runs"
v18_run="${runs_dir}/turtle-league-v18-audit-transfer-v17-1300-to5000"
v19_run="${runs_dir}/turtle-league-v19-opponent-transfer-v18-5000"
v18_checkpoint="${v18_run}/update-005000.pt"
v19_image=sap-rl-v19-frozen:20260831
v18_image=sap-rl-v18-frozen:20260831
v18_container=sap-rl-v18-audit-transfer-v17-1300-to5000
v19_container=sap-rl-v19-opponent-transfer-v18-5000
v18_recovery_script="${host_root}/scripts/run_frozen_v18_to5000.sh"
promote_above="${V19_PROMOTE_ABOVE:-0.52}"
reject_below="${V19_REJECT_BELOW:-0.48}"

wait_for_file() {
  local path="$1"
  while [[ ! -f "${path}" ]]; do
    sleep 30
  done
}

wait_for_container_checkpoint() {
  local container="$1" path="$2"
  while [[ ! -f "${path}" ]]; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null || true)" != "true" ]]; then
      echo "${container} exited before producing ${path}" >&2
      docker logs --tail 100 "${container}" >&2 2>/dev/null || true
      return 1
    fi
    sleep 30
  done
}

wait_for_exit() {
  local container="$1"
  while [[ "$(docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null || true)" == "true" ]]; do
    sleep 15
  done
}

remove_container_if_present() {
  local container="$1"
  if docker inspect "${container}" >/dev/null 2>&1; then
    docker rm "${container}" >/dev/null
  fi
}

start_v19() {
  local target="$1"
  if [[ "$(docker inspect -f '{{.State.Running}}' "${v19_container}" 2>/dev/null || true)" == "true" ]]; then
    echo "v19 already running; retaining live target while awaiting update ${target}"
    return
  fi
  remove_container_if_present "${v19_container}"
  docker run -d \
    --name "${v19_container}" \
    --restart=no \
    --gpus all \
    --ipc=host \
    -e V19_TARGET_UPDATES="${target}" \
    -v "${runs_dir}":/workspace/runs \
    "${v19_image}" \
    bash scripts/run_v19_opponent_transfer_v18_5000.sh >/dev/null
  echo "started v19 target=${target}"
}

run_gate() {
  local update="$1"
  docker run --rm \
    --network none \
    -e V19_GATE_UPDATE="${update}" \
    -e V19_GATE_PAIRS=100 \
    -v "${runs_dir}":/workspace/runs \
    "${v19_image}" \
    bash scripts/evaluate_v19_gate.sh
}

gate_score() {
  local update="$1"
  python3 - "${v19_run}/gate/paired-$(printf '%06d' "${update}")-vs-v18-005600.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["pairedMeanScore"])
PY
}

classify_score() {
  local score="$1"
  python3 - "${score}" "${promote_above}" "${reject_below}" <<'PY'
import sys
score, promote, reject = map(float, sys.argv[1:])
print("promote" if score >= promote else "reject" if score <= reject else "extend")
PY
}

start_v18_fallback() {
  remove_container_if_present "${v18_container}"
  docker run -d \
    --name "${v18_container}" \
    --restart=unless-stopped \
    --gpus all \
    --ipc=host \
    -v "${runs_dir}":/workspace/runs \
    "${v18_image}" \
    bash scripts/run_v18_audit_transfer_v17_1300_to5000.sh >/dev/null
  echo "v19 rejected; resumed frozen v18 from update 5000"
}

complete_v18_fallback() {
  start_v18_fallback
  wait_for_container_checkpoint \
    "${v18_container}" \
    "${v18_run}/update-008300.pt"
  wait_for_exit "${v18_container}"
  echo "frozen v18 fallback completed update 8300"
}

start_v18_to_transfer_boundary() {
  remove_container_if_present "${v18_container}"
  docker run -d \
    --name "${v18_container}" \
    --restart=no \
    --gpus all \
    --ipc=host \
    -v "${runs_dir}":/workspace/runs \
    -v "${v18_recovery_script}":/workspace/run_frozen_v18_to5000.sh:ro \
    "${v18_image}" \
    bash /workspace/run_frozen_v18_to5000.sh >/dev/null
  echo "recovered frozen v18 toward update 5000"
}

if [[ ! -f "${v18_checkpoint}" ]] \
  && [[ "$(docker inspect -f '{{.State.Running}}' "${v18_container}" 2>/dev/null || true)" != "true" ]]; then
  start_v18_to_transfer_boundary
fi
if ! wait_for_container_checkpoint "${v18_container}" "${v18_checkpoint}"; then
  echo "v18 did not reach the transfer boundary; refusing a partial v19 handoff" >&2
  exit 1
fi
wait_for_exit "${v18_container}"
start_v19 1000
if ! wait_for_container_checkpoint "${v19_container}" "${v19_run}/update-001000.pt"; then
  complete_v18_fallback
  exit 0
fi
wait_for_exit "${v19_container}"
if ! run_gate 1000; then
  complete_v18_fallback
  exit 0
fi
score="$(gate_score 1000)"
decision="$(classify_score "${score}")"
echo "v19 gate update=1000 score=${score} decision=${decision}"

if [[ "${decision}" == "extend" ]]; then
  start_v19 2000
  if ! wait_for_container_checkpoint "${v19_container}" "${v19_run}/update-002000.pt"; then
    complete_v18_fallback
    exit 0
  fi
  wait_for_exit "${v19_container}"
  if ! run_gate 2000; then
    complete_v18_fallback
    exit 0
  fi
  score="$(gate_score 2000)"
  decision="$(python3 - "${score}" <<'PY'
import sys
print("promote" if float(sys.argv[1]) >= 0.51 else "reject")
PY
)"
  echo "v19 gate update=2000 score=${score} decision=${decision}"
fi

if [[ "${decision}" == "promote" ]]; then
  docker stop sap-v18-progression-eval >/dev/null 2>&1 || true
  start_v19 5000
  echo "v19 promoted; continuing through update 5000"
  wait_for_container_checkpoint \
    "${v19_container}" \
    "${v19_run}/update-005000.pt"
  wait_for_exit "${v19_container}"
  echo "v19 completed update 5000"
else
  complete_v18_fallback
fi
