#!/bin/bash
set -euo pipefail
job=$(cd "$(dirname "$0")" && pwd);root=$(cd "$job/../.." && pwd)
gpu=(); guard=()
if [[ "${CPU_ONLY:-0}" == 1 ]];then
  cpus=0-4; if [[ -f "$root/runs/train-live-20261123/GPU-FREE" && "${AFTER_RELEASE:-0}" == 1 ]]; then cpus=5-19; fi;gpu=(-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void)
else
  [[ -f "$root/runs/train-live-20261123/GPU-FREE" ]] || { echo 'Waiting for operator GPU-FREE marker; GPU timing disabled.' >&2;exit 3; }
  while [[ ! -s "$job/results/parity-production.json" ]]; do
    if grep -q '^Traceback' "$job/results/parity-production.log"; then echo 'Production parity check failed.' >&2; exit 4; fi
    sleep 10
  done
  exec 9>"$job/tmp/gpu.lock"; flock -x 9
  cpus=5-19;gpu=(--gpus device=0);guard=(bounded.py)
fi
n=$(taskset -c "$cpus" python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
(cd "$root/engine-rs-pinned/v31g" && sha256sum --quiet -c SHA256SUMS)
(cd "$root/engine-rs-pinned/v31g-env" && sha256sum --quiet -c "$job/results/native.SHA256SUMS")
exec taskset -c "$cpus" docker run --rm --network none --read-only --cpuset-cpus="$cpus" --cpus="$n" --memory=23g --memory-swap=23g --shm-size=1g --user "$(id -u):$(id -g)" \
 "${gpu[@]}" -w /job -v "$job:/job" -v "$root:/sap:ro" -v "$root/runs/replay-oracle-20261122/packaged/v31-20261123g:/engine:ro" \
 -v "$root/checkpoints/update-025772.pt:/weights/parent.pt:ro" -e PYTHONPATH=/job:/sap/runs/train-vector-20261123:/sap/engine-rs-pinned/v31g-env:/sap/engine-rs-pinned/v31g:/sap/runs/train-speed-20261123:/sap/runs/train-eval-20261123:/sap/runs/train-v31-prep-20261123 -e TRITON_CACHE_DIR=/job/tmp/triton -e SPEED_FUSED_LOSS="${SPEED_FUSED_LOSS:-1}" -e SPEED_GEOMETRY="${SPEED_GEOMETRY:-block-both}" -e SPEED_BATCH="${SPEED_BATCH:-results/batch.pt}" -e SAP_ENGINE_ROOT=/engine -e SAP_BATTLE_LIBRARY=/disabled/libsap_battle.so -e SAP_GPU_GIB="${SAP_GPU_GIB:-8}" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e TMPDIR=/job/tmp -e MPLCONFIGDIR=/job/tmp/mpl -e CUDA_CACHE_PATH=/job/tmp/cuda -e XDG_CACHE_HOME=/job/tmp/cache \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 "${guard[@]}" "$@"
