#!/bin/bash
set -euo pipefail
job=$(cd "$(dirname "$0")" && pwd);root=$(cd "$job/../.." && pwd)
gpu=(); guard=()
if [[ "${CPU_ONLY:-0}" == 1 ]];then
  cpus=5-9,16;gpu=(-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void)
else
  [[ -f "$root/runs/train-live-20261123/STOPPED" ]] || { echo 'Waiting for operator STOPPED marker; GPU timing disabled.' >&2;exit 3; }
  exec 9>"$job/tmp/gpu.lock"; flock -x 9
  cpus=${SPEED_CPUS:-10-15,17-19};gpu=(--gpus device=0);guard=(gpu_guard.py)
fi
n=$(taskset -c "$cpus" python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
exec taskset -c "$cpus" docker run --rm --network none --read-only --cpuset-cpus="$cpus" --cpus="$n" --memory=23g --memory-swap=23g --shm-size=1g --user "$(id -u):$(id -g)" \
 "${gpu[@]}" -w /job -v "$job:/job" -v "$root:/sap:ro" -v "$root/runs/replay-oracle-20261122/packaged/v31-20261123c:/engine:ro" \
 -e PYTHONPATH=/job:/sap/runs/train-eval-20261123:/sap/runs/train-v31-prep-20261123 -e TRITON_CACHE_DIR=/job/tmp/triton -e SPEED_FUSED_LOSS="${SPEED_FUSED_LOSS:-0}" -e SPEED_GEOMETRY="${SPEED_GEOMETRY:-sorted}" -e SPEED_BATCH="${SPEED_BATCH:-results/batch.pt}" -e SAP_ENGINE_ROOT=/engine -e SAP_BATTLE_LIBRARY=/disabled/libsap_battle.so -e SAP_GPU_GIB="${SAP_GPU_GIB:-8}" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e TMPDIR=/job/tmp -e MPLCONFIGDIR=/job/tmp/mpl -e CUDA_CACHE_PATH=/job/tmp/cuda -e XDG_CACHE_HOME=/job/tmp/cache \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 "${guard[@]}" "$@"
