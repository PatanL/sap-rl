#!/bin/bash
set -euo pipefail
job=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$job/../.." && pwd)
dev=(--gpus device=0)
if [[ "${SCORER_CPU:-0}" == 1 ]];then dev=(-e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=);fi
exec taskset -c 5-9,16 nice -n 10 docker run --rm --network none --read-only --cpus=6 --cpuset-cpus=5-9,16 --memory=12g --memory-swap=12g --shm-size=1g --user "$(id -u):$(id -g)" \
 "${dev[@]}" -w /job -v "$job:/job" -v "$root:/sap:ro" -v "$root/runs/replay-oracle-20261122/packaged/v31-20261123c:/engine:ro" \
 -e SAP_ENGINE_ROOT=/engine -e SAP_GPU_GIB=2 -e SAP_BATTLE_LIBRARY=/disabled/libsap_battle.so -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e TMPDIR=/job/tmp -e MPLCONFIGDIR=/job/tmp/mpl -e CUDA_CACHE_PATH=/job/tmp/cuda -e XDG_CACHE_HOME=/job/tmp/cache \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 "$@"
