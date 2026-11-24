#!/bin/bash
# Launch a long training run with Codex's frozen-opponent collector.
# Code is mounted read-only from runs/train-eval-20261123; everything written goes under this directory.
# Usage: run_train.sh <output name> [collector_join.py train options...]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
name=$1; shift
engine=${SAP_ENGINE_ROOT:-"$root/runs/replay-oracle-20261122/packaged/v31-20261123c"}
cpus=${SAP_CPUSET:-10-15,17-19}
count=$(taskset -c "$cpus" python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
mkdir -p "$here/tmp"
exec taskset -c "$cpus" docker run --rm --network none --read-only --cpus "$count" --cpuset-cpus "$cpus" --memory 16g --memory-swap 16g --shm-size 1g \
 --user "$(id -u):$(id -g)" --gpus device=0 -w /job \
 -v "$root/runs/train-eval-20261123:/job:ro" -v "$here:/out" -v "$root:/sap:ro" -v "$engine:/engine:ro" \
 -e SAP_ENGINE_ROOT=/engine -e SAP_GPU_GIB="${SAP_GPU_GIB:-6}" -e SAP_BATTLE_LIBRARY=/disabled/libsap_battle.so \
 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
 -e TMPDIR=/out/tmp -e MPLCONFIGDIR=/out/tmp/mpl -e CUDA_CACHE_PATH=/out/tmp/cuda -e XDG_CACHE_HOME=/out/tmp/cache \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 \
 collector_join.py train --arm parent --parent /sap/checkpoints/update-025772.pt --output "/out/$name" "$@"
