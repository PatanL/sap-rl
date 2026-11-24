#!/bin/bash
# Run one of Codex's scoring scripts (mounted read-only from runs/train-vector-20261123) with output under this directory.
# Usage: SAP_CPUSET=<cores> score_g.sh <script.py> [args...]   (paths under /out are this directory, /sap the project root)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cpus=${SAP_CPUSET:-5-9}
count=$(taskset -c "$cpus" python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
exec taskset -c "$cpus" docker run --rm --network none --read-only --cpuset-cpus="$cpus" --cpus="$count" --memory=12g --memory-swap=12g --shm-size=1g \
 --user "$(id -u):$(id -g)" --gpus device=0 -w /job \
 -v "$root/runs/train-vector-20261123:/job:ro" -v "$here:/out" -v "$root:/sap:ro" \
 -v "$root/runs/replay-oracle-20261122/packaged/v31-20261123g:/engine:ro" \
 -e PYTHONPATH=/job:/sap/engine-rs-pinned/v31g:/sap/runs/train-speed-20261123:/sap/runs/train-eval-20261123:/sap/runs/train-v31-prep-20261123 \
 -e TRITON_CACHE_DIR=/out/tmp/triton -e SPEED_FUSED_LOSS=1 -e SPEED_GEOMETRY=block-both -e SAP_ENGINE_ROOT=/engine \
 -e SAP_BATTLE_LIBRARY=/disabled/libsap_battle.so \
 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
 -e TMPDIR=/out/tmp -e MPLCONFIGDIR=/out/tmp/mpl -e CUDA_CACHE_PATH=/out/tmp/cuda -e XDG_CACHE_HOME=/out/tmp/cache \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 "$@"
