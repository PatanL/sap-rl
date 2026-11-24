#!/bin/bash
# Training segment with ranked players' boards as a third opponent type (collector_g_ghost.py in this directory).
# Code comes read-only from runs/train-vector-20261123 (Codex's collector_g.py); everything written goes under this directory.
# Usage: SAP_GHOST_SHARE=0.4 SAP_FROZEN_SHARE=0.2 [SAP_BASE_LR=5e-6] [GHOST_AUDIT=/out/<dir>] run_ghost.sh <output name (must not exist)> [collector_g.py options...]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
name=$1; shift
cpus=${SAP_CPUSET:-5-19}
count=$(taskset -c "$cpus" python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
mkdir -p "$here/tmp/triton"
(cd "$root/engine-rs-pinned/v31g" && sha256sum --quiet -c SHA256SUMS)
exec taskset -c "$cpus" docker run --rm --network none --read-only --cpuset-cpus="$cpus" --cpus="$count" --memory=23g --memory-swap=23g --shm-size=1g \
 --user "$(id -u):$(id -g)" --gpus device=0 -w /job \
 -v "$root/runs/train-vector-20261123:/job:ro" -v "$here:/out" -v "$root:/sap:ro" \
 -v "$root/runs/replay-oracle-20261122/packaged/v31-20261123g:/engine:ro" \
 -e SAP_GHOST_SHARE="${SAP_GHOST_SHARE:?set SAP_GHOST_SHARE}" -e SAP_FROZEN_SHARE="${SAP_FROZEN_SHARE:?set SAP_FROZEN_SHARE}" \
 -e SAP_BASE_LR="${SAP_BASE_LR:-}" -e GHOST_AUDIT="${GHOST_AUDIT:-}" \
 -e PYTHONPATH=/job:/sap/engine-rs-pinned/v31g:/sap/runs/train-speed-20261123:/sap/runs/train-eval-20261123:/sap/runs/train-v31-prep-20261123 \
 -e TRITON_CACHE_DIR=/out/tmp/triton -e SPEED_FUSED_LOSS=1 -e SPEED_GEOMETRY=block-both -e SAP_ENGINE_ROOT=/engine \
 -e SAP_BATTLE_LIBRARY=/disabled/libsap_battle.so -e SAP_GPU_GIB="${SAP_GPU_GIB:-8}" \
 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
 -e TMPDIR=/out/tmp -e MPLCONFIGDIR=/out/tmp/mpl -e CUDA_CACHE_PATH=/out/tmp/cuda -e XDG_CACHE_HOME=/out/tmp/cache \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 \
 /out/collector_g_ghost.py --concurrent --priority-inference --workers 12 --games 16 --target 8192 --cap 1536 --max-lag 1 --backlog 0.5 \
 --battle-backend rust --save --out "/out/$name" "$@"
