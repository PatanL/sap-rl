#!/bin/bash
# Run a script of this directory in the training image with the GPU (code of earlier work mounted read-only).
# Usage: [SAP_CPUSET=5-9] [SAP_GPU_GIB=8] run_opt.sh <script.py> [args...]    (/out is this directory)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd); cpus=${SAP_CPUSET:-5-9}
count=$(taskset -c "$cpus" python3 -c 'import os;print(len(os.sched_getaffinity(0)))'); mkdir -p "$here/tmp"
exec taskset -c "$cpus" docker run --rm --network none --read-only --cpuset-cpus="$cpus" --cpus="$count" --memory=32g --memory-swap=32g --shm-size=1g \
 --user "$(id -u):$(id -g)" ${OPT_NOGPU:+-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void} $([ -z "${OPT_NOGPU:-}" ] && echo "--gpus device=0") -w /out -v "$here:/out" -v "$root:/sap:ro" \
 -v "$root/runs/replay-oracle-20261122/packaged/v31-20261123g:/engine:ro" \
 -e PYTHONPATH=/out:/sap/runs/rust-collector-20261123:/sap/runs/train-vector-20261123:/sap/engine-rs-pinned/v31g-env2:/sap/engine-rs-pinned/v31g:/sap/runs/train-speed-20261123:/sap/runs/train-eval-20261123:/sap/runs/train-v31-prep-20261123 \
 -e TRITON_CACHE_DIR=/out/tmp/triton -e SPEED_FUSED_LOSS="${SPEED_FUSED_LOSS:-1}" -e SPEED_GEOMETRY="${SPEED_GEOMETRY:-block-both}" -e SPEED_BATCH=/sap/runs/train-speed-20261123/results/batch.pt \
 -e OPT_ENCODER_MODE="${OPT_ENCODER_MODE:-}" -e OPT_CUDNN_GRU="${OPT_CUDNN_GRU:-0}" -e OPT_COMPILE="${OPT_COMPILE:-0}" -e OPT_COMPACT="${OPT_COMPACT:-0}" -e OPT_CAP="${OPT_CAP:-1536}" ${TORCH_LOGS:+-e TORCH_LOGS=$TORCH_LOGS} -e SAP_ENGINE_ROOT=/engine -e SAP_BATTLE_LIBRARY=/disabled/libsap_battle.so -e SAP_GPU_GIB="${SAP_GPU_GIB:-8}" \
 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
 -e TMPDIR=/out/tmp -e MPLCONFIGDIR=/out/tmp/mpl -e CUDA_CACHE_PATH=/out/tmp/cuda -e XDG_CACHE_HOME=/out/tmp/cache -e TORCHINDUCTOR_CACHE_DIR=/out/tmp/inductor -e TORCHINDUCTOR_COMPILE_THREADS=1 \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 "$@"
