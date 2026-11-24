#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
engine=${SAP_ENGINE_ROOT:-"$here/engine"}
parent=${SAP_PARENT_CHECKPOINT:-"$here/../../checkpoints/update-025772.pt"}
cpus=${SAP_CPUSET:-5-9,16}
count=$(taskset -c "$cpus" python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
[[ -r "$engine/MANIFEST.json" && -r "$parent" ]]
case "${1:-}" in
 inspect) gpu=(-e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void);;
 dry|train) gpu=(--gpus device=0);;
 *) echo 'Usage: run.sh inspect|dry|train --arm parent|fresh [options]' >&2;exit 2;;
esac
mkdir -p "$here/tmp" "$here/results"
exec taskset -c "$cpus" docker run --rm --network none --read-only --cpus "$count" --cpuset-cpus "$cpus" --memory 12g --memory-swap 12g --shm-size 1g --user "$(id -u):$(id -g)" \
 "${gpu[@]}" -v "$here:/work:ro" -v "$here/results:/work/results" -v "$here/tmp:/work/tmp" -v "$engine:/engine:ro" -v "$parent:/weights/parent.pt:ro" \
 -w /work -e SAP_ENGINE_ROOT=/engine -e SAP_BATTLE_LIBRARY=/disabled/libsap_battle.so -e SAP_GPU_GIB="${SAP_GPU_GIB:-1.5}" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/work/tmp/mpl -e CUDA_CACHE_PATH=/work/tmp/cuda -e XDG_CACHE_HOME=/work/tmp/cache -e TMPDIR=/work/tmp \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 /work/join.py "$@"
