#!/usr/bin/env bash
set -eu
d=$(cd "$(dirname "$0")" && pwd)
root=$(realpath "$d/../..")
engine=${SAP_ENGINE_ROOT:-"$root/runs/search-throughput-5x-20260912-v1/reference-source"}
parent_checkpoint=${SAP_PARENT_CHECKPOINT:-/tmp/sap-rl-restore.5dIqhk/runs/ordinary-rl-shared-20260924-v2/outputs/chunk-20/update-025772.pt}
test -f "$parent_checkpoint"
test -f "$engine/sapai/sapai/battle.py"
test -f "$engine/sapai-gym/sapai_gym1/rl/model.py"
# Both packages are mounted as a unit. Repoint SAP_ENGINE_ROOT to a fixed tree.
cpuset=${BENCH_CPUSET:-5-9,16-18}
n=$(taskset -c "$cpuset" python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
test "$n" -le 8
gpu=(--gpus device=0)
if [[ ${BENCH_CPU_ONLY:-0} == 1 ]]; then gpu=(-e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=); fi
exec taskset -c "$cpuset" docker run --rm --network none --read-only "${gpu[@]}" --cpus "$n" --cpuset-cpus "$cpuset" \
 --shm-size 1g --memory 12g --memory-swap 12g --user "$(id -u):$(id -g)" -w /audit \
 -e SAP_ENGINE_ROOT=/engine -e TMPDIR=/audit/tmp -e MPLCONFIGDIR=/audit/tmp/mpl -e XDG_CACHE_HOME=/audit/tmp/cache \
 -e CUDA_CACHE_PATH=/audit/tmp/cuda -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
 -e PYTHONPATH=/audit -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
 -v "$d:/audit" -v "$engine:/engine:ro" -v "$root/runs/population-league-25772-v1/inputs:/examples:ro" -v "$root/runs/population-league-25772-v1/residents:/residents:ro" \
 -v "$parent_checkpoint:/weights/parent.pt:ro" \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 "$@"
