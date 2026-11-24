#!/usr/bin/env bash
set -eu
audit=$(cd "$(dirname "$0")" && pwd)
cpu=$1; shift
n=$(taskset -c "$cpu" python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
test "$n" -le 4
exec taskset -c 16-19 docker run --rm --network none --read-only --cpuset-cpus "$cpu" --cpus "$n" --memory 4g --memory-swap 4g --shm-size 384m --user "$(id -u):$(id -g)" --workdir /audit \
 -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MPLCONFIGDIR=/audit/tmp/mpl -e TMPDIR=/audit/tmp -e PYTHONPATH=/audit:/workspace/runs/claude-end-positioner-20261112-v1 \
 -v /tmp/sap-rl-restore.5dIqhk:/workspace:ro -v "$audit:/audit" \
 --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 "$@"
