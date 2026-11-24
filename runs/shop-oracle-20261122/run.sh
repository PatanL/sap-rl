#!/bin/bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
engine=${SAP_ENGINE_ROOT:-"$here/pinned-v31"}
[[ -f "$engine/sapai/player.py" ]]
exec taskset -c 5-9,16 docker run --rm --network none --read-only --cpuset-cpus=5-9,16 --cpus=6 --memory 6g --user "$(id -u):$(id -g)" -w /audit -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e SAP_ENGINE_ROOT=/engine -e TMPDIR=/audit/tmp -v "$engine:/engine:ro" -v "$here:/audit" --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 "$@"
