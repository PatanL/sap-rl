#!/bin/bash
set -eu
d=$(cd "$(dirname "$0")/.." && pwd)
runs=/home/dzliu/pzliu/sap-rl/runs
exec taskset -c 5-9,16 docker run --rm --network none --read-only --cpuset-cpus=5-9,16 --cpus=6 --memory=7g --user "$(id -u):$(id -g)" -w /audit -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e SAP_ENGINE_ROOT=/engine -e TMPDIR=/audit/tmp -e PYTHONPATH=/audit:/engine:/gym -v "/home/dzliu/pzliu/sap-rl/runs/replay-oracle-20261122/engine-v31:/engine:ro" -v "$d:/audit" -v "$runs/replay-oracle-20261122:/oracle:ro" -v "$runs/expert-replay-ingest-20260926-v1/corpus/matches:/archive:ro" -v "$runs/search-throughput-5x-20260912-v1/reference-source/sapai-gym:/gym:ro" --entrypoint python "$(cat "$d/image-id.txt")" "$@"
