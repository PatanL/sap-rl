#!/bin/bash
# Run a reference-side Python tool in the training image on one core (0-4 only).
# Usage: tools/run_ref.sh <cpu 0-4> <tool.py> [args...]     (paths inside the container: /rs = engine-rs)
set -e
ROOT=/home/dzliu/pzliu/sap-rl
CPU=$1; shift
exec docker run --rm --cpus=1 --cpuset-cpus "$CPU" --network none -w /tmp --user "$(id -u):$(id -g)" \
  -v $ROOT/runs/replay-oracle-20261122/packaged/v31-20261123g:/engine:ro -v $ROOT/engine-rs:/rs \
  -e PYTHONPATH=/engine/sapai:/engine/sapai-gym:/rs/tools:/rs/python -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 --entrypoint python sap-rl-shared:research-25772 "$@"
