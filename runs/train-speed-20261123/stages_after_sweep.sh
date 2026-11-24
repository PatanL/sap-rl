#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
test -f ../train-live-20261123/STOPPED
nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv,noheader,nounits --loop-ms=500 >results/original-gpu.csv &
meter=$!
trap 'kill "$meter" 2>/dev/null || true; wait "$meter" 2>/dev/null || true' EXIT
# The original training entry point/rates, unchanged. Its receipt includes startup and final persistence.
./run.sh /sap/runs/train-eval-20261123/collector_join.py train --arm parent --parent /sap/checkpoints/update-025772.pt --workers 8 --games 96 --transitions 8192 --updates 18 --checkpoint-every 250 --max-padded-rows 128 --frozen-share .5 --pool-size 4 --output /job/results/original >results/original.log 2>&1
