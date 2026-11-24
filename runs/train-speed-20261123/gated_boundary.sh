#!/bin/bash
set -eu
cd "$(dirname "$0")"
while [[ ! -f ../train-live-20261123/STOPPED ]];do sleep 3;done
CPU_ONLY=1 ./run.sh boundary_bench.py >results/boundary-bench.log 2>&1
