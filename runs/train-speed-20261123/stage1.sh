#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
test -f ../train-live-20261123/STOPPED
date -u +%FT%TZ >results/timing-start-utc
nvidia-smi --query-gpu=name,utilization.gpu,temperature.gpu --format=csv >results/gpu-release.csv
./run.sh actor_test.py --device cuda >results/actor-gpu.log 2>&1
./run.sh equivalence.py --prec fp32 --n 8192 --cap 384 --out results/equiv-fp32.json >results/equiv-fp32.log 2>&1
./run.sh equivalence.py --prec bf16 --n 8192 --cap 512 --out results/equiv-bf16.json >results/equiv-bf16.log 2>&1
./run.sh learner_bench.py --sizes ref,128,256,384,512,768,1024,1536 >results/sweep.log 2>&1
./run.sh learner_bench.py --compact --sizes 256,512,768,1024,1536 --out results/compact-sweep.json >results/compact-sweep.log 2>&1
cap=$(python3 choose.py results/compact-sweep.json)
./run.sh equivalence.py --compact --prec bf16 --n 8192 --cap "$cap" --out results/equiv-compact-bf16.json >results/equiv-compact-bf16.log 2>&1
./run.sh equivalence.py --compact --prec fp32 --n 8192 --cap 384 --out results/equiv-compact-fp32.json >results/equiv-compact-fp32.log 2>&1
./stages_after_sweep.sh
