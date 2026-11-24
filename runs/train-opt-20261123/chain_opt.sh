#!/bin/bash
# After the small-network timing: per-call acting cost, the learner-process test, then the mirror-only arm of the frozen-opponent A/B.
cd "$(dirname "$0")"
until grep -q "^done" results/timing-small.log; do sleep 10; done
SAP_CPUSET=5-9 SAP_GPU_GIB=12 ./run_opt.sh /out/act_compile_bench.py /out/results/act-compile-bench.json > results/act-compile-bench.log 2>&1
SAP_CPUSET=5-9,15-19 SAP_GPU_GIB=24 SPEED_GEOMETRY=sorted timeout 500 ./run_opt.sh /out/collect_v3.py --train --learner-process --workers 10 --compact --compile --ghost-share 0 --frozen-share 0 --seconds 300 --arm parent --games 512 --max-lag 3 --backlog 1.0 --cap 3072 --updates 60 --out /out/results/t-v3-parent-512 > results/t-v3-parent-512.log 2>&1
echo "$(date +%T) quick tests done" > results/chain-opt.log
# A/B arm: mirror only, everything else as selfplay-a (collector v2, one process, 512 games, staleness limit 3, from 25,772).
./run_v2.sh mirror-a 2000 --games 512 --max-lag 3 --backlog 1.0 --workers 10 --cap 3072 --compact --compile --ghost-share 0 --frozen-share 0 > results/mirror-a-supervisor.out 2>&1
echo "$(date +%T) mirror-a done" >> results/chain-opt.log
