#!/bin/bash
set -eu
cd "$(dirname "$0")"
for g in 16 24 40;do
 lab=cohort12x$g
 SPEED_CPUS=5-19 ./run.sh collector_g.py --concurrent --priority-inference --workers 12 --games "$g" --target 8192 --cap 1536 --backlog .5 --battle-backend rust --updates 20 --out "results/$lab" >"results/$lab.log" 2>&1
done
