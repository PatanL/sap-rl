#!/usr/bin/env bash
set -eu
while [ ! -s wide-parent-core-cap.json ];do sleep 3;done
sleep 2
BENCH_CPUSET=16-18 ./run.sh actor_exact.py > actor-exact.log 2>&1
