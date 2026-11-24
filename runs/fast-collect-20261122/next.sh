#!/usr/bin/env bash
set -eu
while [ ! -s wide-medium-tape.npz ];do sleep 3;done
sleep 2
for entry in '3 4c-narrow-parent' '48 4c-wide-parent'; do
 read -r games label <<< "$entry"
 BENCH_CPUSET=5-8 ./run.sh collector.py --model parent --workers 4 --games "$games" --seconds 60 --warm-factor 1 --out "$label" > "$label.log" 2>&1
done
./run.sh async_collector.py --model mid --workers 8 --games 48 --cohorts 2 --seconds 80 --tape --out pipe-mid > pipe-mid.log 2>&1
./run.sh async_collector.py --model parent --workers 8 --games 48 --cohorts 2 --seconds 80 --tape --out pipe-parent > pipe-parent.log 2>&1
./run.sh collector.py --model parent --random-parent --workers 8 --games 48 --seconds 65 --warm-factor 1 --out wide-parent-random --tape > wide-parent-random.log 2>&1
