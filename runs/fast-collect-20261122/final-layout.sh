#!/usr/bin/env bash
set -eu
while [ ! -s pipe96-small-update.json ];do sleep 3;done
sleep 2
./run.sh async_collector.py --model parent --workers 8 --games 96 --cohorts 2 --seconds 70 --tape --out pipe96-parent > pipe96-parent.log 2>&1
# Same-policy, same-action replay ceiling for the proposed mid-size configuration.
BENCH_CPU_ONLY=1 ./run.sh replay_cap.py pipe96-mid > pipe96-mid-cap-obs.log 2>&1
