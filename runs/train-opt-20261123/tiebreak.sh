#!/bin/bash
# Last three checkpoints of both arms, each re-recorded on the same five self-play seeds (rule amended 16:48).
cd "$(dirname "$0")"
for seed in 20261201 20261202 20261203 20261204 20261205; do
  for run in ranked-a selfplay-a; do
    for u in 27272 27522 27772; do
      SAP_CPUSET=15-19 ./board_index.sh tb-$run-$u-s$seed /out/results/$run/seg-1/update-0$u.pt $seed >> results/tiebreak.log 2>&1
    done
  done
  SAP_CPUSET=15-19 ./board_index.sh tb-c25772-s$seed /sap/checkpoints/update-025772.pt $seed >> results/tiebreak.log 2>&1
done
echo done >> results/tiebreak.log
