#!/bin/bash
set -eu
cd "$(dirname "$0")"
while [[ ! -e results/tape/receipt.json ]]; do
  if grep -qE 'Traceback|Error:' results/tape.log;then exit 1;fi
  sleep 3
done
for mode in replay python; do
  for lane in act train; do
    arg=(); if [[ "$lane" == train ]];then arg=(--train);fi
    lab=v-$mode-$lane
    SPEED_CPUS=5-19 ./run.sh vector_collect.py --backend "$mode" --workers 12 --games 1024 --seconds 155 "${arg[@]}" --out "results/$lab" >"results/$lab.log" 2>&1
  done
done
