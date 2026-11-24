#!/bin/bash
set -eu
cd "$(dirname "$0")"
while [[ ! -f results/v-replay-act/result.json ]];do sleep 5;done
for spec in 'replay 4096 150 act' 'python 256 120 train' 'python 512 120 train' 'python 128 120 train';do
  read -r backend games duration lane <<<"$spec"
  label="v-$backend-$lane-$games"
  arg=();if [[ "$lane" == train ]]; then arg=(--train);fi
  SPEED_CPUS=5-19 ./run.sh vector_collect.py --backend "$backend" --workers 12 --games "$games" --seconds "$duration" "${arg[@]}" --out "results/$label" >"results/$label.log" 2>&1
done
