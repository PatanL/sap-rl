#!/bin/bash
set -eu
cd "$(dirname "$0")"
for be in replay python; do
  label=v-$be-loose
  SPEED_CPUS=5-19 ./run.sh vector_collect.py --backend "$be" --train --games 1024 --workers 12 --seconds 135 --backlog 4 --out "results/$label" >"results/$label.log" 2>&1
done
