#!/bin/bash
set -eu
cd "$(dirname "$0")"
opts=(--concurrent --priority-inference --workers 8 --games 24 --target 8192 --cap 1536 --backlog 0.5 --battle-backend rust)
for label in matched profile proof; do
  case "$label" in matched) args=(--updates 16);; profile) args=(--updates 14 --profile);; proof) args=(--updates 100 --save);; esac
  ./run.sh collector_g.py "${opts[@]}" "${args[@]}" --out "results/$label" >"results/$label.log" 2>&1
done
