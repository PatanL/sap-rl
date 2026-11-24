#!/usr/bin/env bash
# One bounded replica of the chosen throughput proof. Caller supplies a new output directory.
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# != 1 ]]; then echo "Usage: $0 results/NEW_DIRECTORY" >&2; exit 2; fi
export SPEED_CPUS=5-19
exec ./run.sh collector_g.py --concurrent --priority-inference --workers 12 --games 16 \
    --target 8192 --cap 1536 --max-lag 1 --backlog 0.5 --battle-backend rust \
    --updates 100 --seconds 810 --save --out "$1"
