#!/usr/bin/env bash
# Run the measured configuration for one bounded parent-arm proof.
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# != 1 ]]; then
    echo "Usage: $0 results/NEW_OUTPUT_DIRECTORY" >&2
    exit 2
fi
sha256sum --quiet -c rust-snapshot/SHA256SUMS
export SPEED_GEOMETRY=block-both SPEED_FUSED_LOSS=1 SAP_GPU_GIB=8
exec ./run.sh collector_async.py --concurrent --priority-inference \
    --workers 8 --games 24 --target 8192 --cap 1536 --backlog 0.5 \
    --battle-backend rust --max-lag 1 --updates 100 --seconds 810 \
    --save --audit-battles --out "$1"
