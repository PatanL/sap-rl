#!/bin/bash
# Yardstick-side full matches against ranked boards (runs/player-analysis-20261123/ghost_match.py, greedy, default split = yardstick).
# Usage: fullmatch.sh <name> <checkpoint path under /sap> [matches=2048]
set -euo pipefail
root=/home/dzliu/pzliu/sap-rl; O=/sap/runs/replay-oracle-20261122; name=$1; ck=$2; n=${3:-2048}; cpus=${SAP_CPUSET:-0-4}
out=/sap/runs/train-ghost-20261123/fullmatch/$name.json; mkdir -p $root/runs/train-ghost-20261123/fullmatch
[[ -f $root/runs/train-ghost-20261123/fullmatch/$name.json ]] && exit 0
taskset -c "$cpus" docker run --rm --network none -w /tmp --user "$(id -u):$(id -g)" --gpus device=0 --cpuset-cpus "$cpus" -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
 -v $root:/sap -v $root/runs/replay-oracle-20261122/packaged/v31-20261123g:/engine:ro -e PYTHONPATH=/engine/sapai:/engine/sapai-gym \
 --entrypoint python sap-rl-shared:research-25772 /sap/runs/player-analysis-20261123/ghost_match.py "$ck" "$out" "$n" \
 $O/fixtures-340.json $O/fixtures-fresh-all.json $O/fixtures-fresh2-all.json $O/fixtures-fresh3-all.json $O/fixtures-players.json --greedy --batch 256 2>&1 | grep '^{"matches"' | tail -n1
