#!/bin/bash
# Usage: top_index.sh <name of a recording in runs/train-opt-20261123/results/board (without -battles.jsonl)>
cd "$(dirname "$0")"; n=$1; B=/home/dzliu/pzliu/sap-rl/runs/train-opt-20261123/results/board; mkdir -p out
[[ -f out/$n-fights.json ]] || { python3 -I top_index.py build $B/$n-battles.jsonl out/$n-pairs.json && docker run --rm --network none --cpuset-cpus 0-4 --user $(id -u):$(id -g) -v /home/dzliu/pzliu/sap-rl:/sap -w /sap/runs/player-analysis-20261123/rank --entrypoint python sap-rl-shared:research-25772 /sap/runs/train-vector-20261123/ghost_fights.py --fixtures out/$n-pairs.json --out out/$n-fights.json --data /sap/engine-rs-pinned/v31g/data --samples 8 --threads 5 >/dev/null 2>&1; }
echo -n "$n  "; python3 -I top_index.py report out/$n-pairs.json out/$n-fights.json
