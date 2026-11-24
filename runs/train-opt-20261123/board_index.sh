#!/bin/bash
# Ranked-board index of a checkpoint through the Rust recorder: record 1,000 greedy self-play matches, then fight the
# recorded boards against the yardstick side of the ranked pool (ghost_g.py of runs/train-vector-20261123, unchanged).
# Usage: board_index.sh <name> <checkpoint path under /sap> [seed]     -> results/board/<name>.json and prints the index
set -euo pipefail
cd "$(dirname "$0")"; name=$1; ck=$2; seed=${3:-20261123}; mkdir -p results/board
root=$(cd ../.. && pwd); O=/sap/runs/replay-oracle-20261122
if [[ ! -f results/board/$name.json ]]; then
  [[ -f results/board/$name-battles.jsonl ]] || SAP_CPUSET=${SAP_CPUSET:-0-4} ./run_opt.sh /out/record_boards_rust.py "$ck" /out/results/board/$name-battles.jsonl --seed "$seed" > results/board/$name-record.log 2>&1
  fixtures=$(python3 -c "import json;print(' '.join('$O/'+f for f in json.load(open('../train-vector-20261123/results/fixtures.json'))))")
  mkdir -p tmp; : >> tmp/gpu.lock
  taskset -c ${SAP_CPUSET:-0-4} docker run --rm --network none --read-only --cpuset-cpus=${SAP_CPUSET:-0-4} --user "$(id -u):$(id -g)" -w /job \
    -v "$PWD:/job" -v "$root:/sap:ro" -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR=/job/tmp \
    --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 /sap/runs/train-vector-20261123/ghost_g.py \
    results/board/$name-battles.jsonl results/board/$name.json maxwu 4000 $fixtures -- $O/raw $O/raw-fresh $O/raw-fresh2 $O/raw-fresh3 $O/raw-players > results/board/$name-ghost.log 2>&1
fi
python3 -c "
import json;g=json.load(open('results/board/$name.json'));assert g['split']=='yardstick'
s={t:g['comparisons']['bot v field'][t]['score'] for t in ('3','5','7','10','12')}
print('$name', 'index %.2f'%(100*sum(s.values())/5), {t:round(100*v,1) for t,v in s.items()})"
