#!/bin/bash
set -eu
cd "$(dirname "$0")"
while [[ ! -f results/progression.json ]] || ! python3 -c 'import json;assert json.load(open("results/progression.json"))["checkpoints"]' 2>/dev/null;do sleep 5;done
mapfile -t paths < <(python3 - <<'PY'
from pathlib import Path
import json
r=json.load(open('results/progression.json'))
for m in (r['board_baseline'],r['checkpoints'][-1]['board_strength']):
 print((Path(m['receipt']).parent/'seats-battles.jsonl').relative_to(Path.cwd()))
PY
)
CPU_ONLY=1 SPEED_CPUS=5-9 ./run.sh paired_boards.py "${paths[@]}" >results/paired-board.log 2>&1
