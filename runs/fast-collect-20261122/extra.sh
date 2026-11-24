#!/usr/bin/env bash
set -eu
while ! python3 -c 'import json;from pathlib import Path;p=Path("actor-exact.json");assert p.is_file() and len(json.loads(p.read_text()))==150' 2>/dev/null; do sleep 3;done
for size in mid medium small;do
 ./run.sh async_collector.py --model "$size" --workers 8 --games 96 --cohorts 2 --seconds 70 --tape --out "pipe96-$size" > "pipe96-$size.log" 2>&1
done
