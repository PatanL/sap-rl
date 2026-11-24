#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export SAP_ENGINE_ROOT="$PWD/pinned-handoff"
for cap in 1 64 1024;do
 args=()
 if [[ "$cap" != 1 ]];then args=(--retry "$prev");fi
 ./run.sh replay.py --full-shop --seeds "$cap" "${args[@]}" --out "handoff/full-$cap.jsonl" > "handoff/full-$cap.log" 2>&1
 prev="handoff/full-$cap.jsonl"
done
