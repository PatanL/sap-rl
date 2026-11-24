#!/bin/bash
set -eu
for cap in 1 64 1024;do
 args=()
 if [[ "$cap" != 1 ]];then args=(--retry "$prev");fi
 ./run.sh replay.py --full-shop --seeds "$cap" "${args[@]}" --out "full-$cap.jsonl" > "full-$cap.log" 2>&1
 prev="full-$cap.jsonl"
done
