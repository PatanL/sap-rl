#!/bin/bash
set -eu
for tier in 1 64 1024 8192;do
 prefix="dev-r$tier"
 args=()
 if [[ "$tier" != 1 ]];then args=(--retry "$previous.jsonl");fi
 ./run.sh replay.py --split dev --seeds "$tier" "${args[@]}" --out "$prefix.jsonl" > "$prefix.log" 2>&1
 previous=$prefix
done
