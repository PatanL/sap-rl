#!/bin/bash
set -eu
cd "$(dirname "$0")"
variant=$1
[[ "$variant" =~ ^[a-zA-Z0-9_-]+$ ]]
if [[ -z "${SAP_ENGINE_ROOT:-}" ]];then
 case "$variant" in
 v31) export SAP_ENGINE_ROOT="$PWD/pinned-v31" ;;
 v30) export SAP_ENGINE_ROOT="$PWD/../search-throughput-5x-20260912-v1/reference-source/sapai" ;;
 *) export SAP_ENGINE_ROOT="$PWD/pinned-$variant" ;;
 esac
fi
[[ -f "$SAP_ENGINE_ROOT/sapai/player.py" ]]
for tier in 1 64 1024 8192 65536;do
 prefix="strict-$variant-$tier"
 args=()
 if [[ "$tier" != 1 ]];then args=(--retry "$previous.jsonl");fi
 if [[ "$tier" == 65536 ]];then args+=(--stochastic-only);fi
 ./run.sh replay.py --seeds "$tier" "${args[@]}" --out "$prefix.jsonl" > "$prefix.log" 2>&1
 previous=$prefix
done
