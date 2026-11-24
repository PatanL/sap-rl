#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
label=$1;shift
[[ "$label" =~ ^[a-z0-9_-]+$ ]] && [[ -f ../train-live-20261123/STOPPED ]]
extra=()
if [[ "$label" == profile && ! -f results/final-choice.json ]];then python3 final_tune.py >results/final-tune.log 2>&1;fi
if [[ "$label" == proof && ! -f results/earlier-choice.json ]];then python3 before_proof.py >results/before-proof.log 2>&1;fi
if [[ "$label" == profile || "$label" == proof ]];then
 read -r SPEED_FUSED_LOSS backlog SPEED_GEOMETRY games cap < <(python3 -c 'import json;q=json.load(open("results/final-choice.json")); print(int(q["fused"]),q["backlog"],q["geometry"],q["games"],q["qualifier"])')
 export SPEED_FUSED_LOSS SPEED_GEOMETRY
 extra=(--backlog "$backlog" --games "$games" --cap "$cap")
fi
./run.sh collector_async.py --out "results/$label" "$@" "${extra[@]}" >"results/$label.log" 2>&1
python3 summarize.py
