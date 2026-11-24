#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
read -r cap compact geom < <(python3 -c 'import json;x=json.load(open("results/qualified.json"));print(x["cap"],int(x["compact"]),x["layout"])')
export SPEED_GEOMETRY="$geom"
opts=(--cap "$cap" --updates 18 --target 8192 --workers 8)
if [[ $compact == 1 ]];then opts+=(--compact);fi
./trial.sh sync96 "${opts[@]}" --games 96 --fingerprints
./trial.sh async96 "${opts[@]}" --games 96 --concurrent --fingerprints
./trial.sh actor96 "${opts[@]}" --games 96 --concurrent --fast-actor
