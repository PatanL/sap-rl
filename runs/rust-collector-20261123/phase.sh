#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Keep the supervisor and all pre-release work on the authorized CPU set.
if ! taskset -pc $$ | grep -q '0-4$'; then exec taskset -c 0-4 "$0" "$@"; fi
until [[ -f ../train-live-20261123/GPU-FREE && -s results/parity-256.json ]]; do sleep 20; done
python3 -c 'import json;assert json.load(open("results/parity-256.json"))["exact"]'
nvidia-smi --query-gpu=timestamp,name,utilization.gpu --format=csv >results/gpu-release.csv
for games in 1024 4096; do
 d="results/act-$games"
 if [[ ! -s "$d/result.json" ]];then ./run.sh collect_rust.py --games "$games" --seconds 115 --workers 10 --out "$d" >"$d.log" 2>&1;fi
done
for games in 128 192 256 384 512 1024; do
 d="results/train-$games"
 if [[ ! -s "$d/result.json" ]];then ./run.sh collect_rust.py --train --games "$games" --updates 24 --seconds 750 --workers 10 --out "$d" >"$d.log" 2>&1;fi
done
games=$(python3 select_width.py)
if [[ ! -s results/proof/result.json ]];then ./run.sh collect_rust.py --train --games "$games" --updates 100 --seconds 810 --workers 10 --save --out results/proof >results/proof.log 2>&1;fi
checkpoint=$(python3 -c 'import json,pathlib;r=json.load(open("results/proof/result.json"));assert r["updates"]==100;print(next(pathlib.Path("results/proof").glob("update-*.pt")))')
if [[ ! -s results/strength.json ]];then ./run.sh /sap/runs/train-live-20261123/evaluate_vec.py "/job/$checkpoint" /sap/checkpoints/update-025772.pt results/strength.json --mode confirm --pairs 2048 --batch 4096 --threads 10 --gib 8 >results/strength.log 2>&1;fi
python3 board_phase.py "$checkpoint" >results/board-phase.log 2>&1
