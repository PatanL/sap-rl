#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
proof=$1
candidate=$(ls "$proof"/update-*.pt | sort | tail -1)
[[ -n "$candidate" && -f ../train-live-20261123/STOPPED ]]
parent=/sap/checkpoints/update-025772.pt
old=/sap/runs/train-live-20261123/parent-hist-frozen50/update-026022.pt
ev=/sap/runs/train-eval-20261123/evaluate.py
./run.sh "$ev" "/job/$candidate" "$parent" results/proof-vs-parent.json --mode screen --pairs 64 >results/proof-parent.log 2>&1
./run.sh "$ev" "/job/$candidate" "$old" results/proof-vs-old.json --mode confirm --pairs 256 >results/proof-old.log 2>&1
python3 score_summary.py
