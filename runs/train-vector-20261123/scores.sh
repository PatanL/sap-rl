#!/bin/bash
set -eu
cd "$(dirname "$0")"
python3 board_metric.py ../../checkpoints/update-025772.pt >results/parent-board.log 2>&1
while [[ ! -f results/proof/result.json ]]; do sleep 3; done
python3 watch_g.py --once --run runs/train-vector-20261123/results/proof --progression results/progression.json >results/watch.log 2>&1
