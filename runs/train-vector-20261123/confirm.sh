#!/bin/bash
set -eu
cd "$(dirname "$0")"
./run.sh evaluate.py /job/results/proof/update-025872.pt /sap/checkpoints/update-025772.pt results/proof-confirm.json --mode confirm --pairs 256 >results/proof-confirm.log 2>&1
