#!/bin/bash
# After the 5e-6 run: wait for its scoring to finish, release the GPU, then run the staged Rust-collector measurements.
cd "$(dirname "$0")"
until grep -q "curve complete" curve_5e6.log && grep -q "scored 27772" watch-g-base5e6.log; do sleep 20; done
pkill -f "watch_g.py --run runs/train-live-20261123/g-base5e6" || true
sleep 3
date > GPU-FREE
echo "$(date +%T) GPU released; starting phase.sh"
cd ../rust-collector-20261123 && bash ./phase.sh > results/phase-claude.log 2>&1
echo "$(date +%T) phase.sh exit $?"
