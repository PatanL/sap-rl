#!/bin/bash
# After the ranked arm: the matched self-play control on the same stack, then 2,048 full matches against ranked boards
# (yardstick side) for both final checkpoints.
cd "$(dirname "$0")"
until grep -q "complete:\|giving up\|too many segments" results/ranked-a/supervisor.log 2>/dev/null; do sleep 30; done
./run_v2.sh selfplay-a 2000 --games 512 --max-lag 3 --backlog 1.0 --workers 10 --cap 3072 --compact --compile --ghost-share 0 --frozen-share 0.5 > results/selfplay-a-supervisor.out 2>&1 &
sp=$!
(setsid nohup ./score_run.sh selfplay-a > /dev/null 2>&1 &)
fa=$(ls results/ranked-a/seg-*/update-027772.pt 2>/dev/null | tail -1)
[[ -n "$fa" ]] && SAP_CPUSET=0-4 ../train-ghost-20261123/fullmatch.sh v2-ranked-a-27772 /sap/runs/train-opt-20261123/$fa >> results/chain_ab.log 2>&1
wait $sp
fb=$(ls results/selfplay-a/seg-*/update-027772.pt 2>/dev/null | tail -1)
[[ -n "$fb" ]] && SAP_CPUSET=0-4 ../train-ghost-20261123/fullmatch.sh v2-selfplay-a-27772 /sap/runs/train-opt-20261123/$fb >> results/chain_ab.log 2>&1
echo "$(date +%T) chain complete" >> results/chain_ab.log
