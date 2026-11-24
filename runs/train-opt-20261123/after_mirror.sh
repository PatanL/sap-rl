#!/bin/bash
# When the mirror-only run and its scoring are done: pause the distillation for a clean GPU, profile the network sizes,
# resume it, then take the measures the frozen-opponent verdict needs.
cd "$(dirname "$0")"
until grep -q "complete:" results/mirror-a/supervisor.log 2>/dev/null && grep -q "scoring complete" results/score/mirror-a/table.txt 2>/dev/null; do sleep 30; done
id=$(docker ps --no-trunc --format '{{.ID}} {{.Command}}' | grep "distill.py" | cut -c1-12)
[ -n "$id" ] && docker pause $id >/dev/null
OPT_ENCODER_MODE= SAP_CPUSET=5-9 SAP_GPU_GIB=24 ./run_opt.sh /out/profile_sizes.py /out/results/profile-sizes.json > results/profile-sizes.log 2>&1
[ -n "$id" ] && docker unpause $id >/dev/null
echo "$(date +%T) sizes profiled" > results/after-mirror.log
M=$(ls results/mirror-a/seg-*/update-027772.pt | tail -1); S=results/selfplay-a/seg-1/update-027772.pt
SAP_CPUSET=5-9 ./run_opt.sh /sap/runs/train-live-20261123/evaluate_vec.py /out/$M /out/$S /out/results/score/mirror-a/27772-selfplay27772.json --mode sanity --pairs 2048 --batch 4096 --threads 5 > results/score/mirror-a/27772-selfplay27772.log 2>&1
SAP_CPUSET=5-9 ./run_opt.sh /sap/runs/train-live-20261123/evaluate_vec.py /out/$M /sap/checkpoints/update-025772.pt /out/results/score/mirror-a/27772-25772-confirm.json --mode confirm --pairs 2048 --batch 4096 --threads 5 > results/score/mirror-a/27772-25772-confirm.log 2>&1
for seed in 20261201 20261202 20261203 20261204 20261205; do for u in 27272 27522 27772; do
  f=$(ls results/mirror-a/seg-*/update-0$u.pt | tail -1); SAP_CPUSET=15-19 ./board_index.sh tb-mirror-a-$u-s$seed /out/$f $seed >> results/tiebreak-mirror.log 2>&1
done; done
for pool in top all; do
  extra=""; [ $pool = top ] && extra="--yard /sap/runs/search-20261123/yardstick-top-pool.json"
  (cd ../search-20261123 && SAP_CPUSET=5-9 SAP_GPU_GIB=8 ./run_search.sh /out/search_match.py /sap/runs/train-opt-20261123/$M /out/results/$pool-mirror27772-2048.json --matches 2048 --roots 512 $extra > results/$pool-mirror27772.log 2>&1)
done
echo "$(date +%T) verdict measures done" >> results/after-mirror.log
