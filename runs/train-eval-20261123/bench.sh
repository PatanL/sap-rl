#!/bin/bash
set -eu
cd "$(dirname "$0")"
for arm in off on; do
 share=0; [[ $arm == off ]] || share=.5
 nvidia-smi --query-gpu=name,utilization.gpu --format=csv > "scratch/gpu-before-$arm.csv"
 flock gpu.lock ./run.sh collector_join.py train --parent /sap/checkpoints/update-025772.pt --arm parent --workers 4 --games 32 --transitions 2048 --updates 12 --checkpoint-every 4 --lr-scale 5 --max-padded-rows 96 --frozen-share "$share" --pool-size 2 --output "/job/scratch/$arm" >"scratch/$arm.log" 2>&1
done
