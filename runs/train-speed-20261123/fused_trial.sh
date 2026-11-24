#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export SPEED_FUSED_LOSS=1 SPEED_GEOMETRY=short
for fixture in cpu gpu;do
  if [[ "$fixture" == gpu ]];then export SPEED_BATCH=results/g48/batch.pt;fi
  for prec in fp32 bf16;do
    ./run.sh equivalence.py --prec "$prec" --cap 512 --n 8192 --out "results/fused-$fixture-$prec.json" >"results/fused-$fixture-$prec.log" 2>&1
  done
done
export SPEED_BATCH=results/batch.pt
./run.sh learner_bench.py --sizes 128,256,512,1024 --out results/fused-sweep.json >results/fused-sweep.log 2>&1
