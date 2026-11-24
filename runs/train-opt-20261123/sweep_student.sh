#!/bin/bash
# Self-play rates for the 1.35M student: three short runs from the same checkpoint with fresh optimizer moments, the
# champion's rate set scaled by 1, 5 and 25; each final is then scored against the starting student on 4,096 games.
cd "$(dirname "$0")"
for scale in 1 5 25; do
  n=sweep-student-x$scale
  [ -d results/$n ] || OPT_ENCODER_MODE=max-autotune-no-cudagraphs SAP_CPUSET=0-4,10-14 SAP_GPU_GIB=24 SPEED_GEOMETRY=sorted timeout 2400 ./run_opt.sh /out/collect_v2.py --train --save --arm fresh --resume /out/results/student-start/update-003000.pt --parent-rates --fresh-optimizer --rate-scale $scale \
    --games 2048 --target 32768 --cap 6144 --max-lag 3 --backlog 1.0 --workers 10 --compact --compile --ghost-share 0 --frozen-share 0.5 --checkpoint-every 200 --updates 200 --seconds 2000 --stall 600 --out /out/results/$n > results/$n.log 2>&1
  f=$(ls results/$n/update-*.pt 2>/dev/null | tail -1)
  [ -n "$f" ] && SAP_CPUSET=0-4 ./run_opt.sh /sap/runs/train-live-20261123/evaluate_vec.py /out/$f /out/results/student-start/update-003000.pt /out/results/$n-vs-start.json --mode sanity --pairs 2048 --batch 4096 --threads 5 > results/$n-vs-start.log 2>&1
  python3 -c "
import json,os
r=json.load(open('results/$n/result.json')) if os.path.exists('results/$n/result.json') else {}
s=json.load(open('results/$n-vs-start.json')) if os.path.exists('results/$n-vs-start.json') else {}
print('scale $scale', {k:(round(r[k],3) if isinstance(r[k],float) else r[k]) for k in ('updates','learned_s','seat_s','stale_share','gpu') if k in r}, 'vs start', s.get('score'), s.get('ci95'))" >> results/sweep-student.log
done
echo done >> results/sweep-student.log
