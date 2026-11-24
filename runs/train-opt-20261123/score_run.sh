#!/bin/bash
# Score every checkpoint of a supervised run as it appears: 4,096 games against 25,772 (fresh "sanity" seeds, batch 4,096)
# and the ranked-board index through the Rust recorder. Ends when the supervisor reports completion and all are scored.
# Usage: score_run.sh <run name>      -> results/score/<run>/<update>-25772.json, results/board/<run>-<update>.json, results/score/<run>/table.txt
cd "$(dirname "$0")"; run=$1; mkdir -p results/score/$run
P=/sap/checkpoints/update-025772.pt
while true; do
  finished=$(grep -c "complete:" results/$run/supervisor.log 2>/dev/null)
  for f in $(ls results/$run/seg-*/update-*.pt 2>/dev/null | sort -t/ -k4); do
    u=$(basename $f .pt); u=${u#update-0}
    (( (10#$u - ${SCORE_OFFSET:-22}) % ${SCORE_EVERY:-250} == 0 )) || continue
    if [[ ! -f results/score/$run/$u-25772.json ]]; then
      SAP_CPUSET=0-4 ./run_opt.sh /sap/runs/train-live-20261123/evaluate_vec.py /out/$f $P /out/results/score/$run/$u-25772.json --mode sanity --pairs 2048 --batch 4096 --threads 5 > results/score/$run/$u-25772.log 2>&1
    fi
    [[ -f results/board/$run-$u.json ]] || SAP_CPUSET=0-4 ./board_index.sh $run-$u /out/$f > /dev/null 2>&1
  done
  python3 - "$run" <<'PY' > results/score/$run/table.txt
import json,glob,os,sys
run=sys.argv[1]
for f in sorted(glob.glob(f'results/score/{run}/*-25772.json')):
    u=os.path.basename(f).split('-')[0]; r=json.load(open(f)); b=f'results/board/{run}-{u}.json'; idx=''
    if os.path.exists(b):
        g=json.load(open(b)); s=[g['comparisons']['bot v field'][t]['score'] for t in ('3','5','7','10','12')]; idx='board %.2f'%(100*sum(s)/5)
    print(u, 'vs 25772 %.4f [%.4f, %.4f]'%(r['score'],*r['ci95']), idx)
PY
  [[ "$finished" -ge 1 ]] && { echo "$(date +%T) scoring complete" >> results/score/$run/table.txt; exit 0; }
  sleep 45
done
