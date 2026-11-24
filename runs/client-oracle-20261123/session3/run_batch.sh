#!/bin/bash
# usage: run_batch.sh <manifest-name> [timeout-seconds]   (serial; one browser; cores 10-14)
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export SAP_SHOT_INTERVAL_MS=${SAP_SHOT_INTERVAL_MS:-700}
py=/home/dzliu/pzliu/.venv/bin/python
timeout ${2:-7200} taskset -c 10-14 $py client.py --replay-manifest "$1" > session3/logs/$1.log 2>&1
echo "EXIT $?" >> session3/logs/$1.log
for f in $($py -c "import json;print(' '.join(v['file'][:-5] for v in json.load(open('session2/$1-manifest.json'))))"); do
  d=session2/captures/$f
  if [ -f $d/run.json ] && [ ! -f $d/timeline.txt ]; then taskset -c 10-14 $py timeline.py $d >> session3/logs/$1.log 2>&1; fi
done
echo "BATCH DONE" >> session3/logs/$1.log
