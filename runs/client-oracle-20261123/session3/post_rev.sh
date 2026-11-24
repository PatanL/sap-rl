#!/bin/bash
# second pairing worker: walks the real manifest backwards
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/home/dzliu/pzliu/.venv/bin/python
while true; do
  fin=0; grep -q "BATCH DONE" session3/logs/s3real.log 2>/dev/null && fin=1
  for f in $($py -c "import json;print(' '.join(v['file'][:-5] for v in reversed(json.load(open('session2/s3real-manifest.json'))) if v['file'].startswith('s3f')))"); do
    d=session2/captures/$f
    [ -f $d/pair3.json ] && continue
    [ -f $d/run.json ] || continue
    [ -f $d/pairing.lock ] && continue
    touch $d/pairing.lock
    [ -f $d/timeline.json ] || nice taskset -c 12 $py timeline.py $d > /dev/null 2>&1
    nice taskset -c 12 $py pair3.py $d 200 > /dev/null 2>&1 && echo "paired $f" >> session3/logs/post-s3real.log
  done
  [ $fin = 1 ] && break
  sleep 15
done
