#!/bin/bash
# Incrementally post-process finished real-fight captures of a manifest: timeline + multi-seed pairing.
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/home/dzliu/pzliu/.venv/bin/python
while true; do
  fin=0; grep -q "BATCH DONE" session3/logs/$1.log 2>/dev/null && fin=1
  for f in $($py -c "import json;print(' '.join(v['file'][:-5] for v in json.load(open('session2/$1-manifest.json'))))"); do
    d=session2/captures/$f
    [ -f $d/pair3.json ] && continue
    if [ -f $d/run.json ]; then
      [ -f $d/timeline.json ] || nice taskset -c 14 $py timeline.py $d >> session3/logs/post-$1.log 2>&1
      nice taskset -c 14 $py pair3.py $d ${2:-300} > /dev/null 2>> session3/logs/post-$1.log && echo "paired $f" >> session3/logs/post-$1.log
    fi
  done
  [ $fin = 1 ] && break
  sleep 20
done
echo "POST DONE" >> session3/logs/post-$1.log
