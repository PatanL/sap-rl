#!/bin/bash
# pair every finished constructed capture with the engine (60 seeds) and print one line each
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
export PYTHONDONTWRITEBYTECODE=1
for d in session2/captures/c3-*; do
  [ -f $d/run.json ] || continue
  [ -f $d/timeline.json ] || nice taskset -c 14 /home/dzliu/pzliu/.venv/bin/python timeline.py $d > /dev/null
  [ -f $d/pair3.json ] || nice taskset -c 14 /home/dzliu/pzliu/.venv/bin/python pair3.py $d 60 > /dev/null 2>&1
done
/home/dzliu/pzliu/.venv/bin/python - <<'P'
import json,glob
for p in sorted(glob.glob('session2/captures/c3-*/pair3.json')):
    r=json.load(open(p));n=p.split('/')[2]
    print(n.ljust(48),'OK ' if r['engine_reproduces_all_stages'] else 'DIFF',r['first_difference'] or '',(' | ENG '+r['engine_at_difference'][0]['P']+' || '+r['engine_at_difference'][0]['O']) if r.get('engine_at_difference') else '')
P
