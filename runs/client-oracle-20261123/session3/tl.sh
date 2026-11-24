#!/bin/bash
# usage: tl.sh <prefix> : build missing timelines for finished captures matching prefix and print the tail of each
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
for d in session2/captures/$1*; do
  [ -f $d/run.json ] || continue
  [ -f $d/timeline.json ] || taskset -c 14 /home/dzliu/pzliu/.venv/bin/python timeline.py $d > /dev/null
  echo "=== $(basename $d)"; grep -v "P:   ||  O: *$" $d/timeline.txt | tail -n ${2:-12} | cut -c1-${3:-200}
done
