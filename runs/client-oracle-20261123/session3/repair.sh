#!/bin/bash
# re-pair listed real captures (id8 ...) with the current pair3 (lag-tolerant)
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
export PYTHONDONTWRITEBYTECODE=1
for id in "$@"; do nice taskset -c 13 /home/dzliu/pzliu/.venv/bin/python pair3.py session2/captures/s3f-$id ${N:-300} > /dev/null 2>&1; done
