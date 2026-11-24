#!/bin/bash
cd "$(dirname "$0")"
until grep -q "scoring complete" results/score/long-a/table.txt; do sleep 10; done
run() { n=$1; shift; SAP_CPUSET=5-9,15-19 SAP_GPU_GIB=24 SPEED_GEOMETRY=sorted timeout 400 ./run_opt.sh /out/collect_v2.py --train --workers 10 --compact --compile --ghost-share 0 --frozen-share 0 --seconds 300 --out /out/results/$n "$@" > results/$n.log 2>&1; python3 -c "
import json;r=json.load(open('results/$n/result.json'));s=r['steady'] or {}
print('$n', {k:(round(r[k],3) if isinstance(r[k],float) else r[k]) for k in ('updates','learned_s','seat_s','games_s','stale_share','gpu','learner_wall','wall')}, 'steady', {k:round(v,1) for k,v in s.items()}, {k:round(v,1) for k,v in r['main_seconds'].items()})" 2>&1 | tail -1; }
run t-parent-512 --arm parent --games 512 --max-lag 3 --backlog 1.0 --cap 3072 --updates 40
run t-fresh-512 --arm fresh --games 512 --max-lag 3 --backlog 1.0 --cap 3072 --updates 80
run t-fresh-1024 --arm fresh --games 1024 --max-lag 3 --backlog 1.0 --cap 6144 --target 16384 --updates 60
run t-fresh-2048 --arm fresh --games 2048 --max-lag 3 --backlog 1.0 --cap 12288 --target 32768 --updates 40
echo done
