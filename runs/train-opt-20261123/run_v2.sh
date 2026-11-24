#!/bin/bash
# Supervised training run on the optimised stack (collect_v2.py). Restarts from the newest checkpoint if the process
# ends early or stops writing updates (an in-process watchdog cannot fire when a thread blocks holding the interpreter lock).
# Usage: run_v2.sh <name> <total updates> [collect_v2.py options, without --resume/--updates/--out]
# Segments are written to results/<name>/seg-N; updates of all segments are appended to results/<name>/updates.jsonl.
set -uo pipefail
cd "$(dirname "$0")"; name=$1; total=$2; shift 2
dir=results/$name; mkdir -p "$dir"; seg=0; resume="${RESUME_FROM:-}"; done_updates=0
STALL=${STALL_SECONDS:-420}
while (( done_updates < total )); do
  seg=$((seg+1)); out=$dir/seg-$seg; left=$((total-done_updates))
  echo "$(date +%T) segment $seg: $left updates left ${resume:+from $resume}" | tee -a "$dir/supervisor.log"
  SAP_CPUSET=${SAP_CPUSET:-5-9,15-19} SAP_GPU_GIB=${SAP_GPU_GIB:-24} SPEED_GEOMETRY=sorted ./run_opt.sh /out/collect_v2.py --train --save --updates "$left" --seconds 400000 \
     ${resume:+--resume /out/$resume} --out "/out/$out" "$@" > "$dir/seg-$seg.log" 2>&1 &
  pid=$!; started=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 15
    last=$(stat -c %Y "$out/updates.jsonl" 2>/dev/null || echo "$started")
    if (( $(date +%s) - last > STALL )); then
      echo "$(date +%T) segment $seg stalled (no update for $STALL s); killing" | tee -a "$dir/supervisor.log"
      docker ps --no-trunc --format '{{.ID}} {{.Command}}' | grep "/out/$out" | cut -c1-12 | xargs -r docker kill >/dev/null; kill $pid 2>/dev/null; break
    fi
  done
  wait $pid 2>/dev/null
  [[ -f $out/updates.jsonl ]] && cat "$out/updates.jsonl" >> "$dir/updates.jsonl"
  newest=$(ls "$out"/update-*.pt 2>/dev/null | sort | tail -1)
  if [[ -n "$newest" ]]; then
    resume=$newest; v=$(basename "$newest" .pt); v=$((10#${v#update-}))
    done_updates=$(( v - ${START_VERSION:-25772} ))
  else
    echo "$(date +%T) segment $seg left no checkpoint" | tee -a "$dir/supervisor.log"
    (( seg >= 8 )) && { echo "giving up" | tee -a "$dir/supervisor.log"; exit 1; }
  fi
  (( seg >= 40 )) && { echo "too many segments" | tee -a "$dir/supervisor.log"; exit 1; }
done
echo "$(date +%T) complete: $done_updates updates, newest $resume" | tee -a "$dir/supervisor.log"
