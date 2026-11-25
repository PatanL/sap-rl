#!/bin/bash
# Run several v32 steps back to back, each adding one rule to the kept set (tools/v32_step.sh).
# A rule is kept when the Rust engine reproduces every trace row and it does not break more real
# fights than it fixes; otherwise it is left out of the following sets and the previous trace files
# are put back. Usage: tools/v32_chain.sh <first step number> <kept set> <previous tag> <rule>...
# Logs: runs/replay-oracle-20261122/v32/logs/<tag>.log and chain.log (one line per decision).
HERE=$(cd "$(dirname "$0")" && pwd)   # a frozen copy of this script and v32_step.sh may be run instead
cd /home/dzliu/pzliu/sap-rl/engine-rs
N=$1; KEPT=$2; PREV=$3; shift 3
LOGS=/home/dzliu/pzliu/sap-rl/runs/replay-oracle-20261122/v32/logs
for rule in "$@"; do
  tag=$(printf "s%02d-%s" $N $rule); set="$KEPT,$rule"
  $HERE/v32_step.sh $tag $set $PREV all > $LOGS/$tag.log 2>&1
  status=$?
  bad=$(grep "^rust " $LOGS/$tag.log | grep -vc "mismatched: 0 " || true)
  nrust=$(grep -c "^rust " $LOGS/$tag.log || true)
  total=$(grep "^TOTAL" $LOGS/$tag.log)
  fixed=$(sed -n 's/^TOTAL.* fixed \([0-9]*\)  broken \([0-9]*\)$/\1/p' $LOGS/$tag.log)
  broken=$(sed -n 's/^TOTAL.* fixed \([0-9]*\)  broken \([0-9]*\)$/\2/p' $LOGS/$tag.log)
  if [ $status -eq 0 ] && [ "$nrust" = 8 ] && [ "$bad" = 0 ] && [ -n "$fixed" ] && [ "$broken" -le "$fixed" ]; then
    KEPT=$set; PREV=$tag; verdict=kept
  else
    verdict="LEFT OUT (exit $status, trace files not reproduced $bad of $nrust)"
    for f in golden/prev-v32/v32-*.jsonl; do [ -f $f ] && mv $f golden/; done
  fi
  echo "$(date +%H:%M) $tag SAP_RULES=$set: $verdict | $total | next set $KEPT, previous $PREV" >> $LOGS/chain.log
  N=$((N + 1))
done
echo "$(date +%H:%M) chain done: kept $KEPT, last tag $PREV" >> $LOGS/chain.log
