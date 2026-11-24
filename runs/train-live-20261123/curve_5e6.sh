#!/bin/bash
# Score every checkpoint of g-base5e6 on 4,096 games against 25,772 as it appears; after the last one, the final against v22, v25 and 27,772.
cd "$(dirname "$0")"
R=/sap/runs/train-live-20261123; P=/sap/checkpoints/update-025772.pt
V22=/sap/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt; V25=/sap/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
while true; do
  for f in g-base5e6/update-*.pt; do
    [[ -f $f ]] || continue
    u=$(basename "$f" .pt); u=${u#update-0}
    SAP_CPUSET=0-4 ./rescore_vec.sh "b5e6-$u-25772" "$R/$f" "$P"
  done
  if [[ -f rescore4k/b5e6-27772-25772.json ]]; then
    F=$R/g-base5e6/update-027772.pt
    SAP_CPUSET=0-4 ./rescore_vec.sh b5e6-27772-v22 $F $V22 b5e6-27772-v25 $F $V25 b5e6-27772-a27772 $F $R/g-hist-frozen50/update-027772.pt
    echo "$(date +%T) curve complete"; exit 0
  fi
  sleep 60
done
