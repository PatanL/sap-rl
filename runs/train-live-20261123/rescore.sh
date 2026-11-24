#!/bin/bash
# Larger rescore on a seed set no selection used (evaluate.py mode "sanity": fixed seeds, the same for every pairing), 256 pairs each.
set -uo pipefail
cd "$(dirname "$0")"
R=/sap/runs/train-live-20261123/g-hist-frozen50
P=/sap/checkpoints/update-025772.pt
V22=/sap/runs/turtle-league-v22-curriculum-diverse-v3/update-002000.pt
V25=/sap/runs/turtle-league-v25-recalibration-from-v22-2000/update-000300.pt
job() { # name candidate opponent core
  [[ -f rescore/$1.json ]] && return 0
  SAP_CPUSET=$4 ./score_g.sh evaluate.py "$2" "$3" "/out/rescore/$1.json" --mode sanity --pairs 256 > "rescore/$1.log" 2>&1
  echo "$(date +%T) $1 exit $?"
}
job 25772-v22 $P $V22 5 & job 25772-v25 $P $V25 6 & job 27022-v22 $R/update-027022.pt $V22 7 & job 27022-v25 $R/update-027022.pt $V25 8 &
job 27022-25772 $R/update-027022.pt $P 9 & job 27772-25772 $R/update-027772.pt $P 15 & job 27772-v22 $R/update-027772.pt $V22 16 & job 27772-v25 $R/update-027772.pt $V25 17 &
wait
echo done
