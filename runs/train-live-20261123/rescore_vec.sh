#!/bin/bash
# 4,096-game scores (2,048 fresh pairs, evaluator mode "sanity", Rust vector environment, batch 4096) for a list of pairings.
# Usage: rescore_vec.sh <name> <candidate path under /sap> <opponent path under /sap> [more triples...]
set -uo pipefail
cd "$(dirname "$0")"; mkdir -p rescore4k
while (( $# >= 3 )); do
  name=$1; cand=$2; opp=$3; shift 3
  [[ -f rescore4k/$name.json ]] && continue
  SAP_CPUSET=${SAP_CPUSET:-0-4} ./score_g.sh /out/evaluate_vec.py "$cand" "$opp" "/out/rescore4k/$name.json" --mode sanity --pairs 2048 --batch 4096 --threads 5 > "rescore4k/$name.log" 2>&1
  echo "$(date +%T) $name exit $? $(tail -n1 rescore4k/$name.log)"
done
