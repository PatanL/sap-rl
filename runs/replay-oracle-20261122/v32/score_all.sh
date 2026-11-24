#!/bin/bash
# Score every fixture file with the Rust oracle (5,000 seeds a fight).
# Usage: score_all.sh <tag> [oracle binary]      env SAP_RULES selects the rule set (unset = v31).
set -e
TAG=$1
BIN=${2:-/home/dzliu/pzliu/sap-rl/engine-rs/target/release/oracle}
RO=/home/dzliu/pzliu/sap-rl/runs/replay-oracle-20261122
cd /home/dzliu/pzliu/sap-rl/engine-rs
for pair in 340:fixtures-340.json fresh:fixtures-fresh-all.json fresh2:fixtures-fresh2.json fresh3:fixtures-fresh3.json fresh4:fixtures-fresh4-all.json; do
  name=${pair%%:*}; file=${pair#*:}
  taskset -c 0-4,10-14 $BIN --fixtures $RO/$file --data data --samples 5000 --threads 10 --out $RO/v32/scores/$TAG-$name.json > $RO/v32/scores/$TAG-$name.summary.json
done
echo done $TAG
