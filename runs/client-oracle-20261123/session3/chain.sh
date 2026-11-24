#!/bin/bash
# usage: chain.sh <wait-for-batch> <next-batch> [env assignments...]: start next batch when the previous one has written BATCH DONE
cd /home/dzliu/pzliu/sap-rl/runs/client-oracle-20261123/
while ! grep -q "BATCH DONE" session3/logs/$1.log 2>/dev/null; do sleep 5; done
shift; b=$1; shift
env "$@" session3/run_batch.sh $b 9000
