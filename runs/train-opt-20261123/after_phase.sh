#!/bin/bash
cd "$(dirname "$0")"
until grep -q "phase.sh exit" ../train-live-20261123/after_5e6.log; do sleep 20; done
SAP_GPU_GIB=24 ./run_opt.sh /out/learner_profile.py /out/learner-profile.json > learner-profile.log 2>&1
echo "$(date +%T) profile exit $?" >> learner-profile.log
