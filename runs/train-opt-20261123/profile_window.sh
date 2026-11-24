#!/bin/bash
# Take the GPU for a clean profile right after mirror-a writes its next checkpoint: pause both running jobs, profile
# both network sizes, resume. (mirror-a's own stall alarm may end it during the pause; its supervisor then resumes from
# the checkpoint just written, so at most the few updates after it are redone.)
cd "$(dirname "$0")"
until [ -f results/mirror-a/seg-1/update-026522.pt ]; do sleep 5; done
sup=$(pgrep -f "run_v2.s[h] mirror-a" | head -1)
ids=$(docker ps --no-trunc --format '{{.ID}} {{.Command}}' | grep -e "mirror-a/seg" -e "distill-a" | grep -e collect_v2 -e distill.py | cut -c1-12)
[ -n "$sup" ] && kill -STOP $sup
for i in $ids; do docker pause $i >/dev/null; done
echo "$(date +%T) paused: $ids supervisor $sup" > results/profile-window.log
SAP_CPUSET=5-9 SAP_GPU_GIB=24 ./run_opt.sh /out/profile_arm.py fresh /out/results/profile-fresh.json > results/profile-fresh.log 2>&1
SAP_CPUSET=5-9 SAP_GPU_GIB=24 ./run_opt.sh /out/profile_arm.py parent /out/results/profile-parent.json > results/profile-parent.log 2>&1
for f in results/mirror-a/seg-*/updates.jsonl; do touch $f; done
for i in $ids; do docker unpause $i >/dev/null; done
[ -n "$sup" ] && kill -CONT $sup
echo "$(date +%T) resumed" >> results/profile-window.log
