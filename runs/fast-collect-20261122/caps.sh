#!/usr/bin/env bash
set -eu
while [ ! -s wide-parent-random-tape.npz ];do sleep 3;done
sleep 2
for task in 'wide-mid obs' 'wide-parent obs' 'wide-mid core' 'wide-parent core'; do
 read -r file mode <<< "$task"
 opt=();if [[ $mode == core ]]; then opt=(--noobs);fi
 BENCH_CPU_ONLY=1 ./run.sh replay_cap.py "$file" "${opt[@]}" > "$file-cap-$mode.log" 2>&1
done
