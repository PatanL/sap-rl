#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
export SAP_ENGINE_ROOT="$PWD/pinned-recover"
for corpus in fresh original;do
 for cap in 1 64 1024;do
  prevargs=()
  if [[ $cap != 1 ]];then prevargs=(--retry "$prev");fi
  if [[ $corpus == fresh ]];then
    out="strict-$cap.jsonl"
    ./run.sh -c 'import os,sys,runpy;sys.path.insert(0,"/audit");os.chdir("/audit/recover/fresh");runpy.run_path("/audit/replay.py",run_name="__main__")' --seeds "$cap" "${prevargs[@]}" --out "$out" > "recover/fresh/$cap.log" 2>&1
  else
    out="recover/original-$cap.jsonl"
    ./run.sh replay.py --seeds "$cap" "${prevargs[@]}" --out "$out" > "recover/original-$cap.log" 2>&1
  fi
  prev=$out
 done
done
