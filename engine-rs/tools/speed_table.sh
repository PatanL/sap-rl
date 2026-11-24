#!/bin/bash
# The speed rows of PROGRESS.md: each configuration three times, all three results printed.
# Cores 0-4 only. Usage: tools/speed_table.sh
cd "$(dirname "$0")/.."
B=./target/release/envbench
P="env-golden/policy-sampled.jsonl.gz env-golden/policy-greedy.jsonl.gz"
row() { # label, cpus, args...
  label=$1; cpus=$2; shift 2
  for k in 1 2 3; do
    taskset -c $cpus $B "$@" | sed -E "s/^(replay|random): /$label | /; s/  digest.*//"
  done
}
row "policy tapes x1 (batch 64), 1 core"      2   replay --threads 1 $P
row "policy tapes x16 (batch 1024), 1 core"   2   replay --threads 1 --repeat 16 $P
row "policy tapes x4 (batch 256), 5 cores"    0-4 replay --threads 5 --repeat 4 $P
row "policy tapes x16 (batch 1024), 5 cores"  0-4 replay --threads 5 --repeat 16 $P
row "random tapes random-0 (batch 400), 1 core"  2   replay --threads 1 env-golden/random-0.jsonl.gz
row "random tapes random-0 (batch 400), 5 cores" 0-4 replay --threads 5 env-golden/random-0.jsonl.gz
row "random play batch 64, 1 core"     2   random --threads 1 --matches 2000 --batch 64
row "random play batch 256, 1 core"    2   random --threads 1 --matches 2000 --batch 256
row "random play batch 1024, 1 core"   2   random --threads 1 --matches 2000 --batch 1024
row "random play batch 256, 5 cores"   0-4 random --threads 5 --matches 6000 --batch 256
row "random play batch 1024, 5 cores"  0-4 random --threads 5 --matches 6000 --batch 1024
row "random play batch 4096, 5 cores"  0-4 random --threads 5 --matches 8192 --batch 4096
