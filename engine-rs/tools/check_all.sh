#!/bin/bash
# Every exactness check of the environment port and the battle engine, on cores 0-4.
# Usage: tools/check_all.sh [coverage.json]
cd "$(dirname "$0")/.."
set -e
taskset -c 0-4 ./target/release/envcheck --threads 5 --report 10 ${1:+--coverage $1} env-golden/*.jsonl.gz | tail -4
for f in golden/v31-*.jsonl; do echo -n "$f: "; taskset -c 0-4 ./target/release/golden --golden $f | tail -1; done
