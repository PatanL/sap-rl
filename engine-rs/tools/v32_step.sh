#!/bin/bash
# One evaluation step of the v32 rules work, on cores 10-14 only.
# Usage: tools/v32_step.sh <tag> <SAP_RULES value> [previous tag] [gen|check|score|all]
#   gen:   regenerate golden/v32-*.jsonl from the Python reference engine-v32 under SAP_RULES
#          (host venv; the six real/fuzz files, v32-fuzz3.jsonl from the v32-targeted boards of
#          tools/gen_fuzz_v32.py, and v32-constructed.jsonl) and count the rows that
#          differ from golden/v31-* and from the previous step (kept in golden/prev-v32/)
#   check: the Rust engine must reproduce every v32 file under the same SAP_RULES
#   score: 5,000 seeds a fight over the five real fixture files -> v32/scores/<tag>-*.json,
#          compared with the previous tag (default v31g), and the constructed-test check
set -e -o pipefail
TAG=$1; RULES=$2; PREV=${3:-v31g}; WHAT=${4:-all}
ROOT=/home/dzliu/pzliu/sap-rl
RS=$ROOT/engine-rs; RO=$ROOT/runs/replay-oracle-20261122
PY=/home/dzliu/pzliu/.venv/bin/python
# A chain of steps can run from frozen copies while the live tree is being edited:
ENGINE=${V32_ENGINE:-$RO/engine-v32}       # Python reference to generate from
BIN=${V32_BIN:-$RS/target/release}         # directory holding the golden and oracle binaries
export PYTHONDONTWRITEBYTECODE=1
cd $RS
FILES="340:$RO/fixtures-340.json:3 fresh:$RO/fixtures-fresh.json:2 fresh2:$RO/fixtures-fresh2.json:2 fresh3:$RO/fixtures-fresh3.json:2 fuzz:golden/fuzz-fixtures.json:2 fuzz2:golden/fuzz2-fixtures.json:3 fuzz3:golden/fuzz3-fixtures.json:2 constructed:$RO/v32/constructed-fixtures.json:50"
if [ $WHAT = gen ] || [ $WHAT = all ]; then
  mkdir -p golden/prev-v32
  for spec in $FILES; do
    IFS=: read name file seeds <<< "$spec"
    [ -f golden/v32-$name.jsonl ] && mv golden/v32-$name.jsonl golden/prev-v32/v32-$name.jsonl
    summary=$(SAP_RULES=$RULES taskset -c 10-14 $PY tools/gen_golden.py $ENGINE $file golden/v32-$name.jsonl $seeds 5)
    base=golden/v31-$name.jsonl; [ -f $base ] || base=/dev/null
    prev=golden/prev-v32/v32-$name.jsonl; [ -f $prev ] || prev=$base
    echo "gen $name: $summary  rows differing from v31 $( [ $base = /dev/null ] && echo n/a || (diff $base golden/v32-$name.jsonl | grep -c '^>' || true) )  from previous step $( [ $prev = /dev/null ] && echo n/a || (diff $prev golden/v32-$name.jsonl | grep -c '^>' || true) )"
  done
fi
if [ $WHAT = check ] || [ $WHAT = all ]; then
  for spec in $FILES; do
    IFS=: read name file seeds <<< "$spec"
    echo -n "rust $name: "; SAP_RULES=$RULES taskset -c 10-14 $BIN/golden --golden golden/v32-$name.jsonl | tail -1
  done
fi
if [ $WHAT = score ] || [ $WHAT = all ]; then
  mkdir -p $RO/v32/scores
  for pair in 340:fixtures-340.json fresh:fixtures-fresh-all.json fresh2:fixtures-fresh2.json fresh3:fixtures-fresh3.json fresh4:fixtures-fresh4-all.json; do
    name=${pair%%:*}; file=${pair#*:}
    SAP_RULES=$RULES taskset -c 10-14 $BIN/oracle --fixtures $RO/$file --data data --samples 5000 --threads 5 --out $RO/v32/scores/$TAG-$name.json > $RO/v32/scores/$TAG-$name.summary.json
  done
  $PY $RO/v32/compare_scores.py $PREV $TAG
  CBASE=$PREV; [ -f $RO/v32/constructed/$CBASE.json ] || CBASE=
  $PY $RO/v32/check_constructed.py golden/v32-constructed.jsonl $TAG $CBASE
fi
