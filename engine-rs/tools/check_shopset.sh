#!/bin/bash
# Everything about `set_shop_item`: the shopset-* trace family replayed by envcheck (also part of
# tools/check_all.sh), the tamper test of its trace entries, and python/verify_search.py through
# the C interface: `shopset` (lockstep with the Python reference, edits by name and by id, state
# and observe() right after every edit, refused calls) and `shopsame` (a rolled item replaced by
# itself changes nothing, in the reference and here). Cores 0-4 only; the Python parts run in the
# training image. Called by tools/check_search.sh.
# Usage: tools/check_shopset.sh [quick]        (quick: a tenth of the C-interface check sizes)
cd "$(dirname "$0")/.."
set -e -o pipefail
LIB=/rs/target/release/libsap_engine.so
if [ "$1" = quick ]; then LOCK=10; SAME=10; else LOCK=100; SAME=100; fi
LOG=$(mktemp -d)
echo "== shopset trace family (tools/gen_env_shopset.py) =="
taskset -c 0-4 ./target/release/envcheck --threads 5 --report 10 env-golden/shopset-*.jsonl.gz | tee $LOG/traces.txt | tail -8
grep -q "^TOTAL matches \([0-9]*\)/\1 " $LOG/traces.txt
grep -q "^SHOP EDITS" $LOG/traces.txt
echo "== tamper test of the set_shop_item trace entries =="
python3 tools/tamper_shopset.py ./target/release/envcheck $LOG env-golden/shopset-0.jsonl.gz env-golden/shopset-1.jsonl.gz | tee $LOG/tamper.txt | tail -1
! grep -q "^MISSED" $LOG/tamper.txt
echo "== set_shop_item through the C interface =="
run() { tools/run_ref.sh "$@" 2>&1 | grep -v -e pynvml -e FutureWarning; }
# Three lockstep runs over different matches (first index 0, LOCK, 2 LOCK), the self-replacement check, and five threads.
run 0 /rs/python/verify_search.py shopset $LIB 16 $LOCK 1 0 > $LOG/lock-0.txt &
run 1 /rs/python/verify_search.py shopset $LIB 16 $LOCK 1 $LOCK > $LOG/lock-1.txt &
run 2 /rs/python/verify_search.py shopset $LIB 16 $LOCK 1 $((2 * LOCK)) > $LOG/lock-2.txt &
run 3 /rs/python/verify_search.py shopsame $LIB $SAME 1 > $LOG/same.txt &
wait
cat $LOG/lock-0.txt $LOG/lock-1.txt $LOG/lock-2.txt $LOG/same.txt
run 0-4 /rs/python/verify_search.py shopset $LIB 16 $((LOCK / 5 + 2)) 5 $((3 * LOCK)) | tee $LOG/lock5.txt
for name in lock-0 lock-1 lock-2 same lock5; do grep -q "^ALL EQUAL" $LOG/$name.txt || { echo "FAILED: $name"; exit 1; }; done
echo "SHOPSET CHECKS PASSED"
