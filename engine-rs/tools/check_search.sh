#!/bin/bash
# One command for everything: tools/check_all.sh (every trace file under env-golden/, the families
# ghost-*, ghostmix-*, battles-* and rekey-* of tools/gen_env_ghost.py included, and the battle
# goldens), the unit tests, the tamper test of the new trace keys, the reviewer's lockstep check of
# the unchanged interface, and the checks of the search and ghost primitives through the C interface
# (python/verify_search.py), and tools/check_shopset.sh for `set_shop_item` (the shopset-* family is
# also replayed by check_all.sh). Cores 0-4 only; the Python parts run in the training image.
# Usage: tools/check_search.sh [quick]        (quick: a tenth of the C-interface check sizes)
cd "$(dirname "$0")/.."
set -e -o pipefail
LIB=/rs/target/release/libsap_engine.so
if [ "$1" = quick ]; then GHOST=40; COPY=30; REKEY=500; EVAL=150; VEC=16; else GHOST=400; COPY=300; REKEY=5000; EVAL=1500; VEC=64; fi
LOG=$(mktemp -d)
sha256sum target/release/libsap_engine.so | tee $LOG/sha-before.txt
echo "== trace files and battle goldens (tools/check_all.sh) =="
tools/check_all.sh | tee $LOG/all.txt
grep -q "^TOTAL matches \([0-9]*\)/\1 " $LOG/all.txt
! grep -q "mismatched: [1-9]" $LOG/all.txt
echo "== the 8,668 earlier matches alone: per-file lines and totals against env-golden/baseline-8668.txt =="
OLD=$(sed -n 's/^\(env-golden\/[^:]*\):.*/\1/p' env-golden/baseline-8668.txt)
taskset -c 0-4 ./target/release/envcheck --threads 5 --report 10 $OLD | sed 's/ ([0-9.]*s)$//' > $LOG/old.txt
diff $LOG/old.txt env-golden/baseline-8668.txt && tail -1 $LOG/old.txt && echo "identical to the baseline"
echo "== random-play digest of the unchanged interface (31035309134488a8) =="
taskset -c 0-4 ./target/release/envbench random --threads 5 --matches 1000 --batch 1000 --digest | tail -1 | tee $LOG/digest.txt
grep -q "digest 31035309134488a8" $LOG/digest.txt
echo "== new trace families =="
taskset -c 0-4 ./target/release/envcheck --threads 5 --report 10 env-golden/ghost-*.jsonl.gz env-golden/ghostmix-*.jsonl.gz \
  env-golden/battles-*.jsonl.gz env-golden/rekey-*.jsonl.gz | tail -17
echo "== unit tests =="
taskset -c 0-4 cargo test --release -j5 --offline > $LOG/test.txt 2>&1
grep "test result" $LOG/test.txt | sed -n 1p
echo "== tamper test of the new trace keys =="
python3 tools/tamper_ghost.py ./target/release/envcheck $LOG env-golden/ghost-0.jsonl.gz env-golden/ghostmix-0.jsonl.gz env-golden/rekey-0.jsonl.gz | tail -1
echo "== C interface =="
run() { tools/run_ref.sh "$@" 2>&1 | grep -v -e pynvml -e FutureWarning; }
run 0 /rs/python/verify_search.py ghost $LIB 16 $GHOST 1 > $LOG/ghost.txt &
run 1 /rs/python/verify_search.py copy $LIB $COPY 1 > $LOG/copy.txt &
run 2 /rs/python/verify_search.py rekey $LIB $REKEY 1 > $LOG/rekey.txt &
run 3 /rs/python/verify_search.py eval $LIB $EVAL 1 > $LOG/eval.txt &
run 4 /rs/python/verify_vec_env.py random $LIB 16 $VEC 1 > $LOG/vec.txt &
wait
cat $LOG/ghost.txt $LOG/copy.txt $LOG/rekey.txt $LOG/eval.txt $LOG/vec.txt
# Rekeyed single rolls from fixed states against the uniform roll pools (independent draws).
for t in 2 5 8 11; do run $((t % 5)) /rs/python/verify_search.py rolls $LIB $t $((REKEY * 2)) $t > $LOG/rolls-$t.txt & done
wait
cat $LOG/rolls-*.txt
# The same on five threads (results must not depend on the thread count).
run 0-4 /rs/python/verify_search.py copy $LIB $((COPY / 5)) 5 | tee $LOG/copy5.txt
run 0-4 /rs/python/verify_search.py eval $LIB $((EVAL / 5)) 5 | tee $LOG/eval5.txt
run 0-4 /rs/python/verify_search.py ghost $LIB 16 $((GHOST / 5)) 5 | tee $LOG/ghost5.txt
for name in ghost copy rekey eval vec copy5 eval5 ghost5 rolls-2 rolls-5 rolls-8 rolls-11; do grep -q "^ALL EQUAL" $LOG/$name.txt || { echo "FAILED: $name"; exit 1; }; done
# set_shop_item: its trace family, tamper test and C-interface checks (tools/check_shopset.sh).
tools/check_shopset.sh $1 | tee $LOG/shopset.txt
grep -q "^SHOPSET CHECKS PASSED" $LOG/shopset.txt
sha256sum target/release/libsap_engine.so | tee $LOG/sha-after.txt
cmp $LOG/sha-before.txt $LOG/sha-after.txt
echo "ALL CHECKS PASSED"
