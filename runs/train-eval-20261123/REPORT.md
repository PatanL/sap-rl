# Paired v31 progression, parent continuation and frozen opponents

**Checkpoint 26,022 is a confirmed regression against 25,772.** It won 4 of 128 games on the screen: **3.125%** match score, paired 95% interval **0.78–6.25%**. It won 8 of 256 games on 128 unused paired worlds: **3.125% [1.17%, 5.47%]**. There were no draws. It lost from both seats (screen wins: one from seat 0, three from seat 1). 25,772 remains the evaluation champion. I recommend discontinuing these settings; the live run is untouched and the operator decides. [Screen](receipts/parent-lr5-v31c-26022-champ-0669109e9b6e.json) · [Independent confirmation](receipts/parent-lr5-v31c-26022-confirm-0669109e9b6e.json)

The fixed historical panel corroborates the drop. On the same 32 pairs per opponent, 26,022 scored **6.25% [1.56%, 12.50%]** against both v22 and v25. Relative to 25,772 these are paired losses of **84.375 pp** [75.0, 92.19] and **75.0 pp** [62.5, 85.94], respectively. On 465 identical archived public decision states with each policy’s own recurrent history, the parent chose primitive FREEZE on **24** states and the candidate on **350**. Only 60 complete joint greedy actions agreed. This CPU probe identifies an action-distribution change; it does not attribute it to LR, the separate END/freeze head, self-play mixture or stale-data selection. [Panel v22](receipts/parent-lr5-v31c-26022-panel-v22.json) · [Panel v25](receipts/parent-lr5-v31c-26022-panel-v25.json) · [Fixed-state action counts](public-state-drift.json)

## Scoring and watcher

The scorer uses the verified package `turtle-v31-replay-48218b4eb462-20261123`, payload `80ee15221b5fbfd7b641330cdeaebcc011a2c1dc77da3bce6ffb4979a44bf4b5`. Both policies act **greedily**, with separate recurrent state per game and seat, public observations, and their own native action schema. A legacy policy keeps its primitive freeze actions. The current policy may choose a joint END/freeze mask. Greedy evaluation measures deployed action choices and removes action-sampling variance; the live training actors remain stochastic. Shops, battles and game outcomes are still stochastic across seeds.

Each world starts twice with the same independent-seat paired random tape; the candidate changes seat. Match points are 1/0.5/0 for win/draw/loss. The statistic averages the two games per seed and bootstraps whole pairs, 20,000 resamples, percentile 95% interval. The same-policy sanity check scored 50% with each seat-swapped pair balanced. A screen uses 64 common pairs; confirmation uses 128 unused pairs from a separate namespace keyed by the exact two weight hashes. The fixed legacy panel uses a third namespace. Fixed screens are exploratory under repeated looks. A candidate becomes champion in this **evaluation ledger** only if both head-to-head intervals against the preceding ledger champion are strictly above 50%. A clear negative screen also gets an independent confirmation for a regression flag. Training weights and opponent assignment in the live process are independent of the ledger. [Evaluator](evaluate.py) · [Same-policy check](receipts/same-policy.json)

| Fixed yardstick for 25,772 on v31 | Match score | Paired 95% interval | Games |
|---|---:|---:|---:|
| v22, update 2,000 | 90.625% | 82.81–96.88% | 64 |
| v25, update 300 | 81.25% | 71.88–90.625% | 64 |
| Corrected-observer 10,050, `862e65d2…` | Unavailable | — | 0 |

The v22/v25 digests matched the November historical-panel receipts and both loaded strictly. For 10,050, the canonical weights and initialization backup appear in the November deletion inventory. No copy was located in the project, restored release tree, local archive or remaining SAP image; the published release asset lists contain 25,772 and a different v21 10,800. A different weight at a similarly numbered update would change the yardstick. [v22 games](receipts/baseline-v22.json) · [v25 games](receipts/baseline-v25.json) · [Expected 10,050 hash](../search-throughput-5x-20260912-v1/REFERENCE_MANIFEST.json) · [Deletion inventory](../../DELETED-CHECKPOINTS-2026-11-18.tsv)

The watcher reads completed `update-*.pt` files in update order, scores each against the ledger champion, keeps the 25,772 anchor even if the champion changes, and also scores the available legacy panel. Panel changes vs 25,772 use matched seed pairs. Game receipts are atomic and restartable; the watcher atomically adds a row to the progression ledger after a checkpoint’s comparisons finish. A single watcher lock prevents duplicate processing, and a GPU lock serializes this task’s scoring and scratch training. Evaluation is on CPUs 5–9 and 16 with at most 2 GiB reserved per GPU process, sharing the live GPU. The live run and its output directory were read-only to these containers. [Progression ledger](../train-v31-prep-20261123/results/progression.json) · [Watcher](watch.py) · [State-transition test](watch-test.log)

To stop **only the watcher**, run the following in this directory. It finishes its current scan before exiting; the live training process is separate.

```bash
python3 stop_watcher.py
```

To run it again, use `taskset -c 5-9,16 python3 -u watch.py`, or add `--once` for one scan. To run an explicit fresh-seed comparison, use `./run.sh evaluate.py /sap/path/to/candidate.pt /sap/checkpoints/update-025772.pt receipts/my-confirmation.json --mode confirm --pairs 128`. That command reproduces the candidate/opponent-specific confirmation panel; rerunning it is not a second independent confirmation.

## Learning rate and opponents

**2e-6 predates the plateau.** It appears in a v13 resume from update 2,000 on August 28. The September student trained at that base rate, with approximately half mirrors and half frozen opponents, and scored 61.0% [55.5, 66.5] against its 10,050 initializer. Later students, including 20,806 and 25,772, stayed clearly above 10,050 but did not establish successive confirmed gains over their immediate parents. The last September screen was 50.5% [44.0, 57.0] against 20,806 and 64.0% [58.0, 70.0] against 10,050. November’s population run confirmed at 49.375% [46.25, 52.66]. A rate-only explanation remains unproven. [Learning rates, mixtures, comparisons and recommendations](HISTORY.md) · [Saved-source extract](historical-evidence.json)

Through the first 250 live updates, the log records **1,026,542 learned decisions**, **3,712,494 rejected as stale**, and 1,723.5 seconds elapsed: **596 learned decisions/s**. Stale rows comprised 78.34% of the learned-plus-stale total. This is a measured efficiency cost of the current games-in-flight/update/lag combination. It gives no direct sign for playing strength. For a future throughput test, compare a larger decision target such as the historical 8,192 with the current 4,096 while holding optimizer and opponents fixed. Widening the lag limit also changes the off-policy distribution.

## Optional frozen-opponent collector

[Collector entry point](collector_join.py) · [Worker and inference routing](frozen_wide.py) · [Episode policy pool](opponents.py)

`--frozen-share` defaults to zero. At a positive share, each new match independently selects a mirror or frozen adversary, chooses the frozen seat uniformly, and pins that model until terminal. Before a run snapshot is available the frozen allocation goes to 25,772. Subsequently it is split equally between 25,772 and a uniform draw from the latest `--pool-size` regular snapshots (default four). Snapshots enter at `--checkpoint-every` intervals. An old model remains reachable for any match already using it even after it leaves the sampling pool. Resume seeds the pool from preceding saved checkpoints beside the selected wide checkpoint, subject to the v31 payload and architecture checks. This option targets the parent architecture.

Both policies sample via the actual public network. Each seat keeps its own GRU state and receives the battle result. A frozen seat never adds rows to the learner page. Mirror games deliver both seats; frozen games deliver only the live seat. Workers check episode-stable policy IDs and the coordinator rejects a frozen slot reaching the learner. Final accounting requires live decisions = completed delivered rows + unfinished live rows, and delivered rows = trained + stale + unused. The live trainer’s imported files were unchanged. An example for a **separate authorized continuation** is:

```bash
./run.sh collector_join.py train --arm parent --parent /sap/checkpoints/update-025772.pt \
  --frozen-share .5 --pool-size 4 --checkpoint-every 250 --lr-scale 1 --base-lr 2e-6 \
  --updates "$APPROVED_UPDATES" --output /job/scratch/new-continuation
```

On resume `--lr-scale 1` preserves the rates already stored in that checkpoint. The two proof runs below each started from 25,772 and used 12 nonzero optimizer steps at the same 5× rates, 4 workers × 32 games, 2 cohorts, 2,048 decisions per update, and checkpoints every 4 updates. Only scratch artifacts were written.

| Measured quantity | Mirrors only | 50% frozen games |
|---|---:|---:|
| Learned decisions / timed seconds | 24,643 / 42.34 | 24,670 / 46.27 |
| Learned decisions/s | **582** | **533** |
| Learned decisions/s, updates 3–12 | 651 | 583 |
| All seat decisions/s | 834 | 881 |
| Frozen seat decisions / frozen learner rows | 0 / 0 | **10,513 / 0** |
| Stale learner-eligible rows | 8,708 | 4,108 |
| Peak PyTorch GPU reservation | 618 MiB | 792 MiB |

The mixed run assigned 101 mirrors, 87 parent matches, and 25 snapshot matches after reset. Both snapshots appeared on both seat orientations. The snapshot policies at updates 25,776 and 25,780 took 362 and 361 decisions. Throughput was 8.4% lower on learned decisions/s in this one comparison; pool warm-up, changed trajectories and live-run contention affect that number. The assignment counts include started matches still in flight. This is a collector test, not a strength comparison. [Mirror receipt](scratch/off/RESULT.json) · [Frozen receipt](scratch/on/RESULT.json) · [Benchmark commands](bench.sh)

## Engine follow-up

The current pin reproduces **685/703** previously inference-recovered, representable inputs, up from 680. Eight old misses now match. **Three formerly matching Honey Bee turns broke**: the sole sale of a level-one Bee from a level-two Honey bearer yielded one real gold, while the new rule paid two. The level-two Sheep’s Ram pair payment is now correct even in one turn whose separate frozen-inventory mismatch remains. Sleeping Pill uses the shared post-removal battle resolver. The shop matched the fixed-slot calculation in 560 single-faint and 120 simultaneous-death constructed layouts. Real records identify exact slots in some Deer, Spider and Mushroom examples. None of the usable Sheep replays fixes the immediate Ram slots, although all were compatible. [Shop evidence](SHOP.md)
