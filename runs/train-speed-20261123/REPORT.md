# Parent-arm training speed on the GB10

**Holding the pinned rules, optimizer, learning rates, 50% frozen-match allocation, lag limit, 8,192-decision target and 8 × 24 actor layout fixed**, steady throughput was **911 → 1,845 learned decisions/s (2.02×)**. Stale share in the same window was **0.1% → 4.1%**. The selected path uses rust ordinary battles, overlapped learning, 1536 padded rows on eligible packings, and a queue high-water target of 0.5 learner batches. [Final configuration](results/final-choice.json) · [Per-run timings](results/throughput.json)

The old live run’s first 250 updates averaged 791 learned/s, 40.7% stale, at 8 × 96. In the controlled pre-optimization 8 × 96 run the steady figure was 856/s with 33.7% stale. The optimized proof actually ran **100 updates**, learned 819,948 decisions at 1,850/s overall (1,863/s after warm-up), and saved checkpoint 25,872. It collected 1,957 learner-eligible decisions/s and 2,612 decisions/s across both seats. Its overall stale share was 5.0%. The interval spanned 7.39 minutes. The old matched-layout steady stale share before rounding was 0.06%. [Live baseline](results/incumbent.json) · [Proof](results/proof/result.json)

## Fixed-batch update equivalence

A naive larger minibatch silently changes the loss: the old code weights each small bucket by its share of all decisions and then averages auxiliary terms over the labeled boards or occupied slots *in that bucket*. The new code attaches those denominators to the turns before combining them. It gathers legal padding, computes masked V-trace across sequences, computes the opponent forecast only at the supervised first step, accumulates decision-weighted gradients, clips at 40 and makes one AdamW step. The virtual reference groups still use 128 padded rows and at most 16 turns. The selected physical optimizer batches can exceed 16 turns. On the block-both path the entity encoder, opponent-history encoder and bootstrap retain reference-group GEMM shapes; recurrence, decoding, losses and backward are coalesced. This partition controls BF16 kernel-shape drift. No reward, optimizer rate, action schema, discount or auxiliary coefficient was retuned. [Fast learner](learner_fast.py) · [Predeclared tolerances](correctness.md)

The predeclared limits were absolute loss / relative gradient L2 / relative Adam-delta L2 / maximum absolute post-step weight error. They were 2e−5 / 3e−4 / 1e−3 / 2e−6 in FP32 with TF32 disabled, and 0.003 / 0.035 / 0.035 / 8e−6 in BF16 autocast. The two fixtures contain 16,387 decisions in total.

| Identical saved turns | Rows | Precision | Abs loss Δ | Gradient rel L2 | Gradient cosine | Max abs weight Δ | Adam delta rel L2 |
|---|---|---|---|---|---|---|---|
| CPU-collected | 8194 | FP32 | 6.25e-08 | 0.0001% | 1.0000000 | 1.49e-08 | 0.0008% |
| CPU-collected | 8194 | BF16 | 8.1e-07 | 1.5341% | 0.9998824 | 2.81e-06 | 0.5259% |
| GPU frozen-opponent collector | 8193 | FP32 | 2.84e-09 | 0.0001% | 1.0000000 | 5.96e-08 | 0.0009% |
| GPU frozen-opponent collector | 8193 | BF16 | 0.000812 | 2.9183% | 0.9995768 | 3.01e-06 | 0.5879% |

The comparison restores the exact same weights and optimizer moments for both implementations. The gradient covers every participating parameter *before* clipping. Adam delta uses the reference parameter change as denominator. The public CPU-collection fixture and the independent live-style GPU fixture use real native actions, rewards and bootstrap states. A separate test varies missing battle labels, board presence and occupied slots. [Final independent numerical receipts](results/final-exactness.json) · [Earlier two-fixture qualification](results/crossqual.json) · [CPU unit tests](results/unit-final.log)

| Physical rows | Aligned, BF16 learned/s | Ragged compact, BF16 learned/s | Conservative layout, BF16 learned/s | Conservative + fused losses, BF16 learned/s |
|---|---|---|---|---|
| 128 | 2,259 | 3,078 | 1,782 | 2,567 |
| 256 | 2,564 | 4,292 | 1,976 | 2,269 |
| 512 | 3,003 | 4,364 | 2,535 | 2,717 |
| 768 | 3,595 | 4,958 | — | — |
| 1024 | 3,404 | 4,926 | 2,054 | 2,521 |
| 1536 | 3,374 | 5,176 | — | — |

The original reference measured 1,822 decisions/s, 90 forward/backward passes per saved batch. Timings use median of three synchronized steps after one warm-up on the same fixed turns. Rates in the table are speed measurements, not blanket accuracy passes. Compact 512 reached 4,364/s and passed on the first batch, but gave **10.68%** relative gradient error on the independent frozen-actor batch, exceeding the 3.5% BF16 limit. The numerical screen rejected several faster layouts below the 8-GiB ceiling. The selected cap is the largest tested with passing checks on both fixtures. [Sweep](results/learner-bench.json) · [Compact sweep](results/compact-sweep.json) · [Additional layouts](results/supplement.json) · [Fused sweep](results/fused-sweep.json)

On the independent GPU-collected batch, the physical packing that preserves encoder shapes gave the following real update rates with fused losses. Memory is peak reserved GiB. Only a size that also passes the fixed gradient/weight gates on both fixtures can be used for the final proof.

| Packing cap | Decisions/s | Network/backward groups | Reserved GiB |
|---|---|---|---|
| ref | 1,837 | 90 | 0.62 |
| 128 | 2,311 | 90 | 0.61 |
| 256 | 3,376 | 45 | 0.87 |
| 512 | 4,023 | 23 | 1.41 |
| 768 | 4,250 | 16 | 1.91 |
| 1024 | 4,484 | 13 | 2.43 |

The selected 1,536-row cap measured **4,719 learner decisions/s** on the GPU-collected 8,193 decisions, reducing its 90 backward passes to 9. The matched end-to-end trial reserved 3.78 GiB at peak. [Selected-cap timing and checks](results/upper-bounds.json)

The selected physical learner also uses one Triton recurrence per microbatch for detached V-trace targets, and computes the conditional END distribution once for teacher-forced actions. It keeps native sampling for rollout. This variant passed the four independent precision/fixture checks. [Fused operations](learner_ops.py)

## Asynchronous collection and rebalance

One thread owns the learner and its CUDA stream; the coordinator owns inference, per-seat GRU state and the publication boundary. There is at most one gradient step in flight. The coordinator can return actions and drain completed turns during it. Publication waits for the learner stream and a complete inference wave; subsequent batches see an immutable actor copy. The FIFO checks every segment’s **minimum** sampled version against the current published version on selection. A turn sampled partly more than one version ago is dropped whole. The code retains both seats only for mirror games and only the live seat against a frozen opponent. A high-water queue bounds rollout lead. [Collector](collector_async.py)

Concurrency changes the sampled data even with the same seed and identical objective. In the matched fast-layout serial/async experiment, all 1,479 turns (8,193 rows) in update one were identical. In update two, 342 turns (2,475 rows) were identical. From update three through eight the same-version batches had no identical complete-turn fingerprints. During a pending learner step, additional waves see the previous actor; published weights subsequently change actions, game lengths, resets and frozen assignments. Staleness can rise even though actor idle time falls. Larger decision targets change the number of optimizer steps per learned decision, so the main comparison and proof retain 8,192. [FIFO data-selection evidence](results/data-selection.json)

| Trial | Workers × games | Target | Learned/s overall | Learned/s steady | Delivered/s | Both seats/s | Stale | GPU util |
|---|---|---|---|---|---|---|---|---|
| original | 8 × 96 | 8192 | 833 | 856 | 1,201 | 1,694 | 30.3% | see CSV |
| sync96 | 8 × 96 | 8192 | 1,225 | 1,204 | 1,787 | 2,515 | 31.1% | 24% |
| async96 | 8 × 96 | 8192 | 1,304 | 1,299 | 2,358 | 3,293 | 43.3% | 28% |
| actor96 | 8 × 96 | 8192 | 1,147 | 1,143 | 1,922 | 2,702 | 38.9% | 25% |
| g48 | 8 × 48 | 8192 | 1,456 | 1,505 | 1,844 | 2,555 | 18.0% | 26% |
| verified96 | 8 × 96 | 8192 | 951 | 939 | 1,823 | 2,555 | 45.8% | 27% |
| verified48 | 8 × 48 | 8192 | 1,039 | 1,054 | 1,565 | 2,165 | 30.5% | 30% |
| g24 | 8 × 24 | 8192 | 1,002 | 1,026 | 1,283 | 1,756 | 17.8% | 29% |
| w6g48 | 6 × 48 | 8192 | 963 | 991 | 1,359 | 1,858 | 25.6% | 27% |
| target4096 | 8 × 48 | 4096 | 849 | 848 | 1,644 | 2,286 | 47.0% | 28% |
| target16384 | 8 × 48 | 16384 | 1,153 | 1,215 | 1,497 | 2,060 | 17.5% | 32% |
| rust96 | 8 × 96 | 8192 | 1,168 | 1,150 | 2,252 | 3,151 | 46.4% | 36% |
| rust-w8g48 | 8 × 48 | 8192 | 1,231 | 1,242 | 1,844 | 2,557 | 30.1% | 36% |
| rust-w8g24 | 8 × 24 | 8192 | 1,187 | 1,232 | 1,523 | 2,091 | 17.9% | 34% |
| priority | 8 × 48 | 8192 | 1,273 | 1,289 | 1,928 | 2,656 | 30.9% | 34% |
| fused-final | 8 × 48 | 8192 | 1,295 | 1,314 | 1,952 | 2,698 | 30.6% | 36% |
| tiled-final | 8 × 48 | 8192 | 1,595 | 1,625 | 2,394 | 3,317 | 30.3% | 42% |
| tiled-other | 8 × 24 | 8192 | 1,604 | 1,679 | 2,045 | 2,823 | 17.4% | 41% |
| backpressure | 8 × 24 | 8192 | 1,635 | 1,705 | 1,748 | 2,416 | 3.5% | 42% |
| larger-cap | 8 × 24 | 8192 | 1,780 | 1,845 | 1,908 | 2,642 | 3.8% | 46% |
| matched-original | 8 × 24 | 8192 | 881 | 911 | 883 | 1,232 | 0.0% | see CSV |
| proof | 8 × 24 | 8192 | 1,850 | 1,863 | 1,957 | 2,612 | 5.0% | 46% |

Trials sync96, async96, actor96 and g48 used the faster compact geometry that later failed the second numerical fixture. Their speeds isolate implementation choices. Only the final block-preserving geometry was used in the proof.

Steady excludes the first four updates (one quarter for runs shorter than 16). Both matched rates use updates.jsonl elapsed deltas, so process startup and output cleanup do not enter them. Overall timing of the untouched original also includes startup; the instrumented new clock begins once workers are ready. Delivered denotes completed learner-eligible rows. Stale is discarded / (discarded + learned); shutdown leftovers and ongoing turns are separately accounted. GPU utilization is device-wide NVML sampling. Scratch trials are short, and layouts change sampled trajectories. [Receipts](results/throughput.json) · [Before GPU samples](results/original-gpu.csv)

## Where time goes

| Coordinator critical path, mutually exclusive | % wall |
|---|---|
| engine | 19.69% |
| public observation encoding | 4.28% |
| inference service | 57.63% |
| waiting on learner publication | 12.55% |
| IPC / other worker waiting | 0.89% |
| copy/publish/control | 4.96% |

The learner was active for **56.7%** of the same wall interval; **30.3%** was simultaneous with inference service. 76,483 seat requests in 485 complete waves were served wholly inside learner calls. GPU utilization was 42.0%. These overlapping quantities cannot be added to the coordinator partition. Worker waiting includes time awaiting the next inference response. The profile adds timing/IPC overhead; use uninstrumented trials for rates. [Timeline method](worker_profile.py) · [Breakdown](results/profile-summary.json)

**The largest remaining end-to-end stage is inference service, 57.6% of coordinator wall.** This span includes request preparation, device work, result copies and dispatch, and overlaps the learner. The engine occupied 19.7% of the coordinator critical path. In the isolated Rust-backed environment, the residual Python shop, legality and gym step consumed 88.7% of single-core wall. These different denominators explain why a native shop would help collection while inference remains the first end-to-end target.

| CPU worker lane, pooled worker-wall denominator | % |
|---|---|
| other | 1.16% |
| actor_wait | 62.66% |
| environment | 6.61% |
| shop_api | 22.56% |
| encoding | 6.90% |
| battle | 0.10% |

| Fixed-action boundary, CPU 16 | Median seconds | ms per 1,000 seat decisions |
|---|---|---|
| python | 24.063 | 1319.2 |
| rust | 16.106 | 882.9 |
| cached | 16.026 | 878.6 |
| cached-noobs | 14.288 | 783.3 |
| replay | 0.003 | 0.2 |

On 18,241 identical seat decisions with 762 fights in 64 full matches, real Rust removed **7.957 s (33.1%)** of Python-only environment wall time, with conversion/FFI included. A cached winner removed 8.037 s. Suppressing the encoder with battles already cached removed 1.738 s. Bypassing the remaining shop/environment/legality work using recorded step results removed 14.285 s vs the no-observation/cached-battle arm. This last number is an ideal boundary ablation, not a measured Rust-shop implementation. [Measured alternating replays](results/boundary-bench.json) · [ABI and future shop contract](rust-interface.md)

## Playing strength

Checkpoint **25,872** scored **53.9% [46.88–60.94%]** vs frozen 25,772 on 64 seat-swapped paired seeds. Old-path 26,022 scored 49.2% [42.19–56.25%] on those same seeds. The paired difference is +4.69 pp [-4.69, +14.06]. In a separate direct duel against old-path 26,022, the new policy scored **54.7% [50.78–58.40%]** on 256 unused paired seeds. Intervals bootstrap seed pairs, wins/draws/losses score 1/.5/0. The runs trained for different horizons. [Scores](results/strength.json) · [Common-parent matches](results/proof-vs-parent.json) · [Direct comparison](results/proof-vs-old.json)

All **46,276/46,276** fights actually delegated to Rust during the proof reproduced the same winner when independently reconstructed from the physical END slots and run through Python with the original seed. Python handled 0 unsupported inputs directly. [Proof-run battle audit](results/proof/battle-parity.json)

The direct independent interval meets a zero-margin non-regression test against the old checkpoint.

The launcher mounts the packaged v31c reference read-only. The ordinary outcome-only Rust adapter is pinned separately from the retired v30 search scorer. End-to-end comparisons used the same Python rules, public observation and optimizer. Development used six CPU cores until the explicit release; GPU experiments then used the previously occupied nine-core allocation and an 8-GiB allocator ceiling. The proof used the parent arm, historical four learning rates, no search or league promotion. At a 250-update snapshot interval a 100-update run exercises the initial frozen-parent phase; it does not establish long-run throughput with four resident snapshots.

## Reproducing the short proof

From this report’s directory, with a new output name and the same operator release marker, the command below reproduces the configuration. The checkpoint contains all four optimizer rates. Base is 2e−6, opponent history 1e−4, opponent adapter 1e−5 and END/freeze 5e−5. 

```bash
SPEED_GEOMETRY=block-both SPEED_FUSED_LOSS=1 ./run.sh collector_async.py --concurrent --priority-inference --workers 8 --games 24 --target 8192 --cap 1536 --backlog 0.5 --battle-backend rust --max-lag 1 --updates 100 --seconds 810 --save --audit-battles --out results/reproduction
```

The same preset is available as `./run_verified.sh results/reproduction`. It checks the Rust binary and rule-table digests before launching. [Verified preset](run_verified.sh)

[Pinned container launcher and mounts](run.sh) · [Precision check](exact_final.py) · [Python/Rust fight audit](audit_battles.py) · [Greedy scoring commands and seed counts](score_proof.sh)
