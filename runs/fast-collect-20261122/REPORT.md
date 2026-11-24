# SAP training speed: policy size, collection and the Python ceiling

**For a fresh run on the GB10, I would start with 1,349,741 parameters and eight actor processes, 96 paired games each, split into two cohorts of four.** The randomly initialized policy completed **15.15 matches/s**, produced **6,093 seat decisions/s**, and delivered **4,814 learner-eligible decisions/s**. The native learner separately processed **6,179 decisions/s** on those sampled turns. Together they project to **2,706 learned decisions/s**. This is a throughput recommendation. Only training on the corrected engine could establish the network capacity needed for playing strength. A second sampling seed under the same recording and timing settings measured **18.08 matches/s** and projects to **3,066 learned decisions/s**.

**Does batching alone deliver several-fold faster training with the current policy?** For the 14.35M checkpoint, on the **same four cores** and the same shared-page collector, going from 4 actors × 3 games to 4 × 48 took collection from **448 to 1,283 accepted decisions/s (2.87×)**. The end-to-end projection using the actual collected turns went from **369 to 805 learned decisions/s (2.18×)**. So the claim is supported for GPU forward throughput, nearly true for collection in this test, and too strong as a general promise for the whole training loop. The eight-core, two-cohort run with the current trained network went further: **2,250 accepted/s and a projected 1,246 learned/s**, 3.95× the historical 315.3/s. That includes extra CPU resources, bulk transport and overlap.

## 1. GPU measurements

| Policy | Exact parameters | Entity width | Transformer layers | Attention heads | Feedforward width | GRU state | History width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small | 389,741 | 80 | 2 | 4 | 320 | 96 | 32 |
| Mid | 1,349,741 | 128 | 4 | 4 | 512 | 192 | 48 |
| Medium | 4,467,213 | 224 | 5 | 8 | 896 | 320 | 80 |
| Released | 14,346,925 | 384 | 6 | 8 | 1,536 | 512 | 128 |

All use the actor’s unmodified `entity-public-reorder-budget-v3-20260911` encoder, native legality masks, two-board opponent history, one four-head history Transformer layer, GRU and autoregressive joint action/freeze/value/auxiliary heads. The three small networks were randomly initialized. The released model used checkpoint 25,772. These tests measure speed; no strength curve for the smaller models exists.

Every cell is **ms per call / decisions per second**. Forward rows use the native eager stochastic `policy.act`, with GPU-resident public inputs and memories, frozen parameters and a fresh autocast context on each call. Each cell is the median of three randomized-order windows, each with two warmups and 11 calls synchronized at window boundaries. BF16 is autocast with native FP32 exceptions; the actual population actor used BF16. FP32 disables autocast. These measurements exclude observation generation, IPC and host-device transfers.

### Policy forward, FP32

| Parameters | 1 | 4 | 16 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| 0.390M | 3.11 / 322 | 3.12 / 1,281 | 3.26 / 4,911 | 3.35 / 19,094 | 3.87 / 66,211 | 8.23 / 124,410 |
| 1.350M | 3.35 / 299 | 3.44 / 1,164 | 3.57 / 4,483 | 3.50 / 18,278 | 6.13 / 41,781 | 17.77 / 57,623 |
| 4.467M | 3.47 / 288 | 3.65 / 1,095 | 3.85 / 4,152 | 4.98 / 12,852 | 11.39 / 22,471 | 40.67 / 25,175 |
| 14.347M | 3.80 / 263 | 3.87 / 1,034 | 5.66 / 2,825 | 8.74 / 7,321 | 25.58 / 10,006 | 100.10 / 10,230 |

### Policy forward, BF16

| Parameters | 1 | 4 | 16 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| 0.390M | 3.94 / 254 | 3.94 / 1,015 | 3.87 / 4,134 | 3.92 / 16,325 | 3.84 / 66,678 | 7.28 / 140,688 |
| 1.350M | 4.00 / 250 | 4.16 / 961 | 4.44 / 3,601 | 4.24 / 15,098 | 4.50 / 56,899 | 11.04 / 92,769 |
| 4.467M | 4.72 / 212 | 4.81 / 831 | 4.84 / 3,309 | 5.17 / 12,386 | 8.97 / 28,526 | 31.26 / 32,759 |
| 14.347M | 4.64 / 215 | 4.63 / 865 | 4.70 / 3,401 | 5.35 / 11,973 | 11.41 / 22,438 | 45.44 / 22,533 |

### One optimizer step, FP32

| Parameters | 1 | 4 | 16 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| 0.390M | 14.53 / 69 | 15.24 / 262 | 19.19 / 834 | 22.55 / 2,838 | 60.66 / 4,220 | 235.48 / 4,349 |
| 1.350M | 20.24 / 49 | 22.26 / 180 | 21.19 / 755 | 27.26 / 2,348 | 68.34 / 3,746 | 256.59 / 3,991 |
| 4.467M | 20.07 / 50 | 20.95 / 191 | 23.53 / 680 | 32.74 / 1,955 | 86.27 / 2,967 | 328.24 / 3,120 |
| 14.347M | 23.74 / 42 | 27.47 / 146 | 29.40 / 544 | 43.63 / 1,467 | 126.25 / 2,028 | 476.87 / 2,147 |

### One optimizer step, BF16

| Parameters | 1 | 4 | 16 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| 0.390M | 17.33 / 58 | 19.26 / 208 | 21.83 / 733 | 26.70 / 2,397 | 66.93 / 3,825 | 258.10 / 3,967 |
| 1.350M | 19.07 / 52 | 21.50 / 186 | 23.79 / 673 | 29.95 / 2,137 | 71.83 / 3,564 | 277.69 / 3,688 |
| 4.467M | 21.75 / 46 | 22.17 / 180 | 27.93 / 573 | 32.91 / 1,945 | 78.52 / 3,260 | 310.71 / 3,296 |
| 14.347M | 27.32 / 37 | 29.56 / 135 | 34.20 / 468 | 40.32 / 1,587 | 95.22 / 2,689 | 360.07 / 2,844 |


Weight-update rows include batch packing from CPU, sequence evaluation, V-trace, policy/value/entropy and auxiliary losses, backward, gradient clipping and one AdamW step. LR was zero; moments were allocated during warmup, and every parameter was checked for exact equality afterward. Inputs are archived public observation/action slices with dummy rewards, behavior scores and initial memory. Effective batch is decision rows: length 1/4 for B=1/4, length 8 thereafter. At most 16 sequences per microbatch, with decision-weighted accumulation. Thus B=1,024 still executes one optimizer step. Reported is the median of 3–5 timed steps after two warmups. Real-length, approximately 4,096-decision batches from the collector appear below.

**Small-batch cost.** The 14.35M BF16 actor took 4.64 ms at B=1 and 4.70 ms at B=16. A local linear fit to those two points gives an effective fixed cost of **4.64 ms per call**, about 99% of B=16 latency. For 1.35M it was 3.97 ms and for 0.39M it was 3.95 ms. These are empirical latency floors covering dispatch, GPU kernels and poor small-batch occupancy. The difference between B=1 and B=16 is within run-to-run noise. At B=7 and B=13 the parent took 4.83 and 4.83 ms, or 1.86× as many forward decisions/s. The historical August collection comparison of 826 to about 1,000/s is ~1.21× and covers work outside the network. In a separate instrumented B=1 run, the union of device kernels and copies occupied 1.11 ms of a 7.06 ms profiled call (medians of five, 535 device events). Profiler overhead changes the wall time, so subtracting these figures from the unprofiled 4.64 ms would be invalid.

## 2. Collector prototype, no learning

CPU processes own multiple complete Turtle 1v1 environments. The coordinator collects flat shared-memory requests, groups by policy, retains the GRU state per game and seat on GPU, samples the joint primitive and END subset, and returns sampled actions, behavior likelihoods and next memories. Every live action in the collector is chosen by a network. The CPU engine checks legality, expands freezes, samples the actual battle and advances to native termination. All inference inputs come from the pinned public observation encoder. The separate CPU replay-cap benchmark reuses recorded actions solely to measure environment cost.

Mirrors make both seats eligible. With a frozen resident only seat 0 is eligible. Each slot rotates through three mirrors and three resident assignments per six episodes, with distinct resident IDs. The parent runs use its actual terminal-response and strategy residents plus a frozen parent. The from-scratch runs use matching-size random residents. At battle boundaries the workers publish complete turns, including initial memory, bootstrap observation, action, freeze mask, behavior likelihood/version, native reward, discount, terminal flag and battle label. Before a page can be reused the coordinator copies all fields to learner-owned ragged-dense batches. Timing includes that handoff.

**Denominators.** Seat decisions count both policies. Accepted decisions count learner-eligible rows actually delivered in complete turns. Every run checked that eligible minus accepted equals the change in pending decisions. The historical **315.3/s** counts accepted decisions across collection **and** learning, with 9.22 total seat decisions per inference cycle split into 3.35 per policy call on average. Compare it with projected learned/s. Matches/s counts full native terminations after warmup. Games in flight cross the timing boundaries; one completion per live slot in aggregate was required before timing. Match length, card mix and the fraction of resident decisions vary with the policy.

| Network / run | Actors × games / cohorts | Wave / policy batch | Matches/s | Seat decisions/s | Accepted/s | Learner decisions/s | Projected learned/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 14.347M / 4c-narrow-parent | 4 × 3 / 1 | 20 / 6 | 2.05 | 595 | 448 | 2,083 | 369 |
| 14.347M / 4c-wide-parent | 4 × 48 / 1 | 316 / 79 | 6.07 | 1,682 | 1,283 | 2,162 | 805 |
| 14.347M / wide-parent | 8 × 48 / 1 | 629 / 157 | 8.29 | 2,334 | 1,762 | 2,844* | 1,088* |
| 14.347M / wide-parent-random | 8 × 48 / 1 | 674 / 170 | 2.31 | 4,060 | 3,072 | 2,183 | 1,276 |
| 0.390M / wide-small | 8 × 48 / 1 | 633 / 158 | 6.50 | 3,761 | 2,602 | 5,645 | 1,781 |
| 1.350M / wide-mid | 8 × 48 / 1 | 616 / 154 | 10.48 | 4,319 | 3,432 | 3,688* | 1,778* |
| 4.467M / wide-medium | 8 × 48 / 1 | 635 / 159 | 7.25 | 3,740 | 3,076 | 4,279 | 1,790 |
| 14.347M / pipe-parent | 8 × 48 / 2 | 315 / 79 | 10.22 | 2,979 | 2,250 | 2,792 | 1,246 |
| 1.350M / pipe-mid | 8 × 48 / 2 | 309 / 77 | 12.44 | 5,147 | 4,106 | 6,137 | 2,460 |
| 0.390M / pipe96-small | 8 × 96 / 2 | 636 / 159 | 11.67 | 6,687 | 4,638 | 5,676 | 2,552 |
| 1.350M / pipe96-mid | 8 × 96 / 2 | 616 / 154 | 15.15 | 6,093 | 4,814 | 6,179 | 2,706 |
| 4.467M / pipe96-medium | 8 × 96 / 2 | 635 / 159 | 12.66 | 6,514 | 5,366 | 4,421 | 2,424 |
| 14.347M / pipe96-parent | 8 × 96 / 2 | 629 / 157 | 14.05 | 4,063 | 3,072 | 2,227 | 1,291 |

Layouts use 4 or 8 cores. `wide` uses one synchronized wave. `pipe` divides the actors into two cohorts so that one can advance games while the other uses the GPU. The prototype uses direct stochastic policy sampling throughout. The historical trainer also had an opening exploration setting. The 4-core comparison kept architecture, weights, sampler, resident roster, CPU set, and collector code constant. Wider batches changed the sampled trajectories, so the two live runs are not paired identical game traces. `wide-parent-random` provides a from-scratch 14.35M control. It completed far fewer matches/s than the trained parent while producing more seat decisions/s; that is why match rate alone is a poor model-size comparison.

**Projection.** For collector rate C and update rate L, serial learned/s = 1 / (1/C + 1/L). Primary L is the median zero-LR BF16 native legacy V-trace step on the run's actual sampled ≥4,096-decision batch, grouped by turn length as in the native learner. The benchmark capped microbatches at 16 turns to fit the shared-GPU budget. The stock outer learner permits 128 turns per microbatch. The benchmark retried with a 128-padded-row ceiling if the 1.5-GiB benchmark allocator hit its cap. A star denotes use of the item-1 BF16 B=1,024 microbenchmark instead. Projections exclude checkpoint I/O, new policy publication, stale-sample rejection and evaluation. Live actor/learner overlap or contention could move the result in either direction. They describe the current one-pass learner; PPO with multiple epochs has a different cost.

On the recommended run an average request wave carried **616 seats**; mean forward batches were **486** for the main policy and 37, 58, 35 for the three residents. The **1,066** matches and **338,770** delivered rows were measured over 70.37 s. The sampled learner batch had 4,118 decisions in 185 turns, with 666.4 ms per standard update. Using the optional batched loss on that same batch gave 8,744 learner decisions/s and would project to 3,105 learned/s. The headline keeps the production loss path. For a sensitivity check, replacing the measured turn distribution with the uniform length-eight B=1,024 microbenchmark gives 3,688 learner decisions/s and 2,088 projected learned/s.

**Second sampling seed, tape recording disabled.** With the same 1.35M weights and layout, seed 9020 completed 18.55 matches/s, generated 7,634 seat actions/s, and delivered 6,071 accepted/s. With 6,187 timed learner decisions/s, that projects to **3,064 learned/s**, against 2,706 on the primary seed. The second window lasted 60 s and omitted action-tape recording. Shared-machine load can also affect these rates. The unmodified stock grouping of up to 128 turns exceeded the 1.5-GiB allocator cap on the repeat batch.

**Holding learner data fixed.** Four randomly initialized policies processed exactly the same 4,097 repeat-seed public decision rows and turn lengths with zero initial state of the appropriate width. The standard loss gave 0.390M at 6,904/s, 1.350M at 6,137/s, 4.467M at 4,342/s, 14.347M at 2,045/s. The parent needed the 128-padded-row fallback, while all smaller models fit the 16-turn grouping. The 390K learner was 12% faster than 1.35M on identical work. That learner-only advantage did not translate into a higher fresh self-play projection in the measured configurations.

**Matched repeat.** Seed 9020 also ran with the primary 70-second window and action recording enabled. It completed 18.08 matches/s, generated 7,556 seat decisions/s, and delivered 6,019 eligible decisions/s. With 6,250 learner decisions/s on its own sampled turns, it projects to **3,066 learned/s**. The two matched collection windows cover throughput variability across seeds and shared-machine conditions.

### Collection wall-time shares

Exclusive fractions of coordinator wall. Engine covers both shop/round mechanics and battle execution. Encoding covers native public entities, history and legal masks. Forward includes host launch and synchronized GPU computation. Remaining wait/IPC is time awaiting actor replies beyond critical engine/encoding (including worker-side bookkeeping). Host includes routing, transfers, copying delivered trajectories and loop control. For overlapping cohorts, engine and encoding are intersections of the last-ready actor's measured spans with the coordinator's actual blocked interval. CPU work hidden under the other cohort's inference has already overlapped and is not added again. Raw JSON also gives aggregate worker time.

| Run | Engine | Observation encoding | Policy forward | Remaining wait / IPC | Routing, drain and other host work |
|---|---:|---:|---:|---:|---:|
| 4c-narrow-parent | 41.0% | 3.2% | 45.9% | 5.5% | 4.4% |
| 4c-wide-parent | 73.8% | 9.7% | 12.9% | 2.3% | 1.3% |
| wide-parent | 73.2% | 9.0% | 13.5% | 3.2% | 1.1% |
| wide-parent-random | 62.8% | 8.0% | 22.8% | 4.5% | 1.9% |
| wide-small | 76.3% | 8.2% | 9.1% | 4.7% | 1.6% |
| wide-mid | 68.9% | 11.3% | 12.6% | 5.3% | 2.0% |
| wide-medium | 67.9% | 8.3% | 17.4% | 4.7% | 1.7% |
| pipe-parent | 59.3% | 7.5% | 26.4% | 1.8% | 5.1% |
| pipe-mid | 45.6% | 7.3% | 34.5% | 2.9% | 9.6% |
| pipe96-small | 69.7% | 8.9% | 15.2% | 3.3% | 2.9% |
| pipe96-mid | 59.3% | 10.6% | 21.1% | 3.5% | 5.4% |
| pipe96-medium | 56.3% | 7.3% | 30.1% | 3.2% | 3.0% |
| pipe96-parent | 64.1% | 8.7% | 23.1% | 2.2% | 2.0% |

## 3. Engine cap and network choice

The cap replays entire action tapes on eight CPU workers with the network, learner transport and inference waits removed. Each worker owns the same seeded games as in its collection run. It validates every action, resolves shop and battle rules and resets on terminal. Per-wave action counts, learner-eligible counts, completions, battles and winner checksums must exactly match the source tape. Timings begin after the same warmup and use wall time through the last worker. `Encoded` includes normal public observations. `Mechanics only` suppresses the returned neural observations. These are **measured, workload-specific Python limits on the permitted eight cores**, not a 20-core extrapolation.

| Action tape | Observations | Matches/s | Seat decisions/s | Eligible/s | Battles/s |
|---|---|---:|---:|---:|---:|
| wide-parent | Encoded | 18.94 | 5,333 | 4,015 | 222.4 |
| wide-parent | Suppressed | 21.64 | 6,094 | 4,588 | 254.1 |
| wide-mid | Encoded | 26.04 | 10,731 | 8,521 | 264.1 |
| wide-mid | Suppressed | 29.90 | 12,320 | 9,783 | 303.2 |
| pipe96-mid | Encoded | 29.25 | 11,764 | 9,278 | 292.9 |

The eight-core ceiling on the recommended policy's **exact action tape** was **11,764 seat decisions/s**, **9,278 eligible decisions/s**, or **29.25 matches/s** with public observations. The live collector reached 52% of that seat-decision rate. The replay's eligible counter counts decisions when made, while the collector's accepted counter counts complete turns when delivered.

**How far to shrink.** Under the same 8 × 96 two-cohort layout, the 0.390M, 1.350M and 4.467M designs project to **2,552**, **2,706** and **2,424** learned decisions/s, respectively. The 4.47M model buys 3.3× the parameters at a 10% projected speed cost versus 1.35M. The 390K model has only 29% of the 1.35M model's parameters and its projected rate was 6% lower. Different random policies produce different turn lengths and simulator workloads, so these gaps combine model and trajectory effects.

**Keeping the checkpoint is also viable for collection.** The 14.35M trained network on the same 8 × 96 layout completed 14.05 matches/s and delivered 3,072 accepted/s. Its sampled turns project to 1,291 learned/s, 4.1× the historical rate. The 1.35M model projects to 2.1× that rate, with 9.4% of its parameters. The parent and smaller policy play different game distributions.

**Where Python takes over.** On the four-core parent comparison, the mean per-policy batch rose from 6 to 79. Policy forward fell from 45.9% to 12.9% of collection wall, while engine mechanics rose from 41.0% to 73.8%. On the recommended pipelined layout, mechanics plus public encoding account for 70.0% of the critical path against 21.1% for forward. There is no universal batch-size threshold across layouts or policies.

Absolute rates are one timed window per configuration and a second sampling seed for the recommended layout on a shared GB10. They measure fresh random play and a frozen trained checkpoint. The training distribution will evolve. Sampling quality, win rate, stability after weight publication and eventual strength remain unmeasured.

## Scope and reproduction

The shared GB10 initially reported 0% GPU utilization with two existing compute processes holding 5,868 MiB and 170 MiB. All measurements used bounded Docker processes with read-only source and checkpoint mounts, no network, a 12-GiB host limit, and at most eight CPU cores. The four-core CPU set was 5–8; the eight-core set was 5–9 and 16–18. The initial microbenchmark used the existing 6-GiB allocator convention and peaked below 1 GiB reserved. A subsequent 6-GiB allocator initialization failed under transient shared-memory pressure. The collector and subsequent GPU checks used a 1.5-GiB allocator ceiling. Other jobs were left running. No optimized weights were saved, promoted or used to collect data.

**Engine switch.** Set `SAP_ENGINE_ROOT` to a directory containing the sibling `sapai/` and `sapai-gym/` Python import roots. Its default is `/home/dzliu/pzliu/sap-rl/runs/search-throughput-5x-20260912-v1/reference-source`. The launcher mounts that root read-only as `/engine`, and `common.py` derives and checks both import locations. Model weights and example observations are independent read-only data inputs. If the restored checkpoint moves, set `SAP_PARENT_CHECKPOINT` to the 25,772 file with SHA-256 `0669109e9b6ee2d4f0309a3fec83039fc8a7d74f74e3c9f629d2f7026e515d49`. The measured engine was `turtle-v30-food-right-reorder10-v1-20260911`, manifest `5c14dcbb0739bdab026c79d055e9d99df66c6d4632bf76bc09db49dfa0d4d2e8`. The benchmark used PyTorch 2.9.0a0+50eac811a6.nv25.09, TF32 off, local arm64 image `sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470`. A corrected engine may change both game distributions and throughput.

From this directory, `./run.sh micro.py` reproduces the initial sweep. `./run.sh actor_exact.py` measures randomized-order native actor latencies. The following commands reproduce the recommended collector and its CPU replay ceiling.

```bash
./run.sh async_collector.py --model mid --workers 8 --games 96 --cohorts 2 --seconds 70 --tape --out repeat-fresh
BENCH_CPU_ONLY=1 ./run.sh replay_cap.py repeat-fresh
```

[Machine-readable summary](SUMMARY.json) · [Primary collector result](pipe96-mid.json) · [Primary update result](pipe96-mid-update.json) · [Second seed](pipe96-mid-repeat.json) · [Same-turn update comparison](pipe96-mid-repeat-same-turns.json) · [Actor samples](actor-exact.json) · [Learner samples](micro.json) · [Synchronous collector](collector.py) · [Pipelined collector](async_collector.py) · [Sampled-turn learner timing](real_update.py) · [Action-tape engine cap](replay_cap.py) · [Prior production-speed audit](../speed-audit-20261122/REPORT.md)
