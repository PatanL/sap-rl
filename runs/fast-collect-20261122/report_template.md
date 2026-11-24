# SAP training speed: policy size, collection and the Python ceiling

@@TOP@@

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

@@GPU@@

Weight-update rows include batch packing from CPU, sequence evaluation, V-trace, policy/value/entropy and auxiliary losses, backward, gradient clipping and one AdamW step. LR was zero; moments were allocated during warmup, and every parameter was checked for exact equality afterward. Inputs are archived public observation/action slices with dummy rewards, behavior scores and initial memory. Effective batch is decision rows: length 1/4 for B=1/4, length 8 thereafter. At most 16 sequences per microbatch, with decision-weighted accumulation. Thus B=1,024 still executes one optimizer step. Reported is the median of 3–5 timed steps after two warmups. Real-length, approximately 4,096-decision batches from the collector appear below.

@@FLOOR@@

## 2. Collector prototype, no learning

CPU processes own multiple complete Turtle 1v1 environments. The coordinator collects flat shared-memory requests, groups by policy, retains the GRU state per game and seat on GPU, samples the joint primitive and END subset, and returns sampled actions, behavior likelihoods and next memories. Every live action in the collector is chosen by a network. The CPU engine checks legality, expands freezes, samples the actual battle and advances to native termination. All inference inputs come from the pinned public observation encoder. The separate CPU replay-cap benchmark reuses recorded actions solely to measure environment cost.

Mirrors make both seats eligible. With a frozen resident only seat 0 is eligible. Each slot rotates through three mirrors and three resident assignments per six episodes, with distinct resident IDs. The parent runs use its actual terminal-response and strategy residents plus a frozen parent. The from-scratch runs use matching-size random residents. At battle boundaries the workers publish complete turns, including initial memory, bootstrap observation, action, freeze mask, behavior likelihood/version, native reward, discount, terminal flag and battle label. Before a page can be reused the coordinator copies all fields to learner-owned ragged-dense batches. Timing includes that handoff.

**Denominators.** Seat decisions count both policies. Accepted decisions count learner-eligible rows actually delivered in complete turns. Every run checked that eligible minus accepted equals the change in pending decisions. The historical **315.3/s** counts accepted decisions across collection **and** learning, with 9.22 total seat decisions per inference cycle split into 3.35 per policy call on average. Compare it with projected learned/s. Matches/s counts full native terminations after warmup. Games in flight cross the timing boundaries; one completion per live slot in aggregate was required before timing. Match length, card mix and the fraction of resident decisions vary with the policy.

@@COLLECT@@

Layouts use 4 or 8 cores. `wide` uses one synchronized wave. `pipe` divides the actors into two cohorts so that one can advance games while the other uses the GPU. The prototype uses direct stochastic policy sampling throughout. The historical trainer also had an opening exploration setting. The 4-core comparison kept architecture, weights, sampler, resident roster, CPU set, and collector code constant. Wider batches changed the sampled trajectories, so the two live runs are not paired identical game traces. `wide-parent-random` provides a from-scratch 14.35M control. It completed far fewer matches/s than the trained parent while producing more seat decisions/s; that is why match rate alone is a poor model-size comparison.

**Projection.** For collector rate C and update rate L, serial learned/s = 1 / (1/C + 1/L). Primary L is the median zero-LR BF16 native legacy V-trace step on the run's actual sampled ≥4,096-decision batch, grouped by turn length as in the native learner. The benchmark capped microbatches at 16 turns to fit the shared-GPU budget. The stock outer learner permits 128 turns per microbatch. The benchmark retried with a 128-padded-row ceiling if the 1.5-GiB benchmark allocator hit its cap. A star denotes use of the item-1 BF16 B=1,024 microbenchmark instead. Projections exclude checkpoint I/O, new policy publication, stale-sample rejection and evaluation. Live actor/learner overlap or contention could move the result in either direction. They describe the current one-pass learner; PPO with multiple epochs has a different cost.

@@CROSS@@

### Collection wall-time shares

Exclusive fractions of coordinator wall. Engine covers both shop/round mechanics and battle execution. Encoding covers native public entities, history and legal masks. Forward includes host launch and synchronized GPU computation. Remaining wait/IPC is time awaiting actor replies beyond critical engine/encoding (including worker-side bookkeeping). Host includes routing, transfers, copying delivered trajectories and loop control. For overlapping cohorts, engine and encoding are intersections of the last-ready actor's measured spans with the coordinator's actual blocked interval. CPU work hidden under the other cohort's inference has already overlapped and is not added again. Raw JSON also gives aggregate worker time.

@@WALL@@

## 3. Engine cap and network choice

The cap replays entire action tapes on eight CPU workers with the network, learner transport and inference waits removed. Each worker owns the same seeded games as in its collection run. It validates every action, resolves shop and battle rules and resets on terminal. Per-wave action counts, learner-eligible counts, completions, battles and winner checksums must exactly match the source tape. Timings begin after the same warmup and use wall time through the last worker. `Encoded` includes normal public observations. `Mechanics only` suppresses the returned neural observations. These are **measured, workload-specific Python limits on the permitted eight cores**, not a 20-core extrapolation.

@@CAPS@@

@@DECISION@@

## Scope and reproduction

The shared GB10 initially reported 0% GPU utilization with two existing compute processes holding 5,868 MiB and 170 MiB. All measurements used bounded Docker processes with read-only source and checkpoint mounts, no network, a 12-GiB host limit, and at most eight CPU cores. The four-core CPU set was 5–8; the eight-core set was 5–9 and 16–18. The initial microbenchmark used the existing 6-GiB allocator convention and peaked below 1 GiB reserved. A subsequent 6-GiB allocator initialization failed under transient shared-memory pressure. The collector and subsequent GPU checks used a 1.5-GiB allocator ceiling. Other jobs were left running. No optimized weights were saved, promoted or used to collect data.

**Engine switch.** Set `SAP_ENGINE_ROOT` to a directory containing the sibling `sapai/` and `sapai-gym/` Python import roots. Its default is `/home/dzliu/pzliu/sap-rl/runs/search-throughput-5x-20260912-v1/reference-source`. The launcher mounts that root read-only as `/engine`, and `common.py` derives and checks both import locations. Model weights and example observations are independent read-only data inputs. If the restored checkpoint moves, set `SAP_PARENT_CHECKPOINT` to the 25,772 file with SHA-256 `0669109e9b6ee2d4f0309a3fec83039fc8a7d74f74e3c9f629d2f7026e515d49`. The measured engine was `turtle-v30-food-right-reorder10-v1-20260911`, manifest `5c14dcbb0739bdab026c79d055e9d99df66c6d4632bf76bc09db49dfa0d4d2e8`. The benchmark used PyTorch 2.9.0a0+50eac811a6.nv25.09, TF32 off, local arm64 image `sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470`. A corrected engine may change both game distributions and throughput.

From this directory, `./run.sh micro.py` reproduces the initial sweep. `./run.sh actor_exact.py` measures randomized-order native actor latencies. The following commands reproduce the recommended collector and its CPU replay ceiling.

```bash
@@COMMAND@@
BENCH_CPU_ONLY=1 ./run.sh replay_cap.py repeat-fresh
```

[Machine-readable summary](SUMMARY.json) · [Primary collector result](pipe96-mid.json) · [Primary update result](pipe96-mid-update.json) · [Second seed](pipe96-mid-repeat.json) · [Same-turn update comparison](pipe96-mid-repeat-same-turns.json) · [Actor samples](actor-exact.json) · [Learner samples](micro.json) · [Synchronous collector](collector.py) · [Pipelined collector](async_collector.py) · [Sampled-turn learner timing](real_update.py) · [Action-tape engine cap](replay_cap.py) · [Prior production-speed audit](../speed-audit-20261122/REPORT.md)
