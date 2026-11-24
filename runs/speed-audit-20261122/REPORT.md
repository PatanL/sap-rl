# Ordinary SAP self-play speed audit

**Decision.** Prioritize the policy inference and batching path for faster ordinary self-play. On the pinned Python v30 stack, shop and battle mechanics together occupied **10.37%** of an isolated CPU actor's collection wall time and **4.27%** of one actor's wall time when four actors shared a CPU inference server. In the actual GPU population-league log, the single inference service thread spent **88.45%** of trainer wall time inside policy-group service. The measured **2.98×** planner-assisted training improvement addressed a different workload that scores many hypothetical battles per END.

**Free-engine ceiling.** Holding the GPU run's measured policy-group count and service time fixed, an infinitely fast engine could improve the full training loop by **at most 1.13×**. A port that also changes batch formation, server latency, or actor topology needs a new measurement. An unconditional GPU counterfactual and exact GPU actor engine percentage are unavailable from the saved logs.

## A. Ordinary collection

### Wall time per decision of one actor

A decision is one submitted action by one active seat, including resident seats. Each actor drives three simultaneous matches. A column accounts for the focal actor's entire wall clock, amortized over its submitted decisions. CPU inference used the frozen 14,346,925-parameter checkpoint 25,772, FP32, stochastic actions, no search and no weight updates.

| Exclusive category | Isolated actor, ms/decision | % of actor wall | Focal actor among four, ms/decision | % of actor wall |
|---|---:|---:|---:|---:|
| Shop step, legality, round bookkeeping and reset | 0.730 | 6.90% | 0.797 | 2.90% |
| Construct and resolve the Python battle | 0.367 | 3.47% | 0.378 | 1.37% |
| Encode both players' public observations and masks | 0.188 | 1.77% | 0.197 | 0.72% |
| CPU policy forward for batches containing this actor | 8.863 | 83.76% | 14.200 | 51.60% |
| Wait while the server forwards groups containing only other actors | 0.000 | 0.00% | 10.905 | 39.63% |
| Remaining IPC, observation staging, queue/coalescing and response wait | 0.306 | 2.90% | 0.908 | 3.30% |
| Other actor work, including transition/ring and memory bookkeeping | 0.127 | 1.20% | 0.132 | 0.48% |
| **Total** | **10.581** | **100%** | **27.517** | **100%** |

The isolated column covered **10,206 decisions, 406 battles and 37 match completions in 107.99 s**. The four-actor column covered **5,051 decisions, 210 battles and 19 completions in 138.99 s for the focal actor**. The larger per-actor latency includes waiting for shared inference; it is not an aggregate four-actor throughput figure. Across four contiguous blocks, mechanics occupied 9.91–11.28% and 4.01–4.47% respectively. Including observation encoding, the fractions were **12.14%** and **4.99%**. With all other durations fixed, eliminating mechanics alone gives **1.116×** and **1.045×** focal-actor collection ceilings.

**Method.** The benchmarks ran the actual shared-memory actor, dynamic inference service, joint END/freeze expansion and paired-tape environment. Timers wrapped native calls in the benchmark process. A drain consumed completed dense segments. The isolated run used CPU 16 for the actor and CPU 17 for the service, with the parent routed as both main and frozen resident. The four-actor run shared CPUs 16–19 and used the population league's actual three initial frozen residents plus the main inference replica, at 50% initial mirror probability. The focal actor was actor 0. Both tests used a single Torch computation thread per process, 96 warmup requests, then 2,048 or 1,024 measured request cycles respectively. The four-actor CPU service averaged 3.88 items per policy group including warmup.

**Timer boundaries.** Battle includes construction, team copies, initial priority and the complete `Battle.battle()` call. Observation includes the entity encoder and its mask construction. Shop includes joint validation, native actions, pre-battle board snapshots, score/life settlement, start-turn effects and resets after subtracting nested battle and encoding time. Forward is synchronous CPU wall time inside `policy.act`, including recurrent inference, decoding and sampling. Whole batch time is charged when the focal actor waits for that batch. IPC includes explicit observation writes to shared buffers; remaining actor-side copies fall under other. Forward spans were joined to request/response timestamps across processes and checked against the focal active-slot count for every measured request. Timing excludes model load, process startup and teardown.

These are CPU collection measurements. The production run used a GPU, asynchronous learning and its training-time opening exploration configuration. The actor percentages cannot be transferred directly to GPU training. Raw timings and the partition are in [measurements.json](measurements.json), [actor_cpu.py](actor_cpu.py), [actor_four_cpu.py](actor_four_cpu.py) and [summarize.py](summarize.py).

### One-core engine throughput, uniformly random legal actions

One GB10 performance core (CPU 19), one warmup match per mode, four measured blocks of 16 complete Turtle 1v1 matches, no network and no planner. Wall time includes reset, legal-action selection, both seats and every battle until native termination. All 64 matches completed per row.

| Action sampling and output | Matches/s | Seat decisions/s | Battles/s | Expanded primitive actions/s |
|---|---:|---:|---:|---:|
| **Uniform legal joint action, mechanics only** | **4.270** | **1,965.4** | **51.30** | **2,174.1** |
| Uniform legal primitive, mechanics only | 3.247 | 2,428.0 | 50.54 | 2,428.0 |
| Uniform legal joint action, normal public observations returned | 3.853 | 1,773.7 | 46.30 | 1,962.0 |
| Uniform legal primitive, normal public observations returned | 3.001 | 2,244.1 | 46.71 | 2,244.1 |

The joint action space matches 25,772's interface. Each legal non-END action and each legal END-plus-freeze-subset has equal probability. The primitive sampler chooses uniformly over `core._avail_actions(side)`. These distributions produce different match lengths: 769 battles and 29,460 decisions in the joint run, versus 996 battles and 47,851 decisions in the primitive run. The mechanics-only variant suppresses the public observation arrays returned to an absent network. It still executes validation, shop rules, round bookkeeping, Python battle and the native terminal condition. For each sampler, all four seeded blocks matched the observed variant's counts of matches, battles, decisions and expanded actions. [Benchmark](engine_only.py) · [Mechanics-only blocks](engine-core-only.json) · [Normal-environment blocks](engine-only.json)

### Actual population league, existing logs only

All 1,024 updates in `../population-league-25772-v1/train-from-000000-u8b7b290.log`, 4,405,349 accepted learner decisions. The final recorded cumulative rate was 315.3 decisions/s. The training-loop elapsed time is **13,969.7–13,974.1 s** given that rate's printed precision. Percentages below use its midpoint, 13,971.9 s. This clock covers the learner loop and excludes subsequent evaluation.

| Learner wall-clock accounting | Seconds | Share |
|---|---:|---:|
| Collect/drain sufficient experience | 10,716.809 | 76.70% |
| Weight-update phase | 3,174.195 | 22.72% |
| Residual, including checkpointing and logging | about 80.9 | 0.58% |

Within the two explicitly timed phases the split is 77.15% collection and 22.85% learning. The collection timer measures the learner waiting for and receiving data. It includes many activities besides simulation.

The parallel inference thread logged 626,709 service cycles and 1,727,223 policy groups. A cycle carried about **9.22** active decisions, split into model forwards averaging **3.35** decisions per policy group. The service timer covered 12,472.3 s and its policy-group region covered **12,357.6 s**, or 7.15 ms/group. The separate coalescing timer covered 1,281.7 s. These server times overlap the learner's phases above. The policy-group timer includes tensor staging, model execution, synchronization and copying responses; it is not a CUDA kernel timer. The logs omit the focal actor's shop/observation/battle spans and a separate GPU forward-only span.

For the same number of learned decisions, fixed original grouping and fixed policy-group service costs, the single serial service thread alone needs 12,357.6 s. Hence the **1.13× upper bound** is 13,971.9 / 12,357.6. It already grants perfect overlap of all remaining work and zero coalescing delay. If grouping or service cost can change, the logs only give a much looser weight-update-only ceiling of **4.40×**, conditional on retaining the measured weight-update time. Neither figure is an observed Rust speedup. [Log arithmetic and SHA-256](population-log.json)

## B. Why 5.75% of fights went to Python

The two shipping benchmark blocks sent **5,944 of 103,312** battle requests to the Python scorer. They logged aggregate counts, not a reason per fight. The precise historical split among the following categories cannot be recovered from those receipts. These counts refer to the planner, not ordinary unassisted league play.

The pinned Turtle shop pool has 60 rollable species. At encode time, 21 use the battle-passive representation, 28 have a named native kind (including partially supported Spider), and 11 are unsupported at all three levels. [Two-block scoring receipts](../planner-throughput-25772-v2/update-ship-2/result.json). One unsupported slot on either side sends every seed for that board pair to Python.

| Missing rollable species | Work needed for native coverage |
|---|---|
| Badger, Hedgehog | Adjacent or board-wide faint damage, equipment interactions, simultaneous deaths and cascading priority. |
| Elephant, Snake | Self/friend-ahead after-attack events, repeated shots, random enemy selection and trigger caps. |
| Ox | Snapshot the friend-ahead faint event, grant perk/stats, enforce per-turn triggers through simultaneous removal. |
| Dog | Listen to friendly summons, implement level-specific stat effects and exact random choice where applicable. |
| Rat | Insert Dirty Rats into the opposing team, respecting space, placement and enemy summon observers. |
| Shark, Fly | Post-removal friendly-faint observers, stat growth or replacement summons, counters, recursion exclusions and event order. Both are explicitly delegated. |
| Tiger | Capture the supported pet and repeat the right event at Tiger's level, including simultaneous deaths and nested abilities. Explicitly delegated. |
| Whale | Swallow and preserve the actual pet payload, then restore the correct pet(s), stats, order and ability behavior on faint. |

Additional delegation paths:

- **Spider during execution.** An encoded Spider can finish natively when it survives. A Spider faint delegates the entire seeded fight for random tier-3 summoning. This needs the exact summon pool, draw sequence, token stats and downstream ability semantics.
- **Copied or stored abilities.** Any `override_ability` or nonempty `swallowed_pets` delegates at input. This includes active Parrot copies. A port needs a representation for ability owner, level, counters and nested payloads. An ordinary unmodified shop-only Parrot can be accepted.
- **Input representability.** Unknown species or unsupported current or birth perk, a new non-passive trigger, a board other than five slots, nonintegral/out-of-domain level or stats, invalid counters, or a nonliving nonempty submitted pet delegates. The field must hold an integer level 1–3, attack 0–50, permanent health −50–50, both temporary stats 0–50, each counter 0–99,999 and positive effective submitted health. [Native encoder](../../sapai-gym/research/public_planner/native_scorer.py). Bread/Cake are represented as combat-neutral equipment here. Missing mandatory JSON fields can raise a schema error; a wrong simulator digest or ABI fails initialization.
- **Bounded loops.** The scorer delegates when it exhausts 256 iterations in one hurt/faint resolution, 64 in one knockout resolution, or 512 combat rounds. Reaching 100% of valid *terminating* fights requires resolving remaining event paths and establishing sound termination behavior. Malformed inputs and nonterminating states need an explicit contract. Python fallback currently uses a two-second per-battle timeout and can return no result.

As a concrete attribution check I rescored a separate saved four-root planner corpus with the existing shipping binary. Of **4,704** calls, **4,558** returned native outcomes and **146** delegated after encoding with Spider present. None were rejected at input in that corpus. The ABI returns a single fallback sentinel for runtime causes; Spider presence cannot disambiguate a guard exhaustion from a Spider faint on the same board. This is a different corpus from the 5,944. [Inventory and level-specific triggers](fallback.json) · [Rust dispatch and guards](../../sapai-gym/research/public_planner/native/battle.rs)

For a full battle port, first collect reason counts on representative endgame search boards, then implement and differentially test the remaining event families against terminal teams **and RNG state**. Direct after-attack and damage handlers can reuse existing primitives. Spider needs a new random-summon path. Tiger/Parrot and Whale require representation changes as well as event handlers. The more involved cases share a requirement for identity-preserving deferred events, copied abilities and post-removal summons. A whole-game port additionally needs shop transitions, paired random tapes, generated food, freezes, combines, legal actions and observation parity. No implementation-time estimate was measured here.

## C. Exact Python reference for replay verification

Use **`/home/dzliu/pzliu/sap-rl/runs/search-throughput-5x-20260912-v1/reference-source`**. The engine import root is its `sapai/` directory, and the battle module is `sapai/sapai/battle.py` beneath the reference root. This persistent workspace copy was verified byte-for-byte for all 105 package Python files against the restored release used to evaluate 25,772 at `/tmp/sap-rl-restore.5dIqhk/runs/search-throughput-5x-20260912-v1/reference-source`. The evaluation bootstrap verifies the release manifest. The population training source has identical sapai package code but differences in the distributed actor, model/training wrappers and opening exploration.

| Pin | Value |
|---|---|
| Simulator revision | `turtle-v30-food-right-reorder10-v1-20260911` |
| Reference manifest SHA-256 | `5c14dcbb0739bdab026c79d055e9d99df66c6d4632bf76bc09db49dfa0d4d2e8` |
| 20-file sapai runtime Python content hash | `a6c0280b34719ac3dc3bd80ae957f2081a1c01b43d45f291b7602e4bc1ddff38` |
| Combined 105-file sapai and Gym Python content hash | `7ffafac0cfd6a2e80854ca885fd34bdaf939b885913701a075b760cecc6f2e59` |
| `battle.py` SHA-256 | `03332ca6c673d4c138b6ffed7c4272a497e9660f42ff875964c327ef98ecd923` |
| Audited local CPU-capable arm64 image | `sap-rl-shared:research-25772`, image ID `sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470` |

Tree hashes use sorted lines of `sha256  relative/path\n`, relative to the reference root. [Per-file hashes](python-files.sha256) · [Checkout differences](reference.json). The workspace `sap-rl/sapai` checkout was at Git `977384ed073be025e2afed76737aed7199631bdd`, reporting simulator v28. The runtime differences were `shop.py`, the revision string in `turtle_pack.py`, and v30's additional `paired_random_tape.py`. The battle, effect, pet, team and data files matched. Git also showed three modified calculator tools and an untracked calculator-fixture test. Use the pinned reference path to remove working-tree ambiguity.

**Tested replay command.** The inputs are two serialized `sapai.Team.state` objects, with physical slot 0 in front and END effects already applied. For the included example use the values below. For another battle set `TEAMS` to the directory containing the converted input JSON files, change the filenames and supply its verified unsigned 32-bit battle seed. The script constructs both teams with `Team.from_state` and runs `Battle(left, right, seed=seed, fast_training=True).battle()`. The Rust scorer and policy network are outside this import path.

```bash
AUDIT=/home/dzliu/pzliu/sap-rl/runs/speed-audit-20261122
REF=/home/dzliu/pzliu/sap-rl/runs/search-throughput-5x-20260912-v1/reference-source
TEAMS="$AUDIT"
taskset -c 19 docker run --rm --network none --read-only --cpus=1 --cpuset-cpus=19 \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -e PYTHONPATH=/ref/sapai:/ref/sapai-gym -w /audit \
  -v "$REF:/ref:ro" -v "$AUDIT:/audit:ro" -v "$TEAMS:/teams:ro" \
  --entrypoint python sha256:4e7c18ff4320f735992bef0ea6dbaf14d58df8f6cbc4bcac7af8657b2bb11470 \
  /audit/one_battle.py /teams/left.json /teams/right.json 42
```

For this fixture the result was **0 (left wins)**. The surviving board was Fish 2/3, Ant 2/2 and three empty slots; the right board was empty. The other outcome codes are 1 for right and 2 for draw. Append `--full-state` to emit both terminal `Team.state` objects, including serialized pet fields. Both output modes were executed successfully. [Runner](one_battle.py) · [Left input](left.json) · [Right input](right.json) · [Expected compact output](battle-example-checked.json) · [Expected full output](battle-example-full.json)

For the replay-oracle comparison, establish the replay version, side order, seed mapping and RNG convention, and preserve temporary stats, perks, counters and copied abilities if available. Supplying the same integer to different RNG algorithms can give different sampled outcomes even under identical rules. Live-game disagreement was outside this audit.

**Operating scope.** All new workloads used CPU-only, network-disabled, disposable containers. CPU affinities were subsets of cores 16–19; the random throughput tests used only core 19. No optimizer ran. Sources and existing results were read-only inputs, and audit outputs are local to this directory.
