# Checkpoint 25,772 and fresh 1.35M self-play preparation

**The collector–learner join passed one zero-learning-rate GPU update per arm.** The parent restored the released weights, all four AdamW parameter groups and their moments. The fresh arm started from random weights and an empty optimizer. Each learner consumed complete native shop turns, produced finite gradients, published its weights, and received completed decisions tagged with the new actor version. Both parameter hashes were identical before and after. These artifacts are preparation receipts. Checkpoint 25,772 was not evaluated or re-scored.

## Engine identity

| Component | Pin |
|---|---|
| Engine source | Byte copy of the replay-oracle engine-v31 package at SHA-256 `1737f28de96f2735eb150f1b1b5118bbafac713d62fb727128273246eda56f1c` |
| Gym source | Matching sapai-gym tree from `search-throughput-5x-20260912-v1/reference-source`, including learner and public encoder |
| New simulator revision | `turtle-v31-replay-1737f28de96f-20261123` |
| Combined runnable payload SHA-256 | `46be7d4104eaed258dae6c452fb3d698dec2bfa2f542bab2a9c8bb68ec509052` |
| Input checkpoint | `/home/dzliu/pzliu/sap-rl/checkpoints/update-025772.pt`, SHA-256 `0669109e9b6ee2d4f0309a3fec83039fc8a7d74f74e3c9f629d2f7026e515d49` |
| Observation revision | `entity-public-reorder-budget-v3-20260911` |

The [engine root](engine) contains the two import roots `sapai/` and `sapai-gym/`. Packaging changes the simulator revision and provenance declaration in the copied Turtle module. The rules remain those of the audited copy. The runner verifies the file manifest and import locations, and stamps the revision and payload hash on run metadata and produced checkpoints. The sole v30 transfer exception is the exact released parent. Resuming a wide run checks the new revision, payload, observation revision, architecture and arm. [Manifest](engine/MANIFEST.json) · [Packager](pack_engine.py) · [Verification](runtime.py)

The collector calls `ParallelSelfPlayEnv` with its Python `sapai.battle.Battle`. The native planner, external scorer module and `libsap_battle.so` are absent from the package and its container mounts. The launcher points the scorer’s actual `SAP_BATTLE_LIBRARY` setting at an unavailable path. Runtime checks verify the battle class, check for a mapped scorer, and reject an external scorer configuration. Native PyTorch CUDA kernels remain available to the policy and learner. [Launcher](run.sh)

## Collection, learning and persistence

The parent has 14,346,925 parameters. It continues at update 25,772 and 221,367,362 prior learner transitions. The 1,349,741-parameter arm has width 128, four transformer layers, four attention heads, feedforward width 512, GRU width 192 and opponent-history width 48. Both arms retain the public observation and joint END/freeze action interface.

Each CPU process runs many paired Turtle games. Two cohorts overlap CPU engine work with batched policy inference. **Both seats sample the latest live policy and both supply trajectories.** This prep uses direct stochastic `policy.act` with no planner or league. The earlier throughput benchmark also measured a mixed resident roster. Its matches/s and training projections should not be assigned to this self-play configuration.

Workers preserve per-seat recurrent state across turns, clear it on match reset, and tag each action with the version that sampled it. Shared Linux memfd pages carry public observations, action and freeze mask, behavior log probability/version, starting recurrent state, reward, discount, terminal flag, battle label and bootstrap observation. The coordinator owns a copy of completed turns before releasing the pages. The pinned V-trace learner consumes length-bucketed segments using decision-weighted gradient accumulation, gradient clipping and one AdamW step per update. Actor weights are published only after the step. The default maximum behavior-policy lag is one update. Final receipts account for learned, stale-discarded and unused delivered rows. [Collector](wide.py) · [Learner adapter](learner.py) · [Joined entry point](join.py)

The real-run default uses the parent’s saved rates for base, opponent history, opponent adapter and END/freeze: `2e-6, 1e-4, 1e-5, 5e-5`. The fresh arm uses the pinned trainer’s generic from-scratch default of `3e-4` for all four groups. This fresh rate has not been tuned for the smaller architecture. `--lr-scale` scales the loaded rates. The learner retains per-turn discount 0.99, single-battle rewards, legacy entropy and the parent auxiliary-loss coefficients.

On an authorized real invocation, checkpoints are saved atomically at the requested cadence and at the final update. They include weights, optimizer state, counters, lineage, training semantics and v31 identity. A resume restores those fields and starts new sampled games. In-flight games and partially collected turns are not checkpointed. The restart uses the checkpoint version in its game-seed namespace. Always use a fresh output directory. The v30-to-v31 change is a training distribution change even with identical starting weights.

## Dry-run evidence

| | Continue 25,772 | Fresh 1.35M |
|---|---:|---:|
| Workers × paired games, split into two cohorts | 4 × 64 | 2 × 96 |
| Decisions consumed / complete seat-turns | 4,101 / 727 | 4,097 / 303 |
| Primitive actions represented | 4,778 | 4,165 |
| Adam step / publication | 25,772 → 25,773 | 0 → 1 |
| Decisions observed from new version after publication | 21 | 6 |
| V-trace loss / pre-clip gradient norm | 0.19762 / 1.26044 | 0.64521 / 0.54904 |
| Effective update LR | 0 in every group | 0 in every group |
| Peak PyTorch GPU reservation | 500 MiB | 130 MiB |
| Timed collection/update interval | 10.63 s | 9.08 s |

The runs ended after short batches of completed shop turns, before a full match completed. A separate CPU transport check drove a terminal and continuing battle through the native environment and the shared-page handoff. Terminal discounts were zero, continuing discounts 0.99, and rewards, next observations, labels, GRU state and versions survived page reuse. CPU-only checkpoint round trips verified identical weights, moment tensors, parameter groups, counters and v31 metadata for both arms. The temporary round-trip files were discarded. [Parent GPU receipt](results/parent/RESULT.json) · [Fresh GPU receipt](results/fresh/RESULT.json) · [Transport check](transport-check.log) · [Parent round trip](checkpoint-parent.log) · [Fresh round trip](checkpoint-fresh.log)

The shared GB10 showed 0% utilization before the first run, with two existing processes holding 5,868 MiB and 170 MiB. Two startup attempts failed while initializing CUDA after starting CPU workers, before any optimizer step. The successful launcher initializes the GPU allocator first, then spawns clean CPU workers with lazily mapped shared pages. Successful dry runs used a 1-GiB allocator cap and at most six CPU cores. No other processes were interrupted. Total successful GPU optimizer steps were **two**, with no persisted trained weights. [GPU preflight](gpu-before.csv) · [Startup diagnostic](cuda-preflight.log)

## Before a real run

1. Claude and the user need to freeze the intended rules. The current staging pin still reproduces the food-order issue and the temporary-HP battle handoff failures. The turn-two heart restoration matches next-shop lives but changes the immediate reward. Decide which settlement convention the training objective should use. The stored-release sale and remaining replay misses are documented for that decision. [Shop, battle and settlement findings](../shop-oracle-20261122/REPORT-2.md)
2. If the engine changes, package the frozen source under a new root. The packager derives the revision from the upstream source hash, adds a gym digest suffix when the gym source changes, and preserves existing output roots. If the gym settlement implementation changes, pass the corrected gym source too, verify the new payload, and rerun the affected replay checks. The supplied engine checkout test results have three v31-only expectation failures, two Dragon tests and one Wolverine test, on top of the v30 baseline. Reconcile those expectations against the accepted rules.
3. Wait for explicit authorization for any parent evaluation or nonzero training. Set the real update budget and allocate CPU/GPU resources for the parent. The fresh arm can run separately if its allocation fits; the dry runs were sequential. Parallel-arm throughput and the learning-rate schedule have not been benchmarked here.

The following commands document the prepared interfaces. The real-training commands were not executed.

```bash
# From this directory. CPU-only compatibility checks against the packaged pin.
./run.sh inspect --arm parent --verify-save
./run.sh inspect --arm fresh --verify-save

# If Claude freezes a newer candidate, create a separate root.
python3 pack_engine.py --source /path/to/frozen/engine-v31 \
  --gym /path/to/matching/sapai-gym --out /path/to/new/engine-root
export SAP_ENGINE_ROOT=/path/to/new/engine-root

# After approval: run either arm in its own allocation and output directory.
./run.sh train --arm parent --workers 6 --games 96 --updates "$APPROVED_UPDATES" \
  --output results/parent-run
./run.sh train --arm fresh --workers 2 --games 96 --updates "$APPROVED_FRESH_UPDATES" \
  --output results/fresh-run

# Continue a wide checkpoint on exactly the same engine payload.
./run.sh train --arm parent --resume results/parent-run/update-NNNNNN.pt \
  --updates "$APPROVED_UPDATES" --output results/parent-resume
```

`SAP_CPUSET` selects the allowed CPU IDs, `SAP_GPU_GIB` sets the CUDA allocator ceiling, and `SAP_ENGINE_ROOT` selects the verified root. The launcher mounts the source read-only and writes only under `results/` and `tmp/`. The supplied defaults are six CPU cores and a 1.5-GiB GPU ceiling. The dry parent used `--max-padded-rows 64` at a 1-GiB ceiling.
