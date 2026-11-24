# Training and vector collection on Turtle v31g

The selected 14,346,925-parameter parent arm completed **100 updates** at **2,693 learned decisions/s**, with **4.01% stale**. The historical live path delivered 791/s with 40.7% stale. The selected checkpoint is 25,872. [Proof](results/final-proof/result.json) · [Watcher record](results/final-progression.json)

| 15-core cohort test | Learned/s overall | Collected live/s | Stale | GPU |
|---|---|---|---|---|
| cohort12x16 | 2,634 | 2,805 | 3.68% | 60.6% |
| cohort12x24 | 2,600 | 3,055 | 12.89% | 63.9% |
| cohort12x40 | 2,558 | 3,559 | 26.55% | 62.2% |

The final configuration was chosen from real-training throughput before its scores were read. The screening choice was **cohort12x16**; the 100-update proof is reported separately from the short tuning run. [Selection rule](selection.md) · [Selection receipts](results/chosen.json)

## Same-hardware engine change

The original and replacement runs use CPUs 10–15 and 17–19, 8 workers × 24 games, 1,536 padded learner rows, an 8,192-decision update target, a half-batch queue, maximum lag one, the 25,772 AdamW state and historical rates, and 50% frozen games. Each comparison has 100 updates and begins with new games. The earlier c proof logged every native battle input for an independent audit. The 16-update comparison omits that audit on both engines. The c figures here are from the uncontended final trials. The earlier busy 05:47–05:54 window contributes no baseline.

| Package and trial | Learned/s | Collected live/s | Both seats/s | Games/s | Stale | GPU util |
|---|---|---|---|---|---|---|
| c 16 updates | 1,780 | 1,908 | 2,642 | 8.06 | 3.77% | 45.5% |
| c 100 updates | 1,850 | 1,957 | 2,612 | 8.55 | 5.00% | 46.2% |
| g 16 updates | 2,286 | 2,455 | 3,400 | 10.21 | 3.92% | 54.5% |
| g 100 updates | 2,405 | 2,535 | 3,391 | 11.26 | 4.63% | 60.4% |

| Profiled time share | c | g |
|---|---|---|
| engine | 19.7% | 6.8% |
| encoding | 4.3% | 7.4% |
| inference | 57.6% | 50.9% |
| learner_wait | 12.5% | 27.8% |
| ipc_wait | 0.9% | 1.1% |
| control | 5.0% | 6.1% |
| learner_active | 56.7% | 63.4% |
| overlap | 30.3% | 27.7% |
| worker_wait | 62.7% | 73.1% |

The first six timing rows partition coordinator wall. Learner active and simultaneous learner/inference overlap that wall. Worker waiting has its own denominator, summed worker time. Collection is completed learner-eligible seat decisions. Stale is rejected decisions divided by rejected plus learned decisions. All throughput counts include the cost of both seats, even when only one seat is learned. [Comparison and raw run locations](results/engine-comparison.json)

During the final 12 × 16 proof, one CPU-only checkpoint-loading check briefly used core 16. Excluding startup and all 5 intersecting update intervals gives 2,716 learned/s over 94 remaining updates. The separate matched-layout and tuning trials did not overlap that check. [Timing sensitivity](results/clean-window.json)

## Engine revision and fixed opponents

The scoring and training launchers verify Python package g, revision `turtle-v31-replay-852b274e7eb2-20261123`, payload `d1a4509282965e41b92ed55021e266ca4440c86f52a9055399c5d1eba48d57a2`. The Rust wrapper, rules and shared library come from the read-only v31g pin; the library SHA-256 starts `d8cd765d898f`. The paired evaluator resolves battles in the packaged Python reference. The board recorder and collector use the pinned Rust wrapper. c, f and g have the identical gym payload `3291838d…`, public observations, native action heads, recurrence and optimizer groups. The allow-listed 25,772 initializer (originally v30) loads strictly with its saved Adam moments and four rates. No parameter transform is required to carry it to f or g. Keep the original checkpoint immutable, initialize a new run under the destination stamp, and regenerate its evaluation receipts. A literal same-run resume of a c-stamped checkpoint on g correctly fails the revision check. A strict CPU transfer check on c, f and g compared every restored parameter tensor, optimizer state and parameter group with the original. [Transfer receipts](results/transfer.json) · [New scoring config](config.py) · [Evaluator](evaluate.py)

| Frozen 25,772 vs | Greedy match score | Paired bootstrap 95% CI | Paired worlds |
|---|---|---|---|
| v22 | 90.62% | 82.81–96.88% | 32 |
| v25 | 81.25% | 71.88–90.62% | 32 |

## Ranked board strength and the watcher

Each checkpoint plays 1,000 complete greedy self-play matches. Both pre-battle boards are recorded, including their order, effective stats, perk, level, consumed triggers and copied ability. Ranked fixtures are read only by the evaluator. The exact original ghost sampling order, 4,000 board pairs per turn and 8 fight seeds per pair are retained; the mutable oracle binary is replaced by an outcome adapter to the pinned library. Re-running archived greedy boards reproduced every published score in all four comparisons exactly. [Recorder](record_boards.py) · [Ghost protocol](ghost_g.py) · [Pinned ABI adapter](ghost_fights.py) · [Exact archived control](results/archived-ghost.json)

| 1,000 matches | T3 | T5 | T7 | T10 | T12 | Mean | Recording s | Ghost CPU wall s |
|---|---|---|---|---|---|---|---|---|
| Parent | 40.56% | 39.32% | 33.07% | 24.23% | 18.51% | 31.14% | 117.7 | 23.1 |
| Proof | 40.92% | 38.96% | 31.74% | 25.11% | 18.04% | 30.95% | 116.5 | 14.9 |

Recording seconds span self-play and file writes after loading the model. Ghost CPU wall includes reading fixture pools and all four field controls and excludes waiting for exclusive benchmark cores. The wall-clock receipts separately retain any time waiting to run on the shared benchmark hardware. Future checkpoints reuse the fixed ranked cohort and self-play seeds. The watcher writes board scores and their mean atomically beside the normal paired match and panel scores; identity is bound to checkpoint SHA, engine digest and seed. Start once with `python3 watch_g.py --once --run runs/NAME --progression /absolute/path/to/progression.json`; omit `--once` to poll. The new watcher was exercised in one-shot mode. [Watcher](watch_g.py) · [Board metric](board_metric.py) · [Ranked corpus inputs](results/fixtures.json)

## Proof of playing strength

The final proof scored **53.91% [46.09–61.72%]** against the initializer on 64 paired screening worlds and **54.30% [50.59–58.01%]** on 256 disjoint, seat-swapped confirmation worlds. A win is 1, a draw 0.5. The independent board comparison pairs the two policies on 1,000 greedy world seeds, draws 16 common field opponents per observed seat/turn × 8 battle seeds, and bootstraps whole self-play worlds 10,000 times. Its index difference is +0.19 points [-0.25, +0.64], conditional on the fixed ranked pool. The ordinary watcher index above uses the original ghost sampling scheme. [Head-to-head confirmation](results/final-confirm.json) · [Paired board check](results/final-board-check.json)

The short proof does not statistically establish zero-margin non-regression on both measures. Keep 25,772 as the reference champion.

The separate 8 × 24 engine-change run also produced an update numbered 25,872. Its SHA-256 starts `5e2b30898292`; the selected proof starts `cb96a74c02e1`. The 8 × 24 proof scored 48.05% [44.33–51.95%] vs 25,772 on its fresh worlds and 30.91% on the standard board index. Both runs remain in the record. [Earlier proof progression](results/progression.json) · [Earlier confirmation](results/proof-confirm.json)

On the selected cohort layout, inference service accounted for 51.2% of coordinator wall, learner wait 34.8%, engine 4.2%, encoding 3.4% and IPC plus other control 6.5%. The learner was active 69.7% and overlapped inference for 29.3% of wall. [Selected-layout instrumented run](results/final-profile/result.json)

## Inferred Honey Bee sales

**3/3 turns now match, 0/3 Bees sell for two.** Two level-two Flamingos and one level-two Ant held Honey. Each was pilled; the generated level-one Bee sold for one gold. The saved shop witness now matches the entire recorded endpoint on all three turns. This was the targeted regression check. [Sale traces and gold receipts](results/bee.json)

## Vector environment and collection

All active seats share one GPU GRU bank. Each wave gathers the public observation arrays and serves one sampled forward per active policy identity. The Python stand-in parallelizes persistent scalar environments by shard; the Rust port can implement the same reset/step and output contract. Rewards, terminal bootstraps and versioned behavior log probabilities are captured before reset. Only live-seat turns enter the unchanged learner. Training and rollout overlap on separate CUDA streams, with backpressure and whole-turn lag rejection. [Vector ABI](vector_contract.md) · [Vector collector](vector_collect.py) · [Independent scalar/turn transport tests](results/turns-check.log)

The replay experiment uses every step of 64 complete sampled-parent games, 18,788 seat actions and 64 actual terminal records. It still copies every public observation, samples both policies and updates recurrent state. Actions cannot change tape outcomes. Its learner is disposable at learning rate zero. Games/s is counted at real terminal flags, with no turn cap or incentive for short games. Python games run from turn one. The long enough steady window starts after at least one cohort’s worth of completed games; whole-window rates also count initial in-flight work. [Tape receipt](results/tape/receipt.json) · [Terminal and reset accounting](results/replay-check.log)

| Run | Games in flight | Mean active seats | Seat/s | Games/s | Steady seat/s | Steady games/s | Learned/s | Stale | GPU util |
|---|---|---|---|---|---|---|---|---|---|
| v-python-train-256 | 256 | 420 | 3,699 | 11.61 | 3,665 | 12.52 | 2,391 | 9.74% | 56.7% |
| v-python-act | 1024 | 1,684 | 9,146 | 28.31 | 9,125 | 31.31 | 0 | — | 37.9% |
| v-replay-act-4096 | 4096 | 6,735 | 21,946 | 61.78 | 21,938 | 72.62 | 0 | — | 88.9% |
| v-python-train | 1024 | 1,683 | 5,619 | 14.65 | 5,579 | 16.24 | 1,996 | 51.87% | 56.8% |
| v-replay-act | 1024 | 1,677 | 19,617 | 63.80 | 19,662 | 66.75 | 0 | — | 78.0% |
| v-python-train-512 | 512 | 844 | 4,473 | 13.09 | 4,403 | 14.57 | 2,317 | 29.00% | 56.6% |
| v-python-loose | 1024 | 1,679 | 7,258 | 20.29 | 7,119 | 22.24 | 1,989 | 62.68% | 59.3% |
| v-replay-loose | 1024 | 1,678 | 11,037 | 34.12 | 11,034 | 37.24 | 2,229 | 72.37% | 81.2% |
| v-python-train-128 | 128 | 211 | 3,174 | 10.58 | 3,170 | 11.01 | 2,254 | 0.36% | 55.2% |
| v-replay-train | 1024 | 1,678 | 7,427 | 22.33 | 7,360 | 24.86 | 2,803 | 48.27% | 75.8% |

| Run | Actor service | Vector step | Reset | Turn storage | Learner wait/publication | Learner active (overlaps) |
|---|---|---|---|---|---|---|
| v-python-train-256 | 39.2% | 29.8% | 4.7% | 4.1% | 22.0% | 63.3% |
| v-python-act | 46.5% | 48.9% | 4.6% | 0.0% | 0.0% | 0.0% |
| v-replay-act-4096 | 98.2% | 1.6% | 0.2% | 0.0% | 0.0% | 0.0% |
| v-python-train | 30.8% | 31.3% | 2.4% | 4.7% | 30.5% | 49.8% |
| v-replay-act | 97.1% | 2.2% | 0.6% | 0.0% | 0.0% | 0.0% |
| v-python-train-512 | 31.5% | 33.3% | 3.6% | 4.4% | 27.0% | 58.1% |
| v-python-loose | 44.5% | 43.9% | 3.3% | 7.0% | 0.8% | 66.6% |
| v-replay-loose | 69.0% | 1.8% | 0.6% | 11.2% | 16.8% | 88.6% |
| v-python-train-128 | 54.9% | 26.9% | 4.9% | 4.6% | 8.4% | 64.0% |
| v-replay-train | 38.9% | 0.9% | 0.2% | 5.9% | 53.6% | 67.1% |

With the engine replayed, 4,096 games in flight reached **21,946 seat decisions/s**, 95.0% of this GPU’s independently measured parent acting ceiling. The matched 1,024-game runs with no learning reached 9,146/s on live Python and 19,617/s on replay. Under the half-batch training queue, replay acting fell to **7,427/s** while the unchanged learner absorbed **2,803/s**, with 48.27% stale. The coordinator spent 53.6% of wall time waiting/publishing and 38.9% serving the policy. A four-batch queue tested this tradeoff directly: 11,037 seat/s, 2,229 learned/s and 72.37% stale. Throughput with the Python engine and learning was 5,619 seat/s and 1,996 learned/s at the same 1,024-game width. [Actor-only receipts](results/v-replay-act-4096/result.json) · [Concurrent replay](results/v-replay-train/result.json) · [Larger queue](results/v-replay-loose/result.json)

The original engine-g training proof used the original nine-core layout. Vector experiments have access to CPUs 5–19 and use 12 worker processes for Python. Games per second for replay are traversed recordings. The sweep compares collector capacities and changes which games/turns reach an update. No ranked boards entered training. A 100-update continuation precedes the first periodic frozen snapshot.

## Forward-only network ceilings

| Policy | Parameters | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | Peak decisions/s |
|---|---|---|---|---|---|---|---|
| small | 389,741 | 147,182 | 147,408 | 154,856 | 149,200 | 151,890 | 154,856 |
| mid | 1,349,741 | 88,120 | 93,182 | 87,637 | 94,615 | 95,576 | 95,576 |
| medium | 4,467,213 | 31,783 | 31,797 | 33,257 | 32,708 | 32,847 | 33,257 |
| parent | 14,346,925 | 22,110 | 22,423 | 23,101 | 22,571 | 22,772 | 23,101 |

Full sampled BF16 actor with public observations tiled from real g states. Inputs and seat memory are resident on the GPU; measurement excludes environment, host transfer and learning. Each cell is batch size divided by the median of six synchronized calls after two warm-ups. The random-weight smaller policies use model width / transformer depth / GRU width of 80/2/96, 128/4/192 and 224/5/320 respectively. Their attention head counts are 4, 4, 8, their opponent-history widths are 32, 48, 80 and feed-forward width is 4× model width. The 384/6/512 parent uses its actual checkpoint. At the measured tape mean of 293.6 seat decisions/game, 1,000 games/s requires about 293,562 decisions/s before learning or simulation cost. Even the 0.39M actor’s measured ceiling falls short. At 400 decisions/game the requirement is 400,000/s. No playing-strength inference follows from the smaller-network timings. [All repeated forward measurements](results/forward.json)

## Reproduction

From this directory, `./reproduce.sh results/new-proof` runs the chosen 100-update configuration. For the matched actor-only cases use `SPEED_CPUS=5-19 ./run.sh vector_collect.py --backend python --workers 12 --games 1024 --seconds 155 --out results/new-live`, or substitute `--backend replay` and another output name. Add `--train` to measure the concurrent learner. The launcher uses the pinned container digest, an 8-GiB allocator cap, read-only source mounts and an 870-second hard guard for each GPU invocation. [Proof preset](reproduce.sh) · [Launcher](run.sh) · [Raw small-network timings](results/forward.json)
