# What is in this repository, and what is not

Current state, plans and results: `GOAL.md` (the Status section starts with how to resume).

## In the repository

| Path | What |
| --- | --- |
| `GOAL.md`, `*.md` | Goal, status, results, earlier audits |
| `engine-rs/` | Rust battle engine and full game environment (source, tools, data) |
| `engine-rs-pinned/` | The four pinned builds the training and scoring scripts load by hash, libraries included |
| `runs/replay-oracle-20261122/engine-v31/` | Python reference engine (source of truth for rules) |
| `runs/replay-oracle-20261122/engine-v32/` | Copy for the next rule changes (no change yet) |
| `runs/train-opt-20261123/` | Current collector, learner, distillation, benchmarks and their results |
| `runs/search-20261123/`, `runs/train-ghost-20261123/` | Search harness, ranked-board pools (training side and yardstick side) |
| `runs/client-oracle-20261123/` | Real-client replay injection: scripts, report, findings R1–R20 |
| other `runs/*-2026112[1-3]/` | The rest of the November 2026 campaign: code, reports, results under 1 MB |
| `scripts/` | Earlier campaign scripts |
| `patches/` | Uncommitted local changes in the `sapai` and `sapai-gym` checkouts |

## Other repositories this project uses

| Local folder | Repository | Commit |
| --- | --- | --- |
| `sapai/` | https://github.com/PatanL/sapai (branch `sap-rl-v21-share`) | `977384ed073be025e2afed76737aed7199631bdd` |
| `sapai-gym/` | https://github.com/PatanL/sapai-gym (branch `sap-rl-v21-share`) | `35a8083285d92833c4798af25d407838eb88c803` |
| `sap-replay-bot/` | https://github.com/lgtyqz/sap-replay-bot | `37aa925851845fbaac9fa591db167b9e00b0f49d` |
| `SAP-Calculator/` | https://github.com/robertley/SAP-Calculator | `d165eb0a02f8aa0b54d72ed1d5490a44390d07f4` |

To restore the two forks' local state: check out the commit above, then `git apply patches/<name>-uncommitted.patch`.

## Not in the repository (only on the training machine)

- All network weights (`*.pt`, about 56 GB), including the champion `runs/train-opt-20261123/results/selfplay-a` update 27,772, the long run up to 29,772 and the 1.35M student.
- `checkpoints/`, `replay-data/`, `.human-play-data/`.
- Downloaded replays (`raw*/`), raw battle responses with other players' names, the client network log, browser profiles, screenshots and video.
- Result files over 1 MB (per-battle logs, fixture sets); the exact list is at the end of `.gitignore`.
- `engine-rs/golden/` and `engine-rs/env-golden/` (about 800 MB of generated test fixtures) and `engine-rs/target/`.
- Run folders from before 2026-11-21 (about 300 folders of earlier campaigns; not reviewed for this repository, so left out).

Keep this repository private: it holds replay ids and two players' in-game handles used as reference points.
