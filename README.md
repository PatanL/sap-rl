# Super Auto Pets RL workspace

This workspace contains four checkouts:

- `SAP-Calculator/` — pinned behavioural/content oracle.
- `sapai/` — Python shop and battle simulator being repaired for RL.
- `sapai-gym/` — masked Gymnasium environment over the repaired simulator.
- `sap-replay-bot/` — reference for Team Wood replay retrieval.

The implemented RL design is documented in
[`sapai-gym/ARCHITECTURE.md`](sapai-gym/ARCHITECTURE.md). It includes the
entity-transformer model, nine-type autoregressive action contract, explicit
two-player self-play environment, recurrent batched collector, V-trace learner,
league state, frozen historical opponents, and DGX training entrypoint.

The implemented replay ingestion, conservative reconstruction, expert
recording, and behavior-cloning workflow is in
[`sapai-gym/IMITATION.md`](sapai-gym/IMITATION.md).

The active 10.2-million-transition league run and its recovery commands are
documented in [`runs/turtle-league-v1/RUN.md`](runs/turtle-league-v1/RUN.md).

## Source of truth

Turtle Pack content and battle behaviour are pinned to
[`robertley/SAP-Calculator`](https://github.com/robertley/SAP-Calculator) commit
`d165eb0a02f8aa0b54d72ed1d5490a44390d07f4` (2026-08-02).

Run the synchronization gate from `sapai/`:

```bash
PYTHONPATH=. python tools/check_sap_calculator_sync.py ../SAP-Calculator
```

The gate verifies the exact 60-pet Turtle roster, all base stats and tiers,
and all 18 rollable Turtle foods against SAP-Calculator's canonical JSON.

## Implemented repair pass

- Added Pigeon, Armadillo, and Wolverine; removed Sloth from the rollable pack.
- Synchronized every Turtle pet's base attack, health, and tier.
- Added Cake, Bread, and token Bread Crumbs.
- Updated materially changed abilities and limits, including Beaver, Bison,
  Crab, Deer, Dragon, Giraffe, Hippo, Ox, Penguin, Rabbit, Rat, Rhino, Skunk,
  Snail, Snake, and Turkey.
- Implemented Pigeon stocking, Bread/Cake shop perks, Crab percentage health,
  Wolverine four-friend-hurt handling, Rhino tier-1 doubling, and current
  trigger limits.
- Rebuilt Turtle shop pools uniformly from current unlocked content.
- Fixed nested faint/knockout reprocessing, target indexing, deterministic
  target selection, garlic's two-damage minimum, and shop tier progression.
- Reworked Gym action IDs around physical slots, removed stale action caches,
  allowed legitimate sell/last-gold-roll actions, and prevented actions after
  ending a turn.

## Verification

```bash
cd sapai
PYTHONPATH=. python tests/test_turtle_pack_current.py
PYTHONPATH=. python tools/check_sap_calculator_sync.py ../SAP-Calculator

cd ../sapai-gym
docker run --rm --gpus=all --ipc=host \
  -v "$PWD/..:/workspace" -w /workspace/sapai-gym \
  -e PYTHONPATH=/workspace/sapai:/workspace/sapai-gym \
  nvcr.io/nvidia/pytorch:25.09-py3 \
  bash -lc 'pip install -q -r requirements-dgx.txt && python tests/test_current_env.py'
```

The repair was additionally exercised through all 3,600 ordered one-pet
Turtle matchups, 10,000 seeded random full-team battles, 5,000 masked Gym
steps, and a 2,048-step CUDA MaskablePPO smoke run.

## Known boundary

This is a strong first correctness baseline, not a claim that every interaction
is proven identical. SAP-Calculator is primarily a battle calculator, so shop
effects such as Pigeon, Cake, and Bread use its canonical metadata plus the
corresponding in-game semantics. The old `sapai` test suite mixes historical
patch expectations and expansion-pack tests; `test_turtle_pack_current.py` and
the synchronization gate are the current Turtle correctness gates. More
cross-engine battle fixtures should be added before long production training.
