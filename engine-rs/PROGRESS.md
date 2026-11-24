# Rust battle engine: progress

Goal: reproduce the Python reference (runs/replay-oracle-20261122/engine-v31, table hash in
data/turtle_v31.json `engineSha256`) exactly, seed for seed.

## Status: parity reached

Every row matches exactly: per-round boards (name, attack, health, status, level in all five slots
of both teams), outcome, and the next `random_sample()` after the fight (which checks the number
of random draws).

| Golden file | What it is | Rows | Matched |
|---|---|---:|---:|
| golden/v31-340.jsonl | 4,065 real fights x 3 seeds (the acceptance set) | 12,195 | 12,195 |
| golden/v31-fresh.jsonl | 3,693 real fights from the untouched replay batch x 2 seeds | 7,386 | 7,386 |
| golden/v31-fuzz.jsonl | 6,000 random boards x 2 seeds | 12,000 | 12,000 |
| golden/v31-fuzz2.jsonl | 20,000 random boards x 3 seeds | 60,000 | 60,000 |

The fuzz boards over-sample the hard pets (Tiger, Whale, Parrot, Fly, Shark, summoners, Honey and
Mushroom, many equal attacks) and include tokens; summons, Bees, Chicks, Dirty Rats and Zombie
Flies each appear in over a thousand fuzz rows. The reference raised no error on any row.

Speed: about 35,000 battles/s on one core (release build, no trace, golden set, 6.4 attack rounds
per battle). The reference does roughly 300 per core. Nothing has been optimised yet: abilities
are reference-counted clones and most helpers allocate small vectors.

## Layout

- src/rng.rs: NumPy legacy RandomState (MT19937 seeding, rk_interval, permutation, choice with
  and without replacement, random_sample). Unit test against values printed by NumPy.
- src/data.rs: tables from data/turtle_v31.json and data/turtle_v31_tiers.json. Abilities stay
  data-driven; nothing is hand-coded per pet except the same name checks the reference has
  (Tiger, Shark/Fly, Zombie Fly, Ram, Chick, Bee, Parrot).
- src/engine.rs: the port. Function names from the reference are kept in the doc comments:
  Battle.start / attack / check_battle_result, update_pet_priority, battle_phase_start,
  attack_before, attack, attack_after, battle_phase_hurt_and_faint (records, listeners,
  synchronized removal), _flush_post_removal_events with _slot_insert / _write_layout /
  _resync_layouts and the attack-order sort, battle_phase_knockout with the inherited Tiger,
  tiger_func, get_target, and the effects ModifyStats, DealDamage, SummonPet, SummonRandomPet,
  SummonStoredPets, Swallow, RespawnPet, TransferStats, ReduceHealth, ApplyStatus, AllOf,
  GainHealthPercentHighestFriend; status damage rules; every battle trigger in pets.py.
- src/bin/golden.rs: the checker. Exits non-zero on any mismatch and prints the first diverging
  step with both boards, grouped by pet present.
- tools/gen_golden.py, tools/export_data.py: reference-side generators (run in the training image).
- tools/gen_fuzz_fixtures.py: random boards in the fixtures format.

Not ported (never reached in battle): shop effects (StockFood, DiscountFood, RefillShops,
FoodMultiplier, GainGold), Level2And3Friends and the other shop-only targets, TransferAbility as a
triggered effect (the Parrot copy at team construction is implemented). They return an error if a
battle ever reaches them.

## Commands

```bash
cd /home/dzliu/pzliu/sap-rl/engine-rs
cargo build --release -j3
cargo test --release -j3                                # RNG unit test
./target/release/golden --bench                         # acceptance set + one-core speed
./target/release/golden --golden golden/v31-fuzz2.jsonl # any other golden file
./target/release/golden --only <battle id prefix> --show 5
```

Regenerate after the reference engine changes (rules are data plus the port, so re-export both):

```bash
cd /home/dzliu/pzliu/sap-rl
docker run --rm --cpus=3 --network none \
  -v $PWD/runs/replay-oracle-20261122:/w/oracle:ro -v $PWD/engine-rs:/w/rs \
  sap-rl-shared:research-25772 bash -c "
    python /w/rs/tools/export_data.py /w/oracle/engine-v31 /w/rs/data/turtle_v31.json &&
    python /w/rs/tools/gen_golden.py /w/oracle/engine-v31 /w/oracle/fixtures-340.json /w/rs/golden/v31-340.jsonl 3 3"
```

`data/turtle_v31_tiers.json` (the Spider summon list, `pet_tier_lookup_std`) was exported by a
one-off command; re-export it the same way if tiers change. Files written by the container are
root-owned.

## Open first-divergence groups

None.

## Things in the reference that looked ambiguous or suspicious while porting

Ported as they are; listed for the owner of the rules to check against the real game.

1. Hurt abilities fire for a pet that is already at 0 health (a Blowfish killed by the hit still
   shoots, a dead Camel still buffs, a dead Gorilla gains Coconut and triggers Rabbit).
2. After-attack abilities have no health check: an Elephant that died in the exchange still hits
   the friend behind, and a Snake in slot 1 killed by Chili still shoots.
3. A summon that produces nothing refunds its per-turn trigger, but not when a Tiger stands
   behind (the Tiger path returns nested lists, so `possible == [[]]` is never true). A Fly with a
   Tiger on a full board spends a trigger for nothing.
4. SummonRandomPet (Spider) draws its random pet before checking for room, and reports itself
   activated with no summon.
5. SummonPet's default count is keyed on the summoned token: a Whale releasing a swallowed Ram
   summons two Rams, and a released Chick gets `n = Whale level` copies with attack = half the
   Whale's attack, overriding the stored attack.
6. A Mushroom copy is a brand-new pet: per-turn trigger counters reset (a revived Fly or Rabbit
   gets its uses back), experience is 0, status is the species default.
7. ModifyStats skips a target when its permanent health is <= 0, while "alive" elsewhere means
   permanent + until-end-of-battle health > 0. A Dog kept alive only by its temporary health
   cannot be buffed. Swallow sets permanent health to -1, so a target with temporary health >= 2
   would be stored and also stay on the board.
8. ReduceHealth (Skunk) writes the permanent field from the effective health; a target with an
   in-battle temporary health buff has that buff counted again afterwards. At start of battle
   the buffs are already folded, so this only matters for Tiger-repeated or copied Skunks later.
9. ApplyStatus in battle triggers Rabbit for any perk, including Gorilla's own Coconut.
10. StrongestFriend (Butterfly) indexes the team by position in the living list instead of by
    slot. Unreachable with Turtle Pack pets.
11. update_pet_priority draws random tie-breaks every time it is called (once per loop of the
    hurt/faint resolution), and every knockout event draws a random tie value even when it is
    the only event. Harmless, but any change to how often these run changes every later draw.

## Training-environment speed (2026-11-23)

Measured on one fast core (core 6), 25 full matches of random legal play with observations
encoded, 17,931 decisions and 374 battles (`python/time_env.py`):

| Engine | Battles | ms per decision | Decisions per second |
|---|---|---:|---:|
| engine-v31 before the speed edit | Python | 0.458 | 2,182 |
| engine-v31 before the speed edit | Rust | 0.381 | 2,626 |
| engine-v31 with the speed edit | Python | 0.145 | 6,901 |
| engine-v31 with the speed edit | Rust | 0.127 | 7,871 |

**3.6 times faster, with identical behaviour.** Random play has small boards; games from the
trained policy have longer battles, so the Rust share will be larger there. Not yet measured
inside the collector.

What the speed edit is (three changes in `runs/replay-oracle-20261122/engine-v31/sapai`, diff
against `engine-v31-prespeed`):

1. `rand.LazyRandomState`. numpy needs about 65 microseconds to build a `RandomState` and about
   24 to read or restore its state (measured, numpy 2.1). Every pet, food and shop slot built a
   private generator that the owning Player then replaced with the shop's stream. They now hold
   the saved state and build the generator only if something uses it.
2. `Slot.obj` checked `hasattr(obj, "state")`, which evaluated the whole `state` property (another
   state read) on every slot assignment. It now checks the class.
3. `Player._connect_objects` read the shop stream's state once per pet and per shop slot; it now
   reads it once per call. The stream does not advance inside the call.

Checks that the behaviour is unchanged:
- `python/env_trace.py`: 60 full matches, SHA-256 over every legal-action list, action,
  observation, reward and both players' saved state: 60 of 60 digests equal
  `parity/trace-base.jsonl` (also for the packaged copy `packaged/v31-20261123e`).
- `python/copy_trace.py`: 12 matches with a deep copy, a pickle round trip or a
  `Player.from_state` rebuild every seven steps (902 copies), shop random draws live rather
  than taped: 12 of 12 digests equal the unedited engine.
- `tools/gen_golden.py` re-run on the 4,065 development fights: the 12,195 battle traces are
  byte-identical to `golden/v31-340.jsonl`.
- The engine's own test suite gives the same 37 failures as before (369 pass).

What is left in the environment after this: observation encoding is about 60% of the step
(most of it rebuilding the legal-action list), action resolution about 15%. Going further means
porting the shop and the encoder to Rust; decide that from the collector's measured shares.

The Rust battle drop-in (`python/sap_engine_rs.py`, `src/ffi.rs`) is verified but not used by
the collector yet: the collector's engine packaging refuses compiled files, so it needs a
collector-side switch.

## Rule changes carried on 2026-11-23

Both are in `src/engine.rs` and in the Python reference, and every trace file was regenerated from the
Python reference afterwards:

- `check_summon_triggers`: a summon that lands on the other team (Rat's Dirty Rat) does not run that
  team's friend-summoned abilities.
- `friend_hurt_trigger` / `battle_phase_hurt_and_faint`: a Wolverine activation is queued in
  `pending_counters` when its counter reaches four and run by `run_pending_counter` when the loop
  finds no hurt event and no new death, before removal.

| Trace file | Rows | Matched |
|---|---:|---:|
| golden/v31-340.jsonl | 12,195 | 12,195 |
| golden/v31-fresh.jsonl | 7,386 | 7,386 |
| golden/v31-fresh2.jsonl | 6,656 | 6,656 |
| golden/v31-fresh3.jsonl | 1,756 | 1,756 |
| golden/v31-fuzz.jsonl | 12,000 | 12,000 |
| golden/v31-fuzz2.jsonl | 60,000 | 60,000 |

Real-fight scores for this binary are in `runs/replay-oracle-20261122/rust-v31g-5000-*.json`.
Match fingerprints: `parity/trace-v31g.jsonl` (Python battles) and `parity/trace-v31g-rust.jsonl`
(Rust battles), identical. When writing files from the container, pass `--user $(id -u):$(id -g)`.

# Environment port (shop, turn flow, legality, freeze plan, observations)

2026-11-23. The whole training environment in Rust, step for step identical to the Python
reference `runs/replay-oracle-20261122/packaged/v31-20261123g` on the path the collector uses:
`ParallelSelfPlayEnv()` with default arguments, `reset(seed, paired_tape=True)`, then
`joint_action.step_decisions(env, [JointAction(...) or None, ...])` until `terminated`.

## Status

| Acceptance item | State | Evidence |
|---|---|---|
| 1 Rules and flow | done | `envcheck`: 8,668 matches, 2,231,034 steps, 3,925,805 decisions, 135,092 battles, zero mismatches (the given files alone: 2,068 matches, 887,097 steps, 1,485,238 decisions, 31,416 battles) |
| 2 Observations | done | 403,296 observations byte-compared, 4,076,108 hash-compared, zero mismatches (given files: 200,320 and 1,578,010) |
| 3 Coverage | done, gaps listed below | `env-golden/coverage.txt` (from `envcheck --coverage`), 6,600 targeted matches added |
| 4 Tamper test | done | `tools/tamper_test.py`: 18 single-value alterations, all caught at the right match and step |
| 5 Batch interface | done | `src/batch.rs`, `src/ffi.rs`, `python/sap_env_rs.py`; `python/parity_vec_env.py` passes: 48 games on 5 threads (81 matches, 34,306 steps, 3,022,230 observation arrays) and 16 games on 1 thread (34 matches, 14,429 steps), 14 kinds of illegal input rejected without effect |
| 6 Speed | done | table below |
| Battle checks | unchanged | all six `golden/v31-*.jsonl` files: zero mismatches |

Reference set (`env-golden/`, 389 MB): the given files (`random-0..4` 2,000 matches,
`policy-sampled` 48, `policy-greedy` 16, `smoke` 4) plus targeted files written by
`tools/gen_env_targeted.py` with the extended state block: `targeted-0..4` (1,500 matches, all
scenarios), `deep-0..2` (600, one species to level 3 in the late shop), `sealreward-0` (200),
`wolverine-0` (100), `crowd-0..9` (1,400, ten-slot mostly frozen shops), `cowcrowd-0..9` (1,200),
`crowdloose-0..4` (1,000), `cowfull-0..4` (600, the Cow bought out of ten pet and reward slots).
In the 1,350,537 states of the targeted files the checker also compares
the shop stream's position and key words, the tape cursors, reward groups, food stats and which
shop items are freeze-toggled (`x` block). Battles are compared at this level by outcome and by
everything that follows from it, not round by round (that is what `golden/` does).

## Files

- `src/env.rs`: `Seat` (Player + Team + Shop + the per-seat bookkeeping of SuperAutoPetsEnv) and
  `Game` (ParallelSelfPlayEnv + step_with_end_freeze). Reference function names are in the doc
  comments. Shop: check_rules, roll, _roll_slot, levelup, buy. Player: buy_pet, buy_combine,
  combine, buy_food (Cat, eat, level-ups, Rabbit/Seal, the hurt/faint chain), sell, roll, swap,
  end_turn, start_turn, _shop_trigger_order, _notify_friend_gained_perk, _connect_objects. The
  shop use of battle.battle_phase_hurt_and_faint with its stand-in opponent team is ported in
  full (records, listeners, synchronized removal, _flush_post_removal_events with slot layouts,
  Honey/Mushroom, Shark/Fly). Effects in shop mode: ModifyStats, DealDamage, ApplyStatus (perk
  notification), SummonPet, SummonRandomPet, RespawnPet, AllOf, TransferAbility, GainGold,
  StockFood, RefillShops, DiscountFood, FoodMultiplier. Legality: `avail_actions`
  (`_avail_actions` and everything it calls), `resolve_action`, `expand_end_freeze`.
- `src/tape.rs`: RandomTape (BLAKE2b, 8-byte digest, written by hand and unit-tested against
  hashlib), `Stream` (the same with the message prefix prepared), SHA-256 for the checker.
- `src/envdata.rs`: tables from `data/turtle_v31_env.json` (`tools/export_env_data.py`): turn
  rules, roll pools in the reference's order, food rows, observation item/status ids. Both JSON
  files are compiled into the library (`Tables::global()`): **a table change needs a rebuild**.
- `src/data.rs`: additive only (shop trigger and shop effect enums, `food`, `cost`,
  `includingFuture`, the Level2And3Friends flag). `src/engine.rs`: one added entry point,
  `battle_env` (takes the two shop teams slot by slot with counters and a Parrot's copied
  ability, folds the temporary buffs as Battle.__init__ does); no battle rule changed.
- `src/obs.rs`: the observation encoder. `src/canon.rs`: the canonical state dump.
- `src/batch.rs`, `src/ffi.rs` (`sap_env_*`), `python/sap_env_rs.py`: the batch interface.
- `src/bin/envcheck.rs` (checker), `src/bin/envbench.rs` (speed + a SIGPROF sampling profiler).
  New crates, both already in the offline registry: `flate2` (gzip traces in the two binaries)
  and `libc` (the profiler's signal handler). The library itself uses neither.
- `tools/gen_env_golden.py` (now importable; `--ext` adds the extended state block; the default
  output is unchanged: the four smoke matches regenerate byte-identically apart from the Python
  object ids in `toggled`), `tools/gen_env_targeted.py` (biased driver), `tools/run_ref.sh`
  (docker line), `tools/tamper_test.py`, `tools/coverage_report.py`, `tools/profile_report.py`,
  `tools/check_all.sh`, `python/parity_vec_env.py`.

## Commands

```bash
cd /home/dzliu/pzliu/sap-rl/engine-rs
taskset -c 0-4 cargo build --release -j5 --offline
taskset -c 0-4 cargo test --release -j5 --offline        # NumPy RNG, tape (BLAKE2b, Stream, SHA-256)
tools/check_all.sh /tmp/cov.json                          # items 1+2 on every trace file, coverage counts, battle goldens
python3 tools/coverage_report.py /tmp/cov.json            # item 3 table (kept copy: env-golden/coverage.txt)
python3 tools/tamper_test.py ./target/release/envcheck env-golden/smoke.jsonl.gz /tmp/tamper   # item 4
tools/run_ref.sh 0-4 /rs/python/parity_vec_env.py /rs/target/release/libsap_engine.so 48 772000 5 96   # item 5
tools/run_ref.sh 1   /rs/python/parity_vec_env.py /rs/target/release/libsap_engine.so 16 773000 1 40
# item 6 (see the speed section for the exact rows)
taskset -c 2   ./target/release/envbench random --threads 1 --matches 2000 --batch 256
taskset -c 0-4 ./target/release/envbench random --threads 5 --matches 4000 --batch 1024
taskset -c 0-4 ./target/release/envbench replay --threads 5 --repeat 16 env-golden/policy-sampled.jsonl.gz env-golden/policy-greedy.jsonl.gz
taskset -c 0-4 ./target/release/envbench random --threads 5 --matches 1000 --batch 1000 --digest   # digest 31035309134488a8 for any threads/batch
# reference traces (training image; tools/run_ref.sh <cpu> <tool> args...):
tools/run_ref.sh 0 /rs/tools/gen_env_golden.py random /rs/env-golden/NAME.jsonl.gz <matches> <first> <full every N> [--ext]
tools/run_ref.sh 0 /rs/tools/gen_env_targeted.py /rs/env-golden/NAME.jsonl.gz <matches> <first> <full every N> [scenario]
tools/run_ref.sh 0 /rs/tools/export_env_data.py /rs/data/turtle_v31_env.json     # then rebuild
# profile (perf needs privileges here): build with frame pointers into another target dir
RUSTFLAGS="-C force-frame-pointers=yes" CARGO_TARGET_DIR=/tmp/prof cargo build --release -j5 --offline --bin envbench
taskset -c 2 /tmp/prof/release/envbench random --threads 1 --matches 1500 --batch 64 --sample /tmp/s.txt && python3 tools/profile_report.py /tmp/s.txt
```

## Random streams on this path

T(domain) is `RandomTape(seed).uniform(domain)` (BLAKE2b of `version|seed|len:domain|index`, one
cursor per domain). S is the seat's `shop.rs`, a NumPy legacy `RandomState`.

| # | Call site | Stream | Draws |
|---|---|---|---|
| 1 | `SuperAutoPetsEnv.reset` `next_seed_state` x4 | `RandomState(seed % 2**32)` | four `randint(0, 2**32-1)` (32-bit masked rejection). Children 2 and 4 seed the two teams' `seed_state`; 1 and 3 seed shops that are replaced. Nothing else uses this generator on the path. The traces cannot see these values (no reachable draw uses a team's generator); they are checked against NumPy in the unit test `rng::tests::child_seeds_match_numpy`. |
| 2 | `Shop.enable_paired_tape` | T(`player/{seat}/ability-seed`) | one per seat per match; `int(u * 2**32)` seeds S |
| 3 | `Shop._roll_slot` (reset, `Player.roll`, `start_turn`) | T(`player/{seat}/shop/pet`), T(`.../food`) | one per rolled slot, in slot order after `check_rules`; frozen slots draw nothing. Equal weights: `threshold = u * fsum(w)`, running sum |
| 4 | `Shop.levelup` (`tape.tier_up`) | T(`player/{seat}/shop/pet`) | two per reward pair (without replacement), so a level-up shifts every later pet offer |
| 5 | `Player._shop_trigger_order` | S | `choice(arange(k), k, replace=False)` = `permutation(k)` per group of k > 1 equal attacks, highest attack first. Called by `start_turn`, `end_turn` and `_notify_friend_gained_perk` |
| 6 | `Battle.update_pet_priority(team, Team(), S)` inside `_resolve_shop_hurt_and_faint` | S | `permutation(k)` per tie group; once before the phase, once per loop iteration, once at the end: at least three calls per food purchase, whatever the food |
| 7 | `get_target` RandomFriend (Ant, Beaver, Fish, Otter, Seal; Salad, Sushi, Pizza) | the pet's or food's `rs` | `choice(arange(C(n,k)), (1,), replace=False)` = `permutation(C)[0]` |
| 8 | `get_target` Level2And3Friends (Penguin) | the pet's `rs` | `permutation(m)[:n]`, only when m > n |
| 9 | `SummonRandomPet` (pilled Spider) | the pet's `rs` | `choice(pool, (1,))` = bounded draw; made before the room check |
| 10 | `selfplay_env._next_battle_seed` | T(`battle/turn/{turn}`) | one per battle; `int(u * (2**32-1))` seeds the battle (Rust engine) |

"The pet's `rs`" is S after `Player._connect_objects` (Player construction, `start_turn`, `roll`,
end of every food purchase). Before that: a level-up reward pet holds a private copy of S taken
at creation (`Rs::Private`; a Seal reward fed at once draws from it: 48 times in the reference
set); a pet summoned in the shop holds a generator restored from `team.seed_state`
(`Rs::TeamSeed`, materialised on first use; no draw is reachable with Turtle pets); stocked
foods and the stand-in team's Dirty Rats hold a `MockRandomState` (`Rs::Mock`; a draw would be
NumPy's global generator, so it is an error here; unreachable). NumPy algorithms live in
`src/rng.rs` (`interval`, `permutation`, `choice_replace`, `choice_noreplace`).

## Coverage

`env-golden/coverage.txt` is the full table (per pet, trigger and effect the activations at
levels 1/2/3; foods; shop edge rules; draws by shape; match-flow counts). Summary: every native
shop-phase ability of every rollable pet fired at every level it has; all 20 foods bought; all
60 species bought, bought onto a copy, combined on the team and sold; level-up rewards bought
3,902 times (265 more combined onto a copy); Parrot copied 61 species; Honey 328 and Mushroom 95 faints in the shop; shop
summons of 39 species; capacity squeezed by frozen items 1,207, reward overflow 2,111, stock overflow 361 (1, 2 and 3
slots), discarded reward 11, short stock 12; the Cow's refill displacing one slot 562 times and two
slots 242 times, stocking one Milk instead of two 277 times; private-stream draws 48; termination
by lives 8,467, at the turn cap by lives 140 / by wins 8 / exact tie 53; END freeze plans with 1..8
toggles, 18 plans that use the last budgeted actions, 770 END_TURN at the 50-action cap.

Not hit in the kept files (ported, never executed against the reference):
- The Cow's refill stocking no Milk at all (`shop:cow-milks-stocked:0-of-2`). Unreachable on this
  path: it needs ten frozen pet and reward slots left after the purchase, and a purchase always
  leaves nine or fewer (the only slots added before her ability fires are the unfrozen reward
  pair of her own level-up, and those are the ones displaced).
- Dragon's four-per-turn cap and Fly's three-per-turn cap in the shop (same gate as Rabbit, hit
  53 times, and Ox, 2 times).
- `Rs::TeamSeed` and `Rs::Mock` draws (inferred unreachable, see above).
- Triggers no Turtle pet has: BuyAfterLoss, EndOfTurnWith3PlusGold, EndOfTurnWith4OrLessAnimals,
  EatsShopFood (ported), BuyFood (explicit error).

Cow bought out of a full shop (2026-11-23, `cowfull-0..4`; until then every one of the 2,187 Cow
purchases in the kept files had stocked both Milks without displacing anything). Both remaining
`RefillShops` branches now run against the reference: 600 matches, 177,486 steps, 299,560
decisions, 9,127 battles, zero mismatches, no source change needed. Counts from `envcheck
--coverage` on the five files, each equal to the count taken on the Python side by a wrapper
around `sapai.effects.RefillShops` in the generator (calls / matches):
`shop:cow-overflow-removes:1` 562 / 426, `shop:cow-overflow-removes:2` 242 / 212,
`shop:cow-milks-stocked:1-of-2` 277 / 245 (`2-of-2` 1,592). By case: level-1 Cow, nine slots
left, one displaced, two Milks 485; all nine frozen, one Milk 277; shop Cow bought onto a team Cow
that levels up (reward pair first, then her ability at level 2, Milks +2/+4) on ten slots, two
displaced 242, on nine slots, one displaced 77. Of the 1,046 displaced slots 695 were rewards of
a still complete pair (with one displaced the sibling stays and can be bought; with two it is often
the pair just created), 37 a reward already without its sibling, 314 an ordinary pet; 69 had been
unfrozen earlier in the same turn. Unfrozen slots among those left: none 277, one 180,
two 182, three 104, four or more 338. After the purchase the seat plays on (Milk eaten in the same
turn after 1,037 of the 1,081 purchases, next turn reached after 981). Three one-line mutants of
`refill_shops`, built outside the tree, fail on these files: displacing the first unfrozen slot
instead of the last (393 of 600 matches), displacing frozen slots too (547), always stocking two
Milks (245, exactly the matches with a short stock).
How the state is reached (`cowfull` in `gen_env_targeted.py`, only by name): a turn from 9 on
starts with five pet slots and two foods, and three level-ups of team pets in the purchase turn
give 5 - 1 + 2 + 2 + 2 = 10 pet and reward slots if only one of them costs a shop pet. Two are
free combines prepared on the team (a pet with one experience next to a fresh copy), the third a
pet with one experience whose copy (or a Chocolate) waits in the shop; the two foods are what the
reward overflow removes. Both seats sit out 2-5 turns, then prepare and carry that team (a lost
battle costs one life) until a Cow is offered, freeze her if the gold is short, and on the purchase
turn level up, set a freeze pattern (all / none / pets only / all but one / thaw one to three /
random half) and buy her; then free Milks and uniformly random play. In 30% of the seats the
third level-up is the team Cow's own. In 146 seats one combine is done a turn early and its rewards
and all five pets frozen by the END plan. 1,081 of 1,200 seats reached the purchase (turn 9 to 15+).
Commands (one core per file, about a minute each): `for c in 0 1 2 3 4; do tools/run_ref.sh $c
/rs/tools/gen_env_targeted.py /rs/env-golden/cowfull-$c.jsonl.gz 120 $((20000+c*120)) 10 cowfull;
done`, then `taskset -c 0-4 ./target/release/envcheck --threads 5 --report 10 --coverage
/tmp/cov-cowfull.json env-golden/cowfull-*.jsonl.gz`. The generator's last output line carries the
Python-side counts under `cow` (`events`, `matches`, and how far each seat got under `funnel`).

Targeted driver scenarios (`gen_env_targeted.py`): faint, hurt, sell, turn, buy, levelup, bison,
dragon, parrot, overflow, wolverine, scorpion, chocolate, summons, twostack, deep, crowd,
cowcrowd, crowdloose, sealreward, cowfull; both seats can sit out the first 4-10 turns with empty teams
(draws, no lives lost) so the late shop is reached with all lives.

How the kept targeted files were made (`tools/run_ref.sh <c> /rs/tools/gen_env_targeted.py
/rs/env-golden/<name>.jsonl.gz <matches> <first> <full every N> [scenario]`, one core per file):
`targeted-$c`: 300, c*300, 10; `deep-$c` (c 0-2): 200, 2000+c*200, 10, deep; `sealreward-0`:
200, 3000, 10, sealreward; `wolverine-0`: 100, 3400, 10, wolverine; `crowd-$c` (0-4): 80,
4000+c*80, 10, crowd; `crowd-$((c+5))`: 200, 5000+c*200, 20, crowd; `cowcrowd-$c` (0-4): 80,
7000+c*80, 20, cowcrowd; `cowcrowd-$((c+5))`: 160, 8000+c*160, 20, cowcrowd; `crowdloose-$c`:
200, 10000+c*200, 20, crowdloose; `cowfull-$c`: 120, 20000+c*120, 10, cowfull. The script grew between these runs, so running it again
gives different (equally valid) matches for the older families; the traces carry their own
seeds and decisions and do not depend on the script. No reference match failed to finish.

## Batch interface

`python/sap_env_rs.py` (`SapVecEnv`) documents the contract. Choices made here, not in the brief:
- a seat with no decision (ended its turn) sends type -1; anything else for it is illegal input;
- a finished game is skipped: status 1, rewards and battle flags written as zero, everything
  else (observations, terminated, winner, lives, wins) untouched until `reset`;
- status -1 illegal input (game unchanged), -2 internal error (reset needed), -3 never reset;
- rewards are float64, flags int8, lives/wins int32, status int32;
- the output arrays belong to the wrapper object and are reused by every call;
- threads: a persistent std-thread pool; results do not depend on the thread count.

## Speed

`tools/speed_table.sh` (each row three times; the median is shown). Everything goes through
the batch interface with all fifteen observation arrays written for both seats on every step.
Cores 0-4 only; these are the machine's slower cores (2.8 GHz class). The Python figure in the
brief, 7,871 decisions/s, was taken on core 6 (3.9 GHz class); the same script
(`python/time_env.py 25 --rust-battle ...`, random play, Rust battles) on core 2 gives
**3,200 decisions/s** (three runs: 3,168 / 3,200 / 3,203), which is the like-for-like baseline.

| Workload | Batch | Cores | Decisions/s | Matches/s | vs Python on the same core class (3,200/s per core) |
|---|---:|---:|---:|---:|---:|
| Policy tapes (64 matches, 278 decisions each) | 64 | 1 | 305,000 | 1,096 | 95x |
| Policy tapes, each 16 times | 1,024 | 1 | 200,000 | 720 | 63x |
| Policy tapes, each 4 times | 256 | 5 | 938,000 | 3,369 | 59x per core |
| Policy tapes, each 16 times | 1,024 | 5 | 693,000 | 2,491 | 43x per core |
| Random tapes (`random-0`, 400 matches, 720 decisions each) | 400 | 1 | 740,000 | 1,028 | 231x |
| Random tapes (`random-0`) | 400 | 5 | 2,440,000 | 3,387 | 152x per core |
| Random play (2,000 matches from consecutive seeds) | 64 | 1 | 753,000 | 1,039 | 235x |
| Random play | 256 | 1 | 706,000 | 975 | 221x |
| Random play | 1,024 | 1 | 384,000 | 531 | 120x |
| Random play (6,000 matches) | 256 | 5 | 2,150,000 | 2,967 | 135x per core |
| Random play (6,000 matches) | 1,024 | 5 | 1,620,000 | 2,225 | 101x per core |
| Random play (8,192 matches) | 4,096 | 5 | 1,380,000 | 1,902 | 86x per core |

Against the brief's 7,871/s (a faster core): random play on one core is 49x to 96x depending
on the batch size. Every match is played to its end and every row counts all of them, so short
games are not favoured. The five-core policy rows are short runs (70 to 420 ms) and spread by
about +-20%; the load average of the machine was about 9 during all measurements.

Batch size matters: with more games than fit in the cache, each step touches cold game state
and 6.6 KB of output per game, and one core does about half as many decisions per second at
1,024 games as at 64.

What profiling found and what was done (single core, random play; exactness re-checked after
each change with `tools/check_all.sh`, and the `--digest` of 1,000 random-play matches is
`31035309134488a8` before and after, for any thread count and batch size):
1. Legal actions were recomputed six times per step (validation, resolution, two encodes):
   27% of the time. Now cached per seat until the seat changes (the reference caches behind a
   fingerprint; recomputing is idempotent). Team-order keys are packed into one integer.
2. A thread was spawned per worker per call. Now a persistent pool with a shared work counter;
   five-core random play went from 2.7x to about 4.2x one core at batch 1,024.
3. Mask construction walked all 197 actions with a decode each; now only the legal bits, with a
   decode table. Tape messages were formatted per draw; now the prefix is prepared per stream.
4. The benchmark's own per-step digest and byte-wise action choice were a third of the run;
   the digest is optional now and the choice works on words and in parallel.
The first complete build did 212,000 decisions/s on the policy tapes (batch 64, one core;
replay has no action choice in it), the final one 305,000: that 1.44x is the engine-side gain.
Random play went from 242,000 (batch 1,000) to 384,000 (batch 1,024) on one core, but part of
that is the benchmark's own action choice getting cheaper (item 4), so it overstates the engine.

Where the time goes now (batch 64, one core, random play): observation encoding and the copy
into the caller's arrays about 45% (legal actions 13% of the total inside it), action
resolution 30% (shop rolls 13%, of which BLAKE2b for the tape 5%; battles 5%), the benchmark's
action choice 12%. Not done: encoding straight into the output arrays (saves one 3.3 KB copy
per seat), skipping the rows of a seat that did not change, a more compact game state for
large batches.

## Reference quirks noticed (ported as they are)

1. Every food purchase runs the hurt/faint resolution and draws tie-break permutations from the
   shop stream when two team pets share an attack value (at least three priority computations),
   so buying an Apple changes later random outcomes.
2. A level-up reward pet keeps a private copy of the shop stream until the next
   `_connect_objects`. A Seal reward fed immediately draws its random friends from that copy;
   the shop stream does not advance, so the same numbers are drawn again later.
3. Pets summoned in the shop are created with a generator restored from `team.seed_state`, the
   same state for every such pet of a seat in a match. Stocked foods have a MockRandomState.
4. A pilled Rat summons its Dirty Rats onto a throwaway opponent team; they take part in the
   final priority computation (extra draws when their attack of 1 ties).
5. Computing legality registers the current team order as seen (also for a seat that has ended,
   also through the observation encoder). After ten reorders or at the action cap nothing is
   registered, so the observed seen-order count depends on when legality was computed.
6. END_TURN is legal only at zero gold, or as the only action after 50 actions.
7. The turn-two heart is part of the reward's life swing: on turn two a win at 6-6 lives pays
   0.0, a draw pays +0.1 to a seat that was at 5 lives, and so on. The second seat's reward of a
   zero swing is -0.0.
8. Observations are produced for seats that have ended, with masks as if they could act.
9. `buy_combine` fires the team pet's own Buy ability again (Otter, Cow) but no Summoned trigger
   and no Scorpion perk notification.
10. Cat multiplies Canned Food's permanent shop bonus; Cat's two uses are hard-coded, not
    `maxTriggers`. Canned Food buffs the pets in the shop at once, level-up rewards included.
11. Frozen level-up rewards stay across turns and take pet capacity: once there are as many as
    the turn has pet slots (3 on turns 1-4, 4 on turns 5-8, 5 from turn 9), no ordinary pet is
    offered. Frozen ordinary pets are kept on top of that.
12. `ability_counter` and Wolverine's hurt counter reset when a pet levels up (Rabbit gets its
    three uses back); Wolverine's counter counts shop hurts and is carried into the battle.
13. Rabbit spends a use on a pet that ate a Sleeping Pill (no buff, since health is below zero);
    a pilled Seal still buffs its friends first.
14. A Mushroom copy made in the shop sells for 1 whatever its level; a summoned token sells for
    its summoner's level, a Bee for 1.
15. The freeze-toggled set keeps items that have left the shop (bought); its size is observed.
16. Parrots copy at end of turn in attack order with random ties, so what a Parrot behind a
    Parrot holds can depend on a draw.
17. Level-up choices come from the same tape domain as pet offers.
18. `SuperAutoPetsEnv.allow_additional_reorder_*` is never set; the dump prints False.

## Not ported (explicit errors, never silent)

- Anything but the paired-tape reset: `reset(paired_tape=False)`, `seed=None`, seeds outside
  `[0, 2**63)` (refused), `clone` and the identity remapping it needs (`redeterminize` was on
  this list; it is `rekey` now, and `copy` is the Rust-side clone: see "Search and ghost
  primitives"),
  `SuperAutoPetsEnv.step` with an opponent generator / opponent pool, `record_actions=True`,
  `fast_battles=False` (no battle history), the battle-equity estimator, a causal context
  schema (extra observation keys), `Player.reorder`, `sell_buy`, ShopLearn, ExpansionPack1.
  None of these has an entry point in the Rust API.
- Inside the port, `Err("Unsupported: ...")` for: a draw from a MockRandomState; a BuyFood
  trigger; shop-phase use of Swallow, TransferStats, ReduceHealth, SummonStoredPets,
  GainHealthPercentHighestFriend, StrongestFriend, or an unknown effect or target kind; a Buy
  ability with an unknown `triggeredBy`. Through the batch interface these give status -2 for
  that game.

# Search and ghost primitives

2026-11-23. Four additions to the batch interface so that a game can be copied and looked ahead
in, played against recorded boards, and scored. Nothing that existed changed: `sap_env_reset`,
`sap_env_step`, the `Outputs` struct and every array keep their meaning; the 8,668 earlier matches
give the same per-file lines and totals as before (`env-golden/baseline-8668.txt`), and the
random-play digest is still `31035309134488a8`. A fifth, `set_shop_item` (a chosen pet or food
into a shop slot), was added afterwards the same day; it has its own section below ("Shop edit")
with its semantics, checks, commands and gaps, and all of the above still holds with it.

## Interface (`python/sap_env_rs.py`, class `SapVecEnv`; the docstring there is the contract)

A board is int32 `[5, 7]`: rows front to back = team slots 0..4, columns `[kind id, attack,
health, total experience, perk id, consumed triggers, copied species]`. Ids and Parrot sentinels
are those of the battle entry point (`sap_battle`, `python/sap_engine_rs.py`): kind id =
`env.kind_id(name)` = `sap_kind`, perk id = `env.status_id(name)` = `sap_status`, copied species
-2 (not stated: a Parrot copies the friend ahead, `make_team`'s default rule; every non-Parrot),
-1 (a Parrot that copied nothing), k (the kind a Parrot copied). The one thing added to that
encoding: an empty slot is a row with kind id -1 (`EMPTY_ROW`), because a `[5, 7]` array has no
count. `env.encode_board(specs)` / `env.decode_board(board)` convert from and to the spec dicts
of the ghost pool.

| Python | C entry point | What it does |
|---|---|---|
| `copy(src, dst)` | `sap_env_copy` | game `dst[k]` = exact independent copy of game `src[k]` (the whole `Game`), output rows of `dst[k]` copied from the rows of `src[k]` |
| `rekey(games, seeds)` | `sap_env_rekey` | the reference's `ParallelSelfPlayEnv.redeterminize(seed)` on the paired-tape path |
| `set_team(game, seat, board)` | `sap_env_set_team` | `player.team = make_team(specs); player._connect_objects()` of `ghost_worker.py` |
| `set_gold(game, seat, gold)` | `sap_env_set_gold` | `player.gold = gold` |
| `set_shop_item(game, seat, slot, item, frozen=False)` | `sap_env_set_shop_item` | shop slot `slot` holds the named pet or food as if the shop had just rolled it there (see "Shop edit: `set_shop_item`") |
| `battle_events(carried=False)` -> `(turn [N], seed [N], teams [N,2,5,7])` | `sap_env_battle_events` | the battle the game's last accepted step resolved: one-based turn (0: none), battle seed, both teams as handed to the battle |
| `eval_vs(game, seat, boards[M,5,7], samples, seed)` -> int32 `[M, 3]` | `sap_env_eval_vs` | wins, draws, losses of the seat's would-be battle team against each board; the game is not changed |
| `eval_vs_batch(games, seats, boards[K,M,5,7] or [M,5,7], samples, seeds)` -> `[K, M, 3]` | `sap_env_eval_vs` | the same for K (game, seat) pairs on the environment's threads |
| `step(actions, masks, which=mask)` | `sap_env_step_which` | only the selected games step; the others and their output rows are untouched |
| `observe(which)` | `sap_env_observe` | rewrite the output rows of the selected games from their current state |
| `kind_id`, `kind_name`, `status_id` | `sap_env_kind`, `sap_env_kind_name`, `sap_env_status` | the board ids without `sap_engine_init` |
| `food_id`, `food_name` | `sap_env_food`, `sap_env_food_name` | food ids (index into the food table; their own id space) for `set_shop_item` |

Choices made here, not in the brief:
- **copy**: refused as a whole (ValueError, nothing changed) when a game is in `dst` twice or in
  both lists, or a source was never reset; so every copy reads a game no copy writes, and the
  copies run on the thread pool. The rows of `dst[k]` are byte copies of the rows of `src[k]` as
  they stand (observations, active, lives, wins, terminated, winner, and also the rewards and
  battle flags of the source's last step, which belong to the copied battle event); status 0.
- **rekey**: defined as `redeterminize(seed)`, seed in `[0, 2**63)`. Replaced: both seats' pet and
  food offer streams (new tape, cursors at zero; level-up reward pairs draw from the pet
  stream), both shop streams (re-seeded from the new tape's `ability-seed` domain: trigger-order
  ties, random targets of abilities and foods, priority ties of the shop faint chain; reward pets
  that held a private copy of the old stream are bound to the new one), and the battle-seed tape
  (`battle/turn/{turn}` of the new seed). Not replaced: everything already drawn and visible or
  held, i.e. both current shops (frozen or not), teams, counters, histories; and the per-team
  seed that no reachable draw uses. Nothing visible changes at the call.
- **set_team**: row `i` goes to team slot `i` (gaps allowed; `make_team` itself never makes one).
  Fresh pets with `Pet(id)` defaults for what a row does not state: sell value 1 at any level, no
  temporary buffs, counters zero. The Parrot rule runs front to back as in `make_team`, without a
  draw and without spending a trigger. Gold, shop, action and reorder counters, the seen-order
  set and "turn ended" stay. Refused (ValueError, game unchanged): unknown ids, attack outside
  0..50, health outside 1..50, experience outside 0..5, consumed triggers outside 0..1000 (the
  reference would accept any integer and clamp experience above 5). The swapped-in team has no
  seed of its own (`Team(pets)`), so a pet summoned onto it in the shop holds a `MockRandomState`
  as in the reference; a draw from it is an explicit error (none is reachable with Turtle pets).
- **set_team / set_gold do not rewrite observations** (the reference produces none there, and
  producing one registers the team order as seen); `observe` does it on request.
- **battle_events**: rows are slot positions (a live seat's team can have gaps); stats have the
  temporary shop buffs folded in as `Battle.__init__` does; the experience column is the pet's
  real total (the battle only reads the level); a Parrot states what it holds (-1 or the kind).
  Two values the battle reads do not fit in seven columns: `friends_hurt_counter` (Wolverine's
  shop count) and `_hurt`; `battle_events(carried=True)` returns them as a fourth array
  `[N, 2, 5, 2]`. The event belongs to the game: a refused step (status -1) or a skipped one
  (status 1) leaves it, so a finished game keeps its last battle; `battle_resolved` says whether
  this call resolved one. Copies carry the event.
- **eval_vs**: the seat's end-of-turn effects run on a private copy of the seat by calling the
  end-of-turn routine directly (END need not be legal: gold may be left); a seat that has ended
  is taken as it is; the seat keeps its side of the battle (seat 0 is the first team). The own
  team is handed over exactly as a real battle gets it (carried counters included); the M boards
  are read by the same routine as `set_team`. Battle seed of sample `s` against board `m`:
  `int(RandomTape(seed).uniform(f"eval/{m}") * (2**32 - 1))` at cursor `s`
  (`sap_env_rs.eval_seed`), the construction and range of the match's own battle seeds.

Code: `src/env.rs` (`BoardRows`, `BattleEvent`, `Seat::set_team / set_gold / export_team`,
`Game::rekey / team_if_ended / board_team / eval_board / eval_vs`, `Clone` on `Seat` and `Game`,
`Game::tape_seed`, `Game::last_battle`), `src/batch.rs` (second half of `impl BatchEnv`),
`src/ffi.rs` (the entry points above), `src/bin/envcheck.rs` (optional trace keys `pre`, `teams`,
`carried`, `battleTurn`; a trace without them is checked exactly as before).

## What was checked

Reference traces (`tools/gen_env_ghost.py`, trace format of `gen_env_golden.py` plus the keys
above; replayed by `envcheck`, so `tools/check_all.sh` covers them): every step compares the
legal lists, plan, rewards, battle flag and result, terminated, winner, both observations (bytes
on every battle step and in every 8th or 10th match, SHA-256 otherwise) and the extended state
dump; every battle step also the battle event against the teams the reference's
`Battle.__init__` received (a wrapper around `selfplay_env.Battle`, as `RecordingBattle` in
`bot_games.py`) and its seed and turn.

| Family | What it is | Matches | Steps | Battles | Edits replayed | Result |
|---|---|---:|---:|---:|---|---|
| `ghost-0..4` | `ghost_worker.py`'s flow exactly: ghost seat gold 0 and END, board swapped in on the live seat's END step (own sequence turn by turn, then another player's board of that turn) | 800 | 94,480 | 5,696 | gold 5,696, team 5,696 | zero mismatches |
| `ghostmix-0..4` | the same flow with boards the rule would not give (any turn, empty, cut short, repeated), `parrotCopies` removed, a Parrot behind a Parrot, gaps, and the LIVE seat's team and gold edited mid-turn and before its END | 800 | 163,566 | 7,934 | gold 12,563, team 14,760 | zero mismatches |
| `battles-0..2` | both seats live, no edits | 300 | 88,728 | 4,031 | none | zero mismatches |
| `rekey-0..1` | both seats live, `env.redeterminize(seed)` at random steps | 200 | 59,537 | 2,728 | rekey 1,443 | zero mismatches |
| all four | | 2,100 | 406,311 | 20,389 | gold 18,259, team 20,456, rekey 1,443 | 20,389 battle events equal; 143,766 observations byte-compared, 673,056 hash-compared |

Live seats: uniformly random legal play in half of the matches, a scripted preference of
`gen_env_targeted.py` in the other half; the ghost is seat 0 in half and seat 1 in half. What the
swapped-in boards held: 66 species (the pool's tokens included), 3,490 Parrots (1,531 with an
explicit copy of 44 species, 1,905 by the default rule, 54 that copied nothing, 171 boards with
a Parrot behind a Parrot), 2,037 level-3 pets, 21,975 pets with a perk, 4,230 with consumed
triggers, 1,060 empty boards, 697 boards with a gap. All 1,600 ghost matches end in a battle
whose final observation is byte-compared (the only place where the ghost seat's own masks and
seen-order count after a swap are visible). 7,801 battle events have a team with a gap; 10 carry
a Wolverine counter (the `carried` array equals the reference there). No reference match failed.
`tools/tamper_ghost.py`: 40 single-value alterations of the new keys (a board's attack, health,
perk, experience, species or Parrot copy, a removed edit, a gold value, a rekey seed, a recorded
battle team, turn or counter), all caught at the altered step or, for the rekey seed, at the
next draw.

Through the C interface (`python/verify_search.py`, in the training image; counts of the run on
the final binary, one thread; the five-thread runs of `tools/check_search.sh` repeat a fifth of
each):

| Check | What is compared | Counts | Result |
|---|---|---|---|
| `ghost` | lockstep with the Python environment, edits through `set_gold` / `set_team`: both observations bytewise, rewards, flags, lives, wins, active, winner, state dump, legal lists, and `battle_events()` against the reference's battle teams and seed | 400 matches, 80,606 steps, 2,430,180 observation arrays, 81,006 state dumps, 3,945 battles, 7,312 `set_team`, 6,265 `set_gold` | all equal |
| `copy` (a) | copy, then the same actions to both until the match ends or 5/20/80 steps: every output row, the state dump and the battle event after every step | 300 matches, 2,910 copies (1,859 mid-turn, 765 with one seat ended, 138 right before a battle, 148 after termination), 71,726 lockstep steps with 3,434 battles and 1,761 END freeze plans, 258 followed to the end of the match | identical |
| `copy` (b) | only the copy is stepped (14,872 steps): source's rows, state dump and event before and after; a copy left alone while the source plays on: unchanged, then replays the source's recorded steps exactly (1,031 replays, 24,890 steps); a copy of a copy; 7 refused calls change nothing | 2,349 unchanged checks | never changed |
| `rekey` | same seed twice: identical rows, state and events to the end of the match; another seed and the unrekeyed game: the next roll and the next battle seed | 5,000 points; 356,146 identical steps, 17,167 battles, 1,817 to the end; next roll differs from another seed's in 4,879 of 5,000 (the rest roll one or two slots) and from the unrekeyed one in 4,882; battle seeds differ in 3,183 of 3,183 pairs, both ways; nothing visible changes at the call 5,000 of 5,000 | as required |
| `rekey` frequencies | offers after a roll, rekeyed (77,726 pets, 31,022 foods) and unrekeyed (71,482 pets, 28,612 foods, the sources' own rolls), by pet tier, food tier and food, per rule turn: against the exact pool probabilities and against each other (chi-square) | p = 0.57 / 0.021 / 0.087 (pet tier: unrekeyed against exact, rekeyed against exact, homogeneity), 0.90 / 0.35 / 0.23 (food tier), 0.98 / 0.62 / 0.30 (food) | rekeyed and unrekeyed agree; see "Frequencies, in full" for the tests against the exact probabilities |
| `rolls` | six games played to turn 2, 5, 8 or 11; 10,000 copies of each rekeyed with fresh seeds roll once; offers by name against the uniform pool | 60,000 rekeyed rolls per turn; foods p = 0.22 / 0.23 / 0.80 / 0.91, pets p = 0.34 / 0.98 / 0.047 / 0.65 | uniform |
| `eval` exact | `eval_vs` counts against `sap_engine_rs` fights (`sap_battle`) of the same two teams with the same seeds; the own team is what the real END gives on a copy (right before a battle: the exported team; mid-turn with gold left: a copy with gold set to 0) | 1,500 points (712 before a battle, 788 mid-turn, 198 for the seat that had ended), 6,572 boards, 24,352 fights; 417 boards with a gap, 194 with a default-rule Parrot, 301 with mixed outcomes | counts identical on 6,572 of 6,572 boards |
| `eval` unchanged | rows, state dump and battle event of the game before and after the call; the call repeated | 1,698 | unchanged, repeatable |
| `eval` real battles | the exported teams and seed of the real battle fed to `sap_battle`; copies rekeyed with fresh seeds fight the real battle: battle seed = the one computed from the new tape, outcome reproduced; outcome totals of those real battles against `eval_vs` with as many samples | 712 of 712 and 2,106 of 2,106 reproduced; 2,106 of 2,106 seeds as computed; real 1,083 / 185 / 838 against `eval_vs` 1,080 / 197 / 829 (p = 0.81) | agree |
| `eval` Python engine | a 2% share of the fights on the Python battle engine (`sapai.battle.Battle` on `make_team` of the decoded boards) | 499 | 499 equal |
| unchanged interface | `python/verify_vec_env.py random` | 64 matches, 27,494 steps, 826,740 arrays | ALL EQUAL |

Frequencies, in full. The requirement (rekeyed against unrekeyed) is the homogeneity line: over
seven runs of the play-driven test (5,000 points with driver seeds 97, 97 again after the driver
learned END freeze plans, and 12345; 15,000 points with seeds 1000 to 1003, about 210,000
unrekeyed and 235,000 rekeyed pet offers each) the 21 homogeneity p-values lie between 0.039
and 0.99, with one below 0.05 (about one expected). Against the exact pool probabilities the 8
pet-tier p-values of the four large runs are between 0.35 and 0.80; but over all seven runs the
42 p-values against the exact probabilities include 7 below 0.05 where about 2 are expected
(pet tier 0.021 and 0.030 in the 5,000-point runs; food tier 0.0036, 0.045, 0.050 and food
0.025, 0.046; an earlier 2,500-point run had a food-tier 0.0093), in both the rekeyed and the
unrekeyed arm, while the arms agree with each other. The statistics are not independent (food
and food tier are the same offers), and no cause was found in the tally: this is unexplained,
not shown to be chance. The per-cell residuals of the worst run (seed 1000, `--detail`) have no
common direction across rule turns or arms; its largest cell is tier-1 foods at rule turn 10 in
the rekeyed arm (+3.7 standard deviations of 6,934 offers, tier 3 at -2.9), which the direct
test at turn 10 does not show. The direct test has
none of it: `verify_search.py
rolls` plays six games to a turn, rekeys 10,000 or 20,000 copies of each with fresh seeds, rolls
once and compares the offers by name with the uniform pool (independent draws by construction):
turn 2: 40,000 foods p = 0.22, 130,000 pets p = 0.34; turn 5: 110,000 / 190,000, p = 0.23 / 0.98;
turn 8: 100,000 / 160,000, p = 0.80 / 0.047, and with other states 160,000 / 320,000, p = 0.75 /
0.22; turn 10: 160,000 / 280,000, p = 0.77 / 0.42; turn 11: 60,000 / 130,000, p = 0.91 / 0.65.
(An earlier version of the play-driven test took the unrekeyed side from copies that looked
ahead on the match tape, which counts the same tape entries again at the next point of the same
game; its p-values were too small for that reason and it was replaced by the sources' own
rolls.) `check_search.sh` fails a frequency test at p < 0.001, so about one run in a hundred
fails by chance.

## Shop edit: `set_shop_item` (added 2026-11-23, after the primitives above)

Purpose: start training games from chosen positions (`set_team` places the team, this places a
pet or food in the shop). Only additions: no existing entry point, array or trace key changed.

`env.set_shop_item(game, seat, slot, item, frozen=False)` (`sap_env_set_shop_item(env, game, seat,
slot, item_type, item, frozen)`, `item_type` 0 pet kind id / 1 food id; `env.food_id(name)` /
`env.food_name(id)` = `sap_env_food` / `sap_env_food_name`, next to `kind_id` / `kind_name`).
`slot` indexes the seat's current shop list (the order of `state(game)["seats"][seat]["shop"]` and
of the observation's shop tokens). `item` is a name (`"pet-blowfish"`, `"food-sleeping-pill"`) or a
pair `("pet", kind_id)` / `("food", food_id)`; a bare integer is refused because pets and foods have
separate id spaces.

Code: `src/env.rs` (`ShopItem`, `Seat::set_shop_item`, `Game::set_shop_item`), `src/batch.rs`
(`BatchEnv::set_shop_item`), `src/ffi.rs` (`sap_env_set_shop_item`, `sap_env_food`,
`sap_env_food_name`), `src/bin/envcheck.rs` (the `pre` entries `shop`, `state`, `obs`; a line
"SHOP EDITS ..." in the summary when a trace has them), `python/sap_env_rs.py`,
`tools/gen_env_shopset.py`, `tools/tamper_shopset.py`, `tools/mutation_shopset.py`,
`tools/check_shopset.sh`, `python/verify_search.py` (modes `shopset`, `shopsame`).

Semantics, exactly (the reference edit is `place` in `tools/gen_env_shopset.py`; Rust:
`Seat::set_shop_item` in `src/env.rs`):
- **Pet** into a `pet` or `levelup` slot: `Shop._roll_slot` with the choice given instead of drawn,
  then the stat line `Shop.roll` runs after it: a fresh `Pet(name)` (base stats, level 1, no
  experience, the species' own status, sell value 1, counters zero) with
  `min(50, base + shop_attack / shop_health)`, i.e. the permanent Canned Food bonus the shop has
  accumulated at that point of that game. Cost 3 (the reference never changes a pet slot's cost;
  the generator asserts it). The slot is an ordinary pet slot afterwards.
- **Reward slots**: `_roll_slot`'s own rule is used (`slot_type = "pet"`, `reward_group = None`).
  That branch is dead in normal play (`check_rules` drops unfrozen rewards before any roll), so
  one shop shape is new to the port and the reference alike: an ordinary pet slot in front of a
  `levelup` slot (the reference's own code always puts rewards at the front of the list). The
  sibling reward is not
  touched: it stays a reward, alone in its group (buying it removes only itself; buying the
  placed pet does not remove it). At the next roll `check_rules` treats the placed slot as any
  ordinary pet slot.
- **Food** into a `food` slot: a fresh `Food(name)`, slot cost = the food's table cost (3, Sleeping
  Pill 1). A Squirrel discount on that slot or a stocked item's cost of 0 is gone, as after a roll.
- **Stream binding**: every roll of the reference is followed by `Player._connect_objects`
  (`Player.roll`, `start_turn`, the reset), so a rolled item always draws from the seat's shop
  stream. The edit makes the four assignments of `_connect_objects` for the new item only
  (Rust: `Rs::Shared`). It does not run the whole `_connect_objects`: that would rebind a level-up
  reward that still holds its private copy of the stream, which is another state change.
- **`frozen`** becomes the slot's frozen flag (False also thaws a frozen slot).
- **Position**: the item stays in the slot it was put in; no tier ordering is applied (the next
  roll orders the shop as always).
- **Nothing else changes and nothing is drawn**: gold, the other slots, action and reorder
  counters, the seen-order set, the offer cursors, the shop stream. The freeze-toggled set is
  untouched: the new item is not in it (it can be frozen or unfrozen once this turn like a rolled
  one), the replaced item stays in it if it was (its size is observed). Observations are not
  rewritten (`observe(which)` does that).
- **Accepted items**: what a Turtle shop can roll on some turn, i.e. the 60 pets and 18 foods of the
  turn-11 pools, **whatever the current turn** (tier restrictions are not enforced).
- **Refused** (ValueError, the game unchanged, reason in `error(game)` when the library refused;
  C code -14): a slot index outside the shop list (negative, = length, 10, ...), a pet for a food
  slot, a food for a pet or reward slot, an unknown name or id, and items no roll can produce:
  tokens (`pet-bee`, `pet-zombie-cricket`, ...), `food-milk`, `food-bread-crumbs`, pets of other
  packs. Also -11 game or seat out of range, -12 a game that was never reset.

"As if just rolled" excludes one thing a shop pet can carry: a Duck's "+1 health to shop animals"
(not `includingFuture`) is on the pets that were in the shop when the Duck was sold, not in the
shop's permanent bonus. A placed pet does not have it, exactly like a pet rolled a moment later.
(Checked on the reference: in 40 matches, of 17,848 ordinary pet slots looked at after a roll or
at a turn start, 150 were not at base stats plus the bonus; all 150 were frozen pets that had
stayed, all were +0/+1, and each is accounted for by the Ducks sold while it was in the shop. No
freshly rolled pet differed.)

What the reference does later with an out-of-tier item (a tier-5 or tier-6 pet on turn 2), and the
port with it (demonstrated side by side with a Seal and a Dragon on turn 2; all of it is in the
trace family): it is an ordinary shop item. Its tier is a property of the species (read by the
display order after a roll, the observation and the few abilities that look at a pet's tier);
nothing compares it with the tiers the turn has unlocked, so no later rule knows that the item
could not have been rolled here. A **roll** or the next turn replaces it by an ordinary draw from
the turn's pool unless it is frozen; a **frozen** one stays as long as it is frozen (frozen pets
compact to the left of the pet slots, frozen foods to the right, and frozen ordinary pets are
never dropped even beyond the turn's pet slots). **Freeze / unfreeze** work as for any item.
**Buying** costs 3 (foods their table cost), with the usual room and combine rules. **Sell value**
is 1 at level 1 and follows the level, never the tier. The **level-up reward tier** comes from the
turn (`levelUpTier` of `shop.turn`: tier 2 on turns 1-2, 3 on 3-4, 4 on 5-6, 5 on 7-8, 6 from 9),
not from the pet: three placed Seals combined on turn 2 offer two tier-2 pets. The observation
shows the item's own tier.

### What was checked

Everything that existed, again, on the final library (sha256
`c68cf5a82143526b57cdb47d56c8d1b9f941dc09c3bdd9c44994900f29ea2b85`, the same before and after the
run): `tools/check_search.sh` ends with ALL CHECKS PASSED. In it: `check_all.sh` 11,568 of 11,568
matches (the 8,668 earlier ones, the 2,100 of the four ghost families and the 800 new ones;
2,910,037 steps) and the six battle goldens without a mismatch; the 8,668 earlier matches alone
identical to `env-golden/baseline-8668.txt`; digest `31035309134488a8`; the ghost families with
the same edit and battle-event counts as in the table above; unit tests; `tamper_ghost.py` 40 of
40; `verify_search.py` ghost / copy / rekey / eval / rolls and `verify_vec_env.py` with the same
counts as above (ghost: 400 matches, 80,606 steps, 7,312 `set_team`, 6,265 `set_gold`). Outside
that script, also rerun on this library: `tools/tamper_test.py` on `smoke.jsonl.gz` (18 of 18
caught) and `python/parity_vec_env.py ... 16 773000 1 40` (34 matches, 14,429 steps, 14 kinds of
illegal input rejected without effect). Not rerun: the 48-game five-thread `parity_vec_env.py`
line, the speed table and the coverage report.

Reference traces `env-golden/shopset-0..4.jsonl.gz` (`tools/gen_env_shopset.py`; replayed by
`envcheck`, so `tools/check_all.sh` covers them). The trace format of `gen_env_ghost.py` with
three more `pre` entries: `["shop", seat, slot, name, frozen]` (the edit), `["state", dump]` (the
reference's canonical state right after the edits of the step) and `["obs", [seat 0, seat 1]]`
(both observations right after the edits; taking one registers the team order as seen, on both
sides). Every step compares the legal lists, plan, rewards, battle flag and result, terminated,
winner, the extended state dump and both observations: bytes on every step with an edit, on
every battle step and in every 8th match, SHA-256 otherwise; battle steps also the battle event.

| What | Count |
|---|---:|
| matches (400 random play, 400 scripted preferences; 320 "curriculum" matches: `set_team` + items + `set_gold` at most turn starts; 20 Canned Food farms) | 800, none failed in the reference |
| steps / decisions / battles (matches end after turns 7 to 22) | 272,692 / 430,286 / 9,588 |
| `set_shop_item` edits replayed (pets / foods), in 50,887 steps | 73,631 (50,667 / 22,964) |
| `set_gold` / `set_team` edits in the same matches | 27,304 / 7,700 |
| state dumps / observations compared right after the edits of a step | 50,887 / 51,316 (bytes) |
| observations byte-compared / hash-compared after steps | 204,550 / 342,434 |
| result | zero mismatches |

Where the items were placed (of the 73,631): at the start of the seat's turn 13,690, mid-turn
49,305, for a seat that had already ended its turn 6,992 (the rest are the two scripted kinds
below); right after the seat's roll 21,697; with frozen items elsewhere in the shop 50,518; with
level-up rewards on offer 5,343; into a reward slot 2,818 (1,712 with the sibling still there, 217
frozen rewards); pets placed while the shop had a Canned Food bonus 23,060 (138 of them capped at
50; the largest bonus was +102); out-of-tier pets 15,799 and foods 8,397 (2,079 tier-5 or tier-6
pets on turns 1-2; the scripted Canned Food and Sleeping Pill placements are not in the food
figure, so it is a lower bound); over a frozen item 24,679; over an item in the freeze-toggled set 10,768;
placed frozen 19,076; the species of a team pet 17,131; the item the slot already held 4,286; over
a stocked food 291; over a discounted food 55; in the same step as a `set_team` 13,204. Every one
of the 60 pets was placed at least 636 times and bought at least 295 times, every one of the 18
foods at least 806 and 313 times.

What then happened to them (the seat's choice is biased towards placed items): pet bought 18,474,
bought and combined into a team pet 6,329, food bought 11,549, frozen by an action 19,993,
unfrozen by an action 17,517, rolled away 17,469, kept through a roll 17,432, carried into the
next turn 6,427, replaced at the next turn's roll 9,095; pets that came from a placed one: sold
21,125, fed 8,396, combined on the team 3,028, reached level 2 1,148 times and level 3 164 times.
Two scripted kinds: Canned Food farms (2,154 cans handed to one seat with the gold for them), and
1,490 Sleeping Pills placed next to an untouched level-up reward, of which 1,128 were fed to that
reward right after buying it (45 Spiders and 41 Seals, which then draw from the private copy of
the shop stream a reward holds: this is what tells "bind the new item only" from "run
`_connect_objects`").

| Check | What is compared | Counts (run on the final binary) | Result |
|---|---|---|---|
| `tools/tamper_shopset.py` | one value of a `shop` entry altered (species, frozen flag, slot, seat, entry removed; pet, food and reward-slot edits) with the recorded state kept, and again with the `state` entry removed on steps where the placed item is still in its slot afterwards (only the ordinary comparisons); the recorded state and observation after the edits altered | 54 alterations, 4 controls (entry removed, nothing altered) | 54 of 54 caught at the altered step, 4 of 4 controls pass |
| `tools/mutation_shopset.py` | twelve deliberate errors in a copy of `Seat::set_shop_item`, the trace family replayed with each | matches that differ, of 800: item not bound to the shop stream 161 (pet) / 429 (food); reward group kept 630; slot stays `levelup` 630; no bonus 691; food cost kept 779; legal cache not invalidated 800; `frozen` ignored 800; no cap at 50 18; whole `_connect_objects` 57; toggled set cleared 798; action counter advanced 800 | all twelve noticed; the unpatched copy passes |
| `verify_search.py shopset` (C interface, lockstep with Python) | the generator's edits made through `set_shop_item` (half by name, half by id), `set_team`, `set_gold`; after every step both observations bytewise, rewards, flags, lives, wins, active, winner, the state dump, legal lists, battle events; the state dump right after the edits; `observe()` right after the edits against a Python observation taken at the same point; one edit of each seat's shop after the match has ended | three one-thread runs over different matches: 300 matches, 101,825 steps, 3,351,420 observation arrays, 102,125 state dumps, 3,532 battles; 27,888 `set_shop_item` (13,982 by name, 13,906 by id; 1,076 into reward slots, 6,043 out-of-tier pets, 8,641 pets with a bonus, 75 capped), 2,859 `set_team`, 10,339 `set_gold`; right after the edits 19,206 state dumps and 9,589 `observe()` calls; 9,308 placed pets and 4,329 placed foods bought, 435 reward-and-pill chains (29 Spider or Seal); 594 edits of finished games; all 78 items placed and bought in each run. Five threads: 22 more matches, 7,804 steps, 2,039 `set_shop_item` | all equal |
| refused calls (inside the same runs, at random steps) | ValueError; output rows, state dump and battle event of the game unchanged; a reason in `error(game)` when the library refused; `place` refuses the same slot and item; play goes on in lockstep | 21,174 refused calls: slot index outside the shop list (-1, = length, 10, 1000) 3,031; pet into a food slot 452; food into a pet slot 591; food into a reward slot 13; unknown name (unknown pet, unknown food, no prefix, empty) 2,364; token pet, Milk or Bread Crumbs by name 1,521; token or other-pack pet by id 556; id outside the table or negative 2,965; bare integer or unknown item type 1,209; game or seat out of range 8,468; a game that was never reset 4 | all refused, nothing changed |
| `verify_search.py shopsame` | "as if just rolled", tested on the reference itself: four copies of a match get the same actions: Python, Python with every freshly rolled item replaced by `place` of the same name, Rust, Rust with `set_shop_item` of the same name (after the reset, after every roll, at every turn start; frozen items that stayed included when they are at base stats plus the bonus). Legal lists, observations (bytes), state dumps, rewards, flags and battle events of all four at every step | 100 matches, 27,979 steps, 1,208 battles, 1,678,740 observation arrays, 109,392 state comparisons; 44,376 pets and 18,318 foods replaced by themselves in 8,485 rounds (25,990 frozen, 2,234 pets with a bonus, all 78 items); left alone: 86 reward slots, 243 foods off their table cost, 362 frozen pets above base stats plus the bonus (a sold Duck's +1 health) | all four agree to the end of every match |
| wrapper with the previous library | `python/sap_env_rs.py` on the build from before this addition | reset, step, `set_gold` work; `has_set_shop_item` False, `set_shop_item` raises AttributeError | as intended |

### Commands

```bash
tools/check_shopset.sh [quick]   # the trace family, the tamper test, shopset (3 x 100 matches + 22 on five threads) and shopsame (100);
                                 # called by tools/check_search.sh, which also replays the family through check_all.sh
python3 tools/mutation_shopset.py /tmp/mut      # not in check_search.sh: one extra build, 13 replays
tools/run_ref.sh 0 /rs/python/verify_search.py shopset  /rs/target/release/libsap_engine.so 16 100 1 [first match index]
tools/run_ref.sh 1 /rs/python/verify_search.py shopsame /rs/target/release/libsap_engine.so 100 1
python3 tools/tamper_shopset.py ./target/release/envcheck /tmp/tamper env-golden/shopset-0.jsonl.gz env-golden/shopset-1.jsonl.gz
tools/run_ref.sh 0 /rs/tools/shopset_out_of_tier_demo.py /rs/target/release/libsap_engine.so   # Seals and a Dragon on turn 2, reference and port side by side
tools/run_ref.sh 0 /rs/tools/shopset_cache_hazard.py                                          # the reference's stale legal list after hand edits (see below)
# how the kept files were made (seeds 884400000 + index; summaries with all counts next to them)
for c in 0 1 2 3 4; do tools/run_ref.sh $c /rs/tools/gen_env_shopset.py /rs/env-golden/shopset-$c.jsonl.gz 160 $((40000+c*160)) 8 > env-golden/shopset-$c.summary.json; done
```

### Not covered, and things to know

- The cap at 50 is reached by 138 placed pets, all in the 20 Canned Food farm matches (a bonus
  above +40 does not happen otherwise).
- A game that has ended and a seat that has ended its turn accept the edit (like `set_team`). The
  first is only checked by the state dump in the lockstep runs (no step can follow); the second is
  in the traces (6,992 edits: a frozen item is carried into the next turn, an unfrozen one is
  replaced by the turn's roll).
- A broken game (status -2) refuses the edit as the other edits do; not exercised.
- The state dump does not show which stream an item is bound to. That is checked through later
  draws only (a bought Otter, Salad Bowl, Sushi or Pizza; the reward-and-pill chains): see the
  mutation counts above.
- The reference's legality cache (`_avail_actions`) keys shop items by `id()`, type, cost and
  frozen flag, not by name. A hand edit on the Python side whose displaced objects are freed can
  therefore be handed the address of an earlier item and get that item's legal actions back (seen:
  a team Pigeon, a Pigeon placed in slot 0, legal list read, then a Fish placed twice in slot 0:
  "buy and combine onto the Pigeon" was still listed in 4 of 40 tries, 3 of 40 in a second run;
  `tools/shopset_cache_hazard.py`; which address is reused is up to the allocator). This is the reference's
  behaviour under hand edits, not something to port: the generator keeps displaced objects alive
  until the match ends so that it cannot happen, and the port invalidates its cache on every edit
  and never reuses an identity. Anyone editing the Python environment directly must do the same
  or clear `core._agent_actions_cache` / `_opponent_actions_cache`.
- `place` also runs `Shop._advance_seed_state()` as `_roll_slot` does (`shop.seed_state`); nothing
  reachable reads that field before it is refreshed, and the port does not model it.
- Each edit of a pet slot leaves the displaced pet in the seat's arena until the next turn start
  (memory only; the identity counter allows 33 million objects per match).
- The live seats were random and scripted play, not the network (as for the other primitives).
- Cores 0-4 were shared with another job while these checks ran: no timing here means anything.

## Cost (cores 0-4 class, one core, `python/verify_search.py bench`)

- `copy`: 4.3 to 4.8 microseconds per game (1,024 games at assorted points of real matches, one
  call; 7 to 9 the first time, into slots that were never reset). By estimate (not measured)
  about 20 KB move per game, of which 6.6 KB are the output rows. For scale: a `step` call that
  selects no game (`which` all false, 2,048 games) costs 11 microseconds.
- `eval_vs`, ranked boards of the pool against ranked boards, 8 samples per board: 38,700
  fights/s (turns 3-5), 35,600 (6-8), 32,200 (9-12), 27,500 (13 and later), 38,700 over boards of
  all turns; with 1 sample per board (board conversion every fight) 34,700 / 31,400 / 28,500 /
  24,900 / 34,000. The battle engine alone does about 35,000/s on its golden set, so the wrapper
  costs little. Results do not depend on the thread count (same counts on 1 and 5 threads); a
  five-core figure was not taken (`tools/run_ref.sh` caps the container at one core, so the
  five-thread runs of the check script show that results are the same, not how fast they are).

## Commands

```bash
cd /home/dzliu/pzliu/sap-rl/engine-rs
taskset -c 0-4 cargo build --release -j5 --offline
tools/check_search.sh            # everything: check_all.sh (all trace files, new families included, battle goldens),
                                 # the 8,668 earlier matches against env-golden/baseline-8668.txt, the digest, unit tests,
                                 # tamper test, python/verify_search.py + verify_vec_env.py, and tools/check_shopset.sh
                                 # (5 to 6 minutes before set_shop_item was added; the run recorded for it took 13.6
                                 # minutes with cores 0-4 shared with another job)
tools/check_search.sh quick      # a tenth of the C-interface sizes
# single parts
tools/run_ref.sh 0 /rs/python/verify_search.py ghost /rs/target/release/libsap_engine.so 16 400 1
tools/run_ref.sh 1 /rs/python/verify_search.py copy  /rs/target/release/libsap_engine.so 300 1
tools/run_ref.sh 2 /rs/python/verify_search.py rekey /rs/target/release/libsap_engine.so 5000 1 [driver seed] [--detail]
tools/run_ref.sh 2 /rs/python/verify_search.py rolls /rs/target/release/libsap_engine.so <turn> <copies per state> <driver seed>
tools/run_ref.sh 3 /rs/python/verify_search.py eval  /rs/target/release/libsap_engine.so 1500 1
tools/run_ref.sh 2 /rs/python/verify_search.py bench /rs/target/release/libsap_engine.so 1
python3 tools/tamper_ghost.py ./target/release/envcheck /tmp/tamper env-golden/ghost-0.jsonl.gz env-golden/ghostmix-0.jsonl.gz env-golden/rekey-0.jsonl.gz
# how the kept files were made (one core per file; seeds 884400000 + index)
for c in 0 1 2 3 4; do tools/run_ref.sh $c /rs/tools/gen_env_ghost.py ghost /rs/env-golden/ghost-$c.jsonl.gz 160 $((c*160)) 8; done
for c in 0 1 2 3 4; do tools/run_ref.sh $c /rs/tools/gen_env_ghost.py mix   /rs/env-golden/ghostmix-$c.jsonl.gz 160 $((10000+c*160)) 8; done
for c in 0 1 2;     do tools/run_ref.sh $c /rs/tools/gen_env_ghost.py plain /rs/env-golden/battles-$c.jsonl.gz 100 $((20000+c*100)) 10; done
for c in 0 1;       do tools/run_ref.sh $c /rs/tools/gen_env_ghost.py rekey /rs/env-golden/rekey-$c.jsonl.gz 100 $((30000+c*100)) 10; done
```

`data/ghost-train-pool.json` is a copy of `runs/train-ghost-20261123/ghost-train-pool.json`
(`tools/run_ref.sh` mounts only the engine and this directory); the generator checks the
worker's SHA-256 of it and uses the worker's `load_pool` and `make_team` verbatim.
`env-golden/coverage.txt` and `coverage.json` describe the 8,668 earlier matches only; a
`check_all.sh` coverage run now also counts the new families.

## Not covered, and things to know

- `eval_vs` for a seat whose team carries a Wolverine shop counter: no check hit one (0 of 6,572
  boards; random play rarely hurts a friend next to a Wolverine in the shop). The own team takes
  the same hand-off as the real battle, and the export of such teams is compared with the
  reference (10 battle events), but the counts were not. A board given as an opponent cannot
  carry the counter at all (seven columns), nor can `set_team`.
- The exact worker flow (`ghost-*`) ends by turn 10: random and scripted live seats lose to
  ranked boards (5,585 of 5,696 battles), so its 800 matches cover turns 1 to 7, 78 reach turn
  8, 5 turn 10. Worker-rule boards at later turns (to turn 16) are in `ghostmix-*`, where the
  live seat is handed ranked boards as well.
- The live seat was never the network: the checkpoint is not reachable from `tools/run_ref.sh`.
  `python/verify_vec_env.py` was run in random mode only for the same reason.
- `set_team` with gaps is an extension of `make_team`; the reference side of that check moves
  the pets with `Team.move` after `make_team`.
- The trace format cannot express a legality computation between two edits of one step by
  itself; the generator writes a `["legal"]` marker there (it registers the intermediate team
  order as seen). The first `ghostmix` files lacked it and 2 of 800 matches differed in the
  seen-order count of a terminal observation for that reason; none after the marker.
- Every battle so far gave the same result with gaps as with the team packed to the front (the
  `eval` check feeds gapped boards to `eval_vs` and packed ones to `sap_battle`).
- `copy` takes the output rows of the source as they stand: after `set_team` / `set_gold` on the
  source without a step or `observe`, they are the rows from before the edit.
- An empty directory `env-golden/.tmp` is left from the generator's smoke runs (directories are
  not deleted here).
- `python/sap_env_rs.py` still loads a library built before these additions (checked with the
  previous build: reset and step give the same observations; `has_search_primitives` is False and
  the new methods raise AttributeError). `set_shop_item` and `food_id` are registered separately:
  a library from before `set_shop_item` keeps `has_search_primitives` True, has
  `has_set_shop_item` False, and only those two methods raise AttributeError.

# Rules v32 from the real client

Status 2026-11-23 22:30: **paused before any rule was implemented.** Nothing under `engine-rs/` has
changed: `src/engine.rs` is as it was, nothing was rebuilt, no rules switch exists yet, and no
`golden/v32-*.jsonl` exists. The six `golden/v31-*.jsonl` files pass with the existing binary
(100,193 rows, 0 mismatched). `tools/check_search.sh` was not run, since none of its inputs changed.

Done so far, all outside this directory (`runs/replay-oracle-20261122/`):
- `engine-v32/`: a byte-identical copy of `engine-v31/`, to take the rule changes.
- Baseline reproduced row for row from `rust-v31g-5000-*.json`: 64 of 12,617 fights not reproduced,
  14 wrong winner. The stored fresh2 and fresh3 results use `fixtures-fresh2.json` and
  `fixtures-fresh3.json`; the `-all` files repeat 467 fights of the 340 set.
- `v32/score_all.sh <tag>` and `v32/compare_scores.py A B` for per-rule scoring.

Rules R1, R2, R4, R5, R8, R9, R11, R12, R13, R14, R16: not started. R3 and R17: not to be
implemented. `runs/replay-oracle-20261122/REPORT-v32.md` has not been written (the agent's file
tool refused a new report file); its intended content went to the coordinator as text.

Planned switch (not built): environment variable `SAP_RULES`, read by both engines. `v31` = every
new rule off, `v32` = all on, a comma list such as `R13,R14` = a subset. Default here v31, so
`check_search.sh`, `env-golden/` and `golden/v31-*` stay valid; default in `engine-v32` v32. All
changes behind per-rule flags in `src/engine.rs`.

Reading chosen for each rule (checked against the captures, not coded):
- R13: skip the start-of-battle phase when either board is empty.
- R14: remaining = `floor(h * (1 - p/100))` in strict `f32`, floor 1 (differs from v31 only at
  level 2 on 50 health: 16, not 17).
- R16: a Crab at 0 health or less gains nothing and makes no random draw.
- R12: `modify_stats` skips on total health <= 0, not permanent health.
- R4: next start-of-battle actor = not-yet-acted start-of-battle pet with the highest current
  attack, random pick on a tie.
- R9: a Tiger repeat takes one use of an ability with `maxTriggers` and is skipped when none is left.
- R8: the Tiger for a post-removal summon is the first pet behind that did not die in the same
  step and is alive when the summon runs.
- R5: pending Wolverine activations run after the knock-out queue is empty, before the flush; a
  removed Wolverine still acts. R5b (own flag): a Wolverine holder that died earlier in the same
  attack keeps counting friend hurts through the knock-out phase (fight `48421d51`).
- R2: Fly and Shark holders in attack order, each through every pending death before the next.
- R1: same-team deaths from one `deal_damage` call get a uniform random order (battle stream, drawn
  only when two or more exist and a Fly or Shark holder is present) within their positions in the
  attack-sorted list; the dead pets' own summons keep attack order.
- R11: no `move_forward` before `battle_phase_start` and none in `swallow`; `slot_insert` unchanged.
  Real fixtures are already compact (the converter drops slot positions), so only constructed
  boards with gaps exercise it; plan is a `pet-none` placeholder spec skipped in `Fight::new`.

To resume: (1) add the `SAP_RULES` parser to `engine-v32/sapai` and a rules bitmask to `Fight`, and
confirm `SAP_RULES=v31` still reproduces `golden/v31-*.jsonl` from both; (2) implement R13 in both,
regenerate the six trace files from `engine-v32` with `tools/gen_golden.py` under that flag, check
Rust under the same `SAP_RULES`, run `v32/score_all.sh`, diff with `v32/compare_scores.py`;
(3) continue R14, R16, R12, R4, R9, R8, R5, R5b, R2, R1, R11.
