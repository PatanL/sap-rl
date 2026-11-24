# Python engine vs. real ranked battles

2026-11-22. First check of the `sapai` engine against the real game rather than against another simulator.

## Wolverine and Tiger round (2026-11-23)

One fix kept, three attempts reverted.

- **Kept: a Parrot copying Shark or Fly follows their timing.** Shark and Fly react after dead pets are removed; the engine applied that only to pets named Shark or Fly, so a Parrot copy reacted early and could get a Tiger repeat from a Tiger that died in the same hit. 86 to 84 failing fights, wrong winner 10 to 9, nothing broken.
- **Reverted: where a Mushroom copy or Honey Bee lands** relative to the pet's own summons after several pets die together. Three rules tried (rebuild every summon position from surviving neighbours; Mushroom behind and Honey ahead of own summons; both ahead). Results on all 4,065 fights: 3 fixed and 56 broken; 1 fixed and 10 broken; 2 fixed and 4 broken. The engine's stored position does go stale in these cases, but none of my rules is the real one.
- My round-by-round trace tool skipped a priority refresh the real battle loop does, which briefly made Boar look broken. Corrected.

Current: 84 of 4,065 failing, 9 wrong winner. Wolverine fights 8.4% (26 of 309), Tiger 8.1% (41 of 507). `engine-v31.patch` has 17 hunks. Engine tests unchanged.

### Later the same day: three more rules, from the user's Deer hint

The user said the Bus and the Mushroom Deer land differently depending on how the Deer died. Checked against real fights, that led to these, each tested on all 4,065 fights with nothing broken:

| Rule | Failing fights | Wrong winner |
|---|---:|---:|
| Before | 84 | 9 |
| Summons use fixed slots: a summon goes into the slot where the pet died; if it is taken, pets are pushed forward when a slot ahead is empty, otherwise back | 81 | 8 |
| Pets that died together resolve their summons in attack order, highest first, not in the order the deaths were recorded | 79 | 8 |
| A Rhino that faints in the same attack keeps the Tiger behind it for its chained knockouts | 76 | 7 |

The slot rule's direction matters. Preferring to push back fixed 3 fights and broke 3 (confirmed with 3,000 seeds each); preferring forward fixed 3 and broke none. It explains every case seen: a front Deer with Mushroom returns ahead of its Bus, a Deer sniped behind an empty slot returns behind it, and a Honey Bee lands ahead of the Zombie Cricket.

Now: 76 of 4,065 failing (1.9%), 7 wrong winner. Wolverine 7.8% (24 of 309), Tiger 7.1% (36 of 507). `engine-v31.patch` has 23 hunks; engine tests unchanged. Of the 76, the calculator matches the real game in 32, all of them fights with random effects, and misses in 44. No deterministic fight with a calculator reference is left on the development set, so further progress needs a better reference.

What is missing is a reference for the intermediate steps of a fight: replays give only the end state and the calculator disagrees with the real game on about half of what is left. The user pointed to the SAP Library Extension's replay injection, which makes the real client play any battle JSON; Codex is building a tool on that idea in `../client-oracle-20261123/`.

## Shop-to-battle handoff, lives rule, food order (after Codex's second check)

Codex's checks are in `../shop-oracle-20261122/REPORT-2.md`. What they found and what I did:

- **Temporary health was double-counted in battle.** Starting each fight from the engine's own end-of-turn team instead of the replay snapshot raised failures from 85 to 252 of 3,666 fights. Cause: temporary shop buffs (Bread's +7, Cupcake) stay in a separate field, and effects that write the permanent field (Skunk) left the temporary part to be added again. My snapshot check could not see this because it feeds in combined stats. Fix: a battle now starts from combined stats on its own copy of the team. Re-running Codex's chained check on the fixed engine gives 85 of 3,666, identical to the snapshot start, with no fight failing only when chained.
- **Per-turn trigger counts carry correctly** from shop into battle and reset next turn. 40 of 32,494 counters differ from the replay, all Rabbits.
- **Food order now matches** the real game on all 2,467 pairs (my first attempt read the tier from the wrong place; Codex caught it).
- **Lives and game end match.** Winner and last turn agree on all 331 ordinary matches, and lives before every battle agree on all 8,121 snapshots. One convention differs: the real game shows the turn-two heart being lost and then restored before turn three; the engine applies the restore immediately, so the per-battle reward for a turn-two loss at full hearts is zero. That is the true net effect, and I recommend keeping it.
- A level-2 pet released by Whale would still sell for 1. This cannot happen in the shop, so it is left alone.

After these changes: battles unchanged at 86 of 4,065, shop turns 6,673 of 6,735, engine tests unchanged (3 old-rule failures). `engine-v31.patch` has 12 hunks.

Codex also prepared, without training, a setup that joins its wide collector to the learner for two arms (continue 25,772; fresh 1.35M) in `../train-v31-prep-20261123/`. Its packaged engine predates the temporary-health and food-order fixes and must be repackaged before any run. Two things in that setup need a decision: it is pure self-play of the latest policy against itself with no frozen opponents, and the continue arm inherits the parent's decayed learning rates (2e-6 on the main network).

## With 5,000 seeds per fight, and Parrot's recorded ability

The Rust engine (`../../engine-rs`, exact match to this Python engine on every reference trace) makes 5,000 seeds per fight cheap, about 80 seconds for the development set.

| Set | Fights | Not reproducible | Wrong winner |
|---|---:|---:|---:|
| Development (340 replays) | 4,065 | 39 (0.96%) | 6 (0.15%) |
| Untouched batch one | 3,693 | 34 (0.92%) | 3 (0.08%) |
| Batch two | 3,328 | 42 (1.26%) | 14 (0.42%) |

- **The 200-seed test overcounted.** 33 of the 76 development "failures" were real results the engine does produce, just rarely. The honest rate is about 1%, not 2%.
- **Win probabilities are well calibrated.** On 2,256 fights with random effects, the engine's predicted score and the real score agree in every bucket (for example predicted 0.71, real 0.73; predicted 0.29, real 0.26).
- **Parrot.** Replays record the ability each pet holds. For Parrots that is what it copied: the pet ahead in 3,018 cases, a chain through another Parrot in 70, and its own ability (copied nothing) in 88. Fixtures now use the recorded ability. Engine rule added: a Parrot behind a Parrot that has already copied this turn copies what that Parrot holds; copies resolve in attack order, so a back Parrot with higher attack acts first and gets nothing. This fixed the Dog fights where Sheep, Parrot, Parrot should give five Ram summons.
- Batch two has now been inspected, so it counts as development data from here on.

Remaining clusters: one player's Hedgehog, Hedgehog, Rhino, Wolverine, Parrot team across several turns; Tiger ordering; Wolverine at 4.5% of its development fights.

## Second clean test: 3,328 fights from 275 further untouched matches

Scored once, after the slot, attack-order and Rhino fixes (`raw-fresh2/`, `fixtures-fresh2.json`).

| | v30 as trained | engine-v31 |
|---|---:|---:|
| Fights not reproduced | 227 (6.82%) | 78 (2.34%) |
| Wrong winner | 43 (1.29%) | 17 (0.51%) |

This batch is harder for both engines than the first. The untouched rate has not dropped below about 2% despite the later fixes; Wolverine fights fail 11.0% (34 of 309) and Tiger 8.5% (34 of 398) here.

## Clean test: 3,693 fights from 300 replays no fix was developed against

Downloaded after the fixes below were written (`raw-fresh/`, `fixtures-fresh.json`), no match shared with the earlier set, each engine scored once.

| | v30 as trained | engine-v31 |
|---|---:|---:|
| Fights not reproduced | 201 (5.44%) | 74 (2.00%) |
| Wrong winner | 30 (0.81%) | 6 (0.16%) |
| Matches (of 300) with every fight reproduced | | 248 |

131 fights fixed, 4 newly failing. The rate matches the development set (2.1%), so the fixes are not fitted to it. Highest remaining rates on this set: Armadillo 11.8% (9 of 76), Wolverine 9.7%, Tiger 8.4%. Further fixes should be developed on the earlier set and scored on another fresh download, not on this one.

## Fixes so far (engine-v31)

`engine-v31/` is the pinned v30 engine plus the fixes below; `engine-v31.patch` is the diff (5 hunks in `effects.py` and `status.py`). The real `sapai` sources are untouched. Every fix was checked on all 4,044 fights, split by match into dev (2,579) and held-out (1,465).

| Step | Failing fights | Wrong winner | Newly broken |
|---|---:|---:|---:|
| v30 as trained | 256 (6.3%) | 45 | |
| Wolverine health removal stops at 1 | 161 | 22 | 5 |
| Garlic leaves 1- and 2-damage hits unchanged (it turned 1 into 2) | 134 | 20 | 1 |
| A pet at 0 health cannot be buffed back to life | 130 | 19 | 1 |
| Perks gained in battle trigger Seal and Rabbit, within their per-turn limit | 101 (2.5%) | 12 (0.3%) | 0 |
| (21 token-pet boards added to the test set: 4,065 fights) | 102 | 12 | |
| A Rabbit fainting in the same resolution still triggers | 91 | 11 | 0 |
| A Parrot copying Sheep summons two Rams, not one | 86 (2.1%) | 10 (0.25%) | 0 |

The last step also needed a converter fix: replays record per-turn ability uses already spent in the shop (`TrCo`), and a Rabbit that used its three triggers in the shop does nothing in battle. Without that field, enabling Rabbit broke 9 fights.

Of the 86 remaining failures, the calculator reproduces the real result in 42 (engine bugs with a reference to copy), and misses it too in 44 (suspect remaining converter gaps first). Highest remaining rates: Wolverine 8.7%, Tiger 8.5%, Blowfish 6.0%, Fly 5.6%. The remaining deterministic cases are large late-game boards (Tiger with Parrot, Mushroom with Wolverine, Mammoth with Tiger and Fly), where the difference is in event ordering rather than a single rule.

Tools: `triage.py` (three-way triage and dev/held-out split), `diag.sh <battle id>` (calculator log beside an engine round-by-round trace), `compare.py` (fixed and newly broken fights between two runs).

## Shop turns replayed action by action

Codex built a replay check in `../shop-oracle-20261122/` (its own report is there): start from the real end-of-turn board, apply the real actions of the next turn through the engine, compare with the real next snapshot on board, levels, stats, perks, frozen items and gold. 7,445 completed turns; 710 skipped because a bought pet could not be identified; 6,735 attempted.

| Engine | Turns reproduced | Share |
|---|---:|---:|
| v30, and v31 with battle fixes only | 6,596 | 97.94% |
| v31 with the four shop fixes below | 6,673 | 99.08% |

Shop fixes, applied by me to `engine-v31/` after Codex's findings (77 turns fixed, 29 of them held-out, none broken; battle results unchanged at 86 failing):

- Dragon buffs every friend except itself, including the tier-1 pet just bought. The engine buffed the Dragon and skipped the bought pet.
- A pet summoned at level 2 sells for 2. The engine paid 1.
- Food eaten by shop pets (Canned Food) no longer spends Rabbit's three per-turn uses.
- Scorpion gains Peanut when bought, whatever the shop copy carried.

A fifth shop change, not yet verified by an independent check: freshly rolled foods are now listed higher tier first, as in the real game (the engine reversed 956 of 2,467 real pairs). It leaves the strict shop score unchanged at 6,673.

Engine test suite (the checkout's tests run against this copy): 3 failures that pinned v30 does not have, all tests asserting the old rules (two Dragon, one Wolverine). The suite also caught a mistake in my first food-order change, since corrected.

Not yet checked: whether the engine's own end-of-turn state carries per-turn trigger counts into battle (my battle check injects them from the replay), the lives and game-end rule against real matches, and a score on replays no fix was developed against. The held-out split here was looked at after every fix, so it is no longer a clean test.

Other findings from the shop check:

- Gold reconciles on all 7,445 real turns, and the engine's wallet and food prices matched on every turn that reached the end.
- Roll odds show no departure from the engine's per-tier pet odds (chi-square p between 0.26 and 0.93 across tiers).
- The real shop lists a higher-tier food before a lower-tier one; the engine reverses 956 of 2,467 pairs. This does not change what a turn does, but it changes what the policy sees.
- Still open (62 turns): how Rabbit's uses are shared when one food feeds several pets (33 turns involve Rabbit), upgraded Milk (11), and shop Scorpions showing Peanut before purchase (the engine still creates them with it).

## Shop tables vs. 8,121 real end-of-turn snapshots

`check_shop_tables.py` compares the engine's static shop tables with what real shops contained.

- Pet and food slot counts by turn, and shop tier by turn: no mismatch in any snapshot.
- Every pet the engine can roll appears in real shops at the engine's tier, with the engine's base stats and a price of 3. No engine pet is absent from real shops.
- Level-up reward pets are always exactly one tier above the shop tier, as the engine assumes.
- Food tiers match. Discounted prices (Apple at 2, a few others at 2) are consistent with Worm and Squirrel.
- Gaps: Sloth appears in real shops (3 sightings, 6 board appearances) but the engine never rolls it. The level-scaled generated foods (Better Apple, Best Apple, Better Milk) were not checked.

Not yet checked: what each shop action does. Replays log every action (start turn, buy, roll, combine, food on target, sell, end turn) with item ids that count up as the shop generates items, so bought pets can be identified from later boards and a turn can be replayed through the engine and compared with the next snapshot. Roll results are not logged, so the shop odds can only be checked statistically.

## Larger sample: 4,065 fights from 340 replays

After downloading 300 more ranked Turtle mirrors (all version 48). 4,044 fights were simulated; 21 were skipped because a pre-battle board held a pet outside the engine's Turtle table (Bus 12, Sloth 6, Zombie Fly 2, Ram 1).

| | v30 | v30 + Wolverine health floor |
|---|---:|---:|
| Failing fights | 256 (6.3%) | 161 (4.0%) |
| Wrong winner | 45 (1.1%) | 22 (0.5%) |
| Wrong surviving board only | 211 | 139 |
| Wolverine fights failing (of 307) | 124 (40%) | 29 (9%) |

The floor fixes 100 fights and breaks 5. All 5 are late-game boards with Wolverine and Blowfish, so the floor is not the whole Wolverine story.

Failure rate by pet after the floor (pets in at least 20 fights; pets co-occur, so these are leads):

| Pet | Failing | Fights | Rate |
|---|---:|---:|---:|
| Seal | 6 | 47 | 13% |
| Tiger | 50 | 506 | 10% |
| Wolverine | 29 | 307 | 9% |
| Gorilla | 12 | 141 | 9% |
| Blowfish | 63 | 791 | 8% |
| Hedgehog | 32 | 406 | 8% |
| Parrot | 41 | 533 | 8% |
| Rat | 16 | 214 | 7% |
| Fly | 17 | 230 | 7% |
| Whale | 35 | 482 | 7% |
| Dragon | 19 | 265 | 7% |
| Ox | 53 | 772 | 7% |

By perk the highest rates are Mushroom (11%) and Garlic (9%). Failures rise with the turn: none on turn 1, about 5% from turn 6, 11% on turn 14 and later. 41 of the 161 remaining failures contain none of Fly, Wolverine, Elephant, Tiger, Whale, Parrot or Shark; Ox and Rabbit are the most common pets in those.

Results: `sapai-v30-result-340.json`, `trial-wolverine-floor-result-340.json`, fixtures `fixtures-340.json`, raw downloads in `raw/`.

## First pass: 467 fights from the 40 archived replays

On 467 distinct real fights (40 archived ranked Turtle mirrors, game version 48), 200 seeds per fight:

| | Fights | Share |
|---|---:|---:|
| Real result reproducible (winner and surviving board with exact attack/health) | 444 | 95.1% |
| Winner reproducible, surviving board never reproduced | 19 | 4.1% |
| Real winner never produced | 4 | 0.9% |

The pinned v30 engine (the one checkpoint 25,772 trained on) and the current `sap-rl/sapai` checkout fail on the same 23 fights.

229 fights are fully deterministic in the engine; 13 of those are wrong (4 winner, 9 board).

## These are engine errors, not conversion errors

- Both seats upload every fight separately. All 466 seat pairs convert to mirrored boards with flipped outcomes.
- 444 fights reproduce the real surviving stats exactly, so stats, levels, order and perks are being read correctly.
- SAP-Calculator, run on the same 23 boards, sides with the real game in all 4 wrong-winner fights (200/200 in three, and it allows the real draw in the fourth). On the random fights it also leans to the real result where `sapai` leans away (for example real player win: calculator 162/200, `sapai` 53/200).

The August audit that reported zero `sapai`-vs-calculator discrepancies used 96 random boards with a tolerance band. It did not hit these cases.

## Where the errors are

Share of fights containing the pet that fail (pets in at least 10 fights):

| Pet | Failing | Fights | Rate |
|---|---:|---:|---:|
| Wolverine | 11 | 22 | 50% |
| Fly | 8 | 35 | 23% |
| Rooster | 2 | 10 | 20% |
| Rhino | 4 | 27 | 15% |
| Gorilla | 4 | 26 | 15% |
| Cow | 4 | 30 | 13% |
| Turkey | 8 | 66 | 12% |
| Elephant | 15 | 136 | 11% |
| Blowfish | 7 | 63 | 11% |

Pets co-occur, so these are leads, not a diagnosis. Patterns visible in the individual cases:

- **Wolverine.** Half of all Wolverine fights fail. In one, the engine has the Wolverine side win 200/200 while the real game has the other side win with pets left on 1 health.
- **Fly.** The engine leaves a Zombie Fly on the board that the real game does not.
- **Elephant.** Several boards differ by a few health on the pet behind the Elephant.
- **Gorilla, Kangaroo, Snail, Shark.** Single deterministic cases with different surviving health.

Each case, with both boards and the top simulated end states, is in `sapai-v30-result.json` (rows where `outcomePossible` or `statePossible` is false). Calculator triage is in `triage-calculator.json`.

## First confirmed bug: Wolverine can kill

In `sapai`, Wolverine's "remove 3 health from all enemies" goes through `ModifyStats`, which subtracts health with no floor, so it faints pets. In the real game the removal stops at 1 health (SAP-Calculator's `increaseHealth` clamps the same way).

Trial on a scratch copy of the v30 engine (`engine-trial/`, one change in `effects.py`; the real `sapai` sources are untouched):

| | v30 | v30 + health floor |
|---|---:|---:|
| Failing fights | 23 | 14 |
| Wrong winner | 4 | 3 |
| Failing Wolverine fights (of 22) | 11 | 2 |
| Newly broken fights | | 0 |

So the simulator's Wolverine was stronger than the real one. The remaining 14 failures are not yet diagnosed.

## Limits

- 467 fights is small. Rare pets appear a handful of times; a clean rate here does not clear a pet.
- The game's RNG is not reproduced, so a random fight only fails when the real result is outside everything 200 seeds produced. Wrong probabilities inside the possible set go undetected.
- Only battles are checked. Shop transitions are not covered yet.
- Perk id 41 is treated as Bread (every carrier shows the +7 temporary health); it is missing from the calculator's table.

## Files

- `extract_battles.py`: replay JSON to fixtures (`fixtures.json`).
- `run_sapai.py`: runs an engine checkout on the fixtures.
- `fetch_replays.py`: downloads more replays from the SAP Library API into `raw/`.

## 2026-11-23 05:10: speed edit and Honey Bee sell value

- **Speed, no rule change.** The engine copy now builds per-pet, per-food and per-slot random generators only when used, checks slot contents by class, and snapshots the shop stream once per reconnect. The training environment runs 3.6 times faster (2,182 to 7,871 decisions per second on one core with Rust battles; 6,901 with Python battles). Identical behaviour on 60 fingerprinted matches, 12 matches with copies and pickles, all 12,195 battle traces and the engine's test suite. Details in `sap-rl/engine-rs/PROGRESS.md`. The unedited copy is kept as `engine-v31-prespeed`.
- **Honey Bee sells for 1.** The rule "a summoned token sells for its summoner's level" no longer applies to the Bee, which comes from the perk rather than the pet. Strict shop turns unchanged at 6,674 of 6,735 (nothing fixed or broken there); the three inferred turns where the wrong Bee price was seen have not been re-run.
- Packaged as `packaged/v31-20261123e`, revision `turtle-v31-replay-13c10f72216d-20261123`. Battle rules are the same as `v31-20261123d`.

## 2026-11-23 05:40: first rule taken from the real client

The real web client now replays supplied battles on a guest account and reproduces recorded fights exactly (20 of 20; `runs/client-oracle-20261123/REPORT.md`). It traced 48 development fights frame by frame; 24 of them were among the 39 this engine could not reproduce.

- **A Dirty Rat given to the other team does not trigger that team's Horse, Dog or Turkey.** Seen in five traced fights (frames show the Rams from a Sheep buffed to 4/2 by a level-2 Horse while the gifted Dirty Rat beside them stays 1/1). In the engine: `check_summon_triggers` returns before friend-summoned abilities when the summon lands on the other team. Result with 5,000 seeds, fights not reproducible (wrong winner in brackets):

| Batch | Before | After | Fixed | Broken |
|---|---:|---:|---:|---:|
| First (4,065 fights) | 39 (6) | 32 (5) | 7 | 0 |
| Second (3,693) | 34 (3) | 24 (3) | 10 | 0 |
| Third (3,328) | 42 (14) | 33 (14) | 9 | 0 |

  All three batches have been used for development, so this is not a held-out score. In both engines; Rust matches Python on 12,195 + 6,656 + 60,000 regenerated traces. The engine's own tests now show one more failure, `test_enemy_summon_triggers_opponents_horse`, which asserts the old behaviour with no source (list in `tests-v31-rat.txt`). Strict shop turns unchanged at 6,674 of 6,735.
- **Tried and not kept: a fixed order for a Fly or Shark reacting to several deaths at once.** In fight `4664ea1e` a Whale swallows a Deer and a Crocodile kills the back pet in the same start-of-battle step; the real Zombie Fly appears in the back pet's slot, the engine puts it in the Deer's. Reverse order, back-first and front-first each fixed two to four fights and broke about as many. The open-source calculator breaks this tie at random. Needs a constructed real-client test over several seeds before changing the engine.
- Packaged as `packaged/v31-20261123f`, revision `turtle-v31-replay-37d1495292e8-20261123`: speed edit, Honey Bee sell value, Dirty Rat rule. Match fingerprints for it: `engine-rs/parity/trace-v31f.jsonl` (60 matches; identical with Python and with Rust battles). `engine-v31.patch` regenerated against the pinned v30 source.

Still open from the traces: the Hedgehog, Blowfish and Wolverine cluster (`5fafe4d0`, `6f416aeb`, `b836f265`: the real Blowfish is alive at 1 health where the engine has killed it); Whale release timing (`5ee1a36a`); a Dodo two health high (`d0b33785`) and a Crab one health high (`9001c280`) at battle start; and the real client not closing gaps before start-of-battle abilities (one constructed test).

## 2026-11-23 05:55: Wolverine waits for the chain to finish; first held-out score for today's rules

**Rule.** Wolverine's "four friends hurt" ability does not fire the moment the fourth friend is hurt. It waits until no hurt ability and no faint ability is left in the chain, then runs, before the dead are removed, and it runs even if the Wolverine itself has died. Evidence: the real client's frames for fight `b836f265` (`runs/client-oracle-20261123/session2/captures/dev-b836f265/`). After the first Hedgehog blast four of the Wolverine's friends are already hurt, yet the Blowfish exchange continues for eight seconds, the two remaining Hedgehog blasts land (Blowfish 7 to 3), and only then does the Wolverine, shown dead at -5, light up and take the Blowfish from 3 to 1. The old order ran the Wolverine first and the blasts killed the Blowfish, giving the wrong winner. The open-source calculator gets this fight wrong the same way.

In both engines: the activation is queued when the counter reaches four and run when the hurt and faint loop has nothing else to do (`battle_phase_hurt_and_faint`).

Fights not reproducible with 5,000 seeds (wrong winner in brackets):

| Batch | Yesterday | + Dirty Rat rule | + Wolverine rule | Broken by the Wolverine rule |
|---|---:|---:|---:|---:|
| First (4,065 fights) | 39 (6) | 32 (5) | 21 (3) | 0 |
| Second (3,693) | 34 (3) | 24 (3) | 18 (2) | 1 (`464525d6`) |
| Third (3,328) | 42 (14) | 33 (14) | 21 (9) | 0 |
| **Fourth, untouched (878)** | **8 (2)** | 7 (2) | **3 (0)** | 0 |

- **Held-out result.** The fourth batch (76 matches downloaded 2026-11-23 05:45, none in earlier batches; the download stopped early on server errors) was scored once, after both rules were fixed: 0.34% not reproducible and no wrong winner in 878 fights, against the goal's targets of 1% and 0.1%. One batch of 878 cannot show a rate below about 0.1% reliably; the three development batches together are at 60 of 11,086 (0.54%) with 14 wrong winners (0.13%).
- The one broken fight has a Wolverine on each side, a Tiger and a Rhino; its real result was already rare before (3% of seeds). Open.
- The fourth batch is now development data too.

## 2026-11-23 10:20: fifth batch, untouched, no rule changes since the last score

A new download (`raw-fresh4`, 56 matches not in any earlier batch; the server again returned 502 for most replays and the download stopped itself) was scored once on the unchanged engine g (Rust, 5,000 seeds a fight): **1 of 653 fights not reproducible (0.15%), no wrong winner** (`fixtures-fresh4-all.json`, `rust-v31g-5000-fresh4.json`). The one miss is fight `364ffdfa` (match `01a01880`, turn 12): the real winner is right in 96% of seeds but the real surviving board never appears among 164 distinct simulated end states. Not yet looked at.

Two untouched batches scored after the Wolverine and Dirty Rat rules now stand at 4 of 1,531 not reproducible (0.26%) with no wrong winner. That meets the 1% target with room; 1,531 fights still cannot show a wrong-winner rate below 0.1% (zero observed gives an upper bound of about 0.2% at 95%).

