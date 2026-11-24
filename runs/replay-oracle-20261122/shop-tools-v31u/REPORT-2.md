# Shop verification, battle handoff and match settlement

The frozen Python source for this audit is [pinned-handoff](pinned-handoff), digest `1737f28de96f2735eb150f1b1b5118bbafac713d62fb727128273246eda56f1c`. Checks used the archived 340 matches on at most six CPU cores. Findings below refer to this copy. [Per-file hashes](engine-pin-handoff.json)

## 1. Shop changes

**Food order is still wrong on the same 956 pairs.** The engine matched 1,511 of 2,467 co-generated ordinary food pairs. Both the original shop seeds and the new v31t witness seeds give those same counts. The sorter reads `slot.obj.fd["tier"]`. A `Food` stores its **ability dictionary** in `fd`, and the tier lives in `data["foods"][slot.obj.name]["tier"]`. The fallback returns zero, so stable sorting preserves draw order. Across 6,730 common completed terminal witnesses, **zero** pet-order or food-order comparison flags changed. [Same-pair rerun](handoff/order.json) · [Food construction](pinned-handoff/sapai/foods.py#L49) · [Tier lookup](pinned-handoff/sapai/shop.py#L329)

Best joint reproduction at a common 1,024 shop seeds, with 6,735 attempted turns in each version:

| Endpoint | Original audit | Current pin | Newly matched | Newly missed |
|---|---:|---:|---:|---:|
| requested fields | 6596 | 6673 | 77 | 0 |
| plus remaining offer contents | 6405 | 6484 | 79 | 0 |
| plus pet order | 6442 | 6519 | 77 | 0 |
| plus food order | 4901 | 4956 | 55 | 0 |
| plus all content and both orders | 4664 | 4722 | 58 | 0 |

These order rows also require the requested board, frozen stock and wallet. An increase there can come from a corrected board while the slot order stays the same. [Paired comparison](handoff/full-comparison.json)

**Spider sale is fixed.** On the real turn where a level-two Spider was pilled, all ten possible summons sold for two gold. With the one-gold Dolphin sale, the player could make the recorded three-gold purchase and complete the turn in all 256 seeds. The changed line is in `SummonRandomPet`. [Sale receipt](handoff/spider.json) · [Implementation](pinned-handoff/sapai/effects.py#L1294)

| Other summon probes | Native level | Sale through Player.sell |
|---|---:|---:|
| Level-two Deer pilled, 10/6 Bus with Chili | 1 | 1 |
| Level-three Deer pilled, 15/9 Bus | 1 | 1 |
| Level-two Sheep pilled, two 4/4 Rams | 1 each | 1 each |
| Whale releases a Fish through SummonStoredPets, Whale level two / three | 2 / 3 | **1 / 1** |

Deer scales the Bus’s stats and its descriptor omits `withLevel`. The Bus sale agrees with its native level; the corpus did not provide a confirmed pilled level-two Deer for real-game comparison. The Whale probe invoked native swallow/release and then sold the released pet. Stored releases ordinarily happen in battle, so this is a forced reachability test of the object’s sale value. `SummonStoredPets` delegates to `SummonPet`, which sets `level` from `withLevel` without updating `sell_value`. [All summon probes](handoff/summons.json) · [Deer descriptor](pinned-handoff/sapai/turtle_pack.py#L186) · [Stored release](pinned-handoff/sapai/effects.py#L1420)

## 2. END into battle

**252/3,666 distinct fights (6.87%) failed when the source was the engine’s own END team, versus 85/3,666 (2.32%) from real board snapshots.** Folding temporary HP into the native battle copy’s starting health recovered all 167 extra misses. The input shop team remained unchanged after each battle.

| Cohort | Fights | Snapshot misses | Chained misses | Snapshot outcome absent | Chained outcome absent |
|---|---:|---:|---:|---:|---:|
| Unique, dev | 2,339 | 52 | 178 | 7 | 33 |
| Unique, held-out | 1,327 | 33 | 74 | 2 | 11 |
| Unique, total | 3,666 | 85 | 252 | 9 | 44 |
| All reproduced seat-turns | 6,673 | 147 | 412 | 15 | 73 |

Every reproduced seat-turn used its own turn-replay seed and the live post-END team object with its original slots, buffs and Parrot state. Its opponent was imported from the real canonical fight. Both arms retained the canonical RNG orientation even when the reproduced seat was the opponent. The unique row uses the canonical seat when available (3,306 fights), otherwise the mirror (360). Both seats qualified in 3,007 fights. Each arm tried the same list of up to 200 seeds, stopping at the first exact winner and order-free surviving species/attack/HP multiset. The snapshot pass and outcome-possible flags independently agree with Claude’s 200-seed archive for every compared fight. The corpus-wide snapshot figure of 86/4,065 includes fights outside this paired denominator. [Comparison script](handoff/chain.py) · [Split and paired results](handoff/chain-summary.json)

**Temporary health isolates the gap.** On all 265 seat-level chain-only misses, folding *only HP* into the battle copy recovered the exact recorded result on the snapshot’s first successful seed. Folding attack alone recovered zero. Restoring snapshot counters, Parrot copy or compact slots individually also recovered zero in 200 seeds. In 216 of 265 traces the first visible disagreement occurred during start-of-battle abilities. For example, in match 01a19a7d, turn 8, a Bread Bison starts with 13 effective HP (6 permanent + 7 temporary). After the start phase with Skunk and its Parrot copy, the native Bison has 17 HP and the snapshot Bison has 8. ReduceHealth writes a percentage of effective HP to the permanent field while the getter continues adding the temporary field. Other first divergences include hurt/faint and knockout phases. The HP intervention isolates the representation across the sample; the Skunk assignment explains this particular trace. [First-divergence receipts](handoff/first-divergence.json) · [ReduceHealth](pinned-handoff/sapai/effects.py#L956)

**Counters carry through END.** In the direct Rabbit probe two apples advanced the counter to 2. END, battle construction and battle start each retained 2. The original still had 2 after battle and reset to 0 at the next shop. One reset in pets.py is at start of turn; the other is during level-up. In the same-turn chocolate test a Rabbit with two uses leveled and finished at one, after chocolate consumed a new use. [Lifecycle probe](handoff/lifecycle.json) · [Reset sites](pinned-handoff/sapai/pets.py#L246)

Across 32,494 pet comparisons on all 6,673 reproduced turns, native END matched the replay counter on 32,454. Of 1,415 comparisons with at least one nonzero value, 40 differed across 37 seat-turns, all Rabbits. Native/replay pairs were 3/2 (19 pets), 2/1 (16) and 1/0 (5). The comparison follows the snapshot converter’s maximum TrCo over ability entries, treating an omitted value as zero. Matching board stats alone therefore does not certify the hidden counter. [Per-turn counters and battle results](handoff/chain.jsonl)

## 3. Match settlement and reward timing

**No, settlement does not match at every recorded phase.** In all 331 matches whose two final outcome codes are win/loss, the native winner, terminal battle number and both reported LastTurn values match. All 8,121 available pre-battle life totals and all 8,121 cumulative win counts match. The difference is observable immediately after battle two.

| Recorded endpoint | Agreement |
|---|---:|
| Before battle, player hearts from Battle.UserBoard | 8,121 / 8,121 |
| After battle, player hearts from Build.Bor | 7,779 / 8,121 |
| After battle, opponent lives from the Type 1 mode record | 7,772 / 8,121 |
| Immediate battle life-swing term on paired rounds | 3,784 / 4,056 |
| Ordinary win/loss matches, winner and last turn | 331 / 331 |

All 342 Build heart mismatches are on turn two. The same 342 hearts are restored between the turn-two post-battle and turn-three pre-battle snapshots. The native rule restores them inside _player_fight_outcome, before measuring the battle reward. For example, match 01a1a982 starts turn two at 6/6, the real battle ends 6/5, and the next pre-battle is 6/6. Native settlement immediately reports 6/6. The recorded battle-only life swing implies +0.1, while the actual reward path paid zero. The 272 reward disagreements were all on turn two. Replays provide lives and outcomes; they do not specify which RL shaping convention to use. The observation at the next decision already has the correct hearts. [Phase arithmetic](handoff/settlement-phase.json)

The other nine matches had final codes 1/3 and asymmetric battle histories. Outcome-only settlement stopped on the recorded final battle with the recorded winner for two; seven had no native terminal state after the last supplied battle. No seat pair in this group has two matching native terminal clocks (0/9). The extra seven discrepancies in the mode opponent-lives column occur in this group. The specific cause of code 3 was not established from battle outcomes. Across all 340 matches, native outcome-only settlement matched the recorded winner on 333 and both LastTurn values on 331. The environment increments its turn counter after the terminal battle, matching the ordinary replay clock convention. [Per-match results](handoff/settlement.jsonl) · [Exceptions](handoff/settlement-summary.json)

The test passed END for both players through the normal ParallelSelfPlayEnv.step, substituting only the logged battle winner. Gold was set to zero to make END legal. Lives, wins, the turn cap, termination and rewards came from the environment. All 4,065 distinct recorded battle outcomes were processed. [Executable check](handoff/settlement.py)
