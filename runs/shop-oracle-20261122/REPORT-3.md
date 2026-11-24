# Recovering the 710 skipped Turtle shop turns

**703 of the 710 have representable pet/summon inputs on this pin. 680 of those 703 reproduce the recorded endpoint (96.73%).** There are 23 mapped turns with no matching assignment and seven additional turns whose consumed food enum has no native representation. Across all 710, **30 have no witness**. The fresh strict check reproduced **6,045/6,119 (98.79%)** on 300 separate matches.

Only **516** turns had a single observed assignment. On 15 there were 2–5 assignments and on 149 there were six or more. All 680 count as reproduced on the requested fields. For most purposes a purchase screenshot is still valuable for the 164 multi-assignment turns. It would also supply independent verification for the singleton labels selected by the engine. [All 710 turn results](recover/turns.csv) · [All concrete passing assignments and replay seeds](recover/assignments.jsonl.gz)

## Coverage by match split

| Split | Original skips | Mapped inputs | One assignment | 2–5 assignments | 6+ assignments | Mapped with no witness | Unmapped food with no witness | Reproduced / mapped |
|---|---|---|---|---|---|---|---|---|
| dev | 441 | 436 | 326 | 9 | 87 | 14 | 5 | 422/436 (96.79%) |
| heldout | 269 | 267 | 190 | 6 | 62 | 9 | 2 | 258/267 (96.63%) |
| all | 710 | 703 | 516 | 15 | 149 | 23 | 7 | 680/703 (96.73%) |

| Original skip reason | Turns | Now reproduced | No witness |
|---|---|---|---|
| Purchased/merged pet | 670 | 653 | 17 |
| Referenced board pet after summon | 33 | 27 | 6 |
| Food enum 130/131 | 7 | 0 | 7 |

The newly modelable failure rate was **23/703 (3.27%)**. The earlier strict reference was 6,673/6,735 successes (99.08%). Under this pin those same turns scored **6,674/6,735 (99.09% success, 0.91% failure)**. One previously missed held-out turn passed. Every other strict status stayed the same. The recovered population contains more sales, and searching identities against the answer makes its score an **endpoint compatibility rate**. It is a different test of correctness from replaying known inputs.

## Which identities are independent of the engine result?

The raw records constrain the source UID, destination slot, explicit same-species merges, later board references, and gold. A normal purchase costs three regardless of species. The aggregate sale ledger sometimes isolates Pig after accounting for Swan, level, carried sale bonuses and other sales. Named Milk/Better Milk/Best Milk and Bread Crumbs receipts can constrain which buy stocked Milk or which sale stocked Crumbs. Those identifications rely on the recorded item and the published producer rule. They cannot independently validate that same producer rule. Simple Pills on known pets with a known perk can similarly determine a sole summoned species without consulting its simulated stats.

| Basis for identification | Dev match / miss | Held-out match / miss | Total match / miss |
|---|---|---|---|
| sale proceeds | 6 / 0 | 7 / 0 | 13 / 0 |
| named stock receipt + producer | 138 / 4 | 78 / 2 | 216 / 6 |
| Pill + documented summon/perk | 7 / 2 | 5 / 3 | 12 / 5 |
| terminal effects discriminate | 132 / 0 | 75 / 0 | 207 / 0 |
| engine reconstruction; no action/wallet competitor witnessed | 43 / 0 | 25 / 0 | 68 / 0 |

**252 identities were fixed before testing the endpoint.** 241 matched, 11 missed (4.37% failure). Another **275 singleton matches depended on native reconstruction**. For 207 there is a concrete *different* assignment that completes the recorded actions at the right prices and ends with the right wallet, but differs on the recorded terminal fields. For 68, the recorded constraints alone allow alternatives; the diagnostic found no second action-and-wallet-compatible assignment. Treat neither group as independent evidence for the fitted pet’s behavior. The remaining unique status is empirical within the tested RNG support. [Per-turn evidence classification and competing identities](recover/identity-evidence.csv) · [Input-side certificates](recover/input-certificates.jsonl)

For example, in dev match `01a1b2f1`, turn 6, both Ant and Beaver can be bought and sold with the correct wallet. Only Beaver gave the observed permanent stats among the tested assignments. In dev match `01a1f157`, turn 13, the two UIDs can be Duck and Beaver in either order. No choice between those two joint assignments is justified by the endpoint. A masked control hid the names of 173 otherwise observed, bought-and-sold pets, preserved their recorded frozen flags, and recomputed the input-side constraints. All 173 true names survived; 91 were uniquely predicted before simulating final stats. These controls test the pruning logic, not the identity of the genuinely hidden pets. [Masking control](recover/masked-controls.json)

## No-matching-candidate turns

| Nearest diagnostic discrepancy | Dev | Held-out | Total |
|---|---|---|---|
| Frozen Scorpion perk differs | 3 | 1 | 4 |
| Recorded action or UID generation mismatch | 2 | 2 | 4 |
| Terminal board stats/perk differ | 4 | 2 | 6 |
| Unaffordable action after selling summoned pets | 5 | 4 | 9 |
| Unmapped free food after Chocolate | 5 | 2 | 7 |

These groups describe one closest diagnostic execution or the first unusable input. They are not mutually exclusive explanations of the underlying code. The CSV includes the full match and seat IDs, first blocked action where available, example candidate, exact field deltas and real versus native wallet. Unknown RNG and the mapping of newly generated UIDs can both cause a miss. [All no-witness cases](recover/no-match.csv) · [Action sequences and snapshots](recover/negative.jsonl)

Three dev receipts isolate different issues:

- **Ram sale value.** In `01a1a19a` turn 9, Sheep #24 (one XP) merged into Sheep #142, reaching level 2. It was pilled. The only sales were both new Rams and the level-one Giraffe #63. `GoGa=5` gives **four gold for the Ram pair** after the Giraffe’s one. Native income was three total, one per Ram, and the next real roll was unaffordable. No seller-species guess is needed here. The aggregate receipt establishes the pair’s proceeds; it does not contain separate payouts for each sale.
- **Scorpion shop perk.** In `01a20136` turn 10, a stock receipt identifies the missing Cow. Board and wallet match. Frozen Scorpion #159 is bare in the real shop and has Peanut in the native shop. An alternative pet for the missing buy does not explain the frozen-slot difference.
- **Residual HP.** In `01a1e5bb` turn 10, Milk constrains the missing purchase to Cow. The closest native Tiger finishes 6/4 with Melon; the real Tiger finishes 6/6 with Melon. A level-two Rabbit and a Pill are present. The terminal discrepancy is clear; the log does not localize the trigger that lost the two HP.

## Food IDs 130 and 131

The seven turns contain **four pairs of enum 130 and three pairs of enum 131**, each pair consumed by a targeted Type 8 action. All 14 raw `Spell` objects include `Enu`, an item UID, `Pow: 0`, `Sto: false` and `Buff: true`. **All omit `Pri`**, decoding to price zero; the turn spending also reconciles at zero. None contains a display name or explicit attack/health magnitude. Each pair is the next two food purchases after Chocolate. Six Chocolate recipients are independently named Cows. The seventh is an unnamed purchase constrained to Cow by earlier ordinary Milk receipts.

The level pattern is consistent across all seven: enum **130** follows Chocolate taking its recipient from XP 0 to 1 (still level 1); **131** follows XP 1 to 2 (level 2). The clean enum-130 Elephant case gains **+2 permanent attack and +0 HP** over two uses. In the enum-131 Dodo case it gains **+8/+0** over two uses with a level-one Cat. Together with the Badger/Monkey case these observations support **+1/+0** and **+2/+0** per item before modifiers. The names “Chocolate Milk” or “Better Chocolate Milk” would be guesses. The recorded fact is a distinct, free Cow/Chocolate-associated buff with different enum IDs from ordinary Milk.

I tried 24 represented native food-name/strength aliases per unmapped enum, enforcing the observed price, source UID and natural stock generation. None yielded a joint witness. In `01a19d3e` turn 9, a Chocolate on the existing Cow is followed by casts from UIDs #141 and #140. The native shop has minted only through #139 after the Chocolate. Substituting a name for #141 cannot create it. These seven remain unsupported inputs; forcing two extra foods into stock would have tested a modified engine. [Raw spell fields, recipients and before/after stats](recover/food-receipts.json)

## Sales and actions

**Sell-heavy means at least two Type 9 sales in one seat-turn.** All cells below show **misses / attempted turns**, once per turn. Rows can overlap. A seller is attributed here only if named in the log/identity ledger or identified by the pre-endpoint certificate. A missing seller that fits a failing turn is not assigned an invented species. The independent-ID column is a subset of the new pet/summon column. [All rates by split, seller, action and sale count](recover/sell-action-rates.csv) · [Per-turn seller UIDs](recover/sell-turns.csv) · [Separate possible and necessary fitted sellers](recover/fitted-sellers.csv)

| Sale count | Originally strict | New pet/summon | Independent-ID subset |
|---|---|---|---|
| no sale | 28/3,674 (0.76%) | 4/59 (6.78%) | 0/1 (0.00%) |
| at least one sale | 33/3,061 (1.08%) | 19/644 (2.95%) | 11/251 (4.38%) |
| sell-heavy (2+) | 8/840 (0.95%) | 14/467 (3.00%) | 8/196 (4.08%) |

### By action on sell-heavy turns

| Action present | Originally strict | New candidate search | Independent-ID subset of new |
|---|---|---|---|
| any pet purchase | 8/839 (0.95%) | 14/467 (3.00%) | 8/196 (4.08%) |
| board merge | 2/335 (0.60%) | 5/117 (4.27%) | 3/50 (6.00%) |
| board order request | 8/771 (1.04%) | 12/390 (3.08%) | 8/169 (4.73%) |
| buy | 8/839 (0.95%) | 14/467 (3.00%) | 8/196 (4.08%) |
| buy merge | 4/283 (1.41%) | 5/109 (4.59%) | 1/55 (1.82%) |
| end | 8/840 (0.95%) | 14/467 (3.00%) | 8/196 (4.08%) |
| food | 6/588 (1.02%) | 13/395 (3.29%) | 8/196 (4.08%) |
| freeze/unfreeze request | 8/601 (1.33%) | 14/341 (4.11%) | 8/150 (5.33%) |
| roll | 8/738 (1.08%) | 13/420 (3.10%) | 8/183 (4.37%) |
| sell | 8/840 (0.95%) | 14/467 (3.00%) | 8/196 (4.08%) |
| start | 8/840 (0.95%) | 14/467 (3.00%) | 8/196 (4.08%) |

### By sold species on sell-heavy turns

| Seller | Originally strict | New candidate search | Independent-ID subset of new |
|---|---|---|---|
| (unidentified seller) | — | 5/242 (2.07%) | — |
| Ant | 0/70 (0.00%) | 1/28 (3.57%) | 1/7 (14.29%) |
| Armadillo | 0/4 (0.00%) | 0/1 (0.00%) | 0/1 (0.00%) |
| Badger | 0/9 (0.00%) | 0/1 (0.00%) | — |
| Beaver | 1/131 (0.76%) | 0/29 (0.00%) | 0/5 (0.00%) |
| Bison | 0/2 (0.00%) | — | — |
| Blowfish | 0/9 (0.00%) | 0/5 (0.00%) | 0/1 (0.00%) |
| Boar | — | 0/1 (0.00%) | — |
| Bus | — | 0/2 (0.00%) | 0/2 (0.00%) |
| Camel | 0/16 (0.00%) | 0/6 (0.00%) | 0/4 (0.00%) |
| Cat | 0/14 (0.00%) | 0/4 (0.00%) | 0/3 (0.00%) |
| Cow | 2/103 (1.94%) | 7/193 (3.63%) | 4/156 (2.56%) |
| Crab | 0/13 (0.00%) | 0/8 (0.00%) | 0/4 (0.00%) |
| Cricket | 0/76 (0.00%) | 1/28 (3.57%) | 0/9 (0.00%) |
| Crocodile | 0/16 (0.00%) | 0/6 (0.00%) | 0/2 (0.00%) |
| Deer | 0/40 (0.00%) | 1/20 (5.00%) | 1/12 (8.33%) |
| Dodo | 0/7 (0.00%) | — | — |
| Dog | 1/7 (14.29%) | 0/5 (0.00%) | 0/4 (0.00%) |
| Dolphin | 1/21 (4.76%) | 1/14 (7.14%) | 1/8 (12.50%) |
| Dragon | 0/5 (0.00%) | 0/4 (0.00%) | — |
| Duck | 0/214 (0.00%) | 0/38 (0.00%) | 0/1 (0.00%) |
| Elephant | 1/4 (25.00%) | 0/5 (0.00%) | 0/3 (0.00%) |
| Fish | 0/56 (0.00%) | 0/22 (0.00%) | 0/7 (0.00%) |
| Flamingo | 0/20 (0.00%) | 0/5 (0.00%) | 0/1 (0.00%) |
| Fly | 0/2 (0.00%) | 0/4 (0.00%) | 0/2 (0.00%) |
| Giraffe | 0/47 (0.00%) | 2/34 (5.88%) | 2/20 (10.00%) |
| Gorilla | 0/1 (0.00%) | 0/2 (0.00%) | 0/2 (0.00%) |
| Hedgehog | 1/21 (4.76%) | 0/6 (0.00%) | 0/1 (0.00%) |
| Hippo | 0/4 (0.00%) | — | — |
| Horse | 0/32 (0.00%) | 0/23 (0.00%) | 0/7 (0.00%) |
| Kangaroo | 0/20 (0.00%) | 1/12 (8.33%) | 1/6 (16.67%) |
| Leopard | 0/1 (0.00%) | — | — |
| Mammoth | 0/1 (0.00%) | 0/2 (0.00%) | 0/1 (0.00%) |
| Monkey | 0/17 (0.00%) | 0/22 (0.00%) | 0/12 (0.00%) |
| Mosquito | 0/28 (0.00%) | 0/13 (0.00%) | 0/3 (0.00%) |
| Otter | 1/55 (1.82%) | 0/28 (0.00%) | 0/9 (0.00%) |
| Ox | 0/17 (0.00%) | 0/6 (0.00%) | 0/2 (0.00%) |
| Parrot | 0/11 (0.00%) | 1/8 (12.50%) | 1/4 (25.00%) |
| Peacock | 0/26 (0.00%) | 0/9 (0.00%) | 0/3 (0.00%) |
| Penguin | 1/10 (10.00%) | 1/9 (11.11%) | 1/4 (25.00%) |
| Pig | 0/123 (0.00%) | 0/47 (0.00%) | 0/17 (0.00%) |
| Pigeon | 1/111 (0.90%) | 1/73 (1.37%) | 0/30 (0.00%) |
| Rabbit | 2/63 (3.17%) | 0/10 (0.00%) | 0/3 (0.00%) |
| Ram | — | 6/11 (54.55%) | 4/9 (44.44%) |
| Rat | 1/9 (11.11%) | 1/3 (33.33%) | 1/2 (50.00%) |
| Rhino | 0/3 (0.00%) | 1/5 (20.00%) | 1/3 (33.33%) |
| Rooster | 0/4 (0.00%) | 0/1 (0.00%) | 0/1 (0.00%) |
| Scorpion | 0/17 (0.00%) | 0/7 (0.00%) | 0/4 (0.00%) |
| Seal | 0/1 (0.00%) | 0/1 (0.00%) | 0/1 (0.00%) |
| Shark | 0/8 (0.00%) | 0/3 (0.00%) | 0/1 (0.00%) |
| Sheep | 0/14 (0.00%) | 0/8 (0.00%) | 0/5 (0.00%) |
| Skunk | 0/22 (0.00%) | 0/7 (0.00%) | 0/5 (0.00%) |
| Sloth | 0/1 (0.00%) | — | — |
| Snail | 1/64 (1.56%) | 0/22 (0.00%) | 0/8 (0.00%) |
| Snake | 1/3 (33.33%) | 0/4 (0.00%) | 0/3 (0.00%) |
| Spider | 0/25 (0.00%) | 0/6 (0.00%) | 0/3 (0.00%) |
| Squirrel | 0/11 (0.00%) | 0/4 (0.00%) | — |
| Swan | 1/16 (6.25%) | 0/14 (0.00%) | 0/5 (0.00%) |
| Tiger | 0/4 (0.00%) | 0/4 (0.00%) | 0/1 (0.00%) |
| Turkey | 0/21 (0.00%) | 1/12 (8.33%) | 0/6 (0.00%) |
| Turtle | 0/12 (0.00%) | 0/8 (0.00%) | — |
| Whale | 0/18 (0.00%) | 0/8 (0.00%) | 0/7 (0.00%) |
| Wolverine | 0/1 (0.00%) | 1/2 (50.00%) | 1/1 (100.00%) |
| Worm | 0/31 (0.00%) | 0/8 (0.00%) | 0/5 (0.00%) |
| Zombie Fly | — | 0/1 (0.00%) | 0/1 (0.00%) |

For example, the Ram row concerns the whole turn containing a Ram sale. It is not a direct estimate of the probability that a single Ram sale is wrong. The alternative fitted-seller file counts how often a species is possible, or required somewhere among the hidden sales on all observed passing assignments. Those counts are deliberately separate from seller failure rates.

## Fresh strict comparison

| Fresh match split | Completed T≥2 | Replayable | Skipped | Reproduced | Failure rate |
|---|---|---|---|---|---|
| dev | 4,233 | 3,815 | 418 | 3,773 | 42/3,815 (1.10%) |
| heldout | 2,546 | 2,304 | 242 | 2,272 | 32/2,304 (1.39%) |
| all | 6,779 | 6,119 | 660 | 6,045 | 74/6,119 (1.21%) |

The 300 fresh matches have **zero MatchId overlap** with the original 340. The unchanged strict parser and comparison used the same match-hash split and staged seed schedule. The fresh dataset was excluded from candidate inference, diagnostics and tuning. Fresh skips break down as 608 unnamed purchases/merges, 41 unresolved board references, nine unmapped foods, one missing start/end action and one unidentified purchased item. This is a shop endpoint comparison; the separately supplied 2.00% fresh battle number measures outcomes and survivors. [Fresh input counts](recover/fresh/corpus.json) · [Strict and inference summaries](recover/SUMMARY.json)

## Search contract and reproducibility

Current engine tree SHA-256: `b08ee281b1589ba0d240b76f774baf31970328be0000260c9d47f80f6440375c`, the hash of its sorted per-file Python SHA-256 map. The audit used a read-only mount of the saved copy, with networking and CUDA disabled and all workers confined to CPUs 5–9 and 16. The engine code was not edited. [Exact source pin](engine-pin-recover.json) · [Pinned engine](pinned-recover)

The search treated a candidate as an assignment to **all** missing UIDs together. Ordinary pet offers use the unlocked pool, including rare Sloth as a possible observed override. Linked discovers must come from the next tier (capped at 6), have distinct siblings, and cannot both be purchased. The initial domain is a superset when a previous level-up might have created a reward; the native mint checks the role. The actions enforce purchase slot, same-species merging, sale legality, freezes, all later references and food prices. The wallet prunes impossible Pig combinations when sale levels are known. Named stock receipts constrain possible producers within a refresh interval. Each terminal witness matches board positions **and UIDs**, species, level, XP, permanent/temporary attack and HP, perk, frozen item details and wallet on the same execution.

At seed index zero the uncompressed pass tried **1,552,285 individual purchase assignments** after input pruning, including one empty purchase vector on summon-only cases. It made 1,559,636 native calls including food-alias and Spider alternatives. All implicated Spider choices were explored from the native tier-three pool. Other generated pets came from the native Pill, perk and summon chain. A separate pass on Pill turns also tested alternate UID bindings among simultaneously born pets, preserving their native species, stats and order.

Further seed passes revisited alternatives, including alternatives on turns that already had one match. Broad sets received a shorter refinement; rare/singleton sets were prioritized up to 1,024 seeds and outstanding zero-witness turns up to 8,192, with early exits for deterministic branches. For refinement only, same-tier nontriggering pets bought and immediately sold could share a trial when their slot/merge and discover constraints agreed. **Every retained assignment was then expanded and replayed concretely with its own named inputs.** The receipt file contains 52,947 such assignments. Counts of matching assignments are witnessed lower bounds. In particular, “unique” does not prove another identity could never match on an untested RNG path. [Candidate engine](recover/infer.py) · [Search](recover/search.py) · [Concrete verification](recover/verify.py)

When an input-side freeze list followed a refresh, known newly generated ordinary pet pairs supplied tier-order constraints. Across 12,046 identified pairs there was one strict tier-order exception, the observed Sloth. The filter exempts Sloth and imposes no equal-tier UID order. It removed five candidate assignments without changing a turn’s category. Omitting Sloth entirely yields the same 680 reproduced turns and 516 singletons. Among the remaining matches, 35 have 2–5 native-pool assignments and 129 have six or more. No turn requires Sloth to pass. On the 23 mapped zero-witness turns, a sensitivity pass also removed the gold/stock-producer pruning while retaining recorded actions and endpoint checks. No additional witness was found under its smaller seed budget. [Offer-order check](recover/order_evidence.py) · [Relaxed-pruning results](recover/relaxation.jsonl)

The concrete single-seed pass already found witnesses for 560/710. Multi-seed and summon-branch refinement brought this to 680. Neither the 1,024-seed refinement on the remaining small sets nor the 8,192-seed follow-up on the residual zero-witness cases changed their status. The seven food turns remain outside the comparable pet/summon denominator. All raw-file hashes for the fresh corpus and the 23-hunk patch digest are preserved. [Provenance](recover/provenance.json) · [Replay one stored witness](recover/replay_one.py)
