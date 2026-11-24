# Real Turtle shop turns against the Python engine

**V31 reproduced 6,596 of 6,735 replayable turns (97.94%).** The held-out result was 2,385/2,431 (98.11%). A pass requires one native execution to match all five board slots, pet species, level, XP, permanent and temporary attack/HP, perk, frozen inventory, and the gold ledger. The corpus contains 7,445 completed seat-turns from turn 2 onward. Input identity was unresolved on 710; the confirmed share of **all** completed turns is 88.60%.

**The complete shop is a stricter test.** At a common budget of 1,024 seeds, 6,596 turns matched the requested fields, 6,405 also matched the remaining offers without regard to order, and **4,664/6,735 (69.25%)** also matched both shop arrays in order. Food ordering explains a large part of this gap. The replay stores the order; correspondence with screen coordinates was not measured. [Full-shop results](full-shop-summary.json) · [Food-order evidence](food-order.json)

## Coverage and version comparison

| Match split | Matches | Completed turns | Skipped | Attempted | Reproduced | Misses | Success among attempts |
|---|---|---|---|---|---|---|---|
| dev | 214 | 4,745 | 441 | 4,304 | 4,211 | 93 | 97.84% |
| heldout | 126 | 2,700 | 269 | 2,431 | 2,385 | 46 | 98.11% |
| all | 340 | 7,445 | 710 | 6,735 | 6,596 | 139 | 97.94% |

| Reason for skipping a completed turn | Dev | Held-out | Total |
|---|---|---|---|
| unidentified purchased/merged pet | 418 | 252 | 670 |
| unmapped purchased food enum | 5 | 2 | 7 |
| unidentified referenced board pet after summon | 18 | 15 | 33 |

The 340 ranked Turtle matches comprise 300 local downloads and 40 archived bundles, with 680 perspectives and 8,121 terminal snapshots. The 676 turn-one snapshots supplied initial conditions or were outside the requested turn range. Two additional shop starts, on turns 8 and 13, lack a terminal snapshot. They are outside the completed-turn denominator. Unmapped consumed food enums 130 and 131 account for the seven food skips. [Corpus inventory](corpus.json) · [Immutable source manifest](provenance.json)

All three source pins gave the same pass/miss/skip classification for every turn: the original v31, the later v31 with changes to fainting Rabbit and copied Sheep behavior, and pinned v30. The four battle fixes did not change this corpus’s shop-turn pass count. The snapshots were taken on November 23 at 05:53 and 06:19 UTC. The original v31 package hash is `1f4abd3ef1c578fc207f23612275988f0a802b778d9f0ef2a0fe661646ef4a25`, the refreshed v31 is `bc58ba7d76f454ae4e6db5fc4dc10efc2f2f943d3dce3ff63c7ebabf3d6646e8`, and v30 is `84535ca3636d1a69919ce0de954df171e78b7ab60164d818800c82b69568c6b8`. These are SHA-256 hashes of sorted per-file Python source digests. [Paired turn results](turn-results.csv) · [Versions and split counts](SUMMARY.json)

## Which fields failed

| Observed condition within the native seed budget | Dev | Held-out | Total |
|---|---|---|---|
| Reached end of turn | 4,302 | 2,428 | 6,730 |
| Exact board including levels, split stats and perks | 4,229 | 2,391 | 6,620 |
| Exact set of frozen item IDs | 4,302 | 2,428 | 6,730 |
| Exact frozen item details and costs | 4,283 | 2,422 | 6,705 |
| Exact spending, gain, remaining gold and logged food prices | 4,302 | 2,428 | 6,730 |
| All requested conditions on one seed | 4,211 | 2,385 | 6,596 |

Of 139 primary misses, 134 reached a terminal state and five stopped at an action referencing an item absent from the simulated stock. The closest completed states differed on permanent stats alone in 100 turns, frozen stock alone in 24, board perk alone in nine, and board perk plus frozen stock in one. An absent UID is a turn-replay failure. Diagnosis still has to distinguish native generation, summon placement and inventory eviction from uncertainty in reconstructing a hidden roll. One of these five is the observed Sloth offer. The field rows above are independent existence checks. The last row requires simultaneous agreement. [Failure receipts and exact seeds](turn-results.csv)

### Three recurring mechanics in the requested comparison

1. **Dragon buffs the wrong recipient.** In 44 misses (28 dev, 16 held-out), transferring the level-scaled attack and HP bonus from the Dragon to the purchased tier-one pet accounts for the endpoint difference. In the Pig receipt the real Dragon and Pig finished 3/8 and 5/2; the engine produced 4/9 and 4/1. Native `AllOtherFriends` explicitly removes the purchase trigger from the target list. The calculator’s all-living-pets loop also differs from the exclusion illustrated by this replay. [Dragon/Pig turn](cases/dragon-pig.json) · [Dragon ability](../../SAP-Calculator/src/app/domain/entities/catalog/pets/turtle/tier-6/dragon.class.ts#L64) · [native target exclusion](pinned-v31/sapai/effects.py#L269)
2. **Scorpion gains Peanut at the wrong time.** 30 primary misses have the Scorpion signature (22 dev, 8 held-out). All 244 recorded shop sightings of 239 distinct Scorpion offers had no perk. A fresh native Scorpion already carries Peanut in the shop. When the harness restores a genuinely bare frozen Scorpion and buys it, the native buy can leave it bare on the board. The real pet gains Peanut on summon. A further 166 primary matches showed the wrong Peanut on unsold stock when all offers were inspected on their matching seed. [Bare frozen Scorpion purchase](cases/scorpion-purchase.json) · [Scorpion summon handler](../../SAP-Calculator/src/app/domain/entities/catalog/pets/turtle/tier-5/scorpion.class.ts#L45) · [native pet data](pinned-v31/sapai/data.py#L7796)
3. **Rabbit’s three food triggers can go to different pets.** In 18 misses the observed health split can be obtained by reordering notifications among the same food recipients. In 7 misses, excluding off-board shop pets from Rabbit’s use budget accounts for the endpoint. These are sufficient arithmetic explanations on the recorded action sequence. They do not identify a unique real-game priority rule. Native multi-target food uses ordered combinations and immediately dispatches listeners per eater. Canned Food iterates over shop pets through the same loop. The order matters when a Rabbit has fewer uses left than there are eaters. [Food-target enumeration and health ledgers](food-probes.json) · [Rabbit trigger and use limit](../../SAP-Calculator/src/app/domain/entities/catalog/pets/turtle/tier-3/rabbit.class.ts#L45) · [native food-target ordering](pinned-v31/sapai/effects.py#L485) · [native per-recipient dispatch](pinned-v31/sapai/player.py#L330)

These mechanism counts require a state difference with the indicated signature. Other turns that merely contain the same pet appear in the association tables below. The Rabbit diagnostic only changes the allocation of health bonuses in a separate arithmetic comparison. It never changes the engine’s result. [Signature receipts](mechanic-signatures.json)

### Remaining shop contents and order

| Joint endpoint within 1,024 seeds | Dev | Held-out | All |
|---|---|---|---|
| Requested board, frozen items and wallet | 4,211 | 2,385 | 6,596 |
| Requested fields plus all remaining offer details, any order | 4,089 | 2,316 | 6,405 |
| Requested fields plus pet UID order | 4,108 | 2,334 | 6,442 |
| Requested fields plus food UID order | 3,135 | 1,766 | 4,901 |
| Requested fields, all offer details and both orders | 2,982 | 1,682 | 4,664 |

The complete-offer comparison includes UID, kind, species or food variant, cost, freeze flag, discover flag, and the pet’s level, XP, split stats and perk. It checks each array separately. Hidden unbought offers were supplied by name only when that name is visible elsewhere in the replay; their final stats and flags were never injected. On the primary matching seed, 20 of 6,596 primary matches already disagreed about which unbought IDs remained. [Same-seed inventory results](remaining-shop.jsonl)

Among **2,467 pairs** of ordinary foods known to have been generated together on the last refresh, the replay array put the higher-tier food first in **1,840/1,840** unequal-tier pairs. All 627 same-tier pairs retained ascending creation ID. The combined rule matched 1,597/1,597 dev and 870/870 held-out pairs. The native order matched only 1,511/2,467 and reversed 956 relative to the record. For example, on turn 7 of match `01a19a7d-e2fd-78f6-aacb-0d0ee570adf8`, the replay listed tier-two Sleeping Pill #109 before tier-one Honey #108. The engine listed Honey first. Frozen carryovers, stocked foods and screen rendering are outside this paired check. [Pair counts and examples](food-order.json) · [Native roll sorting](pinned-v31/sapai/shop.py#L309)

## Gold, prices and sale proceeds

**The records reconcile on all 7,445 turns.** One gold per Type 5, three per Type 6, plus the actual price on each Type 8 food receipt sums exactly to `GoSp`. Type 7 board merges add zero. The number of Type 5 rolls also equals `Rold` on every turn. Every snapshot satisfies `10 + GoGa − GoSp = Go` with missing numeric fields read as zero. Spending exceeded ten on 3,999 turns; 19 turns left money unspent. An independent income ledger using Swan level, pet sale level, Pig income and carried Cake sale bonuses reconstructed `GoGa` on **6,912/6,912** calculable turns. The other 533 lacked a required sale identity or level for that independent calculation. [Per-turn ledger](gold-ledger.csv) · [Counts by split](raw-gold.json)

The engine wallet and food prices matched on all 6,730 strict turns that could reach the end. There is a separate native sale bug on a turn excluded from strict scoring because a summoned pet was sold without its species being recorded. A level-two Spider was pilled. The real player received three gold from the ordinary level-one Dolphin and the Spider summon together. Across 256 native seeds, **all ten possible level-two tier-three summons sold for one**. The engine then rejected a real three-gold purchase with only two gold. The summon assigns `level` directly while the pet constructor initializes `sell_value` to one. This isolates the wallet discrepancy without assuming which species appeared in the actual shop phase. [Sale evidence across all ten species](spider-sale.json) · [Native summon](pinned-v31/sapai/effects.py#L1286) · [Sale credit](pinned-v31/sapai/player.py#L424)

## Ten smallest failures of the board, wallet and frozen-shop comparison

Sorted by the number of logged actions, then by occupied starting slots, then match ID, perspective ID and turn. Start and end are included in the sort and omitted from the action column. Each difference reads **real / engine**. HP is permanent unless marked temporary. The comparison uses the closest tested native result; each linked receipt includes both complete boards, frozen inventories, player actions, chosen seed and input identities. Held-out examples are marked so their use for future fitting can be tracked.

| Match and turn | Split | Actions between start and end | Real / engine difference | Mechanic indicated |
|---|---|---|---|---|
| [01a1c61f · T6](cases/01-01a1c61f-t6.json) | dev | buy Rabbit, Salad Bowl, roll, Salad Bowl | Horse x1 HP 5 / 4; Rabbit x4 HP 5 / 6 | Two Salad Bowls and one Rabbit. Permuting notifications within an already-selected Salad pair reproduces the real health split. |
| [01a1d3c6 · T6](cases/02-01a1d3c6-t6.json) | dev | buy Rabbit, Salad Bowl, Salad Bowl, roll | Camel x2 HP 8 / 7; Turtle x3 HP 8 / 9 | Two Salad Bowls reach a Rabbit’s three-use cap. The same recipients can generate the recorded health if the capped notification goes to the other pet. |
| [01a1b9fd · T10](cases/03-01a1b9fd-t10.json) | dev | Canned Food, Canned Food, roll, Bread | Tiger x2 HP 6 / 4 | The two Canned Foods consumed the native Rabbit’s uses on shop pets. The real Tiger received the level-two Rabbit’s +2 HP when it ate Bread. |
| [01a1ccee · T7](cases/04-01a1ccee-t7.json) | heldout | Canned Food, roll, Bread, Salad Bowl | Penguin x1 HP 7 / 6; Snail x2 HP 9 / 8; Flamingo x3 HP 9 / 8 | Ignoring Canned Food’s shop pets for Rabbit’s budget accounts for the three later +1 HP adjustments on the actual board. |
| [01a1ce58 · T12](cases/05-01a1ce58-t12.json) | dev | Steak, roll, Steak, Pear | Frozen Scorpion #250 perk none / Peanut | The real frozen Scorpion was bare. The native constructor attached Peanut while the pet was still in the shop. |
| [01a1d3c6 · T8](cases/06-01a1d3c6-t8.json) | dev | Canned Food, buy merge Squirrel, buy merge Turtle, Cupcake | Squirrel x0 HP 7 / 8; Camel x3 HP 14 / 13 | Canned Food precedes the Squirrel buy-merge and the Camel’s Cupcake. The native shop Squirrel picked up an extra HP that survived the merge; the real Camel received that HP. This matches Rabbit spending a use on shop stock. |
| [01a1d859 · T12](cases/07-01a1d859-t12.json) | heldout | buy merge Tiger, Mushroom, roll, Pear | Frozen Scorpion #203 perk none / Peanut | The recorded frozen Scorpion had no perk while its native counterpart already had Peanut. |
| [01a1db4e · T12](cases/08-01a1db4e-t12.json) | dev | Steak, roll, Sushi, Pear | Rhino x3 HP 16 / 18; Rabbit x4 HP 28 / 26 | Steak consumes one Rabbit use before Sushi. Changing which two Sushi recipients receive the remaining health bonuses accounts for the difference, with the Cat and Sushi stat buffs unchanged. |
| [01a1e117 · T7](cases/09-01a1e117-t7.json) | dev | roll, Salad Bowl, buy merge Ox, Salad Bowl | Rabbit x0 HP 4 / 3; Rabbit x4 HP 11 / 12 | Two Rabbits and two Salad Bowls. Independent recipient order for each Rabbit can split the last health points as recorded. A single shared fixed order does not reproduce this split. |
| [01a194ec · T10](cases/10-01a194ec-t10.json) | dev | Better Apple, roll, Sushi, roll, buy Whale | Cat x1 HP 7 / 9; Shark x2 HP 10 / 8 | Better Apple consumes a Rabbit use before Sushi. Changing the order of the Sushi notifications accounts for the Cat/Shark health difference. |

## Failure rates by action and food

Each turn contributes at most once per row. The denominator is replayable turns containing that action or food. A turn can appear in several rows. Only end snapshots are available for most stat changes, so these rates measure association. `Board order request` and `freeze/unfreeze request` mean a nonempty command array was attached to an action.

| Action occurring in the turn | All misses / attempted | Dev | Held-out |
|---|---|---|---|
| board merge | 28/2,387 (1.17%) | 16/1,529 (1.05%) | 12/858 (1.40%) |
| board order request | 106/5,269 (2.01%) | 75/3,372 (2.22%) | 31/1,897 (1.63%) |
| buy | 116/5,228 (2.22%) | 76/3,313 (2.29%) | 40/1,915 (2.09%) |
| buy merge | 43/2,682 (1.60%) | 29/1,686 (1.72%) | 14/996 (1.41%) |
| end | 139/6,735 (2.06%) | 93/4,304 (2.16%) | 46/2,431 (1.89%) |
| food | 123/4,761 (2.58%) | 81/3,066 (2.64%) | 42/1,695 (2.48%) |
| freeze/unfreeze request | 100/4,452 (2.25%) | 67/2,856 (2.35%) | 33/1,596 (2.07%) |
| roll | 125/6,346 (1.97%) | 84/4,055 (2.07%) | 41/2,291 (1.79%) |
| sell | 92/3,061 (3.01%) | 61/1,925 (3.17%) | 31/1,136 (2.73%) |
| start | 139/6,735 (2.06%) | 93/4,304 (2.16%) | 46/2,431 (1.89%) |
| type 6 pet purchase | 125/6,000 (2.08%) | 84/3,809 (2.21%) | 41/2,191 (1.87%) |

| Food purchased | All misses / attempted | Dev | Held-out |
|---|---|---|---|
| #130 | — | — | — |
| #131 | — | — | — |
| Apple | 5/890 (0.56%) | 3/563 (0.53%) | 2/327 (0.61%) |
| Best Apple | 1/17 (5.88%) | 1/16 (6.25%) | 0/1 (0.00%) |
| Best Milk | 1/2 (50.00%) | 0/1 (0.00%) | 1/1 (100.00%) |
| Better Apple | 7/229 (3.06%) | 6/143 (4.20%) | 1/86 (1.16%) |
| Better Milk | 11/35 (31.43%) | 7/19 (36.84%) | 4/16 (25.00%) |
| Bread | 9/659 (1.37%) | 4/424 (0.94%) | 5/235 (2.13%) |
| Bread Crumbs | 10/262 (3.82%) | 8/167 (4.79%) | 2/95 (2.11%) |
| Cake | 1/344 (0.29%) | 1/227 (0.44%) | 0/117 (0.00%) |
| Canned Food | 23/211 (10.90%) | 13/131 (9.92%) | 10/80 (12.50%) |
| Chili | 5/91 (5.49%) | 3/55 (5.45%) | 2/36 (5.56%) |
| Chocolate | 32/790 (4.05%) | 20/511 (3.91%) | 12/279 (4.30%) |
| Cupcake | 4/118 (3.39%) | 4/76 (5.26%) | 0/42 (0.00%) |
| Garlic | 1/176 (0.57%) | 0/116 (0.00%) | 1/60 (1.67%) |
| Honey | 0/53 (0.00%) | 0/36 (0.00%) | 0/17 (0.00%) |
| Meat Bone | 0/50 (0.00%) | 0/35 (0.00%) | 0/15 (0.00%) |
| Melon | 12/263 (4.56%) | 5/163 (3.07%) | 7/100 (7.00%) |
| Milk | 5/298 (1.68%) | 5/196 (2.55%) | 0/102 (0.00%) |
| Mushroom | 5/189 (2.65%) | 3/120 (2.50%) | 2/69 (2.90%) |
| Pear | 14/762 (1.84%) | 10/518 (1.93%) | 4/244 (1.64%) |
| Pizza | 3/214 (1.40%) | 3/140 (2.14%) | 0/74 (0.00%) |
| Salad Bowl | 18/448 (4.02%) | 14/290 (4.83%) | 4/158 (2.53%) |
| Sleeping Pill | 16/417 (3.84%) | 13/268 (4.85%) | 3/149 (2.01%) |
| Steak | 4/136 (2.94%) | 4/93 (4.30%) | 0/43 (0.00%) |
| Sushi | 11/239 (4.60%) | 9/174 (5.17%) | 2/65 (3.08%) |

## Species frequencies on rolls

Opening inventories expose every item before any purchases or rerolls. All 2,040 opening pet slots on the 680 seats were identified. Against the ten equally likely engine species, Pearson χ² was 11.30 on 9 degrees of freedom (p = 0.255). For later tiers, the main comparison uses ordinary pet offers on paid rolls made with exactly one gold left immediately before the roll. Membership is defined before the offers are drawn. The enclosing turn must replay through the end with matching wallet, prices and allocation counter; future sales are allowed. A name enters the numerator only when witnessed or established by an identity link. Discover choices and retained frozen items are excluded.

| Available tier | Offers | Identified | Missing | Pearson χ² / df | Descriptive p |
|---|---|---|---|---|---|
| 1 | 725 | 725 | 0 | 10.90 / 9 | 0.282 |
| 2 | 1,617 | 1,612 | 5 | 10.83 / 19 | 0.929 |
| 3 | 2,298 | 2,296 | 2 | 31.85 / 29 | 0.327 |
| 4 | 2,426 | 2,426 | 0 | 31.74 / 39 | 0.789 |
| 5 | 2,638 | 2,618 | 20 | 40.90 / 49 | 0.788 |
| 6 | 2,901 | 2,889 | 12 | 62.59 / 59 | 0.350 |

The six one-gold strata show no conspicuous departure for supported species. The p-values condition on identified offers and treat draws as independent; the sample contains repeated turns per match and selection into the replayable subset. Raw end-shop sightings have additional purchase and stopping bias. In the separate all-snapshot cohort **one new tier-three-shop Sloth** appeared even though the native ordinary roll assigns it zero probability. That turn is outside the UID-validated one-gold sample. The per-species file gives counts, native probability, observed fraction and worst-case bounds from unidentified slots for each tier, cohort and split. [All per-species roll frequencies](roll-odds.csv) · [Test statistics](roll-odds-summary.json) · [Native probabilities](engine-odds.json)

## Replay method and limits

The engine began from the preceding real terminal board, previous battle outcome, persistent shop buffs and frozen stock. The parser preserved five physical board positions, with x=4 at the front. Absent Enu and x decode to zero. Permanent and temporary stats remain separate. Each subsequent purchase, merge, sale, food, roll, freeze and end-of-turn trigger runs through native code. The shim binds observed species or food variants to offer IDs at generation. Prior frozen items are hydrated with their real stats and cost. Newly generated named items start with native base stats and receive native modifiers. UID labels are scoped to a seat. A complete input-side freeze inventory can establish the relative positions of otherwise opaque offers. Unknown rolled-away offers can occupy inert slots; they never serve as a purchase of a guessed species. Eviction and rare extra generation remain possible sources of UID divergence. [Replay implementation](replay.py) · [Identity ledger](identity.py)

Type 4 starts the shop, 5 rolls, 6 pays for a shop pet at a square (including buy-merge), 7 combines source onto target, 8 buys the food named in the response, 9 sells, 11 ends the turn, and 12 names the team. Type 0 supplies Build and Battle. The checked state from Build.Bor agreed with the same seat’s pre-battle UserBoard for board, shop, allocation counter and money on all 8,121 snapshots. Type 1 carries mode/opponent updates, type 2 is bookkeeping, and nine type 3 records contained no shop request. No free Type 6 board move or paid Type 7 was needed to reconcile this corpus. Real board reorders come through `Request.Data.BoardOrders`; freeze flags come through `BoardFreezes` on several different action types. Two explicit merges after a Pill use a newly summoned board source. Empty board purchases and shifting a different species toward a vacancy were distinguished from buying a copy onto the resident.

A pet seen in a later snapshot supplies its species as a replay input. Direct explicit merges also establish equal species. For an unnamed purchase onto an occupied square, the decoder can infer absorption when the source has no later board reference, is absent from the ending board, and no later Pill cascade could have destroyed it. It does not read the final pet stats to make that inference. Without any of these implicit source inferences the score was 4,778/4,891 (97.69%). On turns with no pet purchase at all it was 721/735 (98.10%). These are conditional action-replay scores. The experiment does not regenerate the original shop inventory from a real-game seed. [Extraction](extract.py) · [Coverage checks](SUMMARY.json)

Dev and held-out use SHA-256 of the raw MatchId as a big-endian integer, modulo 5, with residues 0–1 held out. Both seats stay in the same split. Unknown real RNG is handled by independent seeded native replays. Primary hits rose from 5,506 at one seed to 6,547 by 64 and 6,596 by 1,024. No additional hits appeared by 8,192 for the remaining failures or by 65,536 for the 49 that showed varied candidate states. A finite-seed miss alone is insufficient to prove a mechanism impossible. The food probes additionally enumerate native recipient subsets for the smallest cases with a fixed remaining random stream. The 95% held-out interval from resampling whole matches is 97.54–98.63%. It describes sampling variation in this corpus, not coverage of unobserved mechanics. [Convergence by split and version](seed-convergence.csv) · [Food diagnostic](food_probe.py)

## Per-pet failure rates

A pet is involved if it was on the starting or ending board, was bought/merged/sold, or occupied a frozen pet slot at either endpoint. The table covers every identified pet present on at least 20 completed turns. Rates use attempted turns. Board-only and directly acted-on variants, skipped counts, and less common pets are in the CSV. [All association counts](association-rates.csv)

| Pet | All misses / attempted | Dev | Held-out |
|---|---|---|---|
| Ant | 17/1,030 (1.65%) | 11/689 (1.60%) | 6/341 (1.76%) |
| Armadillo | 5/84 (5.95%) | 4/56 (7.14%) | 1/28 (3.57%) |
| Badger | 10/521 (1.92%) | 5/307 (1.63%) | 5/214 (2.34%) |
| Beaver | 17/764 (2.23%) | 12/455 (2.64%) | 5/309 (1.62%) |
| Bison | 3/186 (1.61%) | 3/154 (1.95%) | 0/32 (0.00%) |
| Blowfish | 22/844 (2.61%) | 14/508 (2.76%) | 8/336 (2.38%) |
| Boar | 15/311 (4.82%) | 10/218 (4.59%) | 5/93 (5.38%) |
| Camel | 34/1,173 (2.90%) | 23/714 (3.22%) | 11/459 (2.40%) |
| Cat | 28/563 (4.97%) | 17/404 (4.21%) | 11/159 (6.92%) |
| Cow | 31/663 (4.68%) | 24/431 (5.57%) | 7/232 (3.02%) |
| Crab | 8/505 (1.58%) | 3/300 (1.00%) | 5/205 (2.44%) |
| Cricket | 20/957 (2.09%) | 12/594 (2.02%) | 8/363 (2.20%) |
| Crocodile | 3/209 (1.44%) | 2/151 (1.32%) | 1/58 (1.72%) |
| Deer | 22/1,084 (2.03%) | 17/692 (2.46%) | 5/392 (1.28%) |
| Dodo | 10/504 (1.98%) | 6/321 (1.87%) | 4/183 (2.19%) |
| Dog | 8/529 (1.51%) | 3/329 (0.91%) | 5/200 (2.50%) |
| Dolphin | 5/629 (0.79%) | 4/422 (0.95%) | 1/207 (0.48%) |
| Dragon | 46/125 (36.80%) | 29/71 (40.85%) | 17/54 (31.48%) |
| Duck | 12/666 (1.80%) | 8/412 (1.94%) | 4/254 (1.57%) |
| Elephant | 22/1,108 (1.99%) | 16/665 (2.41%) | 6/443 (1.35%) |
| Fish | 16/751 (2.13%) | 11/520 (2.12%) | 5/231 (2.16%) |
| Flamingo | 7/790 (0.89%) | 5/454 (1.10%) | 2/336 (0.60%) |
| Fly | 6/236 (2.54%) | 4/157 (2.55%) | 2/79 (2.53%) |
| Giraffe | 18/982 (1.83%) | 11/599 (1.84%) | 7/383 (1.83%) |
| Gorilla | 5/136 (3.68%) | 3/77 (3.90%) | 2/59 (3.39%) |
| Hedgehog | 1/448 (0.22%) | 0/250 (0.00%) | 1/198 (0.51%) |
| Hippo | 1/107 (0.93%) | 1/75 (1.33%) | 0/32 (0.00%) |
| Horse | 17/609 (2.79%) | 13/439 (2.96%) | 4/170 (2.35%) |
| Kangaroo | 7/676 (1.04%) | 4/424 (0.94%) | 3/252 (1.19%) |
| Leopard | 16/429 (3.73%) | 9/276 (3.26%) | 7/153 (4.58%) |
| Mammoth | 9/468 (1.92%) | 7/308 (2.27%) | 2/160 (1.25%) |
| Monkey | 12/689 (1.74%) | 6/426 (1.41%) | 6/263 (2.28%) |
| Mosquito | 15/244 (6.15%) | 9/135 (6.67%) | 6/109 (5.50%) |
| Otter | 22/732 (3.01%) | 13/472 (2.75%) | 9/260 (3.46%) |
| Ox | 17/818 (2.08%) | 15/529 (2.84%) | 2/289 (0.69%) |
| Parrot | 12/518 (2.32%) | 6/321 (1.87%) | 6/197 (3.05%) |
| Peacock | 5/705 (0.71%) | 3/437 (0.69%) | 2/268 (0.75%) |
| Penguin | 9/403 (2.23%) | 4/255 (1.57%) | 5/148 (3.38%) |
| Pig | 14/762 (1.84%) | 8/485 (1.65%) | 6/277 (2.17%) |
| Pigeon | 21/774 (2.71%) | 16/508 (3.15%) | 5/266 (1.88%) |
| Rabbit | 58/1,122 (5.17%) | 42/797 (5.27%) | 16/325 (4.92%) |
| Rat | 6/246 (2.44%) | 3/166 (1.81%) | 3/80 (3.75%) |
| Rhino | 9/296 (3.04%) | 8/197 (4.06%) | 1/99 (1.01%) |
| Rooster | 8/137 (5.84%) | 7/89 (7.87%) | 1/48 (2.08%) |
| Scorpion | 37/287 (12.89%) | 28/180 (15.56%) | 9/107 (8.41%) |
| Seal | 0/53 (0.00%) | 0/37 (0.00%) | 0/16 (0.00%) |
| Shark | 12/531 (2.26%) | 8/366 (2.19%) | 4/165 (2.42%) |
| Sheep | 2/345 (0.58%) | 2/216 (0.93%) | 0/129 (0.00%) |
| Skunk | 43/1,056 (4.07%) | 26/643 (4.04%) | 17/413 (4.12%) |
| Snail | 10/1,185 (0.84%) | 6/723 (0.83%) | 4/462 (0.87%) |
| Snake | 13/327 (3.98%) | 10/226 (4.42%) | 3/101 (2.97%) |
| Spider | 7/683 (1.02%) | 3/430 (0.70%) | 4/253 (1.58%) |
| Squirrel | 9/267 (3.37%) | 6/186 (3.23%) | 3/81 (3.70%) |
| Swan | 12/639 (1.88%) | 5/376 (1.33%) | 7/263 (2.66%) |
| Tiger | 12/557 (2.15%) | 7/376 (1.86%) | 5/181 (2.76%) |
| Turkey | 5/369 (1.36%) | 3/242 (1.24%) | 2/127 (1.57%) |
| Turtle | 22/730 (3.01%) | 16/488 (3.28%) | 6/242 (2.48%) |
| Whale | 8/510 (1.57%) | 7/311 (2.25%) | 1/199 (0.50%) |
| Wolverine | 12/347 (3.46%) | 9/194 (4.64%) | 3/153 (1.96%) |
| Worm | 16/720 (2.22%) | 13/487 (2.67%) | 3/233 (1.29%) |

## Reproduction

`SAP_ENGINE_ROOT` is the single engine import root. It must contain `sapai/`. The default launcher mounts the saved v31 snapshot read-only. The following commands run in this audit directory. A fresh matrix label keeps the original receipts intact.

```bash
taskset -c 5-9,16 python3 extract.py
taskset -c 5-9,16 python3 raw_checks.py
SAP_ENGINE_ROOT=/path/to/fixed/engine ./run_matrix.sh my-fixed-engine
```

The run used CPU-only Docker with the network disabled and the same six-core affinity for all benchmark work. No learning or engine edits were performed. [Engine pins and input hashes](provenance.json) · [Docker image digest](image-id.txt) · [Reproducible runner](run.sh) · [Machine-readable summary](SUMMARY.json)
