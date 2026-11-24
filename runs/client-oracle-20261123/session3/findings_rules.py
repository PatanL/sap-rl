# Rule table for findings.json (executed by findings_build.py; `rule`, `caps`, `fights` are provided there).
# Slots: 0 = front. Capture labels are directories under session2/captures/.
# real_fights are taken from session3/attribution.json (fights whose first real difference carries the rule id).

rule('R1','3a',
 "When several friends die in the same damage event (one Hedgehog blast), a Fly reacts to those deaths in a random order that changes with the battle seed and does not depend on attack or position. Each zombie fly uses the slot rule at its turn, so the order decides which dead friends are replaced when the Fly's 3 uses or the board run out, and where other summons end up.",
 "Sorts the simultaneous deaths by attack, highest first (ties in recorded order), and lets the Fly react in that fixed order.",
 "Give deaths caused by one multi-target effect a random order drawn from the battle RNG (see R3 for how the Fly then walks through pending deaths). A uniform permutation is the simplest model consistent with the data; the exact distribution is not established.",
 caps('c3-a1-fly4dead-*','c3-a2-fly4tie-*','c3-a5-fly-sheep-*','c3-a6-fly-deer-*'),confidence='high',
 notes="a1: 8 seeds gave 8 different spawn orders; the unreplaced slot was the front one 3 times, slot 2 twice, slot 3 three times. a2 (all four attacks equal) gave the same order as a1 for the same two seeds, so attack plays no part. a6: the two seeds end in different boards (Bus second vs Bus third) only because of this order. a5: Fish (1 attack) then Pig (5) served and Sheep (7) left out on both seeds; the engine serves Sheep and Pig.")

rule('R2','3a',
 "With two pets holding the Fly ability (two Flies, or a Fly and a Parrot that copied Fly), the one with the higher attack reacts to every dead friend first; the other only gets what is left. It is attack, not position or level, that decides.",
 "Goes death by death and lets every Fly react to each death in slot order, so the front-most Fly holder acts first and the two alternate.",
 "Process friend-faint observers observer by observer in attack order (highest first); for each, run through all pending deaths before moving to the next observer.",
 caps('c3-a3-twofly-*','c3-a3r-twofly-rev-*'),confidence='high',
 notes="a3: the level-2 Fly has 8 attack: three 8/8 zombies on both seeds. a3r: attacks swapped (the level-1 Fly has 8 and sits behind): three 4/4 zombies on both seeds. In six real fights the zombie's level is that of the higher-attack Parrot, not of the Fly standing ahead of it.")

rule('R3','3a/3h',
 "When the pending deaths come from different abilities (not one blast), the Fly's order is again not the engine's attack order, and no single rule was found that covers every case. What was seen: (a) two start-of-battle snipes and a Fly with one use left: the zombie replaces the first-killed pet on one seed and the second-killed on the other, and swapping which sniper acts first swaps the result, so the choice is a seeded pick among the deaths; (b) Whale swallows a plain pet and a Crocodile kills another, one use left: either on different seeds; (c) Whale swallows a Deer (whose Bus refills its slot), Crocodile kills the back pet, three uses, one free slot: the back pet is always replaced (8 of 8 seeds, and the real fight this was built from) and the Deer never; (d) in six real fights with one or two free slots the replaced pets were never the engine's choice.",
 "Serves pending deaths by attack, highest first, with every dead pet already removed.",
 "At least: stop ordering the Fly's reactions by the dead pets' attack; draw the order from the battle RNG. Case (c) shows the outcome is not always a free 50/50, so a plain shuffle will still miss some fights; the exact rule (possibly dead pets leaving the board one at a time, so that a not-yet-processed corpse still blocks a slot) needs more tests.",
 caps('c3-h6-fly-two-deaths-sob-*','c3-h9-fly1-*','c3-h9r-fly1-*','c3-h10-fly1-*','c3-h10r-fly1-*','c3-a7-fly-one-use-left-*','c3-c5-whale-fly-*','c3-c5s-whale-shark-*','c3-h7*','c3-h8*'),confidence='low',
 notes="The observations are solid (h6: 8 seeds; h9, h9r, h10: 2 seeds each, the two seeds disagree in each); the rule behind them is open. A corpse-blocks-its-slot model explains (c) but not real fight dda8b4ec nor h7 (seed m: the zombie takes the sniped Deer's slot and the Bus is pushed back into the other dead pet's slot, so that slot was already free); a free shuffle explains those but not (c). h7/h7r/h8 (two snipes, one victim a Deer, three uses): the zombie went to the Deer's slot in 3 captures and to the plain pet's in 3. a7 shows the client honours a recorded use count (TrCo): a Fly with two uses spent makes one zombie. c5/c5s show that a swallowed friend counts as fainted for Fly and Shark.")

rule('R4','3d',
 "Start-of-battle abilities are taken one at a time and the next one is always the pet with the highest attack at that moment. A Dodo's attack gift therefore moves the receiver ahead of pets it now out-attacks (Skunk 5->10 acts before a Crab with 6; a Whale 3->20 swallows before a Dolphin with 10 shoots; a Leopard 13->35 shoots before a Skunk with 14; an Armadillo 5->12 buffs before a Dolphin with 7).",
 "Fixes the order from the attack values before any start-of-battle ability has run.",
 "Re-pick the next start-of-battle actor from current attack after every ability (keep the random tie-break).",
 caps('c3-ddyn-sob-dynamic-*','c3-ddynctl-crab-higher-no-dodo-*'),confidence='high',
 notes="ddyn: real Crab 6/10 (gains 25% of the Fish after Skunk, 26 -> +7), engine Crab 6/13 (gains before Skunk).")

rule('R5','3e',
 "A Wolverine whose four-friends-hurt count is reached during an attack does not act until everything else of that attack has resolved, including Rhino's knock-out damage and its chain. A count that passes 8 in one chain gives two activations. In the one real fight with Wolverines on both sides the higher-attack one acted first.",
 "Runs the queued Wolverine at the end of the hurt/faint phase, before the knock-out phase, so enemies are weakened before Rhino's knock-out damage lands and Rhino chains further.",
 "Move the pending Wolverine activations to after the knock-out phase (and after any hurt/faint chain the knock-outs cause); count hurts that happen in the knock-out phase before deciding how many activations are due.",
 caps('c3-e5b-wolv-vs-rhino-tier2-*','c3-e5-wolv-vs-rhino-*'),confidence='high',
 notes="e5b: real Snail 1/1 and Swan 2/25 after the first exchange (Rhino 5->1, then Wolverine floors at 1); engine has the Snail dead and Swan 2/21. e5 (tier-1 targets take double Rhino damage) ends the same in both, but the movie shows Rhino's two hits before Wolverine's. Rhino deals double damage to tier-1 pets, tokens included (Ram took 8).")

rule('R6','3e',
 "Otherwise Wolverine works as the engine has it: its own hurt does not count; hurts of dying friends and the front pet's attack hit count; the count carries over between attacks; the activation waits for the end of the hurt/faint chain (both Hedgehog blasts land first); a Wolverine that died in that chain still acts; two Wolverines on one team and Wolverines on both teams each act; health removal stops at 1.",
 "Same.","None.",
 caps('c3-e0-wolv-self-*','c3-e1-wolv-timing-*','c3-e2-wolv-dead-*','c3-e3-wolv-both-*','c3-e4-wolv-pair-*'),confidence='high',
 notes="The order of two Wolverines of the same chain cannot change any stat (removal floors at 1 and triggers nothing), so it cannot be tested from outcomes. e4's second exchange shows one of two -3 on the label until the next hit (N1).")

rule('R7','3f',
 "The Tiger's repeat runs directly after the original ability, before the next pet's ability for the same trigger, also when the Tiger's own attack is lower than that next pet's (start of battle: Skunk, repeat, then Crab; faint: Hedgehog blast, repeat blast, then Mammoth's buff, so a 4-health enemy is dead before the buff).",
 "Same.","None.",
 caps('c3-f1a-tiger-sob-low-*','c3-f1b-tiger-sob-mid-*','c3-f1c-notiger-control-*','c3-f2-tiger-faint-*','c3-f2r-tiger-faint-rev-*','c3-f2x-tiger-faint-continues-*'),confidence='high',
 notes="The Tiger's own health label shows only the first of the two blasts until its next change (N1); f2x proves the second was applied (18 on the label, 12 after a 4-damage hit).")

rule('R8','3f',
 "A Tiger that dies in the same step as the pet ahead of it does not repeat that pet's faint summon: Cricket and Tiger killed by one blast give one Zombie Cricket; a Deer whose Tiger dies in the same exchange (Chili splash, or Crocodile shots at start of battle) gives one Bus. With the Tiger alive there are two.",
 "Remembers the Tiger when the faint is queued and repeats anyway: two Zombie Crickets, two Buses.",
 "For summon-type faints resolved after removal, require the remembered Tiger to be still alive (not among the pets that died in the same step).",
 caps('c3-f7-tiger-dies-too-*','c3-f7x-tiger-dies-too-continues-*','c3-f7c-tiger-survives-control-*','c3-f13-*','c3-f14-*'),confidence='high',
 notes="f7x: one Zombie Cricket on 8 of 8 seeds; f7: 2 of 2; control f7c with the Tiger alive: two. The rule is specific to summons, which resolve after the dead are removed: a Hedgehog's blast is still repeated when its Tiger dies in the same step (f14) or is killed by the first blast (f13), and the engine agrees there.")

rule('R9','3f',
 "A Tiger's repeat of an ability with a use limit spends one of that ability's uses. Snake (5 per battle) with a Tiger behind fires 5 shots in total: 2, 2, 1, then nothing. Hippo (3 per battle) gains three times in total, the Tiger's gain at the Tiger's level (Hippo L3 with a level-1 Tiger: +9, +3, +9). Ox (1 per turn at level 1) gains once.",
 "Counts only the pet's own triggers and lets the Tiger repeat each: Snake plus Tiger fires 10 shots, Hippo gains six times, Ox twice.",
 "Increment the ability's use counter for the Tiger's repeat as well, and skip the repeat (or the next own trigger) when no use is left.",
 caps('c3-f9-snake-tiger-shots-*','c3-f11-hippo-tiger-limit-*','c3-f11c-hippo-alone-control-*','c3-f10-snake-tiger-dead-front-*'),confidence='high',
 notes="f9: 10 seeds, 5 shots each. Tested for Snake and Hippo; Ox is from one real fight; not tested for Fly, Gorilla or Elephant. The second shot of a pair reaches the health label only at the next change (N1).")

rule('R10','3b',
 "Empty slots are still open while start-of-battle abilities resolve (they close just before the first attack), but 'nearest friend ahead' skips empty slots and pets already killed: Whale swallows and Dodo buffs the nearest living friend ahead across a gap or a fresh corpse.",
 "Closes gaps first; picks the same targets.",
 "None for targeting. Gaps only matter for where summons land (R11).",
 caps('c3-b1-whale-gap-*','c3-b2-dodo-gap-*','c3-b4-whale-deadahead-*','c3-b6-dodo-deadahead-*'),confidence='high')

rule('R11','3b/3h',
 "Empty slots are still open when a pet dies from start-of-battle damage. Its summon goes into its own slot; a second summon for the same slot (Mushroom copy, second Ram) takes the slot and pushes the first one forward if the slot directly ahead is empty, otherwise back (into a gap behind, or pushing the pets behind). So with a gap ahead the Bus ends in front of the returning Deer, without one behind it.",
 "Applies the same slot rule, but has already closed the gaps before start-of-battle abilities run, so with a gap ahead it pushes the Bus back: Fish, Deer, Bus, Pig instead of Fish, Bus, Deer, Pig.",
 "Keep the board's gaps until all start-of-battle abilities and the deaths they cause have resolved; close them just before the first attack.",
 caps('c3-h1-deer-gap-behind-*','c3-h2-deer-no-gap-*','c3-h3-fly-gap-*','c3-h4-sheep-mid-*','c3-h5-sheep-gap-ahead-*','c3-h11-deer-gap-ahead-and-behind-*','custom-snipe-gaps'),confidence='high',
 notes="Only the gap-ahead case with two summons for one slot differs from the engine (session 2's snipe-gaps, repeated here as h11). No currently failing fight was traced to it.")

rule('R12','(real fights)',
 "A pet whose permanent health is used up but which is alive on battle-only health (a Dog on its own +1 health buffs) is alive and still receives buffs: Dog at 9/1 gains +2/+1 for each of two Rams and a Bus (15/4).",
 "Its 'no buff for a pet at 0 health' check looks at permanent health only: the engine logs the three Dog triggers and leaves the Dog at 9/1.",
 "Test total health (permanent plus battle-only) in the cannot-buff-the-dead check.",
 [],confidence='medium',
 notes="Seen in two real fights; the engine side was confirmed by reading its phase log and the check in effects.py (not edited). Not rebuilt as a constructed test.")

rule('R13','(real fights)',
 "If the opponent has no pets, nothing happens: no start-of-battle ability runs (Armadillo gives no health, Whale does not swallow) and the fight ends with the starting board.",
 "Runs start-of-battle abilities first and reports the changed board.",
 "End the battle before the start-of-battle phase when the opponent's board is empty (only an empty opponent was seen; an empty own board was not tested).",
 caps('c3-em1-empty-opponent-*'),confidence='high')

rule('R14','(real fights)',
 "Skunk level 2 on a pet with 50 health leaves 16, not 17. Every other Skunk result seen agrees with the engine (level 1: 30 to 20, 20 to 13, 40 to 26, 9 to 6, 23 to 15; level 2: 12 to 4, 20 to 6, 19 to 6; level 3 on 50: 1).",
 "Removes 33 and leaves 17.",
 "Any formula that gives 16 for level 2 on 50 and keeps the other values: for example remaining = floor(health x (1 - percent)) in binary floating point (50 x 0.34 is just under 17), or removal = ceil(health x 0.66) in single precision (33.0000013). The data do not tell these apart.",
 caps('c3-sk1-skunk2-on-50-*','c3-sk2-skunk3-on-50-*'),confidence='high',
 notes="sk1: 16 on 2 of 2 seeds. Six real fights show 50 to 16; in two the archive's surviving pet carries the 16 directly.")

rule('R16','3d',
 "A Crab that was killed by an earlier start-of-battle ability stays dead: its health gain does not bring it back and it is removed.",
 "Lets the dead Crab act and adds the gain to its negative health, so the Crab is alive again (and can absorb a second Dolphin shot).",
 "Do not let Crab's gain apply to a pet at 0 health or less (the same rule the engine already has for ordinary buffs). Do not skip dead pets' start-of-battle abilities in general (R20).",
 caps('c3-d7-dead-crab-*'),confidence='high',
 notes="d7: Crab 4/1 shot by a Dolphin with 9 attack is gone on 2 of 2 seeds; engine keeps it at 4/7. Eight real fights show the same.")

rule('R20','3d',
 "A pet killed by a faster enemy start-of-battle ability still uses its own start-of-battle ability when its turn comes, at full effect and with its usual target rule. Tested: Leopard (3 damage, half of its 6 attack, to a random enemy), Dolphin (4 to the lowest-health enemy; killed by a Dolphin and, separately, by a Crocodile), Crocodile (8 to the last enemy), Mosquito (1 to a random enemy), Skunk (a third of the highest-health enemy), Whale (swallows the friend ahead, whose faint ability triggers, and releases it at once), Dodo (gives half its attack). In each the dying pet was the only start-of-battle pet on its side. The movie shows the pet at negative health first and its shot landing afterwards.",
 "Same.","None. (Crab is the one exception, R16, and only because its effect lands on itself.)",
 caps('c3-x1-dead-leopard-*','c3-x2-dead-dolphin-*','c3-x3-dead-crocodile-*','c3-x4-dead-mosquito-*','c3-x5-dead-skunk-*','c3-x6-dead-whale-*','c3-x7-dead-dolphin-killed-by-croc-*','c3-d6-dead-dodo-acts-*'),confidence='high',
 notes="3 seeds per species, all the same: fires. Dodo is one capture (d6).")

rule('R17','(real fights)',
 "A Rhino that died in the exchange keeps chaining knock-outs, with its Tiger's repeats even if the Tiger died too, for as long as each knock-out kills: four enemies in a row in the one fight seen.",
 "Its chain stopped after four hits (two links), leaving two enemies alive.",
 "Let the dead Rhino's chain continue until a knock-out hit fails to kill; no cap.",
 [],confidence='low',
 notes="One real fight (ede958be), read from labels only; the same fight also carries R14.")

rule('R15','3c',
 "Whale. Swallowing triggers the swallowed pet's own faint ability at once (Deer leaves a Bus, Cricket a Zombie Cricket) and counts as a friend fainting for Fly and Shark. When the Whale faints the pet comes back in the Whale's slot with its own attack and health, at the Whale's level (a 6/9 Fish returns 6/9, level 2 from a level-2 Whale). A Honey Bee lands in front of the released pet; a Mushroom Whale's 1/1 copy lands in front of it and holds nothing; a Fly's zombie for the Whale takes the slot and pushes the released pet back; Rams of a Sheep that died in the same blast take their own slots behind it.",
 "Same in every tested case.","None.",
 caps('c3-c1-whale-release-*','c3-c2-whale2-release-*','c3-c3-whale-honey-*','c3-c4-whale-mushroom-*','c3-c5-whale-fly-*','c3-c5s-whale-shark-*','c3-c6-whale-cricket-*','c3-c7-whale-sheep-blast-*'),confidence='high',
 notes="Level read from the level badge in c1 (Lvl 1) and c2 (Lvl 2) frames.")

rule('R18','3d',
 "Dodo and Crab read current totals, temporary part included: a Crab next to a Fish with 20 permanent and 20 temporary health gains 25% of 40; a Dodo with 4 permanent and 4 temporary attack gives 50% of 8. They read the value at the moment they act (a second Dodo gives half of its already raised attack; a Crab after a Skunk reads the reduced health). Crab's gain rounds up, Dodo's gift rounds down. Equal attack is broken at random, within a team and across teams.",
 "Same (random tie-break, same rounding).","None beyond R4.",
 caps('c3-d1-crab-round-*','c3-d1t-crab-temp-*','c3-d3t-dodo-temp-*','c3-d4-dodo-chain-*','c3-d4t-dodo-tie-*','c3-d2c-crab-skunk-tie-*'),confidence='high',
 notes="d2c (Crab 5 against an enemy Skunk 5): Crab first on 3 of 8 seeds, Skunk first on 5. d4t (two Dodos with 8 attack): back Dodo first on 5 of 8 seeds, front Dodo first on 3.")

rule('R19','3g',
 "In battle, a food effect that reaches several pets triggers Rabbit once per pet: a level-2 Turtle's Melon to two friends gives each +1 health from the Rabbit. A Seal that receives Melon from a Turtle gives its +1 attack to three friends. (Shop behaviour was not tested; replay injection only reaches battles.)",
 "Same.","None.",
 caps('c3-g1-rabbit-turtle2-*','c3-g2-seal-turtle-*'),confidence='high')

UNEXPLAINED=[]   # filled by findings_build.py from attribution.json
NOTES=[
 "N1 label lag: when a pet is hit or buffed twice in quick succession (Tiger repeat, two Wolverines, Snake plus Tiger, two Mammoth buffs), the client's label shows the first change only until that pet's next change. Seen in f2x (Tiger label 18, then 12 after a 4-damage hit, so the true value was 16), e4, f9, f10, f11 and in real fights (44f50389 Turtle). A terminal frame can therefore be one change stale; the archived ResultingBoard is the truth where the two disagree (this also explains session 2's 6b2e1d99, client 8/9 against archive 8/8).",
 "The anchor game sits at turn 2 (one turn played); the rewatch arrow is already present there.",
 "Shop-seed playback test (coordinator's late addition) was not run: it needs a locally altered playback response; the tool change was refused by the permission system and the user's limit names battle/get as the only response that may be altered. Needs explicit user approval.",
 "Rounding seen along the way: Crab's gain and Skunk's removal round up (25% of 26 -> 7, of 13 -> 4; 33% of 30 -> 10, of 20 -> 7, of 40 -> 14).",
]
