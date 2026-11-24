# Shop seed replication from replays: not reproduced (2026-11-23)

Written by the session lead from the helper's hand-back; scripts, datasets and logs are in this directory.

**Verdict.** No tested generator, seeded with any logged value, predicts what a roll puts in the shop. Every hypothesis
is at chance on the rolls it was fitted to and on held-out rolls. The test itself works: with species replaced by ones
generated from the real seeds by a known rule (2% noise), the same sweep scores that rule 950 against a chance maximum
near 7.

## What the replays contain
Action types: 4 start turn (builds the shop, keeps frozen items), 5 roll, 6 buy pet, 7 combine, 8 buy food, 9 sell,
11 end turn, 12 name team, 0 battle record with the full end-of-turn state, 1/2/3 markers. Types 4 to 9 and 11 carry a
`Seed` and a `Hash` (254,530 actions). A generation creates pets first, then foods, with consecutive ids, skipping
frozen slots; the displayed list is sorted by tier, so slots must be read in id order. The first shop exists before
any action, so no logged seed precedes it.

## Dataset (`rolls.jsonl`, `genesis.jsonl`)
28,631 turns replayed, 14,985 verified (id counter lands exactly on the snapshot, every buy and every leftover item
matches). 55,140 shop generations with their own seed, the previous action's seed and both hashes; 14,991 fully known
(almost all the last roll of a turn), slot 1 known in 18,749. Split by match: 900 development, 297 held-out.

## Seeds and hash
- Seeds: 1,410 to 2,147,449,700, never negative (the range of .NET `Random.Next()`), uniform, one per roll, never
  shared between the two players. Consecutive seeds are unrelated: correlation +0.003; no linear congruential relation
  for 18 classic constant sets; Seed[i+1] is never among the first 64 outputs of `System.Random(Seed[i])` (0 of 252,144,
  redone with correct 32-bit overflow: the first probe in `../chain/` had ported it without) nor of 32 other
  generators. Battle seeds are a different kind of number (full 32-bit, shared by both players).
- Hash: it is the board's hash (each request sends the previous response's hash back). Not a function of the seed
  alone, nor of the visible shop alone.

## Generator tests
34 generators (.NET System.Random, old Mono, Unity xorshift128, MT19937, PCG32, splitmix64, xorshift variants,
xoshiro256**, xoroshiro128+, java.util.Random, glibc, MSVC, LCGs, MINSTD, mulberry32 and others) x 6 seed sources
(own seed, previous seed, own hash, previous hash, own xor previous, seed + turn) x the first 40 draws x 10 reductions
x 4 slots: 460,000 to 644,000 hypotheses per stratum. Best development score 5.3 to 11.1 against 5.4 to 11.2 on
shuffled labels; held-out scores of the best between -1.6 and +2.4. Redraw-until-eligible and bit-level tests are at
the shuffled level too.

Not tried: a generator with a secret constant, a seed mixed with something not logged (board id, player id), draws
beyond the 40th, a tier-then-pet rule needing two draws above tier 1.

## What would settle it
1. Chosen seeds in the real client: serve a playback whose roll responses carry seeds 0, 1, 2, ... on a fixed board
   and read the displayed shop (also shows whether playback recomputes shops or shows stored snapshots; untested).
2. Read the client build's metadata for the shop-roll method and the random class built from `Seed`.

Limits: generator test vectors were recalled from memory, no reference runtime offline; Unity's xorshift128 has no
checked vector.
