# Can the game's battle seed be reproduced? First probe

2026-11-23. Every real battle record carries a `Seed`. If the game's generator and the order it draws in were known, the engine could replay each fight exactly instead of over many seeds.

## What was tried

`collect.py` found 628 real fights (three replay sets) whose final board reveals exactly one random choice: the engine makes a single random target pick, each option leads to a different final board, and the real board matches exactly one option. 585 have a usable option list.

`test_generators.py` then asked whether a standard generator seeded with the recorded seed predicts the real choice: .NET `System.Random` and Unity's xorshift, each with two ways of turning a draw into an index, counting options from the front or the back, after skipping 0 to 12 earlier draws.

## Result: not cracked

- Pooled over all 585 points, every hypothesis sits at chance (about 0.44).
- Only 7 fights have no engine tie-break before the choice. One hypothesis (Unity xorshift, value mod n, counted from the back, no skip) predicts all 7, but with about a hundred hypotheses tried and seven points that is within luck, and the same hypothesis is at chance on the rest.
- Within groups that share the same number of engine tie-breaks, the best hypothesis differs from group to group and is no better than expected from picking the best of a hundred.

## Why this is hard from replays alone

The real game may draw random numbers the engine does not (or not draw for ties the engine shuffles), so the position of the revealing choice in the game's draw sequence is unknown, and it differs fight to fight. The generator itself and how the seed is turned into its state are also unknown.

## The route that can work

Controlled experiments in the real client with replay injection: one fixed, tie-free board with a single random event (a Mosquito against several enemies), replayed with seeds 0, 1, 2, ... That gives the game's first choice as a clean function of the seed, with nothing else varying. A known generator would be recognisable from that table; a second random event can then be added to learn the draw order. This depends on the real-client tool in `../client-oracle-20261123/`.

Data: `points-340.json`, `points-fresh.json`, `points-fresh2.json`.
