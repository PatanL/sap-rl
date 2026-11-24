# What Max Wu plays, compared with the ranked field

2026-11-23. First descriptive pass. Source: SAP Library ranked Turtle mirrors, game version 48. `analyse.py` reads each seat's end-of-turn boards and food purchases; results in `maxwu.json` and `monkey.json`.

**Max Wu (`maxwu27`): 158 games, 69.6% won.** Field for comparison: 1,933 other seats from the same downloads, 48.6% won. Peak rank seen 1882.

These are counts of what appears in his games. A line that co-occurs with winning is not shown to cause it; he wins 70% whatever he plays.

## The lines the user listed

| Line | Max Wu, share of games | Field | Median turn it first appears |
|---|---:|---:|---:|
| Any Sleeping Pill | 71.5% | 67.0% | 5 |
| Pill on Flamingo | 27.2% | 25.1% | 5 |
| Pill on Turtle | 25.3% | 23.7% | 9 |
| Snail by turn 3 | 20.9% | 14.7% | 3 |
| Horse and Cricket by turn 2 | 8.2% | 9.5% | 1 |
| Sheep by turn 4 | 5.7% | 5.5% | 3 |
| Turkey with two or more summoners | 7.6% | 6.5% | 9 |
| Elephant directly ahead of Camel | 9.5% | 10.2% | 5 |
| Elephant directly ahead of Blowfish | 5.1% | 11.0% | 7 |
| Elephant, Blowfish and Wolverine together | 1.9% | 4.9% | 10 |
| Mushroom Mammoth with Tiger behind | 0.6% | 2.9% | 11 |

- **Sleeping Pill is ordinary play at this level**, in about two thirds of everyone's games and 1.2 per game for Max Wu. The trained bot gave Pill under 1% probability in the September review. This is the largest gap between the bot and ranked players found so far.
- **Snail early** is the one listed line he uses clearly more than the field.
- **Elephant with Blowfish and Mushroom Mammoth with Tiger are field lines more than his.** He plays them at half the field's rate or less.

## What he does differently

Pets on his boards, share of boards, his versus field:

| Phase | He uses more | He uses less |
|---|---|---|
| Turns 3 to 5 | Kangaroo 21 vs 13, Spider 23 vs 17, Horse 18 vs 12, Rabbit 25 vs 19, Peacock 22 vs 17, Crab 14 vs 10 | Fish 8 vs 15, Pigeon 8 vs 13, Beaver 11 vs 15, Elephant 13 vs 16 |
| Turns 6 to 9 | Spider 16 vs 8, Kangaroo 15 vs 9, Crab 12 vs 7, Camel 26 vs 21, Peacock 19 vs 14 | Giraffe 13 vs 19, Elephant 13 vs 19, Blowfish 11 vs 15 |
| Turn 10 on | Gorilla 19 vs 7, Rhino 15 vs 8, Rabbit 15 vs 7, Wolverine 25 vs 16, Deer 25 vs 18 | Elephant 11 vs 17, Leopard 14 vs 21, Tiger 18 vs 24, Blowfish 14 vs 19 |

Perk placement that stands out: Melon on Gorilla (7.0% of games vs 1.8%), Mushroom on Deer (11.4% vs 4.4%), Tiger behind Wolverine (4.4% vs 2.0%). He buys less Chocolate (1.18 per game vs 1.54), Canned Food, Sushi, Pizza and Steak, and slightly more Apple, Bread and Pill. Rolls per turn are the same as the field (1.59 vs 1.62). His games end about a turn earlier (11.3 vs 12.1).

Read together: fast early boards (Kangaroo, Spider, Horse, Crab), a Camel, Rabbit and Peacock middle, and a late game built on Deer, Wolverine, Gorilla with Melon and Rhino rather than on Tiger or Elephant chains.

## monkey

The search for "monkey" matched several accounts. Only two had downloadable games (`monkey12817271821`, rank up to 1841, and `MonkeysAlternativeAccount`): 7 games, 2 won. That is too few to describe a style. If a different account was meant, the name is needed.

## Limits

- 138 of his 226 listed games downloaded; the rest returned errors from the API.
- "Field" includes strong and weak ranked players together (ranks about 1280 to 1880), and includes his opponents.
- Pill targets are identified from boards, so a pet bought and pilled in the same turn shows as "a pet bought that turn" (33% of his games).
- Nothing here measures the bot. The matching counts for the bot's own games have not been produced yet.

# The bot measured against ranked players

2026-11-23. Two measurements of checkpoint 25,772 against the same SAP Library games. Tools: `bot_games.py` (the checkpoint plays both seats; boards, foods and the exact teams entering each battle are written down), `compare_bot.py` (the line counts above, for the bot), `ghost.py` (board against board on the Rust engine).

## Board strength at equal turn

Each cell: boards one side took into battle on that turn, paired at random with boards the other side took into battle on the same turn, 4,000 pairs, 8 seeds each; score is wins plus half the draws. "Field" is 26,000 boards from 1,090 ranked matches. Largest standard error in the table is about 0.8 points.

| Turn | v22 sampled | v25 sampled | 25,772 sampled | 25,772 greedy | 25,772 greedy v Max Wu | Max Wu v field | Field v field |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52.6% | 52.2% | 52.9% | 55.4% | 55.3% | 48.6% | 50.1% |
| 2 | 57.5% | 57.5% | 54.5% | 53.7% | 58.7% | 43.6% | 49.4% |
| 3 | 35.9% | 36.3% | 38.8% | 40.0% | 39.7% | 50.2% | 49.1% |
| 4 | 36.8% | 34.5% | 42.0% | 44.1% | 41.0% | 53.8% | 49.4% |
| 5 | 29.5% | 30.4% | 35.4% | 38.8% | 29.5% | 57.7% | 49.8% |
| 6 | 30.1% | 29.2% | 39.0% | 42.7% | 34.6% | 57.8% | 50.7% |
| 7 | 20.5% | 19.1% | 26.7% | 30.6% | 22.6% | 57.4% | 49.5% |
| 8 | 16.6% | 16.7% | 23.0% | 30.9% | 24.7% | 56.3% | 50.4% |
| 9 | 14.7% | 13.7% | 19.6% | 24.9% | 18.1% | 57.1% | 50.3% |
| 10 | 12.1% | 12.2% | 17.8% | 22.5% | 17.8% | 55.0% | 49.6% |
| 11 | 11.1% | 11.2% | 15.8% | 20.8% | 16.0% | 56.0% | 50.5% |
| 12 | 9.2% | 9.7% | 14.5% | 17.2% | 13.2% | 53.1% | 49.2% |
| 13 | 8.8% | 9.8% | 12.6% | 15.6% | 14.2% | 50.6% | 49.5% |
| 14 | 8.8% | 8.6% | 12.0% | 15.7% | 21.1% | 42.0% | 49.6% |

- **The controls behave.** Field against field is 49 to 51% on every turn. Max Wu's boards beat the field's by 3 to 8 points on turns 4 to 12.
- **The bot's boards are far weaker than ranked players' boards from turn 3 on.** Greedy play scores 40% at turn 3, 31% at turn 7, 23% at turn 10 and 16% at turn 14. Total attack and health match the field's through turn 10 (43.7 and 52.3 against 43.3 and 55.2 at turn 10), so the difference is which pets and perks, not how many stats.
- **The measure follows known strength.** 25,772 beats v22 head to head 90.6% and v25 81.3%; here it is ahead of both on every turn from 3 on (turn 10: 17.8% against 12.1% and 12.2%, same sampling). Self-play gains do carry over, slowly.
- **Greedy is 3 to 5 points better than sampled.** Training games are sampled.
- **Turn 2 is the one turn where Max Wu's boards are weaker than the field's (43.6%).** That fits the deliberate turn-2 loss the user described: a loss at full hearts on turn 2 is restored, so it is a free way to bank a Snail trigger or save gold. The bot does the opposite and has its best relative boards on turns 1 and 2.

## Full matches against the ranked pool

`ghost_match.py`: the bot shops normally with lives tracked by the environment; the other seat buys nothing and, just before each battle, its team is replaced by a board a ranked player took into battle on that turn. The bot sees those boards as its opponent's history.

| Seat 0 | Matches | Won | Score |
|---|---:|---:|---:|
| 25,772 greedy | 512 | 24 | **4.7%** (standard error 0.9) |
| Control: also drawn from the pool | 256 | 122 | 47.7% (standard error 3.1) |

Battle score by turn for 25,772 in these matches: 1: 54% 2: 59% 3: 42% 4: 45% 5: 43% 6: 45% 7: 31% 8: 30% 9: 27% 10: 23% 11: 20% 12: 19% 13: 12% 14: 17%. These are within a few points of the board-against-board table, so the bot does not build differently when it sees human boards.

Limits: each opposing board comes from a different player, so the opponent is the field's typical board for that turn, not one person adapting; and boards from later turns come from matches that lasted that long. The pool is ranked Turtle mirrors at ranks about 1280 to 1880.

## Lines in the bot's own games

3,000 seats of sampled self-play. Share of games, except the pet table.

| Line | 25,772 | Max Wu | Field |
|---|---:|---:|---:|
| Any Sleeping Pill | 8.1% | 71.5% | 67.0% |
| Chocolate | 56.8% | 65.2% | 72.9% |
| Snail by turn 3 | 36.6% | 20.9% | 14.7% |
| Skunk on board | 2.0% | 35.4% | 36.3% |
| Scorpion on board | 0.3% | 13.9% | 15.9% |
| Dolphin late | 25.1% | 4.4% | 13.5% |
| Rhino on board | 9.8% | 13.9% | 11.0% |
| Wolverine on board | 13.4% | 18.4% | 16.9% |
| Fly late | 1.6% | 13.3% | 14.5% |
| Dragon on board | 0.3% | 8.2% | 11.3% |
| Elephant ahead of Blowfish | 0.0% | 5.1% | 11.0% |
| Elephant ahead of Camel | 0.1% | 9.5% | 10.2% |
| Horse and Cricket by turn 2 | 0.6% | 8.2% | 9.5% |
| Turkey with two or more summoners | 0.0% | 7.6% | 6.5% |
| Mammoth with Tiger behind | 0.0% | 0.6% | 4.7% |

Pets on boards at turn 10 and later, share of boards:

| Pet | 25,772 | Max Wu | Field |
|---|---:|---:|---:|
| Skunk | 1% | 25% | 28% |
| Tiger | 0% | 18% | 24% |
| Cat | 0% | 17% | 21% |
| Elephant | 0% | 11% | 17% |
| Leopard | 4% | 14% | 21% |
| Shark | 0% | 15% | 15% |
| Snake | 0% | 14% | 15% |
| Parrot | 0% | 10% | 11% |
| Dragon | 0% | 8% | 11% |
| Scorpion | 0% | 10% | 9% |
| Wolverine | 14% | 25% | 16% |
| Rhino | 10% | 15% | 8% |
| Snail | 41% | 1% | 2% |
| Ant | 33% | 2% | 3% |
| Penguin | 35% | 1% | 3% |
| Crocodile | 34% | 2% | 5% |
| Flamingo | 29% | 0% | 0% |
| Otter | 26% | 4% | 3% |
| Fish | 25% | 2% | 2% |

- The bot almost never uses Sleeping Pill (8% of games against two thirds), and does not play Skunk, Scorpion, Tiger, Cat, Shark, Snake, Parrot or Dragon at all. It never meets them either, because its only opponents are itself.
- It keeps tier-1 and tier-2 pets to the end (Snail on 41% of late boards, Ant 33%, Otter 26%, Fish 25%) and scales them, with Penguin and Crocodile. It rolls 2.5 times a turn against 1.6.
- The "Pill on X" rows are not compared: the bot's target is read when the pill is used, the humans' from the next board.

## Yardstick split and baseline

So that this measure can keep being used after ranked boards are (if the user agrees) put into the training opponent mix, the pool is split by match: matches with sha256(match id) % 5 < 2 are the yardstick side and are never trained on; the rest are reserved for training. `ghost.py` and `ghost_match.py` score on the yardstick side by default. The tables above were made on the whole pool before the split existed.

Index = mean score at turns 3, 5, 7, 10 and 12 on the yardstick side (about 800 field boards per turn), final engine `turtle-v31-replay-852b274e7eb2-20261123`:

| Checkpoint | Index |
|---|---:|
| v22, sampled | 21.6 |
| v25, sampled | 21.9 |
| 25,772, sampled | 26.6 |
| 25,772, greedy | 31.1 |
| Ranked field | 50 |
| Max Wu | about 55 |

Files: `yardstick-*.json`, `yardstick-baseline.json`.
