# Rust collector and battle-recording interface

The sampled single-battle trainer consumes the existing 15 public observations, active bits, immediate rewards, battle event/result and terminal flag. Opponent identity, recurrent state, reset generation, joint log probability, sampled policy version and queue membership belong to the collector. No paired battle seed, turn number, private state, game clone or recording hook is needed for that trainer.

For ranked-board scoring, provide an optional read-only event batch over the games whose most recent step resolved a battle. Capture both teams immediately before calling the battle simulator, after both players' END abilities and freeze plans, and before combat or next-turn triggers. Include terminal battles. Either dense arrays or compact arrays with an `int32[M]` game-index map are sufficient. Return a generation number or guarantee that the event remains valid until the next step/reset; the recorder copies before reset.

| Field | Type | Meaning |
|---|---|---|
| battle_resolved | bool[N] | Existing validity mask |
| battle_turn | int32[N] | One-based turn of the battle just played, shared by both seats |
| battle_seed | uint32[N] | Actual paired tape seed, for deterministic audit or MC, unnecessary to the ghost sampler |
| battle_teams | int32[N,2,5,7] | Seat order and fixed physical slots, front first. In each occupied slot: pinned kind ID, effective attack, effective health, total XP, perk ID, consumed trigger count, copied species. Empty kind −1. Keep the battle ABI copy sentinels −2 (no field) and −1 (explicitly no copy). |
| name map | pinned table or lookup | Map battle ABI kind/status IDs to SAP identifiers. Observation vocabulary indices have a separate namespace. |

The existing board JSON uses level-equivalent XP (0, 2, 5), includes `triggersConsumed`, and writes `parrotCopies` only for Parrots, including an explicit null. Total XP suffices to reproduce that JSON. Exact internal battle re-simulation of specials can require additional payloads beyond those recorded by the legacy JSON. For MC, prefer a method on the engine that retains the actual native teams and returns W/D/L counts, avoiding any loss in exporting and re-importing them. For scalar reward smoothing, also return the three outcome-specific pre-settlement life-swing components or compute the replacement within the engine. Use a separate private seed domain.

The public opponent-history fields contain species, position, perk, effective numeric stats, XP/level and temporary-stat flags. They omit consumed triggers and the actual copied ability. The canonical JSON dump describes the current seats after settlement; it is not a saved copy of both pre-fight objects. Turn numbers for standard recording can be tracked from resets and battle events. Human comparison metadata (names, fixture split, ghost RNG) lives in the scorer. Never include the proposed side channel in learned policy observations.
