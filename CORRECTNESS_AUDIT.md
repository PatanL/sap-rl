# Turtle simulator correctness audit

SAP-Calculator commit `d165eb0a02f8aa0b54d72ed1d5490a44390d07f4` is the
pinned mechanics/content authority. The human Turtle guide is used for policy
strategy and for identifying shop mechanics that the battle calculator does
not model.

## Included in league v2

- two linked, distinct next-tier discover choices
- ten-slot rightmost-unfrozen overflow and atomic all-frozen discard
- stacked experience/stat merging and two-level-two discover suppression
- Worm scaled 2-gold Apple stock
- Seal small-team targeting
- Cow free scaled Milk replacement
- Monkey, Swan, and Squirrel current shop behavior
- discover linkage, food magnitude, and legality observation features

## Queued after league v2 process start

These fixes are tested locally but are not visible to v2's already-running
actors. They form the next simulator/data/training iteration:

- Dragon excludes the purchased tier-one pet from its buff
- Elephant self-after-attack dispatch
- Tiger repeats at Tiger level and re-resolves level-dependent effect kinds
- Parrot copies at Parrot level and restores its native ability next turn
- Whale preserves swallowed stats while applying Whale level
- equal-attack ability priority uses health as the tiebreaker
- Snail retains the previous-round loss flag across start of turn
- team-combine removes the consumed pet before Fish level-up targeting
- Canned Food registers its future modifier exactly once, including empty pet shops
- discover pets inherit Canned Food shop stats
- Skunk uses floored remaining health
- Chili splash preserves a target's unused attack perk
- non-frozen stocked food does not permanently expand later shops
- league PFSP records completed matches instead of individual rounds
- Tiger/Parrot copied abilities preserve the copied route and scale to Tiger level
- Tiger/Whale stores and releases both swallowed pets
- battle randomness uses the seeded environment stream
- food serialization preserves scaled Apple/Milk magnitudes
- replay and live observations use the same absolute-experience feature
- Penguin excludes itself from level-2-or-higher friend targets
- linked multi-reward generation cannot evict choices created in the same operation
- actor failure is fail-fast rather than silently degrading the runtime
- non-frozen excess stock is reconciled at the next shop tier
- direct experience grants stats, clamps at max XP, and level-three combines are illegal
- permanent and temporary stats are internally capped at 50
- Cat ignores Chocolate/perk-only food without consuming a trigger
- Rat's enemy summon activates the opponent's summon observers
- last-battle outcome and ability-use counters are explicit entity features
- behavior cloning now uses complete recurrent shop-turn sequences
- current-v-current games receive 50% of league opponent probability
- simulator revision gates prevent accidental resume/pretraining across mechanics versions
- attack perks affect attack damage only, not stats, trigger priority, or ability scaling
- direct experience stats and level-three legality follow SAP-Calculator runtime behavior
- all permanent/temporary stat storage respects the 50 cap
- Cat's two-use counter ignores Chocolate and perk-only foods
- Honey/Mushroom shop faints reconnect removed pets before status summoning
- recurrent imitation batches complete ordered shop turns rather than one-step samples
- copied Rooster abilities key Chick count/stats on the summoned token rather
  than the native source pet, covering Parrot and Tiger repeats
- summon boundaries reject unresolved generated-unit stat placeholders with
  source/effect diagnostics

Focused current tests: 87 passing after this batch, plus a 20,000-decision
random self-play stress run covering 2,429 battles and 219 complete games. The
final v3 guide stress/dataset run completed 500 games and recorded both
perspectives: 64,385 exact decisions in 10,196 recurrent shop-turn sequences,
17,628 discover states, 2,373 discover purchases, and zero legality-mask
violations.

League v3 fail-fast validation found the copied-Rooster Chick defect at update
27 before its first checkpoint. The run exited instead of silently dropping an
actor. Revision v4 regenerates demonstrations and initialization after the
fix; v3 artifacts are retained for audit and are not resumed.

Revision v6 adds verified P1 match semantics: six starting hearts, one heart
per loss, and a one-heart restore for both players after turn two capped at
six, without clearing the loss flag. Dense self-play reward follows net life
change, so a restored turn-two loss is not automatically penalized. END_TURN
is normally masked until all gold is spent; the 50-action cap remains a safety
escape. The full audited simulator/Gym/guide gate is now 107 passing tests.

Canonical strategy revision `turtle-p1-guide-v2-20260827` is stored in
`sapai-gym/TURTLE_P1_GUIDE.md`. New demonstrations include the guide version,
prose SHA-256, executable-teacher SHA-256, persistent line state, commitments,
sell/flexible slots, required hits, and observed-opponent counters.

Revision v7 replaces overlapping per-slot shop RNG streams with one
authoritative sequential `Shop.rs` stream. Frozen slots consume no draw and
cannot rewind shop state; empty-only rolls draw only eligible slots; discovers
continue from the advanced stream; serialization restores it exactly; and
agent/opponent streams are independently derived from a reproducible match
seed. Seed 123 and empirical categorical/duplicate/triple regressions cover
the original sliding-window failure. The full audited gate is 116 tests.

Revision v8 prevents zero-cost positioning collapse. A shop turn may contain
at most two arbitrary swaps, and a swap whose resulting physical-pet order was
already visited that turn is illegal. The rejected v7 update-200 policy used
four moves on 76/324 sampled turns, including 172 immediate reversals and 164
arrangement revisits, for essentially zero position-score gain. Guide-v2 uses
zero/one/two moves on 10,288/553/39 v8 turns. The full gate is 118 tests.

Revision v9 enforces Player/Team/Shop ownership on every Team insertion,
append, copy, deserialization, summon, and respawn, and reconnects before
start-turn triggers. This fixes a fail-fast update-140 path where a Mushroomed
Squirrel revived by Sleeping Pill had `shop=None` and crashed DiscountFood on
the next turn. V8/v9 guide actions and observations are identical across all
80,946 seeded examples; revision-matched artifacts are still regenerated. The
full gate is 121 tests.

Revision v10 completes battle RNG ownership and fixes late-game faint/perk
semantics found during the v9 workday audit:

- one `Battle.rs` stream now owns priority ties and every existing or newly
  summoned pet's random ability draws; `Battle(seed=...)` reproduces the whole
  battle instead of only priority ties
- Honey/Mushroom `PostRemovalFaint` respawns resolve before Shark/Fly
  `PostRemovalFriendFaints` observers, so a full team preserves the status
  summon and Fly correctly declines without consuming a use
- Shark and Fly observe friendly faints only; the legacy EachFriend route had
  allowed an opposing Shark to scale from enemy deaths
- zero post-defense damage no longer emits Hurt or Wolverine counter events
- Peanut uses normal post-defense damage and becomes lethal only when damage
  remains positive; low-attack Peanut is therefore blocked by Melon/Coconut
  while Garlic still admits positive damage

The audited gate is 126 passing tests plus a 50,013-decision fail-fast stress
run covering 1,316 battles and 88 matches. The regenerated guide-v2 corpus has
80,631 exact decisions in 10,870 recurrent sequences, 500 replay IDs, zero
illegal actions, and 100% zero-gold compliance. V9 artifacts through update
250 are retained as diagnostic baselines and are not promoted or resumed.

Revision v11 narrows the dead Shark/Fly observer guard to native EachFriend
abilities. The old name-only guard also intercepted their temporary
Honey/Mushroom Self override, causing a Mushroomed Shark or Fly to disappear
instead of returning as 1/1. The gate is now 127 passing tests. Re-recording
all 500 seeded guide matches produced 80,631 rows identical to v10 after
removing only `validation.simulatorRevision`; nevertheless, demonstrations and
imitation are saved as fresh v11 artifacts to keep provenance explicit. The
uncheckpointed v10 warm-up was stopped and is not promoted.

Revision v12 exposes the public opponent score to the entity policy. The
global token already encoded the acting player's lives/wins but left two
numeric fields empty, preventing the transformer from applying guide rules
such as increasing tempo when the opponent is low. Those fields now encode
opponent lives/6 and wins/10 with perspective-correct swapping. The gate is
128 passing tests. Re-recording preserved all 80,631 actions, outcomes, masks,
returns, and teacher states while adding 42 distinct opponent-score
combinations to the observations. V11's two-update warm-up was stopped and is
not promoted.

Canonical strategy revision `turtle-p1-guide-v3-20260827` completes the
executable opponent-life rule already present in the prose. The persistent
plan records opponent lives/wins and applies soft pressure at two hearts,
stronger at one: immediate stats/tech and obsolete low-tier conversion rise,
while slow Canned/Cat/Dragon/Monkey/Penguin/Giraffe investments fall without
being prohibited. The guide hash is
`302e1ca9187bb032ccab984a9eabcdb82898ab80e930f6f2be7c370713ffaad1` and
the executable hash is
`cd205eae886234f060eb536749fc204a0abde16ffae6cdbed1a0f35ee4b943d4`.
The v12/guide-v3 corpus contains 82,646 examples in 11,046 sequences, zero
illegal actions, and 100% zero-gold compliance. The guide gate is included in
the 130-test full suite.

The first guide-v3 anchor exhibited behavior-cloning covariate shift despite
66.0% held-out exact accuracy: 40% guide win rate and only 2.45 sells/match.
A DAgger pass labeled 41,175 anchor-visited decisions (19.1% SELL) with guide
v3 and zero illegal actions. Combined pretraining reached 68.5% clean
validation accuracy, 49% guide win rate, 2.73 sells/match, 96% pivot coverage,
and beat the first RL update-50 policy 63-37. This changes learning data and
training, not simulator mechanics; the simulator remains v12.

At update 350, checkpoint-time resident rotation exposed a request-routing
race: `DynamicInferenceBatcher` checked membership before acquiring
`model_lock`, allowing the pool to remove that ID before lookup. Routing and
lookup now occur under the same lock, so stale in-flight IDs atomically fall
back to main. The full focused gate is 131 passing tests, including a direct
stale-resident request regression. The saved update-350 model predates only
the post-save thread exception and is valid.

Revision v13 corrects the current shop-size schedule at tier 2. Turns three
and four now roll three pets plus one food rather than the stale legacy/pinned
calculator value of two foods. Turns one/two remain 3+1 and turns five/six
remain 4+2. This changes shop composition, roll value, authoritative RNG
consumption, action distributions, and learned policy data, so v12 checkpoints
and corpora are retained only as diagnostic/transfer artifacts.

The v13 regressions cover the complete early slot schedule, a hard-coded
tier-2 seed-123 shop sequence, frozen food consuming no draw, turn-2-to-3
physical food-slot/object preservation with updated turn metadata, exact future
RNG state, and Gym observation/action masks exposing one food source while
retaining ten transformer shop tokens. The full simulator/Gym/RL gate is 250
passing tests.

The regenerated guide-v3 corpus contains 80,488 exact decisions in 10,668
recurrent sequences across the same 500 replay IDs, with zero illegal actions
and 100% zero-gold compliance. Relative to v12 it contains 1,116 fewer food
buys and 342 fewer freezes. Fresh v13 imitation pretraining selected epoch 8
at 66.0% held-out exact accuracy with zero rejected examples. The first clean
v13 RL lineage loads no v12 weights, optimizer, league state, or residents.

Revision v18 is documented in `sapai-gym/CORRECTNESS_AUDIT_V18.md`. It adds the
confirmed Badger nearest-living gap fix, exact legality/history features for the
entity policy, RL battle-outcome auxiliary training, conservative replay
identity improvements, stronger game-viewer audits, and a 43% CPU environment
throughput improvement. The focused v18 gate is 321 passing tests. A 40-board,
20,000-total-battle differential sweep against pinned SAP-Calculator covered
all 60 Turtle pets with zero statistically flagged scenarios. V18 transfers
weights from v17 update 1,300 but intentionally starts fresh optimizer,
transition, update, and league state.
