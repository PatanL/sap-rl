# Turtle League Evaluation and Selection Protocol

## Training horizons

- checkpoint every 25 updates for recoverability
- 25-update checks are safety screens only
- interim learning evaluation after about 100 updates / 0.85M transitions
- mature algorithm comparison after at least 250 updates / 2.1M transitions,
  unless a correctness failure, numerical collapse, or catastrophic fixed-gate
  regression ends the branch earlier

The selected checkpoint is immutable rollback protection. A branch may continue
training through interim regressions while it remains healthy. Rollback is not
used to repeatedly reroll short branches until random variation looks good.

## Paired-seat promotion games

Every promotion seed is played twice:

1. candidate as player 0, opponent as player 1;
2. opponent as player 0, candidate as player 1.

Both games use the same environment seed. Results and 95% confidence intervals
are computed over independent seed-pair mean scores. This removes fixed-seat
bias and makes near-50% results easier to interpret.

Recommended sample sizes:

- critical guide/current/tempo comparisons: at least 250 seed pairs
- broader historical, population, and imitation checks: at least 100 pairs
- expand ambiguous intervals rather than interpreting a narrow raw win margin

`scripts/evaluate_paired_policy.py` is authoritative for promotion scores.
`scripts/evaluate_entity_policy.py` remains authoritative for action counts,
build composition, economy, pivot, positioning, and other behavioral diagnostics.

## Promotion policy

Training is permissive; deployment promotion is strict.

A candidate must demonstrate broad improvement across:

- canonical guide
- selected current main
- tempo and late/stat specialists
- historical/population opponents
- imitation anchor
- economy, zero-gold, pivot, movement, and build-diversity diagnostics

No single 100-game, single-seat result is sufficient proof of improvement.
