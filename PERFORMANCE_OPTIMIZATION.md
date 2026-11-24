# Turtle League v12 Throughput Optimization

Date: 2026-08-27

This pass intentionally targeted only two high-return, low-risk paths:

1. omit diagnostic battle trace payloads during RL self-play while retaining
   every semantic event needed by battle resolution;
2. tune the number of games per actor and the inference-request coalescing
   delay under the real resident league.

The selected update-375 checkpoint was read-only throughout profiling.

## Battle fast path

`Battle(..., fast_training=True)` retains phase event counts, attacker indices,
knockout lists, RNG use, and all pet/team mutations. It omits diagnostic pet and
target string snapshots and does not populate `battle_history`. Direct `Battle`
use remains traced by default; `ParallelSelfPlayEnv` enables the fast path.

On 500 seeded summon/hurt-heavy battles:

- traced: 118.1 battles/s
- semantic-only: 127.4 battles/s
- improvement: 7.9%
- all 500 outcomes were identical

Seeded equivalence tests compare outcomes, final team states, and final battle
RNG state across ordinary abilities, summons, Tiger, Rhino, perks, Peanut, and
Chili.

## Inference matrix

Each cell used update 375, 16 actor processes, 8,192 requested transitions per
learner update, the expert anchor, and four frozen resident policies. The first
update was treated as actor warm-up. The table reports the next two updates, so
the small differences should be interpreted as screening measurements.

| Games/actor | Delay cap | Transitions/s | Mean inference batch |
|---:|---:|---:|---:|
| 2 | 2 ms | 826 | 7.0 |
| 2 | 5 ms | 772 | 7.7 |
| 2 | 8 ms | 807 | 9.3 |
| 3 | 2 ms | 859 | 8.6 |
| 3 | 5 ms | 948 | 10.1 |
| 3 | 8 ms | 957 | 11.0 |
| 4 | 2 ms | 915 | 10.2 |
| 4 | 5 ms | 900 | 11.3 |
| 4 | 8 ms | 1,008 | 13.2 |

The two finalists then ran for nine steady updates:

| Configuration | Collection | Learning | Total/update | Transitions/s | Batch |
|---|---:|---:|---:|---:|---:|
| 3 games, 5 ms | 3.694 s | 5.550 s | 9.243 s | 916.2 | 10.48 |
| 4 games, 8 ms | 3.699 s | 5.553 s | 9.252 s | 918.4 | 11.49 |

The long runs are tied within noise. The selected production setting is 3 games
per actor and a 5 ms delay cap because it uses fewer shared slots and has lower
request latency for effectively identical throughput. Compared with the prior
770-830 transitions/s observation, this is approximately a 10-19% improvement.

The configured delay is a cap, not time always spent waiting. The batcher exits
early when every actor buffer is ready. At the selected setting, actual
coalescing averaged 1.22 ms.

## Utilization

Continuous sampling during the 4-game finalist observed 30.8% mean GPU
utilization and 96% peak utilization. The GPU remains bursty because collection
and learning are sequential phases. Sustained saturation would require a larger
pipeline change that overlaps actor collection with learner updates; that is
outside this deliberately narrow pass.

## Verification

- simulator/Gym/guide regression gate: 242 passed
- fast-path equivalence cases: 110 parameterized seeded cases
- no simulator revision bump: the fast path is outcome/RNG equivalent
- no benchmark checkpoint was promoted

## Collection/learning overlap audit

The distributed runtime already overlaps actors with the learner. Actors run
continuously, write turn segments into a bounded shared-memory ring, and block
when no free slot remains. V-trace behavior log probabilities were already
stored. The proposed pipeline rewrite was therefore unnecessary.

Added telemetry records exact ring occupancy, segment counts, inference-policy
version on every main-policy transition, mean/max learner lag, and the fraction
over one update. A typical 16-actor update consumes 1,088-1,152 turn segments.
During a 5.5-second learner step, 16 actors produce only about 450-512 segments.
Increasing the ring from 512 to 1,024 by itself therefore does not remove the
remaining collection time.

Actor-count screening with 3 games/actor and a 5 ms coalescing cap:

| Actors | Queue | Collection | Total/update | Transitions/s | Max lag | Lag > 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1,024 | ~4.3 s | ~9.5-10.1 s | ~830-916 | 1 | 0% |
| 24 | 1,024 | 2.75-3.16 s | 8.68 s | 978 | 1 | 0% |
| 32 | 1,536 | 1.56-1.73 s | 7.55 s | 1,110 | 1 | 0% |
| 36 | 1,536 | 0.96-1.45 s | 7.30 s | 1,164 | 2 | up to 1.2% |
| 40 | 1,536 | 0.66-1.34 s | 7.18 s | 1,171 | 2 | 3.6-5.2% |

The selected setting is 32 actors with a 1,024-slot ring. The observed high
water at 32 actors was 880 slots, so 1,024 retains headroom without the extra
shared-memory allocation of 1,536. It improves throughput by about 21% while
preserving the strict one-update policy-lag bound. The marginal 36/40-actor
gain is rejected because it violates that bound.
