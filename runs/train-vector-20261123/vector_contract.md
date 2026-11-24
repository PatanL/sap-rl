# Vector environment used by the collector

The caller allocates an output dictionary for N persistent games. `reset(indices, seeds, out)` resets exactly the selected games, using the scalar environment’s paired-world seed. `step(actions, end_freeze, out)` steps every live game once. Actions have shape [N,2,3] and END masks [N,2]. Both are signed integers; only active seats submit actions. The two players can finish shopping on different waves.

| Output | Shape | Dtype and meaning |
|---|---|---|
| obs[key] | [N,2,...] | The scalar public observation's exact keys, trailing shapes and dtypes |
| rewards | [N,2] | float32, scalar step reward in learner precision |
| battle_resolved, terminated | [N] | bool |
| battle_result, winner | [N] | int8, seat 0 or 1, draw 2, absent −1 |
| lives, wins | [N,2] | int32 |
| active | [N,2] | bool, seats that require a policy action next wave |

Terminal output includes the actual final public observation. A later step holds that game’s observation and settlement, leaves all seats inactive, and clears its one-shot reward and battle event. Partial reset does not touch other games. The live collector drains both seats’ completed turns using these outputs before resetting. Resetting a game clears its GRU bank and draws its episode-stable opponent. Frozen seats never enter the learner. An immutable actor is published after each complete learner step. Acceptance checks the minimum behaviour version across the entire turn; the maximum lag is one update.

PythonVector spreads the reference scalar engine across forked CPU workers, with a barrier per vector call. Passing the shared output allocated by allocate(..., shared=True) avoids an extra copy. Ordinary caller arrays also work. A native vector can replace PythonVector at construction without touching policy or learner logic. RecordedVector implements the same data plane using all public fields of complete, recorded sampled-policy games. It chooses a recorded game by reset seed and ignores actions. Replay games/s counts traversed terminal records. Its simulated learner uses zero learning rate, and its output cannot be checkpointed.
