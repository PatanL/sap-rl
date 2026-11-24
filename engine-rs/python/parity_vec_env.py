"""Acceptance item 5: the batch wrapper against the Python environment.

Drives N Python `ParallelSelfPlayEnv` matches and one `SapVecEnv` of N games with the same
decisions (uniform random legal play with END freeze masks, as tools/gen_env_golden.py) and
compares after every reset and every step, exactly:
  every observation array (dtype, shape, bytes), rewards, battle_resolved, battle_result,
  terminated, winner, lives, wins, active flags, and the flat legal mask of each active seat.
Games finish at different times: a finished game is stepped along with the others (with "no
decision" rows) and must keep its final rows until it is reset; a third of the finished games
are reset to a new seed while the others keep running. Then the error contract is checked:
illegal input changes nothing and reports status -1 for that game only.

Usage (training image; see tools/run_ref.sh):
  parity_vec_env.py <libsap_engine.so> <games> <first seed> <threads> [max matches]
"""
import random, sys
import numpy as np

from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import AutoregressiveAction, ActionType, FlatActionCodec as C, MAX_SHOP_SLOTS
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.end_freeze import expand_end_freeze, SHOP_ACTION_CAP
import sap_env_rs

LIB, N, FIRST, THREADS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
MAX_MATCHES = int(sys.argv[5]) if len(sys.argv) > 5 else N * 2
rng = random.Random(FIRST)
compared = {"observation arrays": 0, "steps": 0, "resets": 0, "legal masks": 0, "decisions": 0, "battles": 0}


def random_mask(env, side):
    core = env.core
    player = core.agent if side == 0 else core.opponent
    if player.gold != 0:
        return 0
    count = core.shop_action_count_agent if side == 0 else core.shop_action_count_opponent
    mask = 0
    for index in range(min(MAX_SHOP_SLOTS, len(player.shop))):
        if not player.shop[index].empty and rng.random() < 0.2:
            mask |= 1 << index
    if bin(mask).count("1") > max(0, SHOP_ACTION_CAP - count):
        return 0
    try:
        expand_end_freeze(core, side, mask)
    except ValueError:
        return 0
    return mask


def check_game(vec, i, env, observations, where):
    for side in range(2):
        for key, value in observations[side].items():
            mine = vec.obs[key][i, side]
            if mine.dtype != value.dtype or mine.shape != value.shape or mine.tobytes() != np.ascontiguousarray(value).tobytes():
                raise AssertionError(f"{where}: game {i} seat {side} observation {key} differs")
            compared["observation arrays"] += 1
        if set(vec.obs) != set(observations[side]):
            raise AssertionError(f"{where}: observation keys differ")
    core = env.core
    expect = {
        "lives": [core.agent.lives, core.opponent.lives], "wins": [core.agent.wins, core.opponent.wins],
        "active": [int(a) for a in env.active_players()],
    }
    for key, value in expect.items():
        if getattr(vec, key)[i].tolist() != value and not (key == "active" and vec.terminated[i]):
            raise AssertionError(f"{where}: game {i} {key}: reference {value}, wrapper {getattr(vec, key)[i].tolist()}")
    for side, active in enumerate(env.active_players()):
        if active and not vec.terminated[i]:
            legal = np.zeros(197, dtype=np.int8)
            legal[list(core._avail_actions(side))] = 1
            if not np.array_equal(legal, vec.legal(i, side)):
                raise AssertionError(f"{where}: game {i} seat {side} legal mask differs")
            compared["legal masks"] += 1


vec = sap_env_rs.SapVecEnv(N, threads=THREADS, library=LIB)
envs = [ParallelSelfPlayEnv() for _ in range(N)]
seeds = np.array([FIRST + i for i in range(N)], dtype=np.int64)
next_seed = FIRST + N
assert vec.status.tolist() == [sap_env_rs.STATUS_NOT_RESET] * N
assert vec.reset(seeds) == 0
done = [False] * N
final = [None] * N
for i, env in enumerate(envs):
    check_game(vec, i, env, env.reset(seed=int(seeds[i]), paired_tape=True), "reset")
    compared["resets"] += 1
assert (vec.status == 0).all() and (vec.terminated == 0).all() and (vec.winner == -1).all()
matches_started, matches_finished = N, 0

while not all(done):
    actions = np.full((N, 2, 3), sap_env_rs.NO_DECISION, dtype=np.int64)
    masks = np.zeros((N, 2), dtype=np.int64)
    decisions = [None] * N
    for i, env in enumerate(envs):
        if done[i]:
            continue
        row = []
        for side, active in enumerate(env.active_players()):
            if not active:
                row.append(None)
                continue
            action = C.decode(rng.choice(sorted(env.core._avail_actions(side))))
            mask = None
            if action.action_type == int(ActionType.END_TURN):
                mask = random_mask(env, side)
            row.append(JointAction(action, mask))
            actions[i, side] = action.as_tuple()
            masks[i, side] = mask or 0
            compared["decisions"] += 1
        decisions[i] = row
    before = {key: value.copy() for key, value in vec.obs.items()}
    flags_before = (vec.terminated.copy(), vec.winner.copy(), vec.lives.copy(), vec.wins.copy())
    bad = vec.step(actions, masks)
    assert bad == 0, [(i, vec.error(i)) for i in range(N) if vec.status[i] < 0]
    reset_mask = np.zeros(N, dtype=np.uint8)
    for i, env in enumerate(envs):
        if done[i]:
            # A finished game is skipped: status 1, zero rewards, everything else untouched.
            assert vec.status[i] == sap_env_rs.STATUS_FINISHED and vec.rewards[i].tolist() == [0.0, 0.0]
            assert vec.battle_resolved[i] == 0 and vec.battle_result[i] == -1
            assert vec.terminated[i] == 1 and vec.winner[i] == flags_before[1][i]
            assert all(np.array_equal(before[key][i], vec.obs[key][i]) for key in vec.obs), "finished game changed"
            check_game(vec, i, env, final[i], "finished game")
            continue
        result = step_decisions(env, decisions[i]).result
        where = f"seed {int(seeds[i])} step"
        assert vec.status[i] == 0, (where, vec.error(i))
        if vec.rewards[i].tolist() != [float(r) for r in result.rewards]:
            raise AssertionError(f"{where}: rewards {vec.rewards[i].tolist()} vs {result.rewards}")
        if bool(vec.battle_resolved[i]) != bool(result.battle_resolved) or bool(vec.terminated[i]) != bool(result.terminated):
            raise AssertionError(f"{where}: battle/terminated flags differ")
        if int(vec.battle_result[i]) != (-1 if result.battle_result is None else int(result.battle_result)):
            raise AssertionError(f"{where}: battle result differs")
        if int(vec.winner[i]) != (int(_completed_match_result(env.core)) if result.terminated else -1):
            raise AssertionError(f"{where}: winner differs")
        check_game(vec, i, env, result.observations, where)
        compared["steps"] += 1
        compared["battles"] += int(result.battle_resolved)
        if result.terminated:
            matches_finished += 1
            if matches_started < MAX_MATCHES and rng.random() < 0.34:
                reset_mask[i] = 1          # reset only some games, the rest keep their final rows
            else:
                done[i] = True
                final[i] = result.observations
    if reset_mask.any():
        snapshot = {key: value.copy() for key, value in vec.obs.items()}
        for i in np.flatnonzero(reset_mask):
            seeds[i] = next_seed
            next_seed += 1
            matches_started += 1
        assert vec.reset(seeds, which=reset_mask) == 0
        for i, env in enumerate(envs):
            if reset_mask[i]:
                check_game(vec, i, env, env.reset(seed=int(seeds[i]), paired_tape=True), "partial reset")
                compared["resets"] += 1
            else:
                assert all(np.array_equal(snapshot[key][i], vec.obs[key][i]) for key in vec.obs), "reset touched another game"

# ---- error contract ---------------------------------------------------------------------------
assert (vec.terminated == 1).all()
which = np.zeros(N, dtype=np.uint8); which[0] = 1
seeds[0] = next_seed
assert vec.reset(seeds, which=which) == 0 and vec.terminated[0] == 0 and vec.terminated[1:].all()
state = vec.state(0)
snapshot = {key: value.copy() for key, value in vec.obs.items()}
checked = 0


def expect_illegal(action0, action1, mask0=0, mask1=0, text=None):
    global checked
    actions = np.full((N, 2, 3), sap_env_rs.NO_DECISION, dtype=np.int64)
    masks = np.zeros((N, 2), dtype=np.int64)
    actions[0, 0], actions[0, 1], masks[0] = action0, action1, (mask0, mask1)
    assert vec.step(actions, masks) == 1, "exactly one game must report an error"
    assert vec.status[0] == sap_env_rs.STATUS_ILLEGAL, vec.status[0]
    assert (vec.status[1:] == sap_env_rs.STATUS_FINISHED).all()
    assert vec.state(0) == state, "illegal input changed the game"
    assert all(np.array_equal(snapshot[key], vec.obs[key]) for key in vec.obs), "illegal input changed an observation"
    assert vec.rewards[0].tolist() == [0.0, 0.0] and vec.terminated[0] == 0
    message = vec.error(0)
    assert message and (text is None or text in message), message
    checked += 1
    return message


legal0 = np.flatnonzero(vec.legal(0, 0))
good = list(C.decode(int(legal0[0])).as_tuple())
roll = [int(ActionType.ROLL), 10, 5]
print("illegal input messages:")
for name, args in [
    ("END_TURN with gold left", ([0, 10, 5], roll)),
    ("buy from an empty shop position", ([2, 9, 0], roll)),
    ("sell an empty team slot", ([4, 0, 5], roll)),
    ("action type out of range", ([9, 0, 0], roll)),
    ("source out of range", ([2, 11, 0], roll)),
    ("huge field values", ([2, 2**62, -2**62], roll)),
    ("non-canonical unused field", ([1, 0, 5], roll)),
    ("no decision for an active seat", ([-1, -1, -1], roll)),
    ("freeze mask with a non-END action", (roll, roll, 3, 0)),
    ("freeze mask out of range", (roll, roll, 0, 1024)),
    ("negative freeze mask", (roll, roll, -1, 0)),
    ("legal seat 0, illegal seat 1: nothing may be applied", (good, [5, 0, 1])),
]:
    print("  ", name, "->", expect_illegal(*args))
# A legal step still works afterwards, and matches the reference.
env = ParallelSelfPlayEnv(); env.reset(seed=int(seeds[0]), paired_tape=True)
actions = np.full((N, 2, 3), sap_env_rs.NO_DECISION, dtype=np.int64)
actions[0, 0], actions[0, 1] = roll, roll
assert vec.step(actions, np.zeros((N, 2), dtype=np.int64)) == 0 and vec.status[0] == 0
result = step_decisions(env, [JointAction(AutoregressiveAction(*roll)), JointAction(AutoregressiveAction(*roll))]).result
check_game(vec, 0, env, result.observations, "after the rejected inputs")
# A seat that has ended its turn must send "no decision": anything else is rejected for that game.
for _ in range(9):
    assert vec.step(actions, np.zeros((N, 2), dtype=np.int64)) == 0
    result = step_decisions(env, [JointAction(AutoregressiveAction(*roll)), JointAction(AutoregressiveAction(*roll))]).result
end = [int(ActionType.END_TURN), 10, 5]
actions[0, 0], actions[0, 1] = end, [int(ActionType.FREEZE), 0, 5]
assert vec.step(actions, np.zeros((N, 2), dtype=np.int64)) == 0, vec.error(0)
result = step_decisions(env, [JointAction(AutoregressiveAction(*end), 0), JointAction(AutoregressiveAction(int(ActionType.FREEZE), 0, 5))]).result
check_game(vec, 0, env, result.observations, "one seat ended")
assert vec.active[0].tolist() == [0, 1]
state = vec.state(0)
snapshot = {key: value.copy() for key, value in vec.obs.items()}
print("  ", "decision for a seat that has ended its turn", "->", expect_illegal(roll, end))
print("  ", "freeze mask for a seat that has ended its turn", "->", expect_illegal([-1, -1, -1], end, 1, 0))
actions[0, 0], actions[0, 1] = [-1, -1, -1], end
assert vec.step(actions, np.zeros((N, 2), dtype=np.int64)) == 0 and vec.battle_resolved[0] == 1
result = step_decisions(env, [None, JointAction(AutoregressiveAction(*end), 0)]).result
check_game(vec, 0, env, result.observations, "battle after the rejected inputs")
assert vec.rewards[0].tolist() == [float(r) for r in result.rewards] and int(vec.battle_result[0]) == int(result.battle_result)
# Seeds outside [0, 2**63) are refused, the running match is kept.
bad_seeds = seeds.copy(); bad_seeds[0] = -5
assert vec.reset(bad_seeds, which=which) == 1 and vec.status[0] == sap_env_rs.STATUS_ILLEGAL
check_game(vec, 0, env, result.observations, "after a refused reset")
print(json_line := {"games": N, "threads": THREADS, "matches finished": matches_finished, **compared,
                    "illegal inputs rejected without effect": checked})
print("VEC ENV PARITY PASSED")
