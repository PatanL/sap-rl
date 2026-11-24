"""Independent check of the Rust environment against the Python one (written by
the reviewer, not by the port's author).

Drives N Python environments and one Rust SapVecEnv in lockstep on seeds that
appear in no reference file and compares, after every step and for both seats:
every observation array byte for byte, rewards, battle flag and result,
terminated, winner, lives, wins and active flags. Finished games are reset
with new seeds until the requested number of matches has been played.

  random mode: actions are random legal actions (legal list from Python), with random END freeze plans.
  policy mode: the trained network picks actions from the RUST observations.

Usage: verify_vec_env.py random|policy <lib> <games> <matches> <threads> [checkpoint]
"""
import random, sys, time
import numpy as np
from sap_env_rs import SapVecEnv, NO_DECISION
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import AutoregressiveAction, ActionType, FlatActionCodec as C, MAX_SHOP_SLOTS
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.end_freeze import expand_end_freeze, SHOP_ACTION_CAP

MODE, LIB, GAMES, MATCHES, THREADS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rng = random.Random(4242)
next_seed = [977000000 if MODE == "random" else 978000000]
rust = SapVecEnv(GAMES, threads=THREADS, library=LIB) if "library" in SapVecEnv.__init__.__code__.co_varnames else SapVecEnv(GAMES, threads=THREADS)
py = [ParallelSelfPlayEnv() for _ in range(GAMES)]
py_obs = [None] * GAMES
checked = {"steps": 0, "arrays": 0, "matches": 0, "battles": 0, "plans": 0}


def fail(what, game, detail=""):
    print(f"MISMATCH {what} game {game} seed {seeds[game]} step {steps[game]} {detail}"); sys.exit(1)


def compare_obs(game):
    for seat in (0, 1):
        for key, value in py_obs[game][seat].items():
            mine = rust.obs[key][game, seat]
            if mine.dtype != value.dtype or mine.shape != value.shape or mine.tobytes() != np.ascontiguousarray(value).tobytes():
                fail("observation " + key, game, f"seat {seat}")
            checked["arrays"] += 1


seeds, steps = [0] * GAMES, [0] * GAMES
def reset(which):
    mask = np.zeros(GAMES, dtype=bool); new = np.zeros(GAMES, dtype=np.int64)
    for g in which:
        seeds[g] = next_seed[0]; next_seed[0] += 1; steps[g] = 0
        mask[g] = True; new[g] = seeds[g]; py_obs[g] = py[g].reset(seed=seeds[g], paired_tape=True)
    rust.reset(new, which=mask)
    for g in which:
        compare_obs(g)


def random_mask(env, side):
    core = env.core; player = core.agent if side == 0 else core.opponent
    if player.gold != 0:
        return 0
    count = core.shop_action_count_agent if side == 0 else core.shop_action_count_opponent
    mask = 0
    for index in range(min(MAX_SHOP_SLOTS, len(player.shop))):
        if not player.shop[index].empty and rng.random() < 0.25:
            mask |= 1 << index
    if bin(mask).count("1") > max(0, SHOP_ACTION_CAP - count):
        return 0
    try:
        expand_end_freeze(core, side, mask)
    except ValueError:
        return 0
    return mask


model = None
if MODE == "policy":
    import torch
    from sapai_gym1.rl.model import EntityTransformerConfig, EntityTransformerPolicy
    torch.set_num_threads(THREADS); torch.manual_seed(99)
    ck = torch.load(sys.argv[6], map_location="cpu", mmap=True, weights_only=True)
    model = EntityTransformerPolicy(EntityTransformerConfig(**ck["config"]))
    model.load_state_dict(ck["model"], strict=True); model = model.eval().requires_grad_(False)
    memory = model.initial_memory(GAMES * 2, "cpu")

reset(range(GAMES))
finished_total = 0; started = time.time()
live = set(range(GAMES))
while live:
    actions = np.full((GAMES, 2, 3), NO_DECISION, dtype=np.int64); actions[:, :, 1] = 10; actions[:, :, 2] = 5
    masks = np.zeros((GAMES, 2), dtype=np.int64)
    decisions = {g: [None, None] for g in live}
    if model is None:
        for g in live:
            for side, active in enumerate(py[g].active_players()):
                if not active:
                    continue
                action = C.decode(rng.choice(sorted(py[g].core._avail_actions(side))))
                mask = random_mask(py[g], side) if action.action_type == int(ActionType.END_TURN) else 0
                decisions[g][side] = JointAction(action, mask)
    else:
        requests = [(g, s) for g in sorted(live) for s in (0, 1) if rust.active[g, s]]
        for g, s in requests:
            if not py[g].active_players()[s]:
                fail("active flag", g)
        import torch
        with torch.inference_mode():
            inp = {k: torch.from_numpy(np.stack([rust.obs[k][g, s] for g, s in requests])) for k in rust.obs}
            idx = torch.tensor([g * 2 + s for g, s in requests])
            out = model.act(inp, memory=memory.index_select(0, idx), deterministic=False)
            memory.index_copy_(0, idx, out.memory.float())
        ms = [None] * len(requests) if out.end_freeze_mask is None else out.end_freeze_mask.tolist()
        for (g, s), move, m in zip(requests, out.actions.tolist(), ms):
            decisions[g][s] = JointAction(AutoregressiveAction(*move), m)
    for g, pair in decisions.items():
        for side, d in enumerate(pair):
            if d is not None:
                actions[g, side] = d.primitive.as_tuple(); masks[g, side] = int(d.end_freeze_mask or 0)
                checked["plans"] += bool(d.end_freeze_mask)
    rust.step(actions, masks)
    done_now = []
    for g in sorted(live):
        step = step_decisions(py[g], decisions[g]).result; py_obs[g] = step.observations; steps[g] += 1
        if rust.status[g] != 0:
            fail("status", g, str(rust.status[g]))
        compare_obs(g)
        core = py[g].core
        if tuple(float(x) for x in rust.rewards[g]) != tuple(float(x) for x in step.rewards) or np.signbit(rust.rewards[g]).tolist() != [bool(np.signbit(x)) for x in step.rewards]:
            fail("rewards", g, f"{rust.rewards[g]} {step.rewards}")
        if bool(rust.battle_resolved[g]) != bool(step.battle_resolved) or bool(rust.terminated[g]) != bool(step.terminated):
            fail("flags", g)
        if step.battle_resolved and int(rust.battle_result[g]) != int(step.battle_result):
            fail("battle result", g)
        if [int(x) for x in rust.lives[g]] != [int(core.agent.lives), int(core.opponent.lives)] or [int(x) for x in rust.wins[g]] != [int(core.agent.wins), int(core.opponent.wins)]:
            fail("lives or wins", g)
        if not step.terminated and [bool(x) for x in rust.active[g]] != list(py[g].active_players()):
            fail("active", g)
        checked["steps"] += 1; checked["battles"] += bool(step.battle_resolved)
        if step.terminated:
            if int(rust.winner[g]) != int(_completed_match_result(core)):
                fail("winner", g)
            checked["matches"] += 1; done_now.append(g)
    for g in done_now:
        finished_total += 1
        if finished_total + len(live) - len(done_now) < MATCHES + 0 and next_seed[0] - (977000000 if MODE == "random" else 978000000) < MATCHES:
            reset([g])
            if model is not None:
                memory[g * 2:g * 2 + 2] = model.initial_memory(2, "cpu")
        else:
            live.discard(g)
print("ALL EQUAL", MODE, checked, f"{time.time() - started:.0f}s")
