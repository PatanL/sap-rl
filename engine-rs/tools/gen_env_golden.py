"""Step-by-step reference traces of the training environment (Python, frozen engine).

The Rust shop and environment port is checked against these files. One JSON
line per match header and per step, gzip-compressed, so a mismatch names the
first step that differs rather than a whole match.

Every match is played the way the collector plays it:
`ParallelSelfPlayEnv().reset(seed, paired_tape=True)` and `step_decisions`
with one (type, source, target) triple and one END freeze mask per active seat.

Header line: {"match": seed, "obs0": [..], "state0": .., "legal0": [..]}
Step line:
  i        step index
  active   which seats acted
  legal    per seat: sorted flat action ids that were legal BEFORE the step (null if inactive)
  act      per seat: [type, source, target] submitted (null if inactive)
  mask     per seat: END freeze mask id submitted
  plan     per seat: the primitive triples actually executed (freeze toggles, then the action)
  rew      rewards returned
  battle   battle_resolved, result: battle_result, term: terminated
  obs      per seat: observation AFTER the step, base64(zlib(arrays concatenated in OBS_KEYS order)),
           or only "obs_sha" (sha256 of the same bytes) when the match is not a full-observation match
  state    canonical state AFTER the step (see `canonical`)
  winner   on the last step: _completed_match_result

With `--ext` every seat of every state also carries a block `x` of hidden
state that the plain dump leaves out (Rust side: `src/canon.rs`), and battle
steps carry `battleSeed`:
  x.toggledShop  shop positions whose item is in the freeze-toggled set
  x.shop         per shop slot: [reward_group] (empty), [reward_group, attack, health, cost] (food),
                 [reward_group, status, sell_value] (pet)
  x.rs           the shop stream (`shop.rs.get_state()`): [position, first key word, last key word]
  x.cursors      RandomTape cursors of the seat's pet and food offer domains
  x.lf           lf_winner

Usage:
  gen_env_golden.py random <out.jsonl.gz> <matches> <first index> <full obs every N> [--ext]
  gen_env_golden.py policy <out.jsonl.gz> <matches> <first index> <full obs every N> <checkpoint> [--greedy] [--ext]

`tools/gen_env_targeted.py` imports this file for its helpers.
"""
import base64, gzip, hashlib, json, random, sys, zlib
import numpy as np
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import AutoregressiveAction, ActionType, FlatActionCodec as C, MAX_SHOP_SLOTS
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.joint_action import JointAction, step_decisions
from sapai_gym1.rl.end_freeze import expand_end_freeze, SHOP_ACTION_CAP

MAIN = __name__ == "__main__"
if MAIN:
    MODE, OUT, MATCHES, FIRST, FULL_EVERY = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
EXT = "--ext" in sys.argv
OBS_KEYS = None


def obs_bytes(obs):
    global OBS_KEYS
    if OBS_KEYS is None:
        OBS_KEYS = list(obs)
    return b"".join(np.ascontiguousarray(obs[k]).tobytes() for k in OBS_KEYS)


def pack_obs(observations, full):
    raw = [obs_bytes(o) for o in observations]
    if full:
        return {"obs": [base64.b64encode(zlib.compress(r, 6)).decode() for r in raw]}
    return {"obs_sha": [hashlib.sha256(r).hexdigest() for r in raw]}


def pet_row(pet):
    copied = pet.ability.get("_copiedFrom") if pet.override_ability else None
    return [pet.name, int(pet._attack), int(pet._health), int(pet._until_end_of_battle_attack_buff),
            int(pet._until_end_of_battle_health_buff), int(pet.level), int(pet.experience), pet.status,
            int(pet.ability_counter), int(pet.sell_value), int(pet.friends_hurt_counter), int(pet._hurt), copied]


def canonical(core):
    """State both engines can print the same way. Extend on both sides together."""
    seats = []
    for side, player in enumerate((core.agent, core.opponent)):
        tag = "agent" if side == 0 else "opponent"
        team = [None if slot.empty else pet_row(slot.pet) for slot in player.team]
        shop = []
        for slot in player.shop:
            if slot.empty:
                shop.append([slot.slot_type, None, bool(slot.frozen), int(slot.cost)])
            elif slot.slot_type == "food":
                shop.append([slot.slot_type, slot.obj.name, bool(slot.frozen), int(slot.cost)])
            else:
                shop.append([slot.slot_type, slot.obj.name, bool(slot.frozen), int(slot.cost),
                             int(slot.obj._attack), int(slot.obj._health), int(slot.obj.level)])
        toggled = getattr(core, f"_freeze_toggled_{tag}")
        extra = {}
        if EXT:
            rows = []
            for slot in player.shop:
                if slot.empty:
                    rows.append([slot.reward_group])
                elif slot.slot_type == "food":
                    rows.append([slot.reward_group, int(slot.obj.attack), int(slot.obj.health), int(slot.obj.cost)])
                else:
                    rows.append([slot.reward_group, slot.obj.status, int(slot.obj.sell_value)])
            state = player.shop.rs.get_state()
            tape = player.shop._paired_tape
            extra["x"] = {
                "toggledShop": [index for index, slot in enumerate(player.shop) if not slot.empty and id(slot.obj) in toggled],
                "shop": rows,
                "rs": [int(state[2]), int(state[1][0]), int(state[1][623])],
                "cursors": [tape.position(f"player/{side}/shop/pet"), tape.position(f"player/{side}/shop/food")],
                "lf": player.lf_winner,
            }
        seats.append({
            **extra,
            "gold": int(player.gold), "turn": int(player.turn), "lives": int(player.lives), "wins": int(player.wins),
            "ended": bool(player.turn_ended), "last": int(player.last_battle_outcome) if player.last_battle_outcome is not None else None,
            "team": team, "shop": shop,
            "shopAttack": int(player.shop.shop_attack), "shopHealth": int(player.shop.shop_health),
            "actions": int(getattr(core, f"shop_action_count_{tag}")), "reorders": int(getattr(core, f"reorder_count_{tag}")),
            "extraReorder": bool(getattr(core, f"allow_additional_reorder_{tag}")),
            "toggled": sorted(int(x) for x in getattr(core, f"_freeze_toggled_{tag}")),
            "seenOrders": len(getattr(core, f"_seen_team_orders_{tag}")),
        })
    return {"turn": int(core.turn), "seats": seats}


def legal_of(env):
    return [sorted(int(a) for a in env.core._avail_actions(side)) if active else None
            for side, active in enumerate(env.active_players())]


def random_mask(env, side, rng):
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


def write_step(out, index, env, legal, decisions, result, full):
    step = result.result
    line = {"i": index, "active": [d is not None for d in decisions], "legal": legal,
            "act": [None if d is None else list(d.primitive.as_tuple()) for d in decisions],
            "mask": [0 if d is None or d.end_freeze_mask is None else int(d.end_freeze_mask) for d in decisions],
            "plan": [[list(a.as_tuple()) for a in plan] for plan in result.primitive_actions],
            "rew": [float(r) for r in step.rewards], "battle": bool(step.battle_resolved),
            "result": None if step.battle_result is None else int(step.battle_result), "term": bool(step.terminated),
            "state": canonical(env.core)}
    line.update(pack_obs(step.observations, full))
    if EXT and step.battle_resolved:
        line["battleSeed"] = int(env._paired_battle_seeds[-1][1])
    if step.terminated:
        line["winner"] = int(_completed_match_result(env.core))
    out.write(json.dumps(line, separators=(",", ":")) + "\n")
    return step


def header(out, seed, env, obs, full):
    line = {"match": seed, "state0": canonical(env.core), "legal0": legal_of(env)}
    packed = pack_obs(obs, full)
    line.update({k + "0": v for k, v in packed.items()})
    out.write(json.dumps(line, separators=(",", ":")) + "\n")


def play_random(out):
    decisions_total = 0
    for index in range(FIRST, FIRST + MATCHES):
        seed = 881100000 + index
        full = FULL_EVERY > 0 and index % FULL_EVERY == 0
        rng = random.Random(seed)
        env = ParallelSelfPlayEnv(); obs = env.reset(seed=seed, paired_tape=True)
        header(out, seed, env, obs, full)
        for step_index in range(8000):
            legal = legal_of(env); decisions = []
            for side, options in enumerate(legal):
                if options is None:
                    decisions.append(None); continue
                action = C.decode(rng.choice(options)); mask = None
                if action.action_type == int(ActionType.END_TURN):
                    mask = random_mask(env, side, rng)
                decisions.append(JointAction(action, mask)); decisions_total += 1
            step = write_step(out, step_index, env, legal, decisions, step_decisions(env, decisions), full)
            if step.terminated:
                break
        else:
            raise RuntimeError("match did not end")
    return decisions_total


def play_policy(out):
    import torch
    from sapai_gym1.rl.model import EntityTransformerConfig, EntityTransformerPolicy
    torch.set_num_threads(4)
    ck = torch.load(sys.argv[6], map_location="cpu", mmap=True, weights_only=True)
    model = EntityTransformerPolicy(EntityTransformerConfig(**ck["config"]))
    model.load_state_dict(ck["model"], strict=True); model = model.eval().requires_grad_(False)
    greedy = "--greedy" in sys.argv
    torch.manual_seed(FIRST)
    seeds = [882200000 + i for i in range(FIRST, FIRST + MATCHES)]
    envs, obs, files, steps, done = [], [], [], [0] * MATCHES, set()
    import io
    buffers = [io.StringIO() for _ in seeds]
    memory = model.initial_memory(MATCHES * 2, "cpu")
    for k, seed in enumerate(seeds):
        e = ParallelSelfPlayEnv(); o = e.reset(seed=seed, paired_tape=True); envs.append(e); obs.append(o)
        header(buffers[k], seed, e, o, FULL_EVERY > 0 and (FIRST + k) % FULL_EVERY == 0)
    decisions_total = 0
    with torch.inference_mode():
        while len(done) < MATCHES:
            requests = [(k, s) for k in range(MATCHES) if k not in done for s, active in enumerate(envs[k].active_players()) if active]
            inp = {key: torch.from_numpy(np.stack([obs[k][s][key] for k, s in requests])) for key in obs[0][0]}
            indices = torch.tensor([k * 2 + s for k, s in requests])
            result = model.act(inp, memory=memory.index_select(0, indices), deterministic=greedy)
            memory.index_copy_(0, indices, result.memory.float())
            moves = result.actions.tolist()
            masks = [None] * len(requests) if result.end_freeze_mask is None else result.end_freeze_mask.tolist()
            actions = {k: [None, None] for k, _ in requests}
            for (k, s), move, mask in zip(requests, moves, masks):
                actions[k][s] = JointAction(AutoregressiveAction(*move), mask); decisions_total += 1
            for k, decisions in actions.items():
                legal = legal_of(envs[k])
                full = FULL_EVERY > 0 and (FIRST + k) % FULL_EVERY == 0
                step = write_step(buffers[k], steps[k], envs[k], legal, decisions, step_decisions(envs[k], decisions), full)
                steps[k] += 1; obs[k] = step.observations
                if step.terminated:
                    done.add(k)
    for buffer in buffers:
        out.write(buffer.getvalue())
    return decisions_total


if MAIN:
    with gzip.open(OUT, "wt", compresslevel=5) as out:
        total = play_random(out) if MODE == "random" else play_policy(out)
    print(json.dumps({"mode": MODE, "matches": MATCHES, "first": FIRST, "decisions": total, "obs_keys": OBS_KEYS}))
