"""Play the bot against itself and write each seat in the shape analyse.py uses.

One checkpoint plays both seats of every match on the packaged engine. For each
seat it records the board when the seat ends its turn (front first), every food
bought with its target, rolls per turn and the result, so that the line
counts made for human games can be made for the bot.

Usage: bot_games.py <checkpoint> <out.jsonl> <matches> [--greedy] [--batch N] [--seed S]
"""
import json, sys, time
import numpy as np
import torch
from sapai_gym1.rl.model import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.action_schema import AutoregressiveAction, ActionType
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.joint_action import JointAction, step_decisions

CKPT, OUT, MATCHES = sys.argv[1], sys.argv[2], int(sys.argv[3])
GREEDY = "--greedy" in sys.argv
BATCH = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 128
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 20261123
PERK = {"status-honey-bee": "Honey", "status-extra-life": "Mushroom", "status-melon-armor": "Melon",
        "status-garlic-armor": "Garlic", "status-bone-attack": "Meat Bone", "status-steak-attack": "Steak",
        "status-splash-attack": "Chili", "status-coconut-shield": "Coconut", "status-poison-attack": "Peanut",
        "status-weak": "Weak"}


def title(name):
    return " ".join(w.capitalize() for w in name.split("-")[1:])


def board(player):
    pets = []
    for index, slot in enumerate(player.team):
        if slot.empty:
            continue
        pet = slot.pet
        pets.append({"x": 4 - index, "name": title(pet.name), "level": int(pet.level), "perk": PERK.get(pet.status),
                     "attack": int(pet.attack), "health": int(pet.health)})
    return pets


LEVEL_EXPERIENCE = {1: 0, 2: 2, 3: 5}
BATTLES = []          # filled by the recording wrapper below
CURRENT = {"match": None}


def spec(team):
    pets = []
    for slot in team:
        if slot.empty:
            continue
        pet = slot.pet
        row = {"id": pet.name, "attack": int(pet.attack), "health": int(pet.health), "level": int(pet.level),
               "experience": LEVEL_EXPERIENCE[int(pet.level)], "status": pet.status, "triggersConsumed": int(pet.ability_counter)}
        if pet.name == "pet-parrot":
            row["parrotCopies"] = pet.ability.get("_copiedFrom") if pet.override_ability else None
        pets.append(row)
    return pets


import sapai_gym1.rl.selfplay_env as selfplay_env
_RealBattle = selfplay_env.Battle


class RecordingBattle:
    """Write down both teams as the battle receives them, then run it."""

    def __init__(self, t0, t1, **kwargs):
        BATTLES.append({"match": CURRENT["match"], "turn": CURRENT["turn"], "teams": [spec(t0), spec(t1)]})
        self.inner = _RealBattle(t0, t1, **kwargs)

    def battle(self):
        return self.inner.battle()


selfplay_env.Battle = RecordingBattle


def main():
    device = "cuda"
    ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True)
    model = EntityTransformerPolicy(EntityTransformerConfig(**ck["config"]))
    model.load_state_dict(ck["model"], strict=True)
    model = model.eval().requires_grad_(False).to(device)
    torch.manual_seed(SEED)
    name = f"bot-{ck.get('update')}" + ("-greedy" if GREEDY else "")
    out = open(OUT, "w"); battles_out = open(OUT.replace(".jsonl", "") + "-battles.jsonl", "w"); started = time.monotonic(); written = 0; odd = 0
    seeds = [(SEED * 1000003 + i * 7919) % (2**31) for i in range(MATCHES)]
    with torch.inference_mode():
        for start in range(0, MATCHES, BATCH):
            group = seeds[start:start + BATCH]; envs = []; obs = []; done = set()
            memory = model.initial_memory(len(group) * 2, device)
            seats = [[{"name": name, "match": f"{name}-{seed}", "turns": {}, "foods": [], "rolls": {}, "battles": []} for _ in (0, 1)] for seed in group]
            for seed in group:
                e = ParallelSelfPlayEnv(); obs.append(e.reset(seed=seed, paired_tape=True)); envs.append(e)
            for tick in range(4000):
                if len(done) == len(group):
                    break
                requests = [(k, s) for k in range(len(group)) if k not in done for s, active in enumerate(envs[k].active_players()) if active]
                actions = [[None, None] for _ in group]
                if requests:
                    inp = {key: torch.from_numpy(np.stack([obs[k][s][key] for k, s in requests])).to(device) for key in obs[0][0]}
                    indices = torch.tensor([k * 2 + s for k, s in requests], device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = model.act(inp, memory=memory.index_select(0, indices), deterministic=GREEDY)
                    memory.index_copy_(0, indices, result.memory.float())
                    moves = result.actions.cpu().tolist()
                    masks = [None] * len(requests) if result.end_freeze_mask is None else result.end_freeze_mask.cpu().tolist()
                    for (k, s), move, mask in zip(requests, moves, masks):
                        actions[k][s] = JointAction(AutoregressiveAction(*move), mask)
                        player = envs[k].core.agent if s == 0 else envs[k].core.opponent
                        turn = int(player.turn); kind, source, target = move; seat = seats[k][s]
                        if kind == ActionType.ROLL:
                            seat["rolls"][turn] = seat["rolls"].get(turn, 0) + 1
                        elif kind == ActionType.BUY_FOOD:
                            slot = player.shop[source]
                            if slot.slot_type != "food":
                                odd += 1
                            else:
                                aimed = None
                                if target < 5 and not player.team[target].empty:
                                    aimed = title(player.team[target].pet.name)
                                seat["foods"].append((turn, title(slot.obj.name), aimed))
                        elif kind == ActionType.END_TURN:
                            seat["turns"][turn] = {"board": board(player), "wins": int(getattr(player, "wins", 0)),
                                                   "lives": int(getattr(player, "lives", 0)), "rolls": seat["rolls"].get(turn, 0), "tier": None}
                for k in range(len(group)):
                    if k in done:
                        continue
                    CURRENT["match"] = f"{name}-{group[k]}"; CURRENT["turn"] = int(envs[k].core.agent.turn)
                    step = step_decisions(envs[k], actions[k]).result; obs[k] = step.observations
                    if step.terminated:
                        winner = _completed_match_result(envs[k].core); done.add(k)
                        for s in (0, 1):
                            seat = seats[k][s]
                            seat["won"] = int(winner == s); seat["rawOutcome"] = 1 if winner == s else (3 if winner == 2 else 2)
                            seat["last"] = max(seat["turns"]) if seat["turns"] else 0
                            del seat["rolls"]
                            out.write(json.dumps(seat) + "\n"); written += 1
            out.flush()
            for row in BATTLES:
                battles_out.write(json.dumps(row) + "\n")
            BATTLES.clear(); battles_out.flush()
            print(json.dumps({"matches": start + len(group), "seats": written, "seconds": round(time.monotonic() - started, 1), "oddFoodActions": odd}), flush=True)


main()
