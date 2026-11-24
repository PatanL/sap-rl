"""Play the bot through full matches against boards drawn from ranked players.

The bot sits in seat 0 and shops normally. Seat 1 buys nothing; just before
each battle its team is replaced by a board that a ranked player took into
battle on the same turn (SAP Library replays). Lives, wins and the end of the
match follow the environment's own rules, and the bot's view of its opponent's
previous boards shows the human boards. With --control seat 0 is also drawn
from the pool, which must give 50% and checks the harness for seat bias.

Each opposing board comes from a different player, so the opponent is the
ranked field's typical board at that turn, not one person adapting.

Usage: ghost_match.py <checkpoint|none> <out.json> <matches> <fixtures.json>... [--greedy] [--control] [--device cpu|cuda] [--batch N]
"""
import copy, hashlib, json, os, random, sys, time
from collections import Counter, defaultdict
import numpy as np
import torch
from sapai import Pet, Team
from sapai.effects import TransferAbility
from sapai.data import data as ENGINE_DATA
from sapai_gym1.rl.model import EntityTransformerConfig, EntityTransformerPolicy
from sapai_gym1.rl.action_schema import AutoregressiveAction, ActionType
from sapai_gym1.rl.distributed import _completed_match_result
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.joint_action import JointAction, step_decisions

flags = {"--greedy", "--control"}
args = [a for a in sys.argv[1:] if a not in flags]
def option(name, default):
    if name in args:
        at = args.index(name); value = args[at + 1]; del args[at:at + 2]; return value
    return default
DEVICE = option("--device", "cuda"); BATCH = int(option("--batch", 128)); SEED = int(option("--seed", 20261123))
CKPT, OUT, MATCHES, FIXTURES = args[0], args[1], int(args[2]), args[3:]
GREEDY, CONTROL = "--greedy" in sys.argv, "--control" in sys.argv
rng = random.Random(SEED)


def make_team(specs):
    pets = []
    for spec in specs:
        pet = Pet(spec["id"]); pet._attack = int(spec["attack"]); pet._health = int(spec["health"])
        pet.set_total_experience(int(spec["experience"])); pet.status = spec["status"]
        pet.ability_counter = int(spec.get("triggersConsumed", 0)); pets.append(pet)
    team = Team(pets)
    for index, slot in enumerate(team):
        if slot.empty or slot.pet.name != "pet-parrot":
            continue
        if "parrotCopies" in specs[index]:
            source = specs[index]["parrotCopies"]
            if source:
                ability = ENGINE_DATA["pets"][source].get(f"level{slot.pet.level}Ability") or Pet(source).ability
                ability = copy.deepcopy(ability); ability["_copiedFrom"] = source; slot.pet.set_ability(ability)
        else:
            TransferAbility(slot.pet, [0, index], [team])
    return team


pool, seen = defaultdict(list), set()
for path in FIXTURES:
    for f in json.load(open(path))["fixtures"]:
        key = (f["participationId"], f["turn"])
        if key in seen or not f["player"]:
            continue
        # Same match split as ghost.py: score on the yardstick side unless told otherwise.
        split = os.environ.get("GHOST_SPLIT", "yardstick")
        if split != "all" and ("yardstick" if int(hashlib.sha256(f["matchId"].encode()).hexdigest(), 16) % 5 < 2 else "train") != split:
            continue
        seen.add(key); pool[f["turn"]].append(f["player"])
LAST = max(t for t, boards in pool.items() if len(boards) >= 50)


def human_board(turn):
    return rng.choice(pool[min(max(turn, 1), LAST)])


def put(player, specs):
    player.team = make_team(specs)
    player._connect_objects()


model = None
if not CONTROL:
    ck = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True)
    model = EntityTransformerPolicy(EntityTransformerConfig(**ck["config"]))
    model.load_state_dict(ck["model"], strict=True)
    model = model.eval().requires_grad_(False).to(DEVICE)
    torch.manual_seed(SEED)
END = JointAction(AutoregressiveAction(int(ActionType.END_TURN)), None)
results, per_turn = [], defaultdict(Counter)
started = time.monotonic()
seeds = [(SEED * 1000003 + i * 7919) % (2**31) for i in range(MATCHES)]
with torch.inference_mode():
    for start in range(0, MATCHES, BATCH):
        group = seeds[start:start + BATCH]; envs, obs, done = [], [], set()
        memory = None if model is None else model.initial_memory(len(group), DEVICE)
        for seed in group:
            e = ParallelSelfPlayEnv(); obs.append(e.reset(seed=seed, paired_tape=True)); envs.append(e)
        for tick in range(6000):
            if len(done) == len(group):
                break
            actions = [[None, None] for _ in group]; requests = []
            for k in range(len(group)):
                if k in done:
                    continue
                active = envs[k].active_players()
                if active[1]:
                    # The environment refuses to end a turn with gold unspent; the stand-in seat has no use for it.
                    envs[k].core.opponent.gold = 0
                    actions[k][1] = END
                if active[0]:
                    if CONTROL:
                        envs[k].core.agent.gold = 0
                        actions[k][0] = END
                    else:
                        requests.append(k)
            if requests:
                inp = {key: torch.from_numpy(np.stack([obs[k][0][key] for k in requests])).to(DEVICE) for key in obs[0][0]}
                indices = torch.tensor(requests, device=DEVICE)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    out = model.act(inp, memory=memory.index_select(0, indices), deterministic=GREEDY)
                memory.index_copy_(0, indices, out.memory.float())
                moves = out.actions.cpu().tolist()
                masks = [None] * len(requests) if out.end_freeze_mask is None else out.end_freeze_mask.cpu().tolist()
                for k, move, mask in zip(requests, moves, masks):
                    actions[k][0] = JointAction(AutoregressiveAction(*move), mask)
            for k in range(len(group)):
                if k in done:
                    continue
                core = envs[k].core; turn = int(core.agent.turn)
                ends = actions[k][0] is not None and actions[k][0].primitive.action_type == int(ActionType.END_TURN)
                other_done = actions[k][1] is not None or not envs[k].active_players()[1]
                if ends and other_done:
                    # The battle resolves inside this step: swap in the human board(s) now.
                    put(core.opponent, human_board(turn))
                    if CONTROL:
                        put(core.agent, human_board(turn))
                step = step_decisions(envs[k], actions[k]).result; obs[k] = step.observations
                if step.battle_resolved:
                    per_turn[turn][int(core.agent.last_battle_outcome)] += 1
                if step.terminated:
                    winner = _completed_match_result(core); done.add(k)
                    results.append({"seed": group[k], "winner": int(winner), "lastTurn": int(core.turn) - 1,
                                    "lives": [int(core.agent.lives), int(core.opponent.lives)], "wins": [int(core.agent.wins), int(core.opponent.wins)]})
        if len(done) != len(group):
            raise RuntimeError("incomplete matches")
        n = len(results); w = sum(r["winner"] == 0 for r in results); d = sum(r["winner"] == 2 for r in results)
        print(json.dumps({"matches": n, "seat0Wins": w, "draws": d, "score": (w + 0.5 * d) / n, "seconds": round(time.monotonic() - started, 1)}), flush=True)
n = len(results); w = sum(r["winner"] == 0 for r in results); d = sum(r["winner"] == 2 for r in results); score = (w + 0.5 * d) / n
turns = {t: {"battles": sum(c.values()), "won": c[1], "lost": c[-1], "drawn": c[0], "score": (c[1] + 0.5 * c[0]) / max(1, sum(c.values()))} for t, c in sorted(per_turn.items())}
json.dump({"checkpoint": CKPT, "greedy": GREEDY, "control": CONTROL, "matches": n, "seat0Wins": w, "draws": d, "score": score,
           "se": (score * (1 - score) / n) ** 0.5, "meanLastTurn": sum(r["lastTurn"] for r in results) / n, "perTurn": turns, "results": results}, open(OUT, "w"), indent=1)
print("per turn battle score:", {t: round(v["score"], 3) for t, v in turns.items() if v["battles"] >= 20})
