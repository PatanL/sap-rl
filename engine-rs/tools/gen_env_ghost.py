"""Reference traces for the search and ghost primitives (Python, frozen engine).

Matches are played in the Python environment with state edits between steps,
the way `runs/train-ghost-20261123/ghost_worker.py` makes them, and written in
the trace format of `gen_env_golden.py` (extended state block) with three more
keys per step line, which `envcheck` replays through the Rust `Game` methods
behind the C entry points:

  pre         edits made before the step, in order:
              ["gold", seat, value]   player.gold = value
              ["team", seat, board]   player.team = make_team(specs); player._connect_objects()
              ["rekey", seed]         env.redeterminize(seed)
              ["legal"]               the legal actions of the active seats were computed here, between
                                      two edits (computing them registers the current team order as seen)
  teams       on battle steps: both teams as `Battle.__init__` received them (recorded by a
              wrapper around `selfplay_env.Battle`, as `RecordingBattle` in
              runs/player-analysis-20261123/bot_games.py does, but slot by slot and with the
              pet's real total experience)
  carried     with `teams`: per slot [friends_hurt_counter, _hurt]
  battleTurn  with `teams`: core.turn when the battle was fought

A board is five rows or nulls; a row is [id, attack, health, total experience, status,
triggersConsumed, copies], copies false (no `parrotCopies` key: the default copy rule for a
Parrot), null (`parrotCopies: None`) or the copied species. Battle steps always carry both full
observations, so the opponent-history arrays after every battle are byte-compared.

Modes (seeds are 884400000 + index; the ghost seat is index % 2):
  ghost   exactly the worker's flow: one seat live, the other a ghost whose gold is set to 0
          whenever it has to act, which sends END, and whose team is replaced on the step in
          which the live seat ends its turn by a board from the pool: one ranked player's own
          sequence turn by turn, then a board of another player at that turn.
  mix     the same flow with boards the worker's rule would not give at that point (any turn,
          empty, cut short, the same board again), `parrotCopies` removed from some Parrots (the
          default rule), a second Parrot put behind a Parrot, boards with gaps (pets moved back
          with `Team.move`), and edits of the LIVE seat in the middle of its shop phase: its team
          replaced by a pool board, its gold set to another value. The live seat then shops,
          feeds, sells and combines on swapped-in pets. In half of the matches (index // 4 odd) the live seat is
          also handed a ranked board of the same turn right before most of its END actions (its
          end-of-turn abilities then run on that board), so that it wins often enough for the
          match to reach the late turns.
  plain   both seats live, no edits (battle events of ordinary matches).
  rekey   both seats live; `env.redeterminize(seed)` at random steps.
The live seats play uniformly random legal actions in even matches (index // 2 even) and one of
the scripted preferences of `gen_env_targeted.py` in the others.

Usage (training image, tools/run_ref.sh):
  gen_env_ghost.py <mode> <out.jsonl.gz> <matches> <first index> <full obs every N>
"""
import copy, gzip, hashlib, io, json, random, sys, traceback
import numpy as np

import gen_env_golden as G
import sapai_gym1.rl.selfplay_env as selfplay_env
from sapai import Pet, Team
from sapai.effects import TransferAbility
from sapai.data import data as ENGINE_DATA
from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
from sapai_gym1.rl.action_schema import ActionType, FlatActionCodec as C
from sapai_gym1.rl.joint_action import JointAction, step_decisions

MAIN = __name__ == "__main__"
G.EXT = True
POOL_PATH = "/rs/data/ghost-train-pool.json"      # a copy of runs/train-ghost-20261123/ghost-train-pool.json
POOL_SHA256 = "55f6219eca9a173f7e288a837c0669aded85a4822fa9f66c0e0a054731a5b702"


def load_targeted():
    """The scripted live-seat preferences of gen_env_targeted.py (everything above its match loop)."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(G.__file__)), "gen_env_targeted.py")
    source = open(path).read()
    cut = source.index("\ndef play(out):")
    space = {"__name__": "gen_env_targeted_helpers"}
    saved = sys.argv
    sys.argv = ["gen_env_targeted.py", "-", "0", "0", "0"]
    try:
        exec(compile(source[:cut], path, "exec"), space)
    finally:
        sys.argv = saved
    return space


# --- verbatim from runs/train-ghost-20261123/ghost_worker.py (only the pool path differs) ---------

def load_pool():
    raw = open(POOL_PATH, 'rb').read()
    if hashlib.sha256(raw).hexdigest() != POOL_SHA256:
        raise ValueError('ghost pool file changed')
    data = json.loads(raw)
    if data['split'] != 'train':
        raise ValueError('ghost pool is not the training side')
    sequences, by_turn = [], {}
    for row in data['sequences']:
        if int(hashlib.sha256(row['match'].encode()).hexdigest(), 16) % 5 < 2:
            raise ValueError('yardstick-side match in the training pool')
        boards = {int(turn): board for turn, board in row['boards'].items()}
        sequences.append((row['participation'], boards))
        for turn, board in boards.items():
            by_turn.setdefault(turn, []).append(board)
    last = max(turn for turn, boards in by_turn.items() if len(boards) >= 50)
    return sequences, by_turn, last


def make_team(specs):
    """Same construction as runs/player-analysis-20261123/ghost_match.py."""
    pets = []
    for spec in specs:
        pet = Pet(spec['id']); pet._attack = int(spec['attack']); pet._health = int(spec['health'])
        pet.set_total_experience(int(spec['experience'])); pet.status = spec['status']
        pet.ability_counter = int(spec.get('triggersConsumed', 0)); pets.append(pet)
    team = Team(pets)
    for index, slot in enumerate(team):
        if slot.empty or slot.pet.name != 'pet-parrot':
            continue
        if 'parrotCopies' in specs[index]:
            source = specs[index]['parrotCopies']
            if source:
                ability = ENGINE_DATA['pets'][source].get(f'level{slot.pet.level}Ability') or Pet(source).ability
                ability = copy.deepcopy(ability); ability['_copiedFrom'] = source; slot.pet.set_ability(ability)
        else:
            TransferAbility(slot.pet, [0, index], [team])
    return team

# ---------------------------------------------------------------------------------------------------


SEEN = {}
_RealBattle = selfplay_env.Battle


def team_rows(team):
    rows, carried = [], []
    for slot in team:
        if slot.empty:
            rows.append(None); carried.append([0, 0]); continue
        pet = slot.pet
        copies = False
        if pet.name == "pet-parrot":
            copies = pet.ability.get("_copiedFrom") if pet.override_ability else None
        elif pet.override_ability:
            raise RuntimeError("override ability on a non-Parrot")
        rows.append([pet.name, int(pet.attack), int(pet.health), int(pet.total_experience), pet.status,
                     int(pet.ability_counter), copies])
        carried.append([int(pet.friends_hurt_counter), int(pet._hurt)])
    return rows, carried


class RecordingBattle:
    """Write down both teams as the battle receives them, then run it."""

    def __init__(self, t0, t1, **kwargs):
        first, second = team_rows(t0), team_rows(t1)
        SEEN["teams"] = [first[0], second[0]]; SEEN["carried"] = [first[1], second[1]]
        self.inner = _RealBattle(t0, t1, **kwargs)

    def battle(self):
        return self.inner.battle()


def spec_row(spec):
    copies = False
    if "parrotCopies" in spec:
        copies = spec["parrotCopies"] or None
    return [spec["id"], int(spec["attack"]), int(spec["health"]), int(spec["experience"]), spec["status"],
            int(spec.get("triggersConsumed", 0)), copies]


def swap_team(player, specs, slots=None):
    """ghost_worker's swap; `slots` (mix only) moves the pets to these team positions afterwards."""
    team = make_team(specs)
    board = [None] * 5
    if slots is None:
        slots = list(range(len(specs)))
    for index in reversed(range(len(specs))):
        if slots[index] != index:
            team.move(index, slots[index])
    for index, spec in enumerate(specs):
        board[slots[index]] = spec_row(spec)
    player.team = team
    player._connect_objects()
    return board


def write_step(out, index, env, legal, decisions, result, full, pre, turn_before):
    step = result.result
    line = {"i": index}
    if pre:
        line["pre"] = pre
    line.update({"active": [d is not None for d in decisions], "legal": legal,
                 "act": [None if d is None else list(d.primitive.as_tuple()) for d in decisions],
                 "mask": [0 if d is None or d.end_freeze_mask is None else int(d.end_freeze_mask) for d in decisions],
                 "plan": [[list(a.as_tuple()) for a in plan] for plan in result.primitive_actions],
                 "rew": [float(r) for r in step.rewards], "battle": bool(step.battle_resolved),
                 "result": None if step.battle_result is None else int(step.battle_result), "term": bool(step.terminated),
                 "state": G.canonical(env.core)})
    line.update(G.pack_obs(step.observations, full or step.battle_resolved))
    if step.battle_resolved:
        line["battleSeed"] = int(env._paired_battle_seeds[-1][1])
        line["battleTurn"] = turn_before
        line["teams"] = SEEN.pop("teams"); line["carried"] = SEEN.pop("carried")
    if step.terminated:
        line["winner"] = int(G._completed_match_result(env.core))
    out.write(json.dumps(line, separators=(",", ":")) + "\n")
    return step


class Live:
    """One live seat: uniformly random legal actions, or a scripted preference of gen_env_targeted."""

    def __init__(self, targeted, rng, scripted, idle):
        self.t, self.rng, self.sc = targeted, rng, None
        if scripted:
            self.sc = targeted["make_scenario"](rng)
            self.sc.idle = idle if self.sc.fixed_idle is None else min(self.sc.fixed_idle, idle)

    def decide(self, env, side, options):
        flat = None
        if self.sc is not None:
            try:
                flat = self.t["choose"](env, side, options, self.sc, self.rng)
            except Exception:       # the script lost track after an edit of its seat; any legal action will do
                flat = None
            if flat not in options:
                flat = None
        if flat is None:
            flat = self.rng.choice(options)
        action = C.decode(flat); mask = None
        if action.action_type == int(ActionType.END_TURN):
            mask = G.random_mask(env, side, self.rng)
        return JointAction(action, mask)


def worker_board(turn, own_boards, by_turn, last_turn, draw):
    """ghost_worker: the sequence's own board for the turn, else another player's at that turn."""
    if turn in own_boards:
        return own_boards[turn], True
    choices = by_turn[min(max(turn, 1), last_turn)]
    return choices[int(draw.integers(len(choices)))], False


def mix_board(turn, own_boards, by_turn, last_turn, draw, rng, parrots, last):
    """Boards the worker's rule would not give here. Returns (specs, slots or None, tag)."""
    roll = rng.random()
    if roll < 0.30:
        specs, _ = worker_board(turn, own_boards, by_turn, last_turn, draw); tag = "worker"
    elif roll < 0.55:
        other = rng.choice(sorted(by_turn)); specs = rng.choice(by_turn[other]); tag = "any-turn"
    elif roll < 0.75:
        specs = rng.choice(parrots); tag = "parrot"
    elif roll < 0.83:
        specs = []; tag = "empty"
    elif roll < 0.91:
        specs, _ = worker_board(turn, own_boards, by_turn, last_turn, draw)
        specs = specs[:rng.randint(1, max(1, len(specs) - 1))]; tag = "cut"
    else:
        specs = last if last is not None else rng.choice(by_turn[1]); tag = "again"
    specs = [dict(spec) for spec in specs]
    if any(s["id"] == "pet-parrot" for s in specs):
        # A second Parrot right behind a Parrot (what the back one holds depends on the rule order).
        index = next(i for i, s in enumerate(specs) if s["id"] == "pet-parrot")
        if len(specs) < 5 and rng.random() < 0.3:
            twin = dict(specs[index]); twin["attack"] = rng.randint(1, 50); twin["experience"] = rng.choice([0, 1, 2, 3, 5])
            specs.insert(index + 1, twin); tag += "+twin"
        for s in specs:
            if s["id"] == "pet-parrot" and rng.random() < 0.6:
                s.pop("parrotCopies", None); tag += "+default"
    slots = None
    if specs and len(specs) < 5 and rng.random() < 0.35:
        slots = sorted(rng.sample(range(5), len(specs))); tag += "+gap"
    return specs, slots, tag


def play(out, mode, matches, first, full_every):
    targeted = load_targeted()
    sequences, by_turn, last_turn = load_pool()
    parrots = [board for boards in by_turn.values() for board in boards if any(s["id"] == "pet-parrot" for s in board)]
    selfplay_env.Battle = RecordingBattle
    stats = {"decisions": 0, "failed": [], "edits": {}, "turns": {}, "boards": {}, "battles": 0, "ghostResults": [0, 0, 0],
             "scripted": 0}

    def note(table, key):
        stats[table][key] = stats[table].get(key, 0) + 1

    for index in range(first, first + matches):
        seed = 884400000 + index
        full = full_every > 0 and index % full_every == 0
        rng = random.Random(seed)
        draw = np.random.default_rng(np.random.SeedSequence([seed, 20261123]))
        scripted = (index // 2) % 2 == 1
        ghost = index % 2 if mode in ("ghost", "mix") else None
        idle = rng.choice([0, 0, 0, 4]) if mode in ("ghost", "mix") else rng.choice([0, 0, 4, 8])
        lives = [None if side == ghost else Live(targeted, rng, scripted, idle) for side in (0, 1)]
        participation, own_boards = sequences[int(draw.integers(len(sequences)))]
        last_board = None
        strong = mode == "mix" and (index // 4) % 2 == 1
        buffer = io.StringIO()
        local = {"edits": {}, "boards": {}, "turns": {}, "battles": 0, "ghostResults": [0, 0, 0], "decisions": 0}

        def bump(table, key):
            local[table][key] = local[table].get(key, 0) + 1
        try:
            env = ParallelSelfPlayEnv(); obs = env.reset(seed=seed, paired_tape=True)
            G.header(buffer, seed, env, obs, full)
            core = env.core
            players = (core.agent, core.opponent)
            for step_index in range(8000):
                pre = []
                active = env.active_players()
                if ghost is not None and active[ghost]:
                    players[ghost].gold = 0            # END is legal only at zero gold
                    pre.append(["gold", ghost, 0]); bump("edits", "ghost-gold")
                if mode == "mix":
                    live = 1 - ghost
                    if active[live] and rng.random() < 0.02:
                        specs, slots, tag = mix_board(int(players[live].turn), own_boards, by_turn, last_turn, draw, rng, parrots, last_board)
                        pre.append(["team", live, swap_team(players[live], specs, slots)]); bump("edits", "live-team")
                    if active[live] and not active[ghost] and rng.random() < 0.03:
                        value = rng.choice([0, 0, 1, 2, 3, 4, 6, 10, 15])
                        players[live].gold = value
                        pre.append(["gold", live, value]); bump("edits", "live-gold")
                if mode == "rekey" and rng.random() < 1 / 40:
                    new_seed = rng.randrange(2 ** 63)
                    env.redeterminize(new_seed)
                    pre.append(["rekey", new_seed]); bump("edits", "rekey")
                legal = G.legal_of(env); decisions = []
                for side, options in enumerate(legal):
                    if options is None:
                        decisions.append(None); continue
                    if side == ghost:
                        if C.END_TURN not in options:
                            raise RuntimeError("END is not legal for the ghost seat")
                        decisions.append(JointAction(C.decode(C.END_TURN), 0))
                    else:
                        decisions.append(lives[side].decide(env, side, options))
                    local["decisions"] += 1
                if ghost is not None:
                    move = decisions[1 - ghost]
                    if strong and move is not None and int(move.primitive.action_type) == 0 and not active[ghost] and rng.random() < 0.8:
                        specs = rng.choice(by_turn[min(int(players[1 - ghost].turn), last_turn)])
                        pre.append(["legal"])        # the list the live seat chose from, computed above
                        pre.append(["team", 1 - ghost, swap_team(players[1 - ghost], specs)]); bump("edits", "live-team-before-end")
                        legal = G.legal_of(env)      # the list the checker sees: after every edit
                        if C.END_TURN not in legal[1 - ghost]:
                            raise RuntimeError("END stopped being legal after the swap")
                    if move is not None and int(move.primitive.action_type) == 0:
                        # The live seat ends its turn: the battle resolves inside this step.
                        if decisions[ghost] is not None or env.active_players()[ghost]:
                            raise RuntimeError("ghost seat had not ended its turn before the battle")
                        turn = int(players[ghost].turn)
                        if mode == "ghost":
                            specs, own = worker_board(turn, own_boards, by_turn, last_turn, draw)
                            slots, tag = None, "own" if own else "other"
                        else:
                            specs, slots, tag = mix_board(turn, own_boards, by_turn, last_turn, draw, rng, parrots, last_board)
                        last_board = specs
                        pre.append(["team", ghost, swap_team(players[ghost], specs, slots)])
                        bump("edits", "ghost-team"); bump("boards", tag); bump("turns", turn)
                turn_before = int(core.turn)
                step = write_step(buffer, step_index, env, legal, decisions, step_decisions(env, decisions), full, pre, turn_before)
                if step.battle_resolved:
                    local["battles"] += 1
                    if ghost is not None:
                        outcome = int(step.battle_result)
                        local["ghostResults"][2 if outcome == 2 else int(outcome != ghost)] += 1
                    else:
                        bump("turns", turn_before)
                if step.terminated:
                    break
            else:
                raise RuntimeError("match did not end")
        except Exception as error:   # the reference itself failed: keep the evidence, skip the match
            stats["failed"].append({"match": seed, "error": repr(error), "where": traceback.format_exc().strip().splitlines()[-6:]})
            print("reference failed on match", seed, repr(error), file=sys.stderr)
            continue
        out.write(buffer.getvalue())
        stats["scripted"] += int(scripted)
        stats["decisions"] += local["decisions"]; stats["battles"] += local["battles"]
        for table in ("edits", "boards", "turns"):
            for key, value in local[table].items():
                stats[table][key] = stats[table].get(key, 0) + value
        for k in range(3):
            stats["ghostResults"][k] += local["ghostResults"][k]
    stats["turns"] = dict(sorted(stats["turns"].items()))
    return stats


if MAIN:
    MODE, OUT, MATCHES, FIRST, FULL_EVERY = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    if MODE not in ("ghost", "mix", "plain", "rekey"):
        raise SystemExit("mode must be ghost, mix, plain or rekey")
    with gzip.open(OUT, "wt", compresslevel=5) as out:
        summary = play(out, MODE, MATCHES, FIRST, FULL_EVERY)
    summary = {"mode": MODE, "matches": MATCHES, "first": FIRST, "written": MATCHES - len(summary["failed"]), **summary}
    print(json.dumps(summary))
