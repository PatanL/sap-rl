"""Actor worker with a third opponent type: boards ranked players took into battle (training side only).

This is `frozen_wide.worker` (runs/train-eval-20261123, Codex) with these additions, and nothing else changed:
  * a seat whose owner is GHOST never shops: its gold is set to 0 and the coordinator sends END for it;
  * on the step in which the live seat ends its turn (the battle resolves inside that step) the ghost
    seat's team is replaced by a ranked player's board for that turn;
  * one ranked player's own sequence of boards is used for the whole match, turn by turn; once that
    player's game has ended, a board of another player at that turn is drawn;
  * a ghost seat writes no learner trajectory (it is counted with the frozen seats).
Pool: runs/train-ghost-20261123/ghost-train-pool.json (matches with sha256(match id) % 5 >= 2 only).
With GHOST_AUDIT=<dir under /out> every ghost battle is logged with the board that was meant to fight
and the team the battle engine was actually given.
"""
import copy, hashlib, json, os, traceback
import numpy as np
import torch
import frozen_wide as wide
from opponents import LIVE
from runtime import selfplay_env
from sapai import Pet, Team
from sapai.effects import TransferAbility
from sapai.data import data as ENGINE_DATA
from sapai_gym1.rl.action_schema import AutoregressiveAction

GHOST = -3
POOL_PATH = '/sap/runs/train-ghost-20261123/ghost-train-pool.json'
POOL_SHA256 = '55f6219eca9a173f7e288a837c0669aded85a4822fa9f66c0e0a054731a5b702'


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


def summary(team):
    return [[slot.pet.name, int(slot.pet._attack), int(slot.pet._health), slot.pet.status] for slot in team if not slot.empty]


def worker(w, games, width, page, pipe, seed, start_version, profiling, backend, audit_dir):
    try:
        if backend != 'rust' or profiling or audit_dir is not None:
            raise ValueError('ghost worker: Rust battles only, no profiling')
        from rust_g import Bridge
        base = Bridge(None).battle_class()
        seen = {}
        audit = None
        if os.environ.get('GHOST_AUDIT'):
            os.makedirs(os.environ['GHOST_AUDIT'], exist_ok=True)
            audit = open(os.path.join(os.environ['GHOST_AUDIT'], f'ghost-{w}.jsonl'), 'w', buffering=1)

            class Battle(base):
                def __init__(self, *args, **kwargs):
                    seen['teams'] = [summary(team) for team in args[:2]]
                    super().__init__(*args, **kwargs)
            selfplay_env.Battle = Battle
        else:
            selfplay_env.Battle = base
        sequences, by_turn, last_turn = load_pool()
        draw = np.random.default_rng(np.random.SeedSequence([seed, start_version, w, 20261123]))
        torch.set_num_threads(1)
        envs = [selfplay_env.ParallelSelfPlayEnv() for _ in range(games)]
        gens = [0] * games

        def reset(i):
            seq = np.random.SeedSequence([seed, start_version, w, i, gens[i]])
            gens[i] += 1
            world = int(seq.generate_state(1, dtype=np.uint64)[0]) % (2 ** 63)
            return envs[i].reset(seed=world, paired_tape=True)
        obs = [reset(i) for i in range(games)]
        locked = np.full((games, 2), -99, dtype=np.int64)
        fresh = np.ones((games, 2), dtype=bool)
        mem = np.zeros((games, 2, width), 'f4'); initial = mem.copy()
        turns = [[[], []] for _ in envs]
        ghost_boards = [None] * games
        nseg = nrows = matches = 0
        eligible = frozen = 0
        while True:
            active = [(i, s) for i, env in enumerate(envs) for s, yes in enumerate(env.active_players()) if yes]
            n = len(active)
            for k, a in page.obs.items(): np.stack([obs[i][s][k] for i, s in active], out=a[:n])
            page.ids[:n] = [i * 2 + s for i, s in active]
            page.reset[:n] = [fresh[i, s] for i, s in active]
            pipe.send((n, nseg, nrows, matches, eligible, frozen, sum(len(buf) for pair in turns for buf in pair)))
            version = pipe.recv()
            if version is None: break
            moves = [[None, None] for _ in envs]; masks = [[0, 0] for _ in envs]
            eligible = frozen = 0
            for j, (i, s) in enumerate(active):
                action = page.actions[j].copy()
                owner = int(page.owner[j])
                if fresh[i, s]:
                    locked[i, s] = owner
                    if owner == GHOST:
                        ghost_boards[i] = sequences[int(draw.integers(len(sequences)))]
                if locked[i, s] != owner: raise RuntimeError('Opponent changed inside a match')
                if owner == LIVE:
                    turns[i][s].append((obs[i][s], action, float(page.logs[j]), version)); eligible += 1
                else: frozen += 1
                if owner == GHOST:
                    if tuple(map(int, action)) != (0, 10, 5, 0): raise RuntimeError('ghost seat was sent something other than END')
                    core = envs[i].core
                    (core.agent if s == 0 else core.opponent).gold = 0   # END is legal only at zero gold
                moves[i][s] = AutoregressiveAction(*map(int, action[:3])); masks[i][s] = int(action[3])
                mem[i, s] = page.hidden[j]; fresh[i, s] = False
            nseg = nrows = matches = 0
            for i, env in enumerate(envs):
                ghost = 0 if locked[i, 0] == GHOST else 1 if locked[i, 1] == GHOST else None
                expected = None
                if ghost is not None:
                    move = moves[i][1 - ghost]
                    if move is not None and int(move.action_type) == 0:
                        # The live seat ends its turn: the battle resolves inside this step.
                        if moves[i][ghost] is not None or env.active_players()[ghost]:
                            raise RuntimeError('ghost seat had not ended its turn before the battle')
                        core = env.core
                        player = core.agent if ghost == 0 else core.opponent
                        turn = int(player.turn)
                        participation, boards = ghost_boards[i]
                        own = turn in boards
                        specs = boards[turn] if own else by_turn[min(max(turn, 1), last_turn)][int(draw.integers(len(by_turn[min(max(turn, 1), last_turn)])))]
                        player.team = make_team(specs)
                        player._connect_objects()
                        expected = (turn, participation, own, summary(player.team))
                        seen.pop('teams', None)
                step = wide.step_with_end_freeze(env, *moves[i], mask_ids=masks[i]).result
                obs[i] = step.observations
                if expected is not None and not step.battle_resolved:
                    raise RuntimeError('a board was swapped in but no battle followed')
                if ghost is not None and step.battle_resolved and expected is None:
                    raise RuntimeError('a ghost game fought without a swapped-in board')
                if not step.battle_resolved: continue
                if audit is not None and expected is not None:
                    audit.write(json.dumps({'game': i, 'ghostSeat': ghost, 'turn': expected[0], 'participation': expected[1],
                                            'ownSequence': expected[2], 'expected': expected[3], 'fought': seen.get('teams'),
                                            'result': int(step.battle_result), 'terminated': bool(step.terminated)}) + '\n')
                for s in (0, 1):
                    if locked[i, s] != LIVE:
                        if turns[i][s]: raise RuntimeError('Frozen actor produced a learner trajectory')
                        continue
                    label = 2 if step.battle_result == 2 else int(step.battle_result != s)
                    nrows += page.write_turn(turns[i][s], nrows, nseg, initial[i, s], obs[i][s],
                                             step.rewards[s], step.terminated, label, w * page.slots + i * 2 + s)
                    nseg += 1; turns[i][s].clear()
                initial[i] = mem[i]
                if step.terminated:
                    matches += 1; obs[i] = reset(i); fresh[i] = True; mem[i] = 0; initial[i] = 0; ghost_boards[i] = None
    except BaseException:
        try: pipe.send(('error', traceback.format_exc()))
        except (EOFError, BrokenPipeError): pass
        raise
    finally:
        pipe.close()
        try:
            if os.environ.get('GHOST_AUDIT'):
                import sap_engine_rs
                with open(os.path.join(os.environ['GHOST_AUDIT'], f'ghost-{w}-battle-stats.json'), 'w') as fp:
                    json.dump(sap_engine_rs.STATS, fp)
        except Exception:
            pass
