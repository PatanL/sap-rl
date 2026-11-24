"""Checks of the search and ghost primitives of SapVecEnv through the C interface.

  ghost <lib> <games> <matches> <threads>
      Lockstep against the Python reference with ghost_worker.py's edits (the `mix` rules of
      tools/gen_env_ghost.py): after every step both seats' observations byte for byte, rewards,
      battle flag and result, lives, wins, active, terminated, winner, the state dump, and on battle
      steps `battle_events()` against the teams the reference's Battle received and its seed.
  copy <lib> <matches> <threads>
      Rust only. Copies at random steps (mid-turn, right before a battle, after termination):
      (a) copy and source get the same actions until the match ends: all outputs and the state
      dump stay identical; (b) only the copy is stepped: the source's state and output rows do not
      change, and a copy left alone while the source plays on replays the source's recorded steps
      exactly afterwards.
  rekey <lib> <points> <threads> [driver seed]
      Rust only. Copies rekeyed with the same seed stay identical to the end of the match; with
      different seeds (and against the unrekeyed game) the next roll and the next battle seed
      differ; shop offer frequencies by tier of rekeyed and unrekeyed rolls are compared with the
      exact probabilities of the roll pools and with each other (chi-square).
  rolls <lib> <turn> <copies per state> <driver seed>
      Rust only. Six games are played to the given turn; from each, many copies are rekeyed with
      fresh seeds and roll once: the offered pets and foods by name against the uniform roll pool
      of that turn (chi-square, independent draws by construction).
  eval <lib> <points> <threads> [engine data dir]
      `eval_vs` against python/sap_engine_rs.py fights of the same teams with the same seeds;
      the game's state dump and output rows before and after; the real battle of the exported
      teams with the exported seed; a share of the fights also on the Python battle engine.
  bench <lib> [threads]
      Cost of `copy` per game and `eval_vs` fights per second.
  shopset <lib> <games> <matches> <threads> [first match index]
      Lockstep against the Python reference with the edits of tools/gen_env_shopset.py made through
      `set_shop_item` (by name and by id), `set_team` and `set_gold`: everything `ghost` compares,
      the state dump right after the edits of a step, `observe()` right after them against a Python
      observation taken at the same point, and refused calls at random steps (ValueError, the game
      unchanged, the reference edit refuses the same call).
  shopsame <lib> <matches> <threads>
      A freshly rolled item replaced by itself (`set_shop_item` of the same name; in Python the
      reference edit) must be indistinguishable from the rolled one: Python plain, Python with
      replacements, Rust plain and Rust with replacements get the same actions and must agree at
      every step to the end of the match.

Run in the training image: tools/run_ref.sh <cpu> /rs/python/verify_search.py <mode> /rs/target/release/libsap_engine.so ...
"""
import ctypes, json, math, random, sys, time
import numpy as np
from sap_env_rs import SapVecEnv, NO_DECISION, EMPTY_ROW, eval_seed

MODE, LIB = sys.argv[1], sys.argv[2]
END, ROLL = 0, 151
OUTPUT_NAMES = ("rewards", "battle_resolved", "battle_result", "terminated", "winner", "lives", "wins", "active")
STATE_NAMES = ("terminated", "winner", "lives", "wins", "active")


def triples():
    from sapai_gym1.rl.action_schema import FlatActionCodec as C
    return np.array([C.decode(flat).as_tuple() for flat in range(197)], dtype=np.int64)


def fail(what, detail=""):
    print(f"MISMATCH {what} {detail}")
    sys.exit(1)


def rows(env, game, names=OUTPUT_NAMES):
    """Every output row of one game as bytes."""
    parts = [env.obs[key][game].tobytes() for key in env.obs]
    parts += [np.ascontiguousarray(getattr(env, name)[game]).tobytes() for name in names]
    return b"".join(parts)


def event(env, game):
    turn, seed, teams, carried = env.battle_events(carried=True)
    return (int(turn[game]), int(seed[game]), teams[game].tobytes(), carried[game].tobytes())


class Driver:
    """Random legal play on the Rust side: an action type first, then one of its legal actions."""

    def __init__(self, env, seed):
        self.env, self.rng, self.triples = env, random.Random(seed), triples()
        self.types = self.triples[:, 0]

    def choose(self, game, rng=None, want=None):
        """Actions [2, 3] and masks [2] for one game; `want` forces a flat action per seat if legal."""
        rng = rng or self.rng
        actions = np.full((2, 3), NO_DECISION, dtype=np.int64); actions[:, 1] = 10; actions[:, 2] = 5
        flats = [None, None]; masks = np.zeros(2, dtype=np.int64)
        for seat in (0, 1):
            if not self.env.active[game, seat]:
                continue
            legal = np.flatnonzero(self.env.legal(game, seat))
            flat = None
            if want is not None and want[seat] is not None and want[seat] in legal:
                flat = want[seat]
            if flat is None:
                kinds = sorted(set(self.types[legal].tolist()))
                kind = rng.choice(kinds)
                flat = int(rng.choice([f for f in legal.tolist() if self.types[f] == kind]))
            actions[seat] = self.triples[flat]; flats[seat] = flat
            if flat == END and rng.random() < 0.5:
                masks[seat] = self.freeze_plan(game, seat, legal, rng)
        return actions, masks, flats

    def freeze_plan(self, game, seat, legal, rng):
        """A random legal END freeze mask (0 when the seat has gold left or no budget)."""
        state = self.env.state(game)["seats"][seat]
        if state["gold"] != 0:
            return 0
        mask = count = 0
        for index, slot in enumerate(state["shop"][:10]):
            if slot[1] is None or rng.random() >= 0.25:
                continue
            if (162 if slot[2] else 152) + index not in legal:
                return 0
            mask |= 1 << index; count += 1
        return mask if count <= max(0, 50 - state["actions"]) else 0


def step_games(env, games, actions, masks):
    """Step only `games` (dict game -> (actions, masks))."""
    n = env.num_games
    a = np.full((n, 2, 3), NO_DECISION, dtype=np.int64); a[:, :, 1] = 10; a[:, :, 2] = 5
    m = np.zeros((n, 2), dtype=np.int64); which = np.zeros(n, dtype=bool)
    for game in games:
        a[game] = actions[game]; m[game] = masks[game]; which[game] = True
    env.step(a, m, which=which)
    for game in games:
        if env.status[game] < 0:
            fail("status", f"game {game}: {env.status[game]} {env.error(game)}")


# ------------------------------------------------------------------------------------------------
# ghost: lockstep with the Python reference through the C interface
# ------------------------------------------------------------------------------------------------

def run_ghost():
    import gen_env_ghost as H
    import gen_env_golden as G
    from sapai_gym1.rl.action_schema import FlatActionCodec as C
    from sapai_gym1.rl.joint_action import JointAction, step_decisions
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    games, matches, threads = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    G.EXT = True
    targeted = H.load_targeted()
    sequences, by_turn, last_turn = H.load_pool()
    parrots = [b for boards in by_turn.values() for b in boards if any(s["id"] == "pet-parrot" for s in b)]
    H.selfplay_env.Battle = H.RecordingBattle
    env = SapVecEnv(games, threads=threads, library=LIB)
    py = [ParallelSelfPlayEnv() for _ in range(games)]
    info = [None] * games
    counts = {"steps": 0, "arrays": 0, "battles": 0, "matches": 0, "set_team": 0, "set_gold": 0, "states": 0, "ghost_seat": [0, 0],
              "terminal_battles": 0}
    next_index = [0]

    def board_ids(board):
        out = np.tile(np.array(EMPTY_ROW, dtype=np.int32), (5, 1))
        for slot, row in enumerate(board):
            if row is None:
                continue
            copies = -2 if row[6] is False else -1 if row[6] is None else env.kind_id(row[6])
            out[slot] = (env.kind_id(row[0]), row[1], row[2], row[3], env.status_id(row[4]), row[5], copies)
        return out

    def compare(game, step):
        for seat in (0, 1):
            for key, value in info[game]["obs"][seat].items():
                mine = env.obs[key][game, seat]
                if mine.dtype != value.dtype or mine.tobytes() != np.ascontiguousarray(value).tobytes():
                    fail("observation " + key, f"game {game} seed {info[game]['seed']} step {info[game]['steps']} seat {seat}")
                counts["arrays"] += 1
        core = py[game].core
        reference = G.canonical(core)
        mine = env.state(game)
        for seat in (0, 1):
            a, b = dict(reference["seats"][seat]), dict(mine["seats"][seat])
            if len(a.pop("toggled")) != len(b.pop("toggled")) or a != b:
                diff = [k for k in a if a[k] != b.get(k)]
                fail("state", f"game {game} seed {info[game]['seed']} step {info[game]['steps']} seat {seat}: {diff}")
        if reference["turn"] != mine["turn"]:
            fail("state turn", f"game {game}")
        counts["states"] += 1
        if [int(x) for x in env.lives[game]] != [int(core.agent.lives), int(core.opponent.lives)] or \
           [int(x) for x in env.wins[game]] != [int(core.agent.wins), int(core.opponent.wins)]:
            fail("lives or wins", f"game {game}")

    def reset(game):
        index = next_index[0]; next_index[0] += 1
        seed = 885500000 + index
        rng = random.Random(seed)
        draw = np.random.default_rng(np.random.SeedSequence([seed, 20261123]))
        ghost = index % 2
        live = H.Live(targeted, rng, (index // 2) % 2 == 1, rng.choice([0, 0, 0, 4]))
        obs = py[game].reset(seed=seed, paired_tape=True)
        info[game] = {"seed": seed, "rng": rng, "draw": draw, "ghost": ghost, "live": live, "obs": obs, "steps": 0,
                      "own": sequences[int(draw.integers(len(sequences)))][1], "last": None, "strong": (index // 4) % 2 == 1}
        which = np.zeros(games, dtype=bool); which[game] = True
        seeds = np.zeros(games, dtype=np.int64); seeds[game] = seed
        env.reset(seeds, which=which)
        compare(game, None)
        counts["ghost_seat"][ghost] += 1

    for game in range(games):
        reset(game)
    live_games = set(range(games))
    started = time.time()
    while live_games:
        actions = np.full((games, 2, 3), NO_DECISION, dtype=np.int64); actions[:, :, 1] = 10; actions[:, :, 2] = 5
        masks = np.zeros((games, 2), dtype=np.int64); which = np.zeros(games, dtype=bool)
        decisions = {}
        for game in sorted(live_games):
            state = info[game]; rng = state["rng"]; ghost = state["ghost"]; live = 1 - ghost
            core = py[game].core; players = (core.agent, core.opponent)
            active = py[game].active_players()
            if [bool(x) for x in env.active[game]] != list(active):
                fail("active", f"game {game}")
            if active[ghost]:
                players[ghost].gold = 0; env.set_gold(game, ghost, 0); counts["set_gold"] += 1
            if active[live] and rng.random() < 0.02:
                specs, slots, _ = H.mix_board(int(players[live].turn), state["own"], by_turn, last_turn, state["draw"], rng, parrots, state["last"])
                env.set_team(game, live, board_ids(H.swap_team(players[live], specs, slots))); counts["set_team"] += 1
            if active[live] and not active[ghost] and rng.random() < 0.03:
                value = rng.choice([0, 0, 1, 2, 3, 4, 6, 10, 15])
                players[live].gold = value; env.set_gold(game, live, value); counts["set_gold"] += 1
            legal = G.legal_of(py[game])
            for seat in (0, 1):
                if legal[seat] is not None and legal[seat] != np.flatnonzero(env.legal(game, seat)).tolist():
                    fail("legal actions", f"game {game} seat {seat}")
            pair = [None, None]
            for seat, options in enumerate(legal):
                if options is None:
                    continue
                pair[seat] = JointAction(C.decode(C.END_TURN), 0) if seat == ghost else state["live"].decide(py[game], seat, options)
            move = pair[live]
            if move is not None and int(move.primitive.action_type) == 0 and not active[ghost]:
                if state["strong"] and rng.random() < 0.8:
                    specs = rng.choice(by_turn[min(int(players[live].turn), last_turn)])
                    env.set_team(game, live, board_ids(H.swap_team(players[live], specs))); counts["set_team"] += 1
                turn = int(players[ghost].turn)
                if rng.random() < 0.5:
                    specs, _ = H.worker_board(turn, state["own"], by_turn, last_turn, state["draw"]); slots = None
                else:
                    specs, slots, _ = H.mix_board(turn, state["own"], by_turn, last_turn, state["draw"], rng, parrots, state["last"])
                state["last"] = specs
                env.set_team(game, ghost, board_ids(H.swap_team(players[ghost], specs, slots))); counts["set_team"] += 1
            for seat, d in enumerate(pair):
                if d is not None:
                    actions[game, seat] = d.primitive.as_tuple(); masks[game, seat] = int(d.end_freeze_mask or 0)
            decisions[game] = pair; which[game] = True
        env.step(actions, masks, which=which)
        turns, seeds, teams, carried = env.battle_events(carried=True)
        done = []
        for game in sorted(live_games):
            state = info[game]
            turn_before = int(py[game].core.turn)
            step = step_decisions(py[game], decisions[game]).result
            state["obs"] = step.observations; state["steps"] += 1
            if env.status[game] != 0:
                fail("status", f"game {game}: {env.status[game]} {env.error(game)}")
            compare(game, step)
            if tuple(float(x) for x in env.rewards[game]) != tuple(float(x) for x in step.rewards) or \
               np.signbit(env.rewards[game]).tolist() != [bool(np.signbit(x)) for x in step.rewards]:
                fail("rewards", f"game {game}")
            if bool(env.battle_resolved[game]) != bool(step.battle_resolved) or bool(env.terminated[game]) != bool(step.terminated):
                fail("flags", f"game {game}")
            if step.battle_resolved:
                if int(env.battle_result[game]) != int(step.battle_result):
                    fail("battle result", f"game {game}")
                want = np.stack([board_ids(board) for board in H.SEEN.pop("teams")])
                if int(turns[game]) != turn_before or int(seeds[game]) != int(py[game]._paired_battle_seeds[-1][1]) or \
                   not np.array_equal(teams[game], want) or carried[game].tolist() != H.SEEN.pop("carried"):
                    fail("battle event", f"game {game} seed {state['seed']} step {state['steps']}")
                counts["battles"] += 1; counts["terminal_battles"] += bool(step.terminated)
            elif int(turns[game]) != 0:
                fail("battle event without a battle", f"game {game}")
            counts["steps"] += 1
            if step.terminated:
                if int(env.winner[game]) != int(G._completed_match_result(py[game].core)):
                    fail("winner", f"game {game}")
                counts["matches"] += 1; done.append(game)
        for game in done:
            if next_index[0] < matches:
                reset(game)
            else:
                live_games.discard(game)
    print("ALL EQUAL ghost", json.dumps(counts), f"{time.time() - started:.0f}s")


# ------------------------------------------------------------------------------------------------
# copy
# ------------------------------------------------------------------------------------------------

def run_copy():
    matches, threads = int(sys.argv[3]), int(sys.argv[4])
    block = 16
    # Slots: [0, block) sources, [block, 2 block) lockstep copies, [2 block, 3 block) copies stepped
    # on their own, [3 block, 4 block) copies left alone and replayed later.
    env = SapVecEnv(block * 4, threads=threads, library=LIB)
    driver = Driver(env, 20261123)
    rng = driver.rng
    seeds = np.zeros(env.num_games, dtype=np.int64)
    counts = {"matches": 0, "source_steps": 0, "copies": 0, "copy_mid_turn": 0, "copy_before_battle": 0, "copy_after_termination": 0,
              "copy_one_seat_ended": 0, "lockstep_steps": 0, "lockstep_to_the_end": 0, "lockstep_battles": 0,
              "source_untouched_checks": 0, "solo_copy_steps": 0, "delayed_replays": 0, "delayed_steps": 0, "chained_copies": 0,
              "end_freeze_plans": 0, "lockstep_end_freeze_plans": 0}
    next_seed = [886600000]
    twin = [None] * block       # lockstep copy state: remaining steps or None
    delayed = [None] * block    # recorded (actions, masks, digest) since the copy was taken
    last_action = {}

    def digest(game):
        return (rows(env, game), json.dumps(env.state(game), sort_keys=True), event(env, game))

    def equal(a, b, what):
        if rows(env, a) != rows(env, b):
            for key in env.obs:
                if env.obs[key][a].tobytes() != env.obs[key][b].tobytes():
                    fail(what, f"observation {key} of games {a} and {b}")
            for name in OUTPUT_NAMES:
                if np.any(getattr(env, name)[a] != getattr(env, name)[b]):
                    fail(what, f"{name} of games {a} and {b}")
        if env.state(a) != env.state(b):
            fail(what, f"state of games {a} and {b}")
        if event(env, a) != event(env, b):
            fail(what, f"battle event of games {a} and {b}")

    def reset(game):
        which = np.zeros(env.num_games, dtype=bool); which[game] = True
        seeds[game] = next_seed[0]; next_seed[0] += 1
        env.reset(seeds, which=which)
        twin[game] = None; delayed[game] = None

    def kind_of_point(game, flats):
        if env.terminated[game]:
            return "copy_after_termination"
        active = [bool(x) for x in env.active[game]]
        if sum(active) == 1 and flats is not None and flats[active.index(True)] == END:
            return "copy_before_battle"
        if sum(active) == 1:
            return "copy_one_seat_ended"
        return "copy_mid_turn"

    for game in range(block):
        reset(game)
    live = set(range(block)); started_matches = block
    while live:
        # Decide the sources' actions first, so "right before a battle" is known at copy time.
        planned = {game: driver.choose(game) for game in live}
        for game in sorted(live):
            actions, masks, flats = planned[game]
            if twin[game] is None and rng.random() < 0.03:
                # (a) a lockstep copy, compared at once and after every step from now on
                env.copy([game], [block + game]); counts["copies"] += 1; counts[kind_of_point(game, flats)] += 1
                equal(game, block + game, "copy")
                if env.status[block + game] != 0:
                    fail("status of a copy")
                twin[game] = rng.choice([5, 20, 80, 10 ** 9])
            if rng.random() < 0.01:
                # (b) a copy stepped on its own; the source must not notice
                solo = 2 * block + game
                before = digest(game)
                env.copy([game], [solo]); counts["copies"] += 1; counts[kind_of_point(game, flats)] += 1
                equal(game, solo, "copy")
                other = random.Random(rng.random())
                for _ in range(rng.choice([1, 3, 10, 40])):
                    if env.terminated[solo]:
                        break
                    a, m, _f = driver.choose(solo, rng=other)
                    step_games(env, [solo], {solo: a}, {solo: m}); counts["solo_copy_steps"] += 1
                if digest(game) != before:
                    fail("source changed by stepping its copy", f"game {game}")
                counts["source_untouched_checks"] += 1
                # a copy of a copy (chained) is a copy of the source as well
                if rng.random() < 0.3:
                    env.copy([game], [solo]); env.copy([solo], [3 * block + game]) if delayed[game] is None else None
                    if delayed[game] is None:
                        equal(game, 3 * block + game, "chained copy"); counts["chained_copies"] += 1
                        delayed[game] = []
            if delayed[game] is None and rng.random() < 0.01:
                # (b') a copy left alone while the source plays on
                env.copy([game], [3 * block + game]); counts["copies"] += 1; counts[kind_of_point(game, flats)] += 1
                delayed[game] = []
        # Step sources and their lockstep copies together with identical actions.
        stepping, acts, msks = [], {}, {}
        for game in sorted(live):
            actions, masks, flats = planned[game]
            stepping.append(game); acts[game] = actions; msks[game] = masks
            counts["end_freeze_plans"] += int((masks != 0).sum())
            counts["lockstep_end_freeze_plans"] += int((masks != 0).sum()) if twin[game] is not None else 0
            if twin[game] is not None:
                stepping.append(block + game); acts[block + game] = actions; msks[block + game] = masks
        frozen = {game: digest(3 * block + game) for game in live if delayed[game] is not None and rng.random() < 0.05}
        step_games(env, stepping, acts, msks)
        for game, before in frozen.items():
            if digest(3 * block + game) != before:
                fail("copy changed by stepping its source", f"game {game}")
            counts["source_untouched_checks"] += 1
        for game in sorted(live):
            counts["source_steps"] += 1
            if twin[game] is not None:
                equal(game, block + game, "lockstep")
                counts["lockstep_steps"] += 1; counts["lockstep_battles"] += int(env.battle_resolved[game])
                twin[game] -= 1
                if env.terminated[game]:
                    counts["lockstep_to_the_end"] += 1; twin[game] = None
                elif twin[game] <= 0:
                    twin[game] = None
            if delayed[game] is not None:
                delayed[game].append((planned[game][0], planned[game][1], digest(game)))
                if len(delayed[game]) >= 25 or env.terminated[game]:
                    late = 3 * block + game
                    for a, m, want in delayed[game]:
                        step_games(env, [late], {late: a}, {late: m})
                        if digest(late) != want:
                            fail("delayed replay of a copy", f"game {game}")
                        counts["delayed_steps"] += 1
                    counts["delayed_replays"] += 1; delayed[game] = None
            if env.terminated[game]:
                # A finished game copies like any other: same rows, same state, and a step is skipped.
                if rng.random() < 0.5:
                    spare = block + game
                    env.copy([game], [spare]); counts["copies"] += 1; counts["copy_after_termination"] += 1
                    equal(game, spare, "copy of a finished game")
                    a, m, _f = driver.choose(spare)
                    before = (rows(env, spare, STATE_NAMES), env.state(spare), event(env, spare))
                    if before[2] != event(env, game) or before[2][0] == 0:
                        fail("battle event of a finished game", f"game {game}")
                    n = env.num_games
                    aa = np.full((n, 2, 3), NO_DECISION, dtype=np.int64); mm = np.zeros((n, 2), dtype=np.int64)
                    which = np.zeros(n, dtype=bool); which[spare] = True
                    env.step(aa, mm, which=which)
                    if env.status[spare] != 1 or (rows(env, spare, STATE_NAMES), env.state(spare), event(env, spare)) != before \
                       or env.battle_resolved[spare] != 0:
                        fail("step of a copied finished game", f"game {game}")
                counts["matches"] += 1
                if started_matches < matches:
                    reset(game); started_matches += 1
                else:
                    live.discard(game)
    # Refused calls change nothing.
    env2 = SapVecEnv(4, threads=1, library=LIB)
    env2.reset(np.array([1, 2, 3, 4], dtype=np.int64), which=np.array([1, 1, 0, 0], dtype=bool))
    before = [json.dumps(env2.state(g)) for g in (0, 1)]
    refused = 0
    for src, dst in (([0], [0]), ([0, 1], [1, 2]), ([0, 0], [2, 2]), ([2], [3]), ([0], [4]), ([-1], [1]), ([0, 1], [2])):
        try:
            env2.copy(src, dst)
        except ValueError:
            refused += 1
    if refused != 7 or [json.dumps(env2.state(g)) for g in (0, 1)] != before or env2.status[2] != -3:
        fail("refused copy calls", str(refused))
    env2.copy([0, 0], [2, 3])
    if env2.state(2) != env2.state(0) or env2.state(3) != env2.state(0):
        fail("one source copied twice")
    counts["refused_calls"] = refused
    print("ALL EQUAL copy", json.dumps(counts))


# ------------------------------------------------------------------------------------------------
# rekey
# ------------------------------------------------------------------------------------------------

def chi_square_p(chi2, df):
    """Upper tail of the chi-square distribution (regularized incomplete gamma, series / continued fraction)."""
    if df <= 0:
        return 1.0
    a, x = df / 2.0, chi2 / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:
        term = total = 1.0 / a
        for n in range(1, 500):
            term *= x / (a + n); total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d
    for n in range(1, 500):
        an = -n * (n - a); b += 2
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d; delta = d * c; h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return min(1.0, h * math.exp(-x + a * math.log(x) - math.lgamma(a)))


def run_rekey():
    points, threads = int(sys.argv[3]), int(sys.argv[4])
    tables = json.load(open("/rs/data/turtle_v31_env.json"))
    tier = {name: entry["tier"] for name, entry in json.load(open("/rs/data/turtle_v31.json"))["pets"].items()}
    block = 16
    # Slots: sources, then per source: unrekeyed copy U, rekeyed A1, A2 (same seed), B (another seed).
    env = SapVecEnv(block * 5, threads=threads, library=LIB)
    driver_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 97
    driver = Driver(env, driver_seed)
    rng = driver.rng
    seeds = np.zeros(env.num_games, dtype=np.int64)
    next_seed = [887700000 + 100000 * (driver_seed % 1000)]
    counts = {"points": 0, "same_seed_identical_steps": 0, "same_seed_to_the_end": 0, "same_seed_battles": 0,
              "roll_differs_other_seed": 0, "roll_differs_from_unrekeyed": 0, "roll_same_as_unrekeyed": 0, "rolls_compared": 0,
              "battle_seed_pairs": 0, "battle_seed_differs_other_seed": 0, "battle_seed_differs_from_unrekeyed": 0,
              "rekey_changes_nothing_visible": 0}
    tiers = {"unrekeyed": {}, "rekeyed": {}}       # rule turn -> tier -> count of offered pets
    foods = {"unrekeyed": {}, "rekeyed": {}}       # rule turn -> food -> count

    def reset(game):
        which = np.zeros(env.num_games, dtype=bool); which[game] = True
        seeds[game] = next_seed[0]; next_seed[0] += 1
        env.reset(seeds, which=which)

    def shop(game, seat):
        return env.state(game)["seats"][seat]["shop"]

    def tally(arm, game, seat, frozen_before):
        state = env.state(game)["seats"][seat]
        rule = str(min(state["turn"], 11))
        for index, slot in enumerate(state["shop"]):
            if slot[2] or slot[1] is None:        # frozen items were not rolled
                continue
            if slot[0] == "food":
                foods[arm].setdefault(rule, {}).setdefault(slot[1], 0); foods[arm][rule][slot[1]] += 1
            else:
                tiers[arm].setdefault(rule, {}).setdefault(tier[slot[1]], 0); tiers[arm][rule][tier[slot[1]]] += 1

    def next_battle_seed(game, own):
        """Play `game` with its own random actions to its next battle; return (turn, seed)."""
        for _ in range(400):
            a, m, _f = driver.choose(game, rng=own, want=[END, END])
            step_games(env, [game], {game: a}, {game: m})
            if env.battle_resolved[game]:
                turn, seed, _t, _c = env.battle_events(carried=True)
                return int(turn[game]), int(seed[game])
        fail("no battle within 400 steps")

    for game in range(block):
        reset(game)
    while counts["points"] < points:
        for game in range(block):
            if env.terminated[game]:
                reset(game)
            a, m, flats = driver.choose(game)
            seat = next((s for s in (0, 1) if env.active[game, s] and env.legal(game, s)[ROLL]), None)
            if seat is not None and rng.random() < 0.08 and counts["points"] < points:
                counts["points"] += 1
                u, a1, a2, b = (block * k + game for k in (1, 2, 3, 4))
                env.copy([game] * 4, [u, a1, a2, b])
                key_a, key_b = rng.randrange(2 ** 63), rng.randrange(2 ** 63)
                visible = [(rows(env, g, STATE_NAMES), env.state(g)["seats"][0]["shop"], env.state(g)["seats"][1]["team"]) for g in (a1, b)]
                env.rekey([a1, a2, b], [key_a, key_a, key_b])
                which = np.zeros(env.num_games, dtype=bool); which[[a1, b]] = True
                env.observe(which)
                if visible != [(rows(env, g, STATE_NAMES), env.state(g)["seats"][0]["shop"], env.state(g)["seats"][1]["team"]) for g in (a1, b)] \
                   or rows(env, a1, STATE_NAMES) != rows(env, u, STATE_NAMES):
                    fail("rekey changed something visible", f"game {game}")
                counts["rekey_changes_nothing_visible"] += 1
                # The same roll in all four: the seat rolls, the other seat (if active) plays what the source plays.
                roll, _m, _f = driver.choose(game, want=[ROLL if s == seat else flats[s] for s in (0, 1)])
                group = [u, a1, a2, b]
                step_games(env, group, {g: roll for g in group}, {g: np.zeros(2, dtype=np.int64) for g in group})
                if rows(env, a1) != rows(env, a2) or env.state(a1) != env.state(a2):
                    fail("same rekey seed, different roll", f"game {game}")
                counts["rolls_compared"] += 1
                counts["roll_differs_other_seed"] += shop(a1, seat) != shop(b, seat)
                same = shop(a1, seat) == shop(u, seat)
                counts["roll_differs_from_unrekeyed"] += not same; counts["roll_same_as_unrekeyed"] += same
                tally("rekeyed", a1, seat, None); tally("rekeyed", b, seat, None)
                # More rekeyed rolls for the frequency comparison. (The unrekeyed side is tallied from
                # the sources' own rolls below: a copy that looks ahead on the match tape would count
                # the same tape entries again at the next point of the same game.)
                for _ in range(3):
                    if not all(env.active[g, seat] and env.legal(g, seat)[ROLL] for g in (a1, b)):
                        break
                    for g in (a1, b):
                        step_one, _m, _f = driver.choose(g, want=[ROLL if s == seat else END for s in (0, 1)])
                        step_games(env, [g], {g: step_one}, {g: np.zeros(2, dtype=np.int64)})
                        tally("rekeyed", g, seat, None)
                env.copy([a2], [a1])     # A1 went its own way above; A2 is still at the first roll
                # Same seed twice: identical to the end of the match under identical actions.
                if rng.random() < 0.35:
                    for _ in range(4000):
                        if env.terminated[a1]:
                            counts["same_seed_to_the_end"] += 1
                            break
                        act, msk, _f = driver.choose(a1)
                        step_games(env, [a1, a2], {a1: act, a2: act}, {a1: msk, a2: msk})
                        if rows(env, a1) != rows(env, a2) or env.state(a1) != env.state(a2) or event(env, a1) != event(env, a2):
                            fail("same rekey seed diverged", f"game {game}")
                        counts["same_seed_identical_steps"] += 1; counts["same_seed_battles"] += int(env.battle_resolved[a1])
                else:
                    # Battle seeds: other seed and unrekeyed differ at the same turn.
                    own = random.Random(rng.random())
                    env.copy([game], [a1]); env.rekey([a1], [key_a])
                    env.copy([game], [b]); env.rekey([b], [key_b])
                    env.copy([game], [u])
                    got = {g: next_battle_seed(g, random.Random(own.random())) for g in (u, a1, b)}
                    if got[u][0] == got[a1][0] == got[b][0]:
                        counts["battle_seed_pairs"] += 1
                        counts["battle_seed_differs_other_seed"] += got[a1][1] != got[b][1]
                        counts["battle_seed_differs_from_unrekeyed"] += got[a1][1] != got[u][1]
            step_games(env, [game], {game: a}, {game: m})
            for s in (0, 1):
                if flats[s] == ROLL:
                    tally("unrekeyed", game, s, None)

    # Frequencies against the exact probabilities of the roll pool of each rule turn, and arm against
    # arm. Cells with an expected count below 5 are merged into one cell per rule turn (the
    # chi-square approximation needs that); a rule turn left with one cell is skipped.
    food_tier = {name: int(t) for t, names in tables["foodTierStd"].items() for name in names}
    report = {}

    def merged(expected, observed_list):
        """expected: label -> expected count (of the first list's total); returns label groups."""
        small = [label for label, e in expected.items() if e < 5]
        groups = [[label] for label in expected if label not in small]
        if small:
            if sum(expected[label] for label in small) >= 5 or not groups:
                groups.append(small)
            else:
                groups[-1] = groups[-1] + small
        return groups

    for name, table, key, weights, label_of in (("pet tier", tiers, "availPets", "pp", lambda item: tier[item]),
                                                ("food tier", foods, "availFoods", "fp", lambda item: food_tier[item]),
                                                ("food", foods, "availFoods", "fp", lambda item: item)):
        def relabel(seen):
            out = {}
            for item, n in seen.items():
                label = item if name == "pet tier" else label_of(item)
                out[label] = out.get(label, 0) + n
            return out
        for arm in ("unrekeyed", "rekeyed"):
            chi2 = 0.0; df = 0; total = 0; detail = {}
            for rule, raw in sorted(table[arm].items(), key=lambda kv: int(kv[0])):
                seen = relabel(raw)
                pool = tables["turns"][rule][key]; w = tables["turns"][rule][weights]
                share = {}
                for item, weight in zip(pool, w):
                    share[label_of(item)] = share.get(label_of(item), 0.0) + weight / sum(w)
                n = sum(seen.values()); total += n
                if set(seen) - set(share):
                    fail("an offer outside the roll pool", f"{arm} rule {rule} {set(seen) - set(share)}")
                groups = merged({label: n * p for label, p in share.items()}, None)
                if len(groups) < 2:
                    continue
                part = 0.0
                for group in groups:
                    e = sum(n * share[label] for label in group); o = sum(seen.get(label, 0) for label in group)
                    part += (o - e) ** 2 / e
                chi2 += part; df += len(groups) - 1
                detail[rule] = [n, round(part, 1), len(groups) - 1,
                                {str(label): [seen.get(label, 0), round(n * share[label], 1)] for label in sorted(share, key=str)}]
            report[f"{name}: {arm} vs exact pool probabilities"] = {"offers": total, "chi2": round(chi2, 2), "df": df, "p": round(chi_square_p(chi2, df), 4)}
            if "--detail" in sys.argv:
                report[f"{name}: {arm} vs exact pool probabilities"]["per rule turn [offers, chi2, df, {label: [seen, expected]}]"] = detail
        chi2 = 0.0; df = 0
        for rule in sorted(set(table["unrekeyed"]) & set(table["rekeyed"]), key=int):
            a, b = relabel(table["unrekeyed"][rule]), relabel(table["rekeyed"][rule])
            na, nb = sum(a.values()), sum(b.values())
            labels = sorted(set(a) | set(b), key=str)
            pooled = {label: (a.get(label, 0) + b.get(label, 0)) / (na + nb) for label in labels}
            groups = merged({label: min(na, nb) * pooled[label] for label in labels}, None)
            if len(groups) < 2:
                continue
            for group in groups:
                share = sum(pooled[label] for label in group)
                oa = sum(a.get(label, 0) for label in group); ob = sum(b.get(label, 0) for label in group)
                chi2 += (oa - na * share) ** 2 / (na * share) + (ob - nb * share) ** 2 / (nb * share)
            df += len(groups) - 1
        report[f"{name}: rekeyed vs unrekeyed (homogeneity)"] = {"chi2": round(chi2, 2), "df": df, "p": round(chi_square_p(chi2, df), 4)}
    by_tier = {arm: {} for arm in tiers}
    for arm in tiers:
        for rule, seen in tiers[arm].items():
            for label, n in seen.items():
                by_tier[arm][label] = by_tier[arm].get(label, 0) + n
    shares = {arm: {str(k): round(v / sum(by_tier[arm].values()), 4) for k, v in sorted(by_tier[arm].items())} for arm in by_tier}
    problems = [k for k, v in report.items() if v["p"] < 0.001]
    if counts["roll_differs_other_seed"] < 0.95 * counts["rolls_compared"] or counts["battle_seed_differs_other_seed"] != counts["battle_seed_pairs"] \
       or counts["battle_seed_differs_from_unrekeyed"] != counts["battle_seed_pairs"] or problems:
        fail("rekey statistics", json.dumps({"counts": counts, "report": report}))
    print("ALL EQUAL rekey", json.dumps(counts))
    print("pet offers by tier, share over all rolls:", json.dumps(shares))
    for key, value in report.items():
        print(" ", key, json.dumps(value))


def run_rolls():
    turn, copies, seed = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    tables = json.load(open("/rs/data/turtle_v31_env.json"))
    env = SapVecEnv(1 + 256, threads=1, library=LIB)
    driver = Driver(env, seed); rng = driver.rng
    total = {"food": {}, "pet": {}}; states = 0
    seeds = np.zeros(env.num_games, dtype=np.int64)
    while states < 6:
        seeds[0] = 891000000 + rng.randrange(10 ** 6)
        which = np.zeros(env.num_games, dtype=bool); which[0] = True
        env.reset(seeds, which=which)
        seat = None
        for _ in range(5000):
            if env.terminated[0]:
                break
            state = env.state(0)
            if state["turn"] == turn:
                seat = next((s for s in (0, 1) if env.active[0, s] and env.legal(0, s)[ROLL]), None)
                if seat is not None:
                    break
            a, m, _f = driver.choose(0)
            step_games(env, [0], {0: a}, {0: m})
        if seat is None:
            continue
        states += 1
        roll, _m, _f = driver.choose(0, want=[ROLL if s == seat else None for s in (0, 1)])
        done = 0
        while done < copies:
            k = min(256, copies - done)
            dst = list(range(1, 1 + k))
            env.copy([0] * k, dst)
            env.rekey(dst, [rng.randrange(2 ** 63) for _ in dst])
            step_games(env, dst, {g: roll for g in dst}, {g: np.zeros(2, dtype=np.int64) for g in dst})
            for g in dst:
                for slot in env.state(g)["seats"][seat]["shop"]:
                    if slot[2] or slot[1] is None:        # frozen items were not rolled
                        continue
                    kind = "food" if slot[0] == "food" else "pet"
                    total[kind][slot[1]] = total[kind].get(slot[1], 0) + 1
            done += k
    rule = str(min(turn, 11)); out = {"turn": turn, "states": states, "rekeyed_copies": states * copies}
    for kind, key in (("food", "availFoods"), ("pet", "availPets")):
        pool = tables["turns"][rule][key]; seen = total[kind]; n = sum(seen.values())
        if set(seen) - set(pool):
            fail("an offer outside the roll pool", str(set(seen) - set(pool)))
        chi2 = sum((seen.get(x, 0) - n / len(pool)) ** 2 / (n / len(pool)) for x in pool)
        out[kind] = {"offers": n, "pool": len(pool), "chi2": round(chi2, 2), "df": len(pool) - 1, "p": round(chi_square_p(chi2, len(pool) - 1), 4)}
    if min(out["food"]["p"], out["pet"]["p"]) < 0.001:
        fail("rekeyed rolls are not uniform over the pool", json.dumps(out))
    print("ALL EQUAL rolls", json.dumps(out))


# ------------------------------------------------------------------------------------------------
# eval_vs
# ------------------------------------------------------------------------------------------------

def state_board(env, team):
    """A seat's team of the state dump as a board (what the battle would be handed)."""
    board = np.tile(np.array(EMPTY_ROW, dtype=np.int32), (5, 1))
    carried = False
    for slot, row in enumerate(team):
        if row is None:
            continue
        name, attack, health, tmp_attack, tmp_health, level, experience, status, counter, _sell, friends_hurt, hurt, copied = row
        copies = -2
        if name == "pet-parrot":
            copies = -1 if copied is None else env.kind_id(copied)
        board[slot] = (env.kind_id(name), min(attack + tmp_attack, 50), min(health + tmp_health, 50), {1: 0, 2: 2, 3: 5}[level] + experience,
                       env.status_id(status), counter, copies)
        carried |= bool(friends_hurt or hurt)
    return board, carried


def run_eval():
    points, threads = int(sys.argv[3]), int(sys.argv[4])
    data_dir = sys.argv[5] if len(sys.argv) > 5 else "/rs/data"
    import sap_engine_rs
    import gen_env_ghost as H
    from sapai.battle import Battle
    sap_engine_rs.install(LIB, data_dir)
    lib = sap_engine_rs._LIB
    sequences, by_turn, last_turn = H.load_pool()
    pool = [board for boards in by_turn.values() for board in boards]
    block = 16
    env = SapVecEnv(block * 2, threads=threads, library=LIB)
    for name in ("pet-ant", "pet-parrot", "pet-zombie-fly", "pet-wolverine"):
        if env.kind_id(name) != lib.sap_kind(name.encode()):
            fail("kind ids of the environment and of the battle entry point differ")
    for kind in range(70):
        if lib.sap_kind(env.kind_name(kind).encode()) != kind:
            fail("kind ids differ", str(kind))
    driver = Driver(env, 4711)
    rng = driver.rng
    seeds = np.zeros(env.num_games, dtype=np.int64)
    next_seed = [888800000]
    counts = {"points": 0, "points_before_battle": 0, "points_mid_turn": 0, "points_seat_already_ended": 0, "boards": 0, "fights": 0,
              "boards_identical": 0, "boards_with_carried_counter": 0, "boards_with_carried_counter_identical": 0,
              "boards_with_gap": 0, "real_battles": 0, "real_battles_reproduced": 0, "real_battles_with_carried_counter": 0,
              "python_engine_fights": 0, "python_engine_equal": 0, "unchanged_checks": 0, "seat0": 0, "seat1": 0,
              "default_parrot_boards": 0, "outcomes": [0, 0, 0], "boards_with_mixed_outcomes": 0, "own_team_set_from_pool": 0,
              "rekeyed_real_battles": 0, "rekeyed_battle_seed_as_computed": 0, "rekeyed_real_battles_reproduced": 0,
              "rekeyed_points": 0, "rekeyed_points_same_teams": 0, "real_outcomes": [0, 0, 0], "eval_outcomes_same_points": [0, 0, 0],
              "points_real_all_one_outcome": 0, "of_those_eval_all_the_same_outcome": 0, "end_freeze_plans": 0}

    def tape_seed(seed, domain, index=0):
        import hashlib
        message = f"paired-random-tape-v1-20260911|{seed}|{len(domain)}:{domain}|{index}".encode()
        bits = int.from_bytes(hashlib.blake2b(message, digest_size=8).digest(), "big")
        return int((bits >> 11) * (2.0 ** -53) * (2 ** 32 - 1))

    def packed(board):
        cells = [int(v) for row in board.tolist() if row[0] >= 0 for v in row]
        return (ctypes.c_int32 * max(len(cells), 1))(*cells), len(cells) // 7

    def fight(first, second, seed):
        a, na = packed(first); b, nb = packed(second)
        result = lib.sap_battle(a, na, b, nb, ctypes.c_uint32(seed))
        if result < 0:
            fail("sap_battle", str(result))
        return result

    def python_fight(first, second, seed):
        teams = []
        for board in (first, second):
            specs = [s for s in env.decode_board(board) if s is not None]
            teams.append(H.make_team(specs))
        return int(Battle(teams[0], teams[1], seed=seed, fast_training=True).battle())

    def reset(game):
        which = np.zeros(env.num_games, dtype=bool); which[game] = True
        seeds[game] = next_seed[0]; next_seed[0] += 1
        env.reset(seeds, which=which)

    def near(turn):
        turn = min(max(turn + rng.choice([-1, 0, 0, 1]), 1), last_turn)
        return by_turn[turn]

    def random_board(turn=None):
        specs = [dict(s) for s in rng.choice(pool if turn is None or rng.random() < 0.2 else near(turn))]
        default = False
        for s in specs:
            if s["id"] == "pet-parrot" and rng.random() < 0.5:
                s.pop("parrotCopies", None); default = True
        slots = sorted(rng.sample(range(5), len(specs))) if specs and rng.random() < 0.2 else None
        return env.encode_board(specs, slots), default

    def check_point(game, seat, own_board, own_carried, boards, flags, samples, seed):
        before = (rows(env, game), json.dumps(env.state(game), sort_keys=True), event(env, game))
        got = env.eval_vs(game, seat, boards, samples, seed)
        if (rows(env, game), json.dumps(env.state(game), sort_keys=True), event(env, game)) != before:
            fail("eval_vs changed the game", f"game {game}")
        counts["unchanged_checks"] += 1
        again = env.eval_vs_batch([game, game], [seat, seat], boards, samples, [seed, seed])
        if not (np.array_equal(again[0], got) and np.array_equal(again[1], got)):
            fail("eval_vs is not repeatable", f"game {game}")
        for m, board in enumerate(boards):
            want = [0, 0, 0]
            for s in range(samples):
                battle_seed = eval_seed(seed, m, s)
                outcome = fight(own_board, board, battle_seed) if seat == 0 else fight(board, own_board, battle_seed)
                column = 1 if outcome == 2 else 0 if outcome == seat else 2
                want[column] += 1
                if rng.random() < 0.02 and not own_carried:
                    reference = python_fight(own_board, board, battle_seed) if seat == 0 else python_fight(board, own_board, battle_seed)
                    counts["python_engine_fights"] += 1; counts["python_engine_equal"] += reference == outcome
            same = got[m].tolist() == want
            counts["boards"] += 1; counts["fights"] += samples
            counts["boards_with_gap"] += bool(flags[m].get("gap"))
            counts["default_parrot_boards"] += bool(flags[m].get("default"))
            for k in range(3):
                counts["outcomes"][k] += int(got[m][k])
            counts["boards_with_mixed_outcomes"] += int(max(got[m]) < samples)
            if own_carried:
                counts["boards_with_carried_counter"] += 1; counts["boards_with_carried_counter_identical"] += same
            elif not same:
                fail("eval_vs counts", f"game {game} seat {seat} board {m}: {got[m].tolist()} here, {want} from sap_battle; own {own_board.tolist()} other {board.tolist()}")
            else:
                counts["boards_identical"] += 1
        counts["seat0" if seat == 0 else "seat1"] += 1

    for game in range(block):
        reset(game)
    while counts["points"] < points:
        for game in range(block):
            if env.terminated[game]:
                reset(game)
            a, m, flats = driver.choose(game)
            active = [bool(x) for x in env.active[game]]
            scratch = block + game
            samples = rng.choice([1, 2, 4, 8]); seed = rng.randrange(2 ** 63)
            turn_now = int(env.state(game)["turn"])
            point = (sum(active) == 1 and flats[active.index(True)] == END) or (any(active) and rng.random() < 0.04)
            if point and rng.random() < 0.6:
                # Half of the points: a ranked board of this turn as the seat's own team, so that the
                # fights against ranked boards are even and their outcomes depend on the seeds.
                for seat in (0, 1):
                    if active[seat] or rng.random() < 0.5:
                        env.set_team(game, seat, random_board(turn_now)[0]); counts["own_team_set_from_pool"] += 1
                a, m, flats = driver.choose(game, want=flats)
            if sum(active) == 1 and flats[active.index(True)] == END:
                # Right before a battle: the real END on a copy gives the teams and the real seed.
                seat = active.index(True)
                env.copy([game], [scratch])
                step_games(env, [scratch], {scratch: a}, {scratch: m})
                if not env.battle_resolved[scratch]:
                    fail("no battle after the last END")
                turn, battle_seed, teams, carried = env.battle_events(carried=True)
                teams = teams[scratch].copy(); own_carried = bool(carried[scratch, seat].any()); any_carried = bool(carried[scratch].any())
                result = int(env.battle_result[scratch])
                counts["real_battles"] += 1; counts["real_battles_with_carried_counter"] += any_carried
                reproduced = fight(teams[0], teams[1], int(battle_seed[scratch])) == result
                counts["real_battles_reproduced"] += reproduced
                if not reproduced and not any_carried:
                    fail("the exported teams and seed do not reproduce the battle", f"game {game}")
                boards = [teams[1 - seat]]; flags = [{"gap": any(teams[1 - seat][i][0] >= 0 and teams[1 - seat][i - 1][0] < 0 for i in range(1, 5))}]
                for _ in range(rng.choice([0, 3, 7])):
                    board, default = random_board(turn_now)
                    boards.append(board); flags.append({"default": default, "gap": any(board[i][0] >= 0 and board[i - 1][0] < 0 for i in range(1, 5))})
                check_point(game, seat, teams[seat], own_carried, np.stack(boards), flags, samples, seed)
                counts["end_freeze_plans"] += int((m != 0).sum())
                if rng.random() < 0.5:
                    # Repeated real battles: copies rekeyed with fresh seeds fight the real battle of this
                    # turn, each with the battle seed of its own tape; eval_vs gets as many samples.
                    repeats = 6; real = [0, 0, 0]; same_teams = True
                    for _ in range(repeats):
                        key = rng.randrange(2 ** 63)
                        env.copy([game], [scratch]); env.rekey([scratch], [key])
                        step_games(env, [scratch], {scratch: a}, {scratch: m})
                        t2, s2, teams2, carried2 = env.battle_events(carried=True)
                        counts["rekeyed_real_battles"] += 1
                        if int(t2[scratch]) != turn_now:
                            fail("battle turn of a rekeyed copy")
                        if int(s2[scratch]) != tape_seed(key, f"battle/turn/{turn_now}"):
                            fail("battle seed of a rekeyed copy is not the one of its tape", f"game {game}")
                        counts["rekeyed_battle_seed_as_computed"] += 1
                        outcome = int(env.battle_result[scratch])
                        again = fight(teams2[scratch, 0], teams2[scratch, 1], int(s2[scratch]))
                        if again != outcome and not carried2[scratch].any():
                            fail("a rekeyed real battle is not reproduced by its exported teams and seed", f"game {game}")
                        counts["rekeyed_real_battles_reproduced"] += again == outcome
                        same_teams &= bool(np.array_equal(teams2[scratch], teams))
                        real[1 if outcome == 2 else 0 if outcome == seat else 2] += 1
                    counts["rekeyed_points"] += 1
                    if same_teams:      # (end-of-turn abilities with random targets can give another team)
                        counts["rekeyed_points_same_teams"] += 1
                        got = env.eval_vs(game, seat, np.stack([teams[1 - seat]]), repeats, rng.randrange(2 ** 63))[0].tolist()
                        for k in range(3):
                            counts["real_outcomes"][k] += real[k]; counts["eval_outcomes_same_points"][k] += got[k]
                        if max(real) == repeats:
                            counts["points_real_all_one_outcome"] += 1
                            counts["of_those_eval_all_the_same_outcome"] += got == real
                # The seat that had already ended: its team is taken as it is.
                if rng.random() < 0.3:
                    check_point(game, 1 - seat, teams[1 - seat], bool(carried[scratch, 1 - seat].any()), np.stack([teams[seat]]), [{}], samples, seed)
                    counts["points_seat_already_ended"] += 1
                counts["points"] += 1; counts["points_before_battle"] += 1
            elif point:
                # In the middle of a turn, usually with gold left: what the real END would give is read
                # from a copy whose gold is set to 0 so that END is legal.
                seat = rng.choice([s for s in (0, 1) if active[s]])
                env.copy([game], [scratch])
                env.set_gold(scratch, seat, 0)
                forced, fm, _f = driver.choose(scratch, want=[END if s == seat else flats[s] for s in (0, 1)])
                if forced[seat][0] != 0:
                    fail("END not legal at zero gold")
                step_games(env, [scratch], {scratch: forced}, {scratch: fm})
                if env.battle_resolved[scratch]:
                    turn, battle_seed, teams, carried = env.battle_events(carried=True)
                    own, own_carried = teams[scratch, seat].copy(), bool(carried[scratch, seat].any())
                else:
                    own, own_carried = state_board(env, env.state(scratch)["seats"][seat]["team"])
                boards, flags = [], []
                for _ in range(rng.choice([1, 4, 8])):
                    board, default = random_board(turn_now)
                    boards.append(board); flags.append({"default": default, "gap": any(board[i][0] >= 0 and board[i - 1][0] < 0 for i in range(1, 5))})
                check_point(game, seat, own, own_carried, np.stack(boards), flags, samples, seed)
                counts["points"] += 1; counts["points_mid_turn"] += 1
            step_games(env, [game], {game: a}, {game: m})
    # Refused boards change nothing and say why.
    bad = np.tile(np.array(EMPTY_ROW, dtype=np.int32), (1, 5, 1)); bad[0, 0] = (999, 1, 1, 0, 0, 0, -2)
    try:
        env.eval_vs(0, 0, bad, 1, 1)
        fail("a bad board was accepted")
    except ValueError as error:
        counts["refused_board"] = str(error)
    if counts["python_engine_fights"] != counts["python_engine_equal"]:
        fail("Python battle engine disagrees", json.dumps(counts))
    a, b = counts["real_outcomes"], counts["eval_outcomes_same_points"]
    chi2 = sum((x - (x + y) / 2) ** 2 / ((x + y) / 2) + (y - (x + y) / 2) ** 2 / ((x + y) / 2) for x, y in zip(a, b) if x + y)
    counts["real_vs_eval_chi2_df2_p"] = [round(chi2, 3), round(chi_square_p(chi2, 2), 4)]
    if counts["real_vs_eval_chi2_df2_p"][1] < 0.001:
        fail("eval_vs outcome distribution differs from repeated real battles", json.dumps(counts))
    print("ALL EQUAL eval", json.dumps(counts))


# ------------------------------------------------------------------------------------------------
# bench
# ------------------------------------------------------------------------------------------------

def run_bench():
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    import gen_env_ghost as H
    sequences, by_turn, last_turn = H.load_pool()
    n = 1024
    env = SapVecEnv(2 * n, threads=threads, library=LIB)
    driver = Driver(env, 5)
    env.reset(np.arange(2 * n, dtype=np.int64) + 889900000, which=np.arange(2 * n) < n)
    # Sources at assorted points of real matches.
    for _ in range(300):
        acts, msks = {}, {}
        for game in range(n):
            if not env.terminated[game]:
                a, m, _f = driver.choose(game); acts[game] = a; msks[game] = m
        step_games(env, list(acts), acts, msks)
    src = np.arange(n, dtype=np.int32); dst = src + n
    best = 1e9
    for _ in range(20):
        t = time.perf_counter(); env.copy(src, dst); best = min(best, time.perf_counter() - t)
    first = 1e9
    env_fresh = SapVecEnv(2 * n, threads=threads, library=LIB)
    env_fresh.reset(np.arange(2 * n, dtype=np.int64) + 889900000, which=np.arange(2 * n) < n)
    t = time.perf_counter(); env_fresh.copy(src, dst); first = time.perf_counter() - t
    a = np.full((2 * n, 2, 3), NO_DECISION, dtype=np.int64); m = np.zeros((2 * n, 2), dtype=np.int64)
    t = time.perf_counter()
    for _ in range(20):
        env.step(a, m, which=np.zeros(2 * n, dtype=bool))
    idle = (time.perf_counter() - t) / 20
    print(json.dumps({"copy": {"games": n, "threads": threads, "microseconds_per_game": round(best / n * 1e6, 3),
                               "first_copy_into_never_reset_slots_microseconds_per_game": round(first / n * 1e6, 3),
                               "empty_call_microseconds": round(idle * 1e6, 1)}}))
    # eval_vs: ranked boards against ranked boards.
    rng = random.Random(1)
    for label, turns in (("turn 3-5", (3, 4, 5)), ("turn 6-8", (6, 7, 8)), ("turn 9-12", (9, 10, 11, 12)), ("turn 13+", tuple(t for t in by_turn if t >= 13)),
                         ("all turns", tuple(by_turn))):
        boards = [b for t in turns for b in by_turn.get(t, [])]
        games = 32; m_boards = 64; samples = 8
        for game in range(games):
            env.set_team(game, 0, env.encode_board(rng.choice(boards)))
        others = np.stack([np.stack([env.encode_board(rng.choice(boards)) for _ in range(m_boards)]) for _ in range(games)])
        seeds = [rng.randrange(2 ** 63) for _ in range(games)]
        env.eval_vs_batch(list(range(games)), [0] * games, others, 1, seeds)
        t = time.perf_counter()
        counts = env.eval_vs_batch(list(range(games)), [0] * games, others, samples, seeds)
        elapsed = time.perf_counter() - t
        fights = games * m_boards * samples
        t = time.perf_counter()
        env.eval_vs_batch(list(range(games)), [0] * games, others, 1, seeds)
        single = time.perf_counter() - t
        print(json.dumps({"eval_vs": label, "threads": threads, "pool_boards": len(boards), "fights": fights, "seconds": round(elapsed, 3),
                          "fights_per_second": round(fights / elapsed), "fights_per_second_at_1_sample_per_board": round(games * m_boards / single),
                          "win_draw_loss": counts.sum(axis=(0, 1)).tolist()}))


# ------------------------------------------------------------------------------------------------
# shopset / shopsame: set_shop_item through the C interface
# ------------------------------------------------------------------------------------------------

def _board_ids(env, board):
    out = np.tile(np.array(EMPTY_ROW, dtype=np.int32), (5, 1))
    for slot, row in enumerate(board):
        if row is None:
            continue
        copies = -2 if row[6] is False else -1 if row[6] is None else env.kind_id(row[6])
        out[slot] = (env.kind_id(row[0]), row[1], row[2], row[3], env.status_id(row[4]), row[5], copies)
    return out


def _same_state(reference, mine):
    """Canonical dumps equal (the freeze-toggled list by length: Python ids on one side)."""
    if reference["turn"] != mine["turn"]:
        return ["turn"]
    for seat in (0, 1):
        a, b = dict(reference["seats"][seat]), dict(mine["seats"][seat])
        if len(a.pop("toggled")) != len(b.pop("toggled")) or a != b:
            keys = [k for k in a if a[k] != b.get(k)]
            detail = "; ".join(f"{k}: {json.dumps(a[k])} != {json.dumps(b.get(k))}" for k in keys)
            return [f"seat {seat}: " + (detail[:1500] or "toggled count")]
    return []


def _compare_obs(env, game, observations, where, counts):
    for seat in (0, 1):
        for key, value in observations[seat].items():
            mine = env.obs[key][game, seat]
            if mine.dtype != value.dtype or mine.tobytes() != np.ascontiguousarray(value).tobytes():
                fail("observation " + key, f"{where} seat {seat}")
            counts["arrays"] += 1


def _digest(env, game):
    return (rows(env, game), json.dumps(env.state(game), sort_keys=True), event(env, game))


def _refusals(env, game, py_env, shopset, rng, counts, where):
    """Calls that must be refused: on the Rust side a ValueError with the game unchanged (output rows,
    state dump, battle event) and, for the refusals made inside the library, a reason in error(game);
    the reference edit (`gen_env_shopset.place`) refuses the same slot / item."""
    import gen_env_golden as G
    core = py_env.core
    players = (core.agent, core.opponent)
    seat = rng.randrange(2)
    slots = players[seat].shop.slots
    pet_slots = [i for i, s in enumerate(slots) if s.slot_type in ("pet", "levelup")]
    reward_slots = [i for i, s in enumerate(slots) if s.slot_type == "levelup"]
    food_slots = [i for i, s in enumerate(slots) if s.slot_type == "food"]
    pets, foods = shopset.POOL_PETS, shopset.POOL_FOODS
    tokens = [name for name in ("pet-bee", "pet-zombie-cricket", "pet-dirty-rat", "pet-ram", "pet-bus", "pet-chick", "pet-zombie-fly",
                                "pet-butterfly", "pet-none") if env.kind_id(name) >= 0]
    cases = [("slot index = shop length", len(slots), rng.choice(pets), True),
             ("slot index = shop length (food)", len(slots), rng.choice(foods), True),
             ("slot index -1", -1, rng.choice(pets), True),
             ("slot index 10", 10, rng.choice(foods), True),
             ("slot index 1000", 1000, rng.choice(pets), True),
             ("unknown pet name", pet_slots[0] if pet_slots else 0, "pet-unicorn", True),
             ("unknown food name", food_slots[0] if food_slots else 0, "food-nothing", True),
             ("name without prefix", pet_slots[0] if pet_slots else 0, "blowfish", True),
             ("empty name", 0, "", True),
             ("pet id out of range", pet_slots[0] if pet_slots else 0, ("pet", 100000), False),
             ("negative pet id", pet_slots[0] if pet_slots else 0, ("pet", -1), False),
             ("food id out of range", food_slots[0] if food_slots else 0, ("food", 20), False),
             ("negative food id", food_slots[0] if food_slots else 0, ("food", -1), False),
             ("bad item type", 0, ("egg", 1), False),
             ("bare integer", 0, 3, False)]
    if food_slots:
        cases.append(("pet into a food slot", rng.choice(food_slots), rng.choice(pets), True))
        cases.append(("Milk (a stocked token) into a food slot", rng.choice(food_slots), "food-milk", True))
        cases.append(("Bread Crumbs into a food slot", rng.choice(food_slots), "food-bread-crumbs", True))
        cases.append(("food id past the table", rng.choice(food_slots), ("food", 20 + rng.randrange(80)), False))
    if pet_slots:
        cases.append(("food into a pet slot", rng.choice(pet_slots), rng.choice(foods), True))
        if tokens:
            cases.append(("token pet into a pet slot", rng.choice(pet_slots), rng.choice(tokens), True))
        cases.append(("a pet of another pack or a token by id", rng.choice(pet_slots), ("pet", rng.choice(_other_kinds(env, pets))), False))
    if reward_slots:
        cases.append(("food into a level-up reward slot", rng.choice(reward_slots), rng.choice(foods), True))
    before = _digest(env, game)
    reference_before = json.dumps(G.canonical(core), sort_keys=True)
    for label, slot, item, on_reference in rng.sample(cases, min(len(cases), 6)):
        frozen = rng.random() < 0.5
        try:
            env.set_shop_item(game, seat, slot, item, frozen)
        except ValueError as error:
            text = str(error)
        else:
            fail("refusal", f"{where}: accepted {label}: slot {slot} item {item!r}")
        library = "refused value" in text
        if library and not env.error(game):
            fail("refusal", f"{where}: no reason reported for {label}")
        if on_reference:
            try:
                shopset.place(players[seat], slot, item, frozen, [])
            except (IndexError, ValueError):
                pass
            else:
                fail("refusal", f"{where}: the reference edit accepted {label}")
        counts["refused"][label] = counts["refused"].get(label, 0) + 1
        counts["refusals"] += 1
    if _digest(env, game) != before or json.dumps(G.canonical(core), sort_keys=True) != reference_before:
        fail("refusal", f"{where}: a refused call changed the game")
    for bad_game, bad_seat in ((env.num_games, 0), (-1, 0), (game, 2), (game, -1)):
        try:
            env.set_shop_item(bad_game, bad_seat, 0, pets[0])
        except ValueError:
            counts["refusals"] += 1; counts["refused"]["game or seat out of range"] = counts["refused"].get("game or seat out of range", 0) + 1
        else:
            fail("refusal", "accepted a game or seat out of range")
    if _digest(env, game) != before:
        fail("refusal", f"{where}: a refused call changed the game")


_OTHER = {}


def _other_kinds(env, pool):
    """Pet kind ids the library knows that the roll pools do not hold (tokens, pets of other packs)."""
    if "kinds" not in _OTHER:
        inside = {env.kind_id(name) for name in pool}
        kinds, kind = [], 0
        while True:
            try:
                env.kind_name(kind)
            except ValueError:
                break
            if kind not in inside:
                kinds.append(kind)
            kind += 1
        _OTHER["kinds"] = kinds
    return _OTHER["kinds"]


def run_shopset():
    """Lockstep with the Python reference, edits of tools/gen_env_shopset.py through the C interface."""
    import gen_env_shopset as S
    import gen_env_ghost as H
    import gen_env_golden as G
    from sapai_gym1.rl.joint_action import step_decisions
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    games, matches, threads = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    first = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    games = min(games, matches)
    G.EXT = True
    targeted = H.load_targeted()
    pool = H.load_pool()
    H.selfplay_env.Battle = H.RecordingBattle
    env = SapVecEnv(games + 1, threads=threads, library=LIB)        # the last game is never reset
    py = [ParallelSelfPlayEnv() for _ in range(games)]
    info = [None] * games
    counts = {"matches": 0, "steps": 0, "arrays": 0, "states": 0, "battles": 0, "set_shop_item": 0, "by_name": 0, "by_id": 0,
              "set_team": 0, "set_gold": 0, "legal_markers": 0, "states_right_after_edit": 0, "observe_right_after_edit": 0,
              "refusals": 0, "refused": {}, "edits": {}, "where": {}, "fate": {}, "placed_species": 0, "bought_species": 0,
              "after_termination": 0}
    placed, bought = {}, {}
    next_index = [0]
    chooser = random.Random(20261123)

    try:
        env.set_shop_item(games, 0, 0, "pet-ant")
    except ValueError as error:
        if "never reset" not in str(error):
            fail("refusal", f"a game that was never reset: {error}")
        counts["refusals"] += 1; counts["refused"]["a game that was never reset"] = 1
    else:
        fail("refusal", "accepted a game that was never reset")
    for name in S.POOL_PETS:
        if env.kind_name(env.kind_id(name)) != name:
            fail("kind ids", name)
    for name in S.POOL_FOODS + ["food-milk", "food-bread-crumbs"]:
        if env.food_id(name) < 0 or env.food_name(env.food_id(name)) != name:
            fail("food ids", name)
    if env.food_id("food-nothing") != -1 or env.food_id("pet-ant") != -1:
        fail("food ids", "an unknown food has an id")

    def where(game):
        return f"game {game} seed {info[game]['seed']} step {info[game]['steps']}"

    def compare(game):
        _compare_obs(env, game, info[game]["obs"], where(game), counts)
        core = py[game].core
        diff = _same_state(G.canonical(core), env.state(game))
        if diff:
            fail("state", f"{where(game)}: {diff}")
        counts["states"] += 1
        if [int(x) for x in env.lives[game]] != [int(core.agent.lives), int(core.opponent.lives)] or \
           [int(x) for x in env.wins[game]] != [int(core.agent.wins), int(core.opponent.wins)]:
            fail("lives or wins", where(game))

    def reset(game):
        index = first + next_index[0]; next_index[0] += 1
        seed = 887700000 + index
        obs = py[game].reset(seed=seed, paired_tape=True)
        info[game] = {"seed": seed, "obs": obs, "steps": 0, "session": S.Session(py[game], index, targeted, pool, seed)}
        which = np.zeros(env.num_games, dtype=bool); which[game] = True
        seeds = np.zeros(env.num_games, dtype=np.int64); seeds[game] = seed
        env.reset(seeds, which=which)
        compare(game)

    for game in range(games):
        reset(game)
    live_games = set(range(games))
    started = time.time()
    n = env.num_games
    while live_games:
        actions = np.full((n, 2, 3), NO_DECISION, dtype=np.int64); actions[:, :, 1] = 10; actions[:, :, 2] = 5
        masks = np.zeros((n, 2), dtype=np.int64); which = np.zeros(n, dtype=bool)
        decisions = {}
        for game in sorted(live_games):
            state = info[game]; session = state["session"]
            active = py[game].active_players()
            if [bool(x) for x in env.active[game]] != list(active):
                fail("active", where(game))
            if chooser.random() < 0.02:
                _refusals(env, game, py[game], S, chooser, counts, where(game))
            for op in session.edits():
                if op[0] == "gold":
                    env.set_gold(game, op[1], op[2]); counts["set_gold"] += 1
                elif op[0] == "team":
                    env.set_team(game, op[1], _board_ids(env, op[2])); counts["set_team"] += 1
                elif op[0] == "shop":
                    _, seat, slot, name, frozen = op
                    if chooser.random() < 0.5:
                        env.set_shop_item(game, seat, slot, name, frozen); counts["by_name"] += 1
                    else:
                        item = ("pet", env.kind_id(name)) if name.startswith("pet-") else ("food", env.food_id(name))
                        env.set_shop_item(game, seat, slot, item, frozen); counts["by_id"] += 1
                    counts["set_shop_item"] += 1
                elif op[0] == "legal":
                    for seat in (0, 1):
                        if active[seat]:
                            env.legal(game, seat)
                    counts["legal_markers"] += 1
                elif op[0] == "state":
                    diff = _same_state(op[1], env.state(game))
                    if diff:
                        fail("state right after an edit", f"{where(game)}: {diff}")
                    counts["states_right_after_edit"] += 1
                elif op[0] == "obs":
                    only = np.zeros(n, dtype=bool); only[game] = True
                    before = {name: getattr(env, name)[game].copy() for name in ("rewards", "battle_resolved", "battle_result")}
                    env.observe(only)
                    _compare_obs(env, game, op[1], where(game) + " (observe right after an edit)", counts)
                    counts["observe_right_after_edit"] += 1
                    core = py[game].core
                    if [int(x) for x in env.lives[game]] != [int(core.agent.lives), int(core.opponent.lives)] or \
                       [bool(x) for x in env.active[game]] != list(active) or \
                       any(not np.array_equal(getattr(env, name)[game], value) for name, value in before.items()):
                        fail("observe rows", where(game))
            legal = G.legal_of(py[game])
            for seat in (0, 1):
                if legal[seat] is not None and legal[seat] != np.flatnonzero(env.legal(game, seat)).tolist():
                    fail("legal actions", f"{where(game)} seat {seat}")
            pair = session.decide(legal)
            for seat, d in enumerate(pair):
                if d is not None:
                    actions[game, seat] = d.primitive.as_tuple(); masks[game, seat] = int(d.end_freeze_mask or 0)
            decisions[game] = pair; which[game] = True
        env.step(actions, masks, which=which)
        turns, seeds, teams, carried = env.battle_events(carried=True)
        done = []
        for game in sorted(live_games):
            state = info[game]
            turn_before = int(py[game].core.turn)
            result = step_decisions(py[game], decisions[game])
            step = result.result
            state["obs"] = step.observations; state["steps"] += 1
            if env.status[game] != 0:
                fail("status", f"{where(game)}: {env.status[game]} {env.error(game)}")
            compare(game)
            if tuple(float(x) for x in env.rewards[game]) != tuple(float(x) for x in step.rewards) or \
               np.signbit(env.rewards[game]).tolist() != [bool(np.signbit(x)) for x in step.rewards]:
                fail("rewards", where(game))
            if bool(env.battle_resolved[game]) != bool(step.battle_resolved) or bool(env.terminated[game]) != bool(step.terminated):
                fail("flags", where(game))
            if step.battle_resolved:
                if int(env.battle_result[game]) != int(step.battle_result):
                    fail("battle result", where(game))
                want = np.stack([_board_ids(env, board) for board in H.SEEN.pop("teams")])
                if int(turns[game]) != turn_before or int(seeds[game]) != int(py[game]._paired_battle_seeds[-1][1]) or \
                   not np.array_equal(teams[game], want) or carried[game].tolist() != H.SEEN.pop("carried"):
                    fail("battle event", where(game))
                counts["battles"] += 1
            elif int(turns[game]) != 0:
                fail("battle event without a battle", where(game))
            state["session"].after(result)
            counts["steps"] += 1
            if step.terminated:
                if int(env.winner[game]) != int(G._completed_match_result(py[game].core)):
                    fail("winner", where(game))
                counts["matches"] += 1; done.append(game)
                # A finished game still takes the edit (like set_team); only the state dump can show it.
                core = py[game].core
                for seat, player in enumerate((core.agent, core.opponent)):
                    if player.shop.slots:
                        slot = chooser.randrange(len(player.shop.slots))
                        name = chooser.choice(S.POOL_FOODS if player.shop.slots[slot].slot_type == "food" else S.POOL_PETS)
                        frozen = chooser.random() < 0.5
                        S.place(player, slot, name, frozen, state["session"].keep)
                        env.set_shop_item(game, seat, slot, name, frozen)
                        counts["after_termination"] += 1
                diff = _same_state(G.canonical(core), env.state(game))
                if diff:
                    fail("state after an edit of a finished game", f"{where(game)}: {diff}")
                stats = state["session"].stats
                for table in ("edits", "where", "fate"):
                    for key, value in stats[table].items():
                        counts[table][key] = counts[table].get(key, 0) + value
                for table, total in (("placed", placed), ("bought", bought)):
                    for key, value in stats[table].items():
                        total[key] = total.get(key, 0) + value
        for game in done:
            if next_index[0] < matches:
                reset(game)
            else:
                live_games.discard(game)
    counts["placed_species"] = len(placed); counts["bought_species"] = len(bought)
    chain = "placed pill fed to a reward bought right after the edit"
    fed = {key: value for key, value in counts["fate"].items() if key.startswith(chain)}
    counts["fate"] = {key: value for key, value in counts["fate"].items() if key not in fed}
    counts["fate"][chain] = sum(fed.values())
    counts["fate"][chain + " (Spider or Seal: draws from its private stream)"] = sum(
        value for key, value in fed.items() if key.endswith(("pet-spider", "pet-seal")))
    for table in ("edits", "where", "fate", "refused"):
        counts[table] = dict(sorted(counts[table].items()))
    print("ALL EQUAL shopset", json.dumps(counts), f"{time.time() - started:.0f}s")


def run_shopsame():
    """A freshly rolled item replaced by itself must change nothing, now or later. Four copies of
    every match get the same actions: Python plain, Python with every rolled item replaced by
    `gen_env_shopset.place` of the same name, Rust plain, Rust with `set_shop_item` of the same
    name. Replaced: after the reset, after every roll of a seat and at every turn start, every
    ordinary pet slot whose pet is at base stats plus the shop's permanent bonus (every pet that
    was just rolled is; a frozen pet that stayed can carry a Duck's +1 health and is skipped then)
    and every food slot that holds a roll-pool food at its table cost; frozen items that stayed
    through the roll included, with their frozen flag. All four must agree at
    every step: legal lists, observations (bytes), state dumps, rewards, flags."""
    import gen_env_shopset as S
    import gen_env_ghost as H
    import gen_env_golden as G
    from sapai_gym1.rl.joint_action import step_decisions
    from sapai_gym1.rl.selfplay_env import ParallelSelfPlayEnv
    matches, threads = int(sys.argv[3]), int(sys.argv[4])
    G.EXT = True
    targeted = H.load_targeted()
    env = SapVecEnv(2, threads=threads, library=LIB)
    counts = {"matches": 0, "steps": 0, "arrays": 0, "states": 0, "battles": 0, "replaced_pets": 0, "replaced_foods": 0,
              "replaced_frozen": 0, "replaced_with_canned_bonus": 0, "skipped_reward_slots": 0, "skipped_foods_off_table_cost": 0, "skipped_frozen_pets_buffed_in_the_shop": 0,
              "replacement_rounds": 0, "species": 0}
    species = set()
    started = time.time()
    n = 2
    for index in range(matches):
        seed = 888800000 + index
        rng = random.Random(seed)
        plain, edited = ParallelSelfPlayEnv(), ParallelSelfPlayEnv()
        obs = [plain.reset(seed=seed, paired_tape=True), edited.reset(seed=seed, paired_tape=True)]
        env.reset(np.array([seed, seed], dtype=np.int64))
        live = [H.Live(targeted, rng, (index // 2) % 2 == 1, rng.choice([0, 0, 4, 8])) for _ in (0, 1)]
        keep = []
        steps = 0

        def replace(seats):
            core = edited.core
            for seat in seats:
                player = (core.agent, core.opponent)[seat]
                for slot_index, slot in enumerate(player.shop.slots):
                    name = slot.obj.name
                    if slot.slot_type == "levelup":
                        counts["skipped_reward_slots"] += 1; continue
                    if slot.slot_type == "food" and (name not in S.FOOD_TIER or slot.cost != ENGINE_FOOD_COST[name]):
                        counts["skipped_foods_off_table_cost"] += 1; continue
                    if slot.slot_type == "pet":
                        # A frozen pet that stayed through the roll can carry a Duck's +1 health, which a
                        # fresh pet does not have: only pets at base stats plus the permanent bonus qualify.
                        fd = engine_data["pets"][name]
                        if slot.obj._attack != min(50, fd["baseAttack"] + player.shop.shop_attack) or \
                           slot.obj._health != min(50, fd["baseHealth"] + player.shop.shop_health):
                            if not slot.frozen:
                                fail("a pet that was just rolled is not at base stats plus the shop bonus", f"seed {seed} step {steps}")
                            counts["skipped_frozen_pets_buffed_in_the_shop"] += 1; continue
                    frozen = bool(slot.frozen)
                    S.place(player, slot_index, name, frozen, keep)
                    env.set_shop_item(1, seat, slot_index, name, frozen)
                    counts["replaced_pets" if slot.slot_type == "pet" else "replaced_foods"] += 1
                    counts["replaced_frozen"] += frozen
                    counts["replaced_with_canned_bonus"] += slot.slot_type == "pet" and bool(player.shop.shop_attack or player.shop.shop_health)
                    species.add(name)
            counts["replacement_rounds"] += 1

        def agree(label):
            where = f"seed {seed} step {steps} ({label})"
            reference = G.canonical(plain.core)
            for other, name in ((G.canonical(edited.core), "Python with replacements"), (env.state(0), "Rust plain"),
                                (env.state(1), "Rust with replacements")):
                diff = _same_state(reference, other)
                if diff:
                    fail("state: " + name, f"{where}: {diff}")
            counts["states"] += 3

        from sapai.data import data as engine_data
        ENGINE_FOOD_COST = {name: int(engine_data["foods"][name].get("cost", 3)) for name in S.FOOD_TIER}
        replace((0, 1))
        agree("reset")
        while True:
            legal = G.legal_of(plain)
            if legal != G.legal_of(edited):
                fail("legal: Python with replacements", f"seed {seed} step {steps}")
            for game in (0, 1):
                for seat in (0, 1):
                    if legal[seat] is not None and legal[seat] != np.flatnonzero(env.legal(game, seat)).tolist():
                        fail("legal: Rust", f"seed {seed} step {steps} game {game} seat {seat}")
            pair = [None if options is None else live[seat].decide(plain, seat, options) for seat, options in enumerate(legal)]
            actions = np.full((n, 2, 3), NO_DECISION, dtype=np.int64); actions[:, :, 1] = 10; actions[:, :, 2] = 5
            masks = np.zeros((n, 2), dtype=np.int64)
            for seat, d in enumerate(pair):
                if d is not None:
                    actions[:, seat] = d.primitive.as_tuple(); masks[:, seat] = int(d.end_freeze_mask or 0)
            result = step_decisions(plain, pair)
            other = step_decisions(edited, pair)
            env.step(actions, masks)
            steps += 1; counts["steps"] += 1
            step = result.result
            if (env.status != 0).any():
                fail("status", f"seed {seed} step {steps}: {env.error(0)} {env.error(1)}")
            if rows(env, 0) != rows(env, 1):
                fail("Rust rows: plain against replaced", f"seed {seed} step {steps}")
            _compare_obs(env, 0, step.observations, f"seed {seed} step {steps} Rust plain", counts)
            _compare_obs(env, 1, other.result.observations, f"seed {seed} step {steps} Rust with replacements", counts)
            if tuple(step.rewards) != tuple(other.result.rewards) or bool(step.battle_resolved) != bool(other.result.battle_resolved) or \
               step.battle_result != other.result.battle_result or bool(step.terminated) != bool(other.result.terminated):
                fail("Python step: plain against replaced", f"seed {seed} step {steps}")
            if tuple(float(x) for x in env.rewards[0]) != tuple(float(x) for x in step.rewards) or \
               bool(env.battle_resolved[0]) != bool(step.battle_resolved) or bool(env.terminated[0]) != bool(step.terminated):
                fail("Rust step against Python", f"seed {seed} step {steps}")
            if event(env, 0) != event(env, 1):
                fail("battle event: plain against replaced", f"seed {seed} step {steps}")
            counts["battles"] += bool(step.battle_resolved)
            agree("after the step, before replacing")
            if step.terminated:
                break
            rolled = [seat for seat, plan in enumerate(result.primitive_actions) if any(a.as_tuple()[0] == 1 for a in plan)]
            if step.battle_resolved:
                rolled = [0, 1]
            if rolled:
                replace(rolled)
                agree("after replacing")
        counts["matches"] += 1
    counts["species"] = len(species)
    print("ALL EQUAL shopsame", json.dumps(counts), f"{time.time() - started:.0f}s")


if MODE == "none":      # imported for its helpers
    pass
elif MODE == "ghost":
    run_ghost()
elif MODE == "copy":
    run_copy()
elif MODE == "rekey":
    run_rekey()
elif MODE == "rolls":
    run_rolls()
elif MODE == "eval":
    run_eval()
elif MODE == "bench":
    run_bench()
elif MODE == "shopset":
    run_shopset()
elif MODE == "shopsame":
    run_shopsame()
else:
    raise SystemExit("mode must be ghost, copy, rekey, rolls, eval, bench, shopset or shopsame")
