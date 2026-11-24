"""Batch interface to the Rust training environment (ctypes + numpy only).

    from sap_env_rs import SapVecEnv
    env = SapVecEnv(num_games, threads=5)            # loads target/release/libsap_engine.so
    env.reset(seeds)                                 # all games; or env.reset(seeds, which=bool_mask)
    env.step(actions, masks)                         # actions int64 [N, 2, 3], masks int64 [N, 2]

Every game is one `ParallelSelfPlayEnv().reset(seed, paired_tape=True)` match driven by
`joint_action.step_decisions`: per seat one `[type, source, target]` triple (FlatActionCodec /
ActionType numbering, unused fields at their NO_SOURCE=10 / NO_TARGET=5 values) and one END
freeze mask. A seat that has already ended its turn must send type -1 (`NO_DECISION`) and mask 0.

Results are written into numpy arrays owned by this object and reused by every call:

    env.obs[key]         [N, 2, ...]  the reference's observation arrays, same dtypes
    env.rewards          [N, 2] float64
    env.battle_resolved  [N] int8
    env.battle_result    [N] int8     0 / 1 / 2, -1 when the step resolved no battle
    env.terminated       [N] int8
    env.winner           [N] int8     _completed_match_result once terminated, else -1
    env.lives, env.wins  [N, 2] int32
    env.active           [N, 2] int8  the seat still has to act this turn
    env.status           [N] int32    0 ok; 1 skipped, the match is over; -1 illegal input (the game
                                      is unchanged); -2 internal error; -3 never reset

There is no automatic reset: a finished game keeps its final observation, `terminated`, `winner`,
lives and wins until `reset` is called for it; stepping it only writes status 1 and zero rewards.
Illegal input never raises here and never touches another game: read `status` and `error(i)`.

Search and ghost primitives (additions; `reset` and `step(actions, masks)` behave as before).
A bad argument to one of these raises ValueError and changes nothing.

Boards. A board is int32 `[5, 7]`, rows front to back = team slots 0..4, columns
`[kind id, attack, health, total experience, perk id, consumed triggers, copied species]` with the
ids and Parrot sentinels of the battle entry point (`python/sap_engine_rs.py`): kind id =
`env.kind_id("pet-ant")` (= `sap_kind`), perk id = `env.status_id("status-garlic-armor")`
(= `sap_status`; "none" is 0), copied species -2 = not stated (a Parrot then copies the friend
ahead, the default rule of `make_team`; on export every non-Parrot has -2), -1 = a Parrot that
copied nothing, k = the kind a Parrot copied. An empty slot is a row with kind id -1
(`EMPTY_ROW`). Accepted: attack 0..50, health 1..50, total experience 0..5, consumed triggers
0..1000. `env.encode_board(specs)` / `env.decode_board(board)` convert to and from the spec
dicts of the ghost pool (`id, attack, health, experience, status, triggersConsumed, parrotCopies`).

    env.step(actions, masks, which=mask)   only the selected games take a step; the others and
                                     their output rows are not touched at all
    env.copy(src, dst)               game dst[k] becomes an exact, independent copy of game src[k]:
                                     both seats, shops, hidden shop state, random streams and tape
                                     cursors, counters, histories, the last battle event. The
                                     output rows of dst[k] (obs, active, lives, wins, terminated,
                                     winner, and the rewards and battle flags of the source's last
                                     step, which go with the copied battle event) become copies of
                                     the rows of src[k] as they stand; status of dst[k] is 0. A
                                     game may not be in dst twice or in both lists; a source that
                                     was never reset is refused.
    env.rekey(games, seeds)          from now on every random draw of these games comes from the
                                     tape of the given seed (int in [0, 2**63)), exactly as the
                                     reference's `ParallelSelfPlayEnv.redeterminize(seed)` does on
                                     the paired-tape path. Replaced: both seats' pet and food offer
                                     streams (level-up rewards come from the pet stream), both shop
                                     streams (trigger-order ties, random targets of abilities and
                                     foods, priority ties in the shop) and every later battle
                                     seed. Not replaced: what was already drawn, i.e. the current
                                     shops (frozen or not), teams, counters and histories. Nothing
                                     visible changes at the call; observations are not rewritten.
    env.set_team(game, seat, board)  `player.team = make_team(specs); player._connect_objects()`
                                     of runs/train-ghost-20261123/ghost_worker.py. Fresh pets with
                                     `Pet(id)` defaults for whatever a row does not state (sell
                                     value 1, no temporary buffs, counters zero). Gold, shop,
                                     action counters, the seen-order set and "turn ended" stay.
    env.set_gold(game, seat, gold)   `player.gold = gold` (0..1000)
                                     Neither call rewrites the output rows (the reference produces
                                     no observation there); `env.observe(which)` does, and like any
                                     observation registers the current team order as seen.
    env.battle_events(carried=False) after a `step`: (turn, seed, teams) for all games, arrays
                                     owned by this object: turn int32 [N], the one-based turn of
                                     the battle the game's last accepted step resolved (0: it
                                     resolved none; a refused step, status -1, or a skipped one,
                                     status 1, leaves the game and therefore its event as they
                                     were, so a finished game keeps its last battle: read
                                     `battle_resolved` for "in this call"), seed
                                     int64 [N], the battle seed used, teams int32 [N, 2, 5, 7],
                                     both teams (seat 0 first) as handed to the battle engine:
                                     after both seats' end-of-turn effects, before combat, stats
                                     with temporary shop buffs folded in as `Battle.__init__` does.
                                     Valid until the next step/reset of that game. With
                                     carried=True a fourth array int32 [N, 2, 5, 2]: per slot
                                     `friends_hurt_counter` and `_hurt`, the two values the battle
                                     reads that the seven columns cannot carry.
    env.eval_vs(game, seat, boards, samples, seed) -> int32 [M, 3]
                                     wins, draws, losses of `seat`: the team the seat would take
                                     into battle if it ended its turn now with no freezes (its
                                     end-of-turn effects are applied to a private copy; a seat that
                                     has ended is taken as it is) fights each of the M boards
                                     `samples` times. The seat keeps its side of the battle (seat 0
                                     is the first team). Battle seed of sample s against board m:
                                     `eval_seed(seed, m, s)` = int(RandomTape(seed).uniform(
                                     f"eval/{m}") * (2**32 - 1)) at cursor s, the same construction
                                     and range as the match's own battle seeds. The game is not
                                     changed.
    env.eval_vs_batch(games, seats, boards, samples, seeds) -> int32 [K, M, 3]
                                     the same for K (game, seat) pairs, boards [K, M, 5, 7] or
                                     [M, 5, 7] for all pairs, on the environment's threads.
"""
import ctypes
import hashlib
import json
import os

import numpy as np

NO_DECISION = -1
STATUS_OK, STATUS_FINISHED, STATUS_ILLEGAL, STATUS_BROKEN, STATUS_NOT_RESET = 0, 1, -1, -2, -3
BOARD_SLOTS, BOARD_FIELDS = 5, 7
EMPTY_ROW = (-1, 0, 0, 0, 0, 0, -2)
COPIES_DEFAULT, COPIES_NOTHING = -2, -1
_CALL_ERRORS = {-10: "null pointer or bad count", -11: "game or seat index out of range", -12: "a game that was never reset",
                -13: "a game is in dst twice or in both src and dst", -14: "refused value", -15: "a fight failed inside the engine",
                -100: "panic inside the library"}


def eval_seed(seed, board, sample):
    """Battle seed of sample `sample` against board `board` in `eval_vs(..., seed)`."""
    domain = f"eval/{board}"
    message = f"paired-random-tape-v1-20260911|{int(seed)}|{len(domain)}:{domain}|{int(sample)}".encode()
    bits = int.from_bytes(hashlib.blake2b(message, digest_size=8).digest(), "big")
    return int((bits >> 11) * (2.0 ** -53) * (2 ** 32 - 1))

OBS_LAYOUT = (
    ("item_ids", np.int64, (21,)),
    ("kind_ids", np.int64, (21,)),
    ("status_ids", np.int64, (21,)),
    ("positions", np.int64, (21,)),
    ("numeric", np.float32, (21, 16)),
    ("entity_mask", np.int8, (21,)),
    ("type_mask", np.int8, (9,)),
    ("source_mask", np.int8, (9, 11)),
    ("target_mask", np.int8, (9, 11, 6)),
    ("opponent_history_item_ids", np.int64, (2, 5)),
    ("opponent_history_status_ids", np.int64, (2, 5)),
    ("opponent_history_positions", np.int64, (2, 5)),
    ("opponent_history_numeric", np.float32, (2, 5, 8)),
    ("opponent_history_mask", np.int8, (2, 5)),
    ("opponent_history_board_mask", np.int8, (2,)),
)


class _Outputs(ctypes.Structure):
    _fields_ = [
        ("obs", ctypes.c_void_p * 15),
        ("rewards", ctypes.c_void_p),
        ("battle_resolved", ctypes.c_void_p),
        ("battle_result", ctypes.c_void_p),
        ("terminated", ctypes.c_void_p),
        ("winner", ctypes.c_void_p),
        ("lives", ctypes.c_void_p),
        ("wins", ctypes.c_void_p),
        ("active", ctypes.c_void_p),
        ("status", ctypes.c_void_p),
    ]


def default_library():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "target", "release", "libsap_engine.so")


def load_library(path=None):
    lib = ctypes.CDLL(path or default_library())
    lib.sap_env_create.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.sap_env_create.restype = ctypes.c_void_p
    lib.sap_env_destroy.argtypes = [ctypes.c_void_p]
    lib.sap_env_destroy.restype = None
    lib.sap_env_len.argtypes = [ctypes.c_void_p]
    lib.sap_env_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.sap_env_reset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_Outputs)]
    lib.sap_env_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_Outputs)]
    lib.sap_env_error.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
    lib.sap_env_legal.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p]
    lib.sap_env_state_json.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
    lib.sap_env_obs_layout.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    for name in ("sap_env_len", "sap_env_set_threads", "sap_env_reset", "sap_env_step", "sap_env_error",
                 "sap_env_legal", "sap_env_state_json", "sap_env_obs_layout"):
        getattr(lib, name).restype = ctypes.c_int32
    # Search and ghost primitives (absent from libraries built before they were added).
    try:
        p, i32 = ctypes.c_void_p, ctypes.c_int32
        lib.sap_env_kind.argtypes = [ctypes.c_char_p]
        lib.sap_env_status.argtypes = [ctypes.c_char_p]
        lib.sap_env_kind_name.argtypes = [i32, ctypes.c_char_p, i32]
        lib.sap_env_step_which.argtypes = [p, p, p, p, ctypes.POINTER(_Outputs)]
        lib.sap_env_observe.argtypes = [p, p, ctypes.POINTER(_Outputs)]
        lib.sap_env_copy.argtypes = [p, p, p, i32, ctypes.POINTER(_Outputs)]
        lib.sap_env_rekey.argtypes = [p, p, p, i32]
        lib.sap_env_set_team.argtypes = [p, i32, i32, p]
        lib.sap_env_set_gold.argtypes = [p, i32, i32, i32]
        lib.sap_env_battle_events.argtypes = [p, p, p, p, p]
        lib.sap_env_eval_vs.argtypes = [p, p, p, i32, p, i32, i32, p, p]
        for name in ("sap_env_kind", "sap_env_status", "sap_env_kind_name", "sap_env_step_which", "sap_env_observe", "sap_env_copy",
                     "sap_env_rekey", "sap_env_set_team", "sap_env_set_gold", "sap_env_battle_events", "sap_env_eval_vs"):
            getattr(lib, name).restype = i32
        lib.has_search_primitives = True
    except AttributeError:
        lib.has_search_primitives = False
    # The library's layout must be the one this file allocates.
    buffer = ctypes.create_string_buffer(4096)
    lib.sap_env_obs_layout(buffer, len(buffer))
    theirs = [(name, dtype, tuple(shape)) for name, dtype, shape in json.loads(buffer.value.decode())]
    ours = [(name, np.dtype(dtype).name, shape) for name, dtype, shape in OBS_LAYOUT]
    if theirs != ours:
        raise RuntimeError(f"observation layout mismatch between the library and sap_env_rs.py: {theirs} != {ours}")
    return lib


class SapVecEnv:
    def __init__(self, num_games, threads=1, library=None):
        if int(num_games) <= 0 or int(threads) <= 0:
            raise ValueError("num_games and threads must be positive")
        self._lib = load_library(library) if library is None or isinstance(library, str) else library
        self.num_games = int(num_games)
        self._env = self._lib.sap_env_create(self.num_games, int(threads))
        if not self._env:
            raise RuntimeError("sap_env_create failed")
        n = self.num_games
        self.obs = {name: np.zeros((n, 2) + shape, dtype=dtype) for name, dtype, shape in OBS_LAYOUT}
        self.rewards = np.zeros((n, 2), dtype=np.float64)
        self.battle_resolved = np.zeros(n, dtype=np.int8)
        self.battle_result = np.full(n, -1, dtype=np.int8)
        self.terminated = np.zeros(n, dtype=np.int8)
        self.winner = np.full(n, -1, dtype=np.int8)
        self.lives = np.zeros((n, 2), dtype=np.int32)
        self.wins = np.zeros((n, 2), dtype=np.int32)
        self.active = np.zeros((n, 2), dtype=np.int8)
        self.status = np.full(n, STATUS_NOT_RESET, dtype=np.int32)
        out = _Outputs()
        for index, (name, _, _) in enumerate(OBS_LAYOUT):
            out.obs[index] = self.obs[name].ctypes.data
        for name in ("rewards", "battle_resolved", "battle_result", "terminated", "winner", "lives", "wins", "active", "status"):
            setattr(out, name, getattr(self, name).ctypes.data)
        self._out = out

    def close(self):
        if getattr(self, "_env", None):
            self._lib.sap_env_destroy(self._env)
            self._env = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def set_threads(self, threads):
        if self._lib.sap_env_set_threads(self._env, int(threads)) != 0:
            raise ValueError("threads must be positive")

    def reset(self, seeds, which=None):
        """Start new matches. `seeds`: one int in [0, 2**63) per game (entries of games that are
        not selected are ignored). `which`: optional boolean array selecting the games to reset.
        Returns the number of refused resets (status -1 for those games)."""
        seeds = np.ascontiguousarray(seeds, dtype=np.int64)
        if seeds.shape != (self.num_games,):
            raise ValueError(f"seeds must have shape ({self.num_games},)")
        pointer = None
        if which is not None:
            which = np.ascontiguousarray(which, dtype=np.uint8)
            if which.shape != (self.num_games,):
                raise ValueError(f"which must have shape ({self.num_games},)")
            pointer = which.ctypes.data
        code = self._lib.sap_env_reset(self._env, pointer, seeds.ctypes.data, ctypes.byref(self._out))
        if code < 0:
            raise RuntimeError(f"sap_env_reset failed with code {code}")
        return code

    def step(self, actions, masks, which=None):
        """One decision per active seat of every running game. Returns the number of games whose
        status is negative after the call. `which`: optional boolean array; only the selected
        games take the step, the others and their output rows are not touched."""
        actions = np.ascontiguousarray(actions, dtype=np.int64)
        masks = np.ascontiguousarray(masks, dtype=np.int64)
        if actions.shape != (self.num_games, 2, 3) or masks.shape != (self.num_games, 2):
            raise ValueError(f"expected actions [{self.num_games}, 2, 3] and masks [{self.num_games}, 2]")
        if which is None:
            code = self._lib.sap_env_step(self._env, actions.ctypes.data, masks.ctypes.data, ctypes.byref(self._out))
        else:
            which = self._which(which)
            code = self._lib.sap_env_step_which(self._env, which.ctypes.data, actions.ctypes.data, masks.ctypes.data,
                                                ctypes.byref(self._out))
        if code < 0:
            raise RuntimeError(f"sap_env_step failed with code {code}")
        return code

    # ---- search and ghost primitives (see the module docstring) ----

    def _which(self, which):
        which = np.ascontiguousarray(which, dtype=np.uint8)
        if which.shape != (self.num_games,):
            raise ValueError(f"which must have shape ({self.num_games},)")
        return which

    def _check(self, code, game=None):
        if code < 0:
            detail = self.error(game) if code == -14 and game is not None else ""
            raise ValueError(_CALL_ERRORS.get(code, f"code {code}") + (f": {detail}" if detail else ""))
        return code

    def kind_id(self, name):
        """Kind id of a pet name ("pet-ant"), the id `sap_kind` gives; -1 if unknown."""
        return int(self._lib.sap_env_kind(name.encode()))

    def kind_name(self, kind):
        buffer = ctypes.create_string_buffer(64)
        if self._lib.sap_env_kind_name(int(kind), buffer, len(buffer)) < 0:
            raise ValueError(f"no pet kind {kind}")
        return buffer.value.decode()

    def status_id(self, name):
        """Perk id of a status name ("none", "status-garlic-armor"), the id `sap_status` gives; -1 if unknown."""
        return int(self._lib.sap_env_status(name.encode()))

    def encode_board(self, specs, slots=None):
        """Spec dicts (ghost pool / `make_team`: id, attack, health, experience, status, optional
        triggersConsumed and parrotCopies) -> int32 [5, 7]. Pet k goes to row k, or to row slots[k]."""
        board = np.tile(np.array(EMPTY_ROW, dtype=np.int32), (BOARD_SLOTS, 1))
        if len(specs) > BOARD_SLOTS:
            raise ValueError("more than five pets")
        for index, spec in enumerate(specs):
            kind, status = self.kind_id(spec["id"]), self.status_id(spec["status"])
            if kind < 0 or status < 0:
                raise ValueError(f"unknown pet or status in {spec}")
            copies = COPIES_DEFAULT
            if "parrotCopies" in spec:
                copies = self.kind_id(spec["parrotCopies"]) if spec["parrotCopies"] else COPIES_NOTHING
                if spec["parrotCopies"] and copies < 0:
                    raise ValueError(f"unknown copied pet in {spec}")
            board[index if slots is None else slots[index]] = (kind, int(spec["attack"]), int(spec["health"]), int(spec["experience"]),
                                                              status, int(spec.get("triggersConsumed", 0)), copies)
        return board

    def decode_board(self, board):
        """int32 [5, 7] -> five spec dicts or None (empty slot). A copied-species column of -2 gives
        no `parrotCopies` key."""
        if not hasattr(self, "_status_names"):
            names = ["none", "status-bread", "status-cake", "status-bone-attack", "status-coconut-shield", "status-extra-life",
                     "status-garlic-armor", "status-honey-bee", "status-melon-armor", "status-poison-attack", "status-splash-attack",
                     "status-steak-attack", "status-weak"]
            if [self.status_id(name) for name in names] != list(range(len(names))):
                raise RuntimeError("status ids of the library differ from this file")
            self._status_names = names
        out = []
        for row in np.asarray(board).reshape(BOARD_SLOTS, BOARD_FIELDS).tolist():
            if row[0] < 0:
                out.append(None); continue
            spec = {"id": self.kind_name(row[0]), "attack": row[1], "health": row[2], "experience": row[3],
                    "status": self._status_names[row[4]], "triggersConsumed": row[5]}
            if row[6] != COPIES_DEFAULT:
                spec["parrotCopies"] = None if row[6] == COPIES_NOTHING else self.kind_name(row[6])
            out.append(spec)
        return out

    def observe(self, which):
        """Write the output rows (obs, lives, wins, active, terminated, winner) of the selected games
        again from their current state. Returns the number of games written."""
        which = self._which(which)
        return self._check(self._lib.sap_env_observe(self._env, which.ctypes.data, ctypes.byref(self._out)))

    def copy(self, src, dst):
        """Game dst[k] becomes an exact, independent copy of game src[k]."""
        src = np.ascontiguousarray(src, dtype=np.int32).reshape(-1)
        dst = np.ascontiguousarray(dst, dtype=np.int32).reshape(-1)
        if src.shape != dst.shape:
            raise ValueError("src and dst must have the same length")
        self._check(self._lib.sap_env_copy(self._env, src.ctypes.data, dst.ctypes.data, len(src), ctypes.byref(self._out)))

    def rekey(self, games, seeds):
        """Every later random draw of games[k] comes from the tape of seeds[k] (int in [0, 2**63))."""
        games = np.ascontiguousarray(games, dtype=np.int32).reshape(-1)
        seeds = np.ascontiguousarray(seeds, dtype=np.int64).reshape(-1)
        if games.shape != seeds.shape:
            raise ValueError("games and seeds must have the same length")
        self._check(self._lib.sap_env_rekey(self._env, games.ctypes.data, seeds.ctypes.data, len(games)))

    def set_team(self, game, seat, board):
        """Replace the team of `seat` by `board` (int32 [5, 7])."""
        board = np.ascontiguousarray(board, dtype=np.int32)
        if board.shape != (BOARD_SLOTS, BOARD_FIELDS):
            raise ValueError("a board is int32 [5, 7]")
        self._check(self._lib.sap_env_set_team(self._env, int(game), int(seat), board.ctypes.data), game)

    def set_gold(self, game, seat, gold):
        self._check(self._lib.sap_env_set_gold(self._env, int(game), int(seat), int(gold)), game)

    def battle_events(self, carried=False):
        """(turn [N], seed [N], teams [N, 2, 5, 7]) of the battle each game's last accepted step
        resolved; turn 0 where it resolved none. With carried=True also [N, 2, 5, 2]."""
        n = self.num_games
        if not hasattr(self, "battle_turn"):
            self.battle_turn = np.zeros(n, dtype=np.int32)
            self.battle_seed = np.zeros(n, dtype=np.int64)
            self.battle_teams = np.zeros((n, 2, BOARD_SLOTS, BOARD_FIELDS), dtype=np.int32)
            self.battle_carried = np.zeros((n, 2, BOARD_SLOTS, 2), dtype=np.int32)
        self._check(self._lib.sap_env_battle_events(self._env, self.battle_turn.ctypes.data, self.battle_seed.ctypes.data,
                                                    self.battle_teams.ctypes.data, self.battle_carried.ctypes.data if carried else None))
        if carried:
            return self.battle_turn, self.battle_seed, self.battle_teams, self.battle_carried
        return self.battle_turn, self.battle_seed, self.battle_teams

    def eval_vs(self, game, seat, boards, samples, seed):
        """Wins, draws, losses (int32 [M, 3]) of `seat` of `game` against each of the M boards."""
        boards = np.ascontiguousarray(boards, dtype=np.int32)
        if boards.ndim != 3 or boards.shape[1:] != (BOARD_SLOTS, BOARD_FIELDS):
            raise ValueError("boards must be int32 [M, 5, 7]")
        return self.eval_vs_batch([game], [seat], boards[None], samples, [seed])[0]

    def eval_vs_batch(self, games, seats, boards, samples, seeds):
        """`eval_vs` for K (game, seat) pairs: boards [K, M, 5, 7] (or [M, 5, 7] for all pairs),
        seeds [K]; returns int32 [K, M, 3]."""
        games = np.ascontiguousarray(games, dtype=np.int32).reshape(-1)
        seats = np.ascontiguousarray(seats, dtype=np.int32).reshape(-1)
        seeds = np.ascontiguousarray(seeds, dtype=np.int64).reshape(-1)
        boards = np.asarray(boards, dtype=np.int32)
        k = len(games)
        if boards.ndim == 3:
            boards = np.broadcast_to(boards, (k,) + boards.shape)
        boards = np.ascontiguousarray(boards)
        if seats.shape != (k,) or seeds.shape != (k,) or boards.ndim != 4 or boards.shape[0] != k or boards.shape[2:] != (BOARD_SLOTS, BOARD_FIELDS):
            raise ValueError("expected games [K], seats [K], seeds [K] and boards [K, M, 5, 7]")
        if int(samples) < 0:
            raise ValueError("samples must not be negative")
        m = boards.shape[1]
        counts = np.zeros((k, m, 3), dtype=np.int32)
        code = self._lib.sap_env_eval_vs(self._env, games.ctypes.data, seats.ctypes.data, k, boards.ctypes.data, m, int(samples),
                                         seeds.ctypes.data, counts.ctypes.data)
        if code == -14:
            for game in games.tolist():
                if self.error(game):
                    self._check(code, game)
        self._check(code)
        return counts

    def error(self, game):
        buffer = ctypes.create_string_buffer(2048)
        if self._lib.sap_env_error(self._env, int(game), buffer, len(buffer)) < 0:
            raise IndexError(game)
        return buffer.value.decode(errors="replace")

    def legal(self, game, seat):
        """Flat legal-action mask (FlatActionCodec ids), int8 [197]."""
        mask = np.zeros(197, dtype=np.int8)
        if self._lib.sap_env_legal(self._env, int(game), int(seat), mask.ctypes.data) < 0:
            raise ValueError("no such game or seat, or the game was never reset")
        return mask

    def state(self, game):
        """The canonical state dump (tools/gen_env_golden.py --ext layout; `toggled` differs, see src/canon.rs)."""
        buffer = ctypes.create_string_buffer(1 << 16)
        if self._lib.sap_env_state_json(self._env, int(game), buffer, len(buffer)) < 0:
            raise ValueError("no such game, or the game was never reset")
        return json.loads(buffer.value.decode())
