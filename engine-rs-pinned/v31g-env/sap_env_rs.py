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
"""
import ctypes
import json
import os

import numpy as np

NO_DECISION = -1
STATUS_OK, STATUS_FINISHED, STATUS_ILLEGAL, STATUS_BROKEN, STATUS_NOT_RESET = 0, 1, -1, -2, -3

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

    def step(self, actions, masks):
        """One decision per active seat of every running game. Returns the number of games whose
        status is negative after the call."""
        actions = np.ascontiguousarray(actions, dtype=np.int64)
        masks = np.ascontiguousarray(masks, dtype=np.int64)
        if actions.shape != (self.num_games, 2, 3) or masks.shape != (self.num_games, 2):
            raise ValueError(f"expected actions [{self.num_games}, 2, 3] and masks [{self.num_games}, 2]")
        code = self._lib.sap_env_step(self._env, actions.ctypes.data, masks.ctypes.data, ctypes.byref(self._out))
        if code < 0:
            raise RuntimeError(f"sap_env_step failed with code {code}")
        return code

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
