"""Drop-in Rust replacement for sapai.battle.Battle(...).battle().

The Rust engine matches the Python reference seed for seed, so the training
environment can call it wherever it only needs the outcome of a seeded fight.
Anything the Rust side does not represent falls back to the Python battle.

    import sap_engine_rs
    sap_engine_rs.install(lib_path, data_dir)        # once per process
    sapai_gym1.rl.selfplay_env.Battle = sap_engine_rs.RustBattle
"""
import ctypes

from sapai.battle import Battle as PythonBattle

_LIB = None
_KIND = {}
_STATUS = {}
STATS = {"rust": 0, "fallback": 0, "reasons": {}}
_LEVEL_EXPERIENCE = {1: 0, 2: 2, 3: 5}


def install(lib_path, data_dir):
    global _LIB
    lib = ctypes.CDLL(lib_path)
    lib.sap_engine_init.argtypes = [ctypes.c_char_p]
    lib.sap_kind.argtypes = [ctypes.c_char_p]
    lib.sap_status.argtypes = [ctypes.c_char_p]
    lib.sap_battle.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                               ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_uint32]
    lib.sap_battle.restype = ctypes.c_int32
    if lib.sap_engine_init(str(data_dir).encode()) != 0:
        raise RuntimeError(f"could not load engine tables from {data_dir}")
    _LIB = lib


def _kind(name):
    if name not in _KIND:
        _KIND[name] = _LIB.sap_kind(name.encode())
    return _KIND[name]


def _status(name):
    if name not in _STATUS:
        _STATUS[name] = _LIB.sap_status(name.encode())
    return _STATUS[name]


class Unsupported(Exception):
    pass


def encode(team):
    """Pack a sapai Team into the flat record the Rust engine takes."""
    cells = []
    count = 0
    for slot in team:
        if slot.empty:
            continue
        pet = slot.pet
        kind, status = _kind(pet.name), _status(pet.status)
        if kind < 0:
            raise Unsupported(f"pet {pet.name}")
        if status < 0:
            raise Unsupported(f"status {pet.status}")
        parrot = -2
        if pet.override_ability:
            origin = pet.ability.get("_copiedFrom")
            if pet.name != "pet-parrot" or origin is None:
                raise Unsupported("override ability on a non-Parrot")
            parrot = _kind(origin)
            if parrot < 0:
                raise Unsupported(f"copied {origin}")
        elif pet.name == "pet-parrot":
            parrot = -1
        if pet.health <= 0 or getattr(pet, "friends_hurt_counter", 0) or getattr(pet, "_hurt", 0) or getattr(pet, "swallowed_pets", None):
            raise Unsupported("pre-battle state the Rust engine does not carry")
        level = int(pet.level)
        if level not in _LEVEL_EXPERIENCE:
            raise Unsupported(f"level {level}")
        cells += [kind, int(pet.attack), int(pet.health), _LEVEL_EXPERIENCE[level], status, int(pet.ability_counter), parrot]
        count += 1
    if count > 5:
        raise Unsupported("more than five pets")
    return (ctypes.c_int32 * len(cells))(*cells), count


def battle(t0, t1, seed):
    first, n_first = encode(t0)
    second, n_second = encode(t1)
    result = _LIB.sap_battle(first, n_first, second, n_second, ctypes.c_uint32(int(seed) % (2 ** 32)))
    if result < 0:
        raise Unsupported(f"engine code {result}")
    return result


class RustBattle:
    """Same call shape as sapai.battle.Battle; only .battle() is provided."""

    def __init__(self, t0, t1, seed=None, fast_training=False):
        self.t0, self.t1, self.seed, self.fast_training = t0, t1, seed, fast_training

    def battle(self):
        if _LIB is not None and self.seed is not None:
            try:
                result = battle(self.t0, self.t1, self.seed)
                STATS["rust"] += 1
                return result
            except Unsupported as reason:
                key = str(reason)
                STATS["reasons"][key] = STATS["reasons"].get(key, 0) + 1
        STATS["fallback"] += 1
        return PythonBattle(self.t0, self.t1, seed=self.seed, fast_training=self.fast_training).battle()
