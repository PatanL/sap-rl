"""Shadows runs/rust-collector-20261123/rust_adapter.py on purpose (this directory is first on PYTHONPATH): the same code on the newer pinned library engine-rs-pinned/v31g-env2, which adds copy, rekey, set_team, set_gold, battle_events and eval_vs and is unchanged on everything older. The library file is hashed at import."""
from pathlib import Path

import numpy as np
from sap_env_rs import SapVecEnv

PIN = Path('/sap/engine-rs-pinned/v31g-env2')
LIB_SHA = '68147ae0dd09304979b2a661cab375a3d871d73c41957a07a8ea7ad2599e6eac'
INACTIVE = (-1, 10, 5)


class RustVector:
    def __init__(self, games, threads):
        self.native = SapVecEnv(games, threads=threads, library=str(PIN/'libsap_engine.so'))
        n = self.native
        self.output = {key: getattr(n, key) for key in ('obs','winner','battle_result','lives','wins')}
        # Boolean indexing and torch terminals require bool; Rust guarantees int8 0/1.
        for key in ('active','battle_resolved','terminated'):
            self.output[key] = getattr(n, key).view(np.bool_)
        self.output['rewards'] = np.zeros((games, 2), dtype=np.float32)
        self.actions = np.empty((games,2,3),dtype=np.int64)
        self.freeze = np.empty((games,2),dtype=np.int64)
        self.seeds = np.zeros(games,dtype=np.int64)
        self.selected = np.zeros(games,dtype=np.bool_)

    def check(self, code):
        if code:
            i = int(np.flatnonzero(self.native.status < 0)[0])
            raise RuntimeError(f'Rust environment status {self.native.status[i]} at game {i}: {self.native.error(i)}')

    def reset(self, indices, seeds, output):
        if output is not self.output: raise ValueError('Use the vector-owned output views')
        if len(indices) != len(seeds): raise ValueError('Reset selection length mismatch')
        self.selected.fill(False)
        self.selected[indices] = True
        self.seeds[indices] = seeds
        self.check(self.native.reset(self.seeds, which=self.selected))
        output['rewards'][indices] = self.native.rewards[indices]

    def step(self, actions, freeze, output):
        if output is not self.output: raise ValueError('Use the vector-owned output views')
        np.copyto(self.actions,actions)
        np.copyto(self.freeze,freeze)
        inactive = ~output['active']
        self.actions[inactive] = INACTIVE
        self.freeze[inactive] = 0
        self.check(self.native.step(self.actions,self.freeze))
        # Scalar rewards are double; both collectors store training rewards as float32.
        np.copyto(output['rewards'],self.native.rewards,casting='same_kind')

    def close(self):
        self.native.close()

import hashlib as _hashlib
with open(PIN/'libsap_engine.so', 'rb') as _fp:
    if _hashlib.file_digest(_fp, 'sha256').hexdigest() != LIB_SHA:
        raise RuntimeError('pinned Rust library changed')
