"""Speed of the Rust environment as Python sees it: random legal play chosen
from the observation masks in numpy, through SapVecEnv, with resets.
Usage: time_vec_env.py <lib> <games> <threads> <seconds>"""
import sys, time
import numpy as np
from sap_env_rs import SapVecEnv, NO_DECISION
LIB, N, T, SECS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
env = SapVecEnv(N, threads=T, library=LIB); rng = np.random.default_rng(1)
seed = 5_000_000; env.reset(np.arange(seed, seed + N, dtype=np.int64)); seed += N
decisions = matches = calls = 0; t_env = 0.0; start = time.perf_counter()
while time.perf_counter() - start < SECS:
    tm = env.obs["target_mask"].reshape(N, 2, -1).astype(np.float64)          # [N,2,9*11*6]
    tm += rng.random(tm.shape) * (tm > 0)                                     # random legal triple
    flat = tm.argmax(-1); act = np.stack(np.unravel_index(flat, (9, 11, 6)), -1).astype(np.int64)
    active = env.active.astype(bool); act[~active] = (NO_DECISION, 10, 5)
    masks = np.zeros((N, 2), dtype=np.int64)
    t0 = time.perf_counter(); env.step(act, masks); t_env += time.perf_counter() - t0
    decisions += int(active.sum()); calls += 1
    done = env.terminated.astype(bool)
    if done.any():
        k = int(done.sum()); matches += k
        new = np.zeros(N, dtype=np.int64); new[done] = np.arange(seed, seed + k); seed += k
        env.reset(new, which=done)
total = time.perf_counter() - start
bad = int((env.status < 0).sum())
print(f"games {N} threads {T}: {decisions/total:,.0f} decisions/s, {matches/total:,.0f} matches/s overall; env.step alone {decisions/t_env:,.0f} decisions/s ({1e3*t_env/calls:.2f} ms per call); bad status {bad}")
