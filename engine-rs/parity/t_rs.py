import numpy as np, timeit, sys, cProfile, pstats
print(np.__version__)
import numpy._core._ufunc_config as u
import inspect
src = inspect.getsource(u).splitlines()
print("\n".join(src[440:470]))
st = np.random.RandomState(123).get_state()
r = np.random.RandomState(0)
N=20000
print("ctor(0)", 1e6*timeit.timeit(lambda: np.random.RandomState(0), number=N)/N)
print("ctor()", 1e6*timeit.timeit(lambda: np.random.RandomState(), number=N)/N)
print("set_state", 1e6*timeit.timeit(lambda: r.set_state(st), number=N)/N)
print("get_state", 1e6*timeit.timeit(lambda: r.get_state(), number=N)/N)
bg = r._bit_generator
print("bg.state get", 1e6*timeit.timeit(lambda: bg.state, number=N)/N)
d = bg.state
def setd(): bg.state = d
print("bg.state set", 1e6*timeit.timeit(setd, number=N)/N)
print("MT19937()", 1e6*timeit.timeit(lambda: np.random.MT19937(0), number=N)/N)
print("zeros", 1e6*timeit.timeit(lambda: np.zeros(624, dtype=np.uint32), number=N)/N)
import copy, pickle
print("deepcopy", 1e6*timeit.timeit(lambda: copy.deepcopy(r), number=2000)/2000)
p = cProfile.Profile(); p.enable()
for _ in range(2000): r.get_state(); r.set_state(st)
p.disable(); pstats.Stats(p).sort_stats("tottime").print_callers(6)
