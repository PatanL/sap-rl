import numpy as np


class MockRandomState:
    """
    Numpy RandomState is actually extremely slow, requiring about 300 microseconds
    for any operation involving state. Therefore, when reproducibility is not
    necessary, this class should be used to immensly improve efficiency.

    Tests were run for Player:
        %timeit _ = Player.from_state(pstate)
        # ORIGINAL: 26.8 ms ± 788 µs per loop
        # MockRandomState: 1.15 ms ± 107 µs per loop
    Use of MockRandomState improved the performance by 10x. This is very important.

    """

    def __init__(self):
        pass

    def set_state(self):
        """Doesn't do anything"""
        return None

    def get_state(self):
        return None

    def choice(self, *args, **kwargs):
        return np.random.choice(*args, **kwargs)


class LazyRandomState:
    """A ``RandomState`` restored from ``seed_state`` only when first used.

    Speed only, no behaviour change. Building a numpy ``RandomState`` and
    restoring its state costs about 90 microseconds, and pets, foods and shop
    slots each built a private one that the owning Player then replaced with
    the shop's stream. Every attribute is forwarded to the real generator,
    which is created on first use from the same state, so any draw it makes is
    identical to the eager version.
    """

    __slots__ = ("_seed_state", "_real")

    def __init__(self, seed_state):
        self._seed_state = seed_state
        self._real = None

    def _materialize(self):
        real = self._real
        if real is None:
            real = np.random.RandomState(0)
            real.set_state(self._seed_state)
            self._real = real
        return real

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._materialize(), name)
