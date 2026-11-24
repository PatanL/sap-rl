"""Private simulator randomness tape prototype; never an observation feature.

Uniform draws are keyed by immutable world seed/domain/index. Cloning copies
owned cursors, not mutable PRNG state. This module does not install itself into
the simulator; integration and distribution gates remain required.
"""
import hashlib
import math

VERSION = 'paired-random-tape-v1-20260911'


class RandomTape:
    def __init__(self, seed, cursors=None):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Explicit nonnegative 63-bit world seed required')
        self._seed = seed
        self._cursors = dict(cursors or {})
        for domain, index in self._cursors.items():
            self._domain(domain)
            if type(index) is not int or index < 0:
                raise ValueError('Invalid tape cursor')

    @staticmethod
    def _domain(domain):
        if not isinstance(domain, str) or not domain or len(domain) > 96 or not domain.isascii():
            raise ValueError('Explicit bounded ASCII stream domain required')

    def clone(self):
        return type(self)(self._seed, self._cursors)

    def position(self, domain):
        self._domain(domain)
        return self._cursors.get(domain, 0)

    def uniform(self, domain):
        self._domain(domain)
        index = self.position(domain)
        message = f'{VERSION}|{self._seed}|{len(domain)}:{domain}|{index}'.encode()
        bits = int.from_bytes(hashlib.blake2b(message, digest_size=8).digest(), 'big')
        self._cursors[domain] = index+1
        return (bits >> 11) * (2.**-53)

    def sample(self, domain, items, *, weights=None, count=1, replace=True):
        self._domain(domain)
        items = tuple(items)
        if type(count) is not int or count < 0 or type(replace) is not bool or not items:
            raise ValueError('Explicit nonempty pool/count/replacement contract required')
        w = [1.]*len(items) if weights is None else [float(x) for x in weights]
        if len(w) != len(items) or any(not math.isfinite(x) or x < 0 for x in w):
            raise ValueError('Invalid categorical weights')
        if not math.isfinite(math.fsum(w)) or math.fsum(w) <= 0:
            raise ValueError('Positive finite total weight required')
        if not replace and count > sum(x > 0 for x in w):
            raise ValueError('Insufficient positive-weight outcomes without replacement')
        # Validate everything before advancing any cursor.
        chosen=[]
        for _ in range(count):
            threshold=self.uniform(domain)*math.fsum(w)
            cumulative=0.; index=None
            for i, weight in enumerate(w):
                cumulative += weight
                if weight > 0 and threshold < cumulative:
                    index=i; break
            if index is None:
                index=max(i for i,x in enumerate(w) if x > 0)
            chosen.append(items[index])
            if not replace: w[index]=0.
        return chosen

    def shop_draw(self, seat, kind, pool, *, weights=None, frozen=False):
        if type(seat) is not int or seat not in (0,1) or kind not in ('pet','food') or type(frozen) is not bool:
            raise ValueError('Explicit seat/category/frozen contract required')
        if frozen: return None
        return self.sample(f'player/{seat}/shop/{kind}', pool, weights=weights)[0]

    def tier_up(self, seat, eligible_pool):
        if type(seat) is not int or seat not in (0,1): raise ValueError('Invalid seat')
        pool=tuple(eligible_pool)
        if len(set(pool)) != len(pool): raise ValueError('Distinct discover species required')
        return self.sample(f'player/{seat}/shop/pet', pool, count=2, replace=False)

    def private_state(self):
        return dict(schema=VERSION, seed=self._seed, cursors=dict(self._cursors))

    @classmethod
    def from_private_state(cls, value):
        if not isinstance(value,dict) or set(value)!={'schema','seed','cursors'} or value['schema']!=VERSION:
            raise ValueError('Wrong private tape schema')
        if not isinstance(value['cursors'],dict): raise ValueError('Owned cursor mapping required')
        return cls(value['seed'], value['cursors'])
