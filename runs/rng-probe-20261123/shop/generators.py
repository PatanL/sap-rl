#!/usr/bin/env python3
"""Reference implementations of candidate generators (stdlib only). Each returns (list of K raw outputs, range R)
for an integer seed.  Run this file directly for the self-test against published vectors."""
import random as _pyrandom

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
MBIG = 2147483647


def _i32(x):
    x &= M32
    return x - (1 << 32) if x & 0x80000000 else x


def _dotnet_core(seed, k, inextp0, ms_quirk):
    """C# reference source, line by line, with 32-bit wrap-around emulated (C# int arithmetic is unchecked).
    The wrap matters: SeedArray[55] = 161803398 - |seed| is negative for every seed above 161803398, and
    the mixing rounds then overflow int32.  A port without the wrap (as in ../chain/chain_probe.py) is only
    guaranteed correct for |seed| <= 161803398, about 7.5% of the logged seeds; prime_candidates.py measures
    how often it actually differs.  ../test_generators.py (the battle probe) does emulate the wrap and agrees
    with this port."""
    seed = _i32(seed)
    sub = MBIG if seed == -2147483648 else abs(seed)
    mj = 161803398 - sub
    a = [0] * 56
    a[55] = mj
    mk = 1
    for i in range(1, 55):
        ii = (21 * i) % 55
        a[ii] = mk
        mk = _i32(mj - mk)
        if mk < 0:
            mk = _i32(mk + MBIG)
        mj = a[ii]
    for _ in range(4):
        for i in range(1, 56):
            v = _i32(a[i] - a[1 + (i + 30) % 55])
            if v < 0:
                v = _i32(v + MBIG)
            a[i] = v
    inext, inextp, out = 0, inextp0, []
    for _ in range(k):
        inext = inext + 1 if inext + 1 < 56 else 1
        inextp = inextp + 1 if inextp + 1 < 56 else 1
        r = _i32(a[inext] - a[inextp])
        if ms_quirk and r == MBIG:
            r -= 1
        if r < 0:
            r = _i32(r + MBIG)
        a[inext] = r
        out.append(r & 0x7FFFFFFF)       # a negative value can only arise from the int.MinValue corner
    return out, MBIG


def dotnet(seed, k):
    """.NET System.Random(seed): Knuth subtractive generator (inextp = 21). Same outputs in .NET Framework,
    current Mono/Unity (reference-source corlib) and .NET Core / 5+ when a seed is given (CompatPrng).
    Output = InternalSample() = Next();  Next(n) = (int)(Sample * n) with Sample = output / 2147483647."""
    return _dotnet_core(seed, k, 21, True)


def mono_old(seed, k):
    """System.Random of old Mono (pre reference-source corlib, as in old Unity): same seeding, inextp = 31,
    no 'retVal == MBIG' adjustment."""
    return _dotnet_core(seed, k, 31, False)


def dotnet_nowrap(seed, k):
    """The port WITHOUT int32 wrap-around, as in ../chain/chain_probe.py. Kept only to show the difference."""
    sub = MBIG if seed == -2147483648 else abs(seed)
    mj = 161803398 - sub
    a = [0] * 56
    a[55] = mj
    mk = 1
    for i in range(1, 55):
        ii = (21 * i) % 55
        a[ii] = mk
        mk = mj - mk
        if mk < 0:
            mk += MBIG
        mj = a[ii]
    for _ in range(4):
        for i in range(1, 56):
            a[i] -= a[1 + (i + 30) % 55]
            if a[i] < 0:
                a[i] += MBIG
    inext, inextp, out = 0, 21, []
    for _ in range(k):
        inext = inext + 1 if inext + 1 < 56 else 1
        inextp = inextp + 1 if inextp + 1 < 56 else 1
        r = a[inext] - a[inextp]
        if r == MBIG:
            r -= 1
        if r < 0:
            r += MBIG
        a[inext] = r
        out.append(r)
    return out, MBIG


def unity(seed, k):
    """UnityEngine.Random.InitState(seed): xorshift128, state filled with the MT-style multiplier 1812433253."""
    s = [seed & M32]
    for _ in range(3):
        s.append((s[-1] * 1812433253 + 1) & M32)
    x, y, z, w = s
    out = []
    for _ in range(k):
        t = (x ^ (x << 11)) & M32
        x, y, z = y, z, w
        w = (w ^ (w >> 19) ^ t ^ (t >> 8)) & M32
        out.append(w)
    return out, 1 << 32


_R = _pyrandom.Random()


def mt19937(seed, k):
    """Mersenne Twister, init_genrand(seed), genrand_int32()."""
    mt = [seed & M32]
    for i in range(1, 624):
        p = mt[-1]
        mt.append((1812433253 * (p ^ (p >> 30)) + i) & M32)
    _R.setstate((3, tuple(mt) + (624,), None))
    return [_R.getrandbits(32) for _ in range(k)], 1 << 32


def _pcg(initstate, initseq, k):
    mult = 6364136223846793005
    inc = ((initseq << 1) | 1) & M64
    state = inc                                   # (0*mult + inc)
    state = (state + initstate) & M64
    state = (state * mult + inc) & M64
    out = []
    for _ in range(k):
        old = state
        state = (old * mult + inc) & M64
        xs = (((old >> 18) ^ old) >> 27) & M32
        rot = old >> 59
        out.append(((xs >> rot) | (xs << ((-rot) & 31))) & M32)
    return out, 1 << 32


def pcg32_seq54(seed, k):
    return _pcg(seed & M64, 54, k)


def pcg32_default(seed, k):
    """pcg32 rng(seed) from pcg-cpp: default stream constant 1442695040888963407."""
    return _pcg(seed & M64, 721347520444481703, k)


def pcg32_seqseed(seed, k):
    return _pcg(seed & M64, seed & M64, k)


def _splitmix_stream(seed, k):
    x = seed & M64
    out = []
    for _ in range(k):
        x = (x + 0x9E3779B97F4A7C15) & M64
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        out.append(z ^ (z >> 31))
    return out


def splitmix64_hi(seed, k):
    return [v >> 32 for v in _splitmix_stream(seed, k)], 1 << 32


def splitmix64_lo(seed, k):
    return [v & M32 for v in _splitmix_stream(seed, k)], 1 << 32


def xorshift32(seed, k):
    x = (seed & M32) or 1
    out = []
    for _ in range(k):
        x ^= (x << 13) & M32
        x ^= x >> 17
        x ^= (x << 5) & M32
        out.append(x)
    return out, 1 << 32


def _xorshift64(seed, k):
    x = (seed & M64) or 1
    out = []
    for _ in range(k):
        x ^= (x << 13) & M64
        x ^= x >> 7
        x ^= (x << 17) & M64
        out.append(x)
    return out


def xorshift64_lo(seed, k):
    return [v & M32 for v in _xorshift64(seed, k)], 1 << 32


def xorshift64_hi(seed, k):
    return [v >> 32 for v in _xorshift64(seed, k)], 1 << 32


def xorshift64star(seed, k):
    x = (seed & M64) or 1
    out = []
    for _ in range(k):
        x ^= x >> 12
        x ^= (x << 25) & M64
        x ^= x >> 27
        out.append(((x * 2685821657736338717) & M64) >> 32)
    return out, 1 << 32


def _rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & M64


def _xoshiro256ss(seed, k):
    s = _splitmix_stream(seed, 4)
    out = []
    for _ in range(k):
        out.append((_rotl((s[1] * 5) & M64, 7) * 9) & M64)
        t = (s[1] << 17) & M64
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
    return out


def xoshiro256ss_hi(seed, k):
    return [v >> 32 for v in _xoshiro256ss(seed, k)], 1 << 32


def xoshiro256ss_lo(seed, k):
    return [v & M32 for v in _xoshiro256ss(seed, k)], 1 << 32


def xoroshiro128plus_hi(seed, k):
    s0, s1 = _splitmix_stream(seed, 2)
    out = []
    for _ in range(k):
        out.append(((s0 + s1) & M64) >> 32)
        s1 ^= s0
        s0 = _rotl(s0, 24) ^ s1 ^ ((s1 << 16) & M64)
        s1 = _rotl(s1, 37)
    return out, 1 << 32


def java_util_random(seed, k):
    """java.util.Random(seed): next(31), the value nextInt(bound) is built from."""
    mask = (1 << 48) - 1
    s = (seed ^ 0x5DEECE66D) & mask
    out = []
    for _ in range(k):
        s = (s * 0x5DEECE66D + 0xB) & mask
        out.append(s >> 17)
    return out, 1 << 31


def glibc_random(seed, k):
    """glibc srandom(seed); random() (TYPE_3 additive feedback)."""
    r = [(seed & M32) or 1]
    if r[0] & 0x80000000:
        r[0] -= 1 << 32
    for i in range(1, 31):
        hi, lo = divmod(r[i - 1], 127773) if r[i - 1] >= 0 else (-((-r[i - 1]) // 127773), -((-r[i - 1]) % 127773))
        word = 16807 * lo - 2836 * hi
        if word < 0:
            word += 2147483647
        r.append(word)
    for i in range(31, 34):
        r.append(r[i - 31])
    for i in range(34, 344):
        r.append((r[i - 31] + r[i - 3]) & M32)
    out = []
    for i in range(344, 344 + k):
        v = (r[i - 31] + r[i - 3]) & M32
        r.append(v)
        out.append(v >> 1)
    return out, 1 << 31


def _lcg(a, c, m, shift, mask, rng):
    def f(seed, k):
        s = seed % m
        out = []
        for _ in range(k):
            s = (a * s + c) % m
            out.append((s >> shift) & mask)
        return out, rng
    return f


msvc_rand = _lcg(214013, 2531011, 1 << 32, 16, 0x7FFF, 1 << 15)
ansi_c_31 = _lcg(1103515245, 12345, 1 << 31, 0, 0x7FFFFFFF, 1 << 31)
posix_rand = _lcg(1103515245, 12345, 1 << 32, 16, 0x7FFF, 1 << 15)
nr_lcg = _lcg(1664525, 1013904223, 1 << 32, 0, M32, 1 << 32)
nr_lcg_hi16 = _lcg(1664525, 1013904223, 1 << 32, 16, 0xFFFF, 1 << 16)
borland = _lcg(22695477, 1, 1 << 32, 16, 0x7FFF, 1 << 15)
knuth_mmix_hi = _lcg(6364136223846793005, 1442695040888963407, 1 << 64, 32, M32, 1 << 32)


def _minstd(a):
    def f(seed, k):
        s = seed % MBIG or 1
        out = []
        for _ in range(k):
            s = (a * s) % MBIG
            out.append(s)
        return out, MBIG
    return f


minstd_16807 = _minstd(16807)
minstd_48271 = _minstd(48271)


def _fmix32(h):
    h &= M32
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & M32
    h ^= h >> 16
    return h


def murmur_fmix_counter(seed, k):
    """Stateless hash generator: fmix32(seed + i)."""
    return [_fmix32(seed + i) for i in range(k)], 1 << 32


def murmur_fmix_golden(seed, k):
    return [_fmix32((seed & M32) ^ ((i * 0x9E3779B9) & M32)) for i in range(k)], 1 << 32


def identity(seed, k):
    """The seed itself (output 0), then seed + i: catches 'the seed IS the random number'."""
    return [(seed + i) & M32 for i in range(k)], 1 << 32


def knuth_mult_hash(seed, k):
    """(seed + i) * 2654435761 mod 2^32."""
    return [((seed + i) * 2654435761) & M32 for i in range(k)], 1 << 32


GENERATORS = {
    'dotnet System.Random': dotnet,
    'old Mono System.Random (inextp 31)': mono_old,
    'unity xorshift128 InitState': unity,
    'mt19937 init_genrand': mt19937,
    'pcg32 (seed, seq 54)': pcg32_seq54,
    'pcg32 (seed, default stream)': pcg32_default,
    'pcg32 (seed, seq seed)': pcg32_seqseed,
    'splitmix64 high 32': splitmix64_hi,
    'splitmix64 low 32': splitmix64_lo,
    'xorshift32 (13,17,5)': xorshift32,
    'xorshift64 (13,7,17) low 32': xorshift64_lo,
    'xorshift64 (13,7,17) high 32': xorshift64_hi,
    'xorshift64*': xorshift64star,
    'xoshiro256** (splitmix seeded) high 32': xoshiro256ss_hi,
    'xoshiro256** (splitmix seeded) low 32': xoshiro256ss_lo,
    'xoroshiro128+ (splitmix seeded) high 32': xoroshiro128plus_hi,
    'java.util.Random next(31)': java_util_random,
    'glibc random()': glibc_random,
    'MSVC rand()': msvc_rand,
    'LCG ANSI C mod 2^31': ansi_c_31,
    'LCG POSIX rand() example': posix_rand,
    'LCG Numerical Recipes full state': nr_lcg,
    'LCG Numerical Recipes high 16': nr_lcg_hi16,
    'LCG Borland high 15': borland,
    'LCG Knuth MMIX high 32': knuth_mmix_hi,
    'MINSTD 16807': minstd_16807,
    'MINSTD 48271': minstd_48271,
    'murmur fmix32(seed+i)': murmur_fmix_counter,
    'murmur fmix32(seed^i*phi)': murmur_fmix_golden,
    'identity seed+i': identity,
    'Knuth multiplicative hash (seed+i)': knuth_mult_hash,
}


js_lcg_233280 = _lcg(9301, 49297, 233280, 0, (1 << 32) - 1, 233280)


def mulberry32(seed, k):
    a = seed & M32
    out = []
    for _ in range(k):
        a = (a + 0x6D2B79F5) & M32
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        out.append((t ^ (t >> 14)) & M32)
    return out, 1 << 32


def xorshift128plus_hi(seed, k):
    s0, s1 = _splitmix_stream(seed, 2)
    out = []
    for _ in range(k):
        x, y = s0, s1
        s0 = y
        x ^= (x << 23) & M64
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        out.append(((s1 + y) & M64) >> 32)
    return out, 1 << 32


EXTRA = {
    'LCG 9301/49297 mod 233280': js_lcg_233280,
    'mulberry32': mulberry32,
    'xorshift128+ (splitmix seeded) high 32': xorshift128plus_hi,
}
GENERATORS.update(EXTRA)


def selftest():
    ok = True

    def chk(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(('ok   ' if good else 'FAIL ') + name, '' if good else f'got {got} want {want}')
    chk('.NET Random(0) first three', dotnet(0, 3)[0], [1559595546, 1755192844, 1649316166])
    chk('.NET Random(1) first three', dotnet(1, 3)[0], [534011718, 237820880, 1002897798])
    chk('.NET Random(42) first', dotnet(42, 1)[0], [1434747710])
    chk('MT19937 seed 5489 first', mt19937(5489, 1)[0], [3499211612])
    chk('MT19937 seed 1 first', mt19937(1, 1)[0], [1791095845])
    chk('PCG32 (42, 54) first three', pcg32_seq54(42, 3)[0], [0xa15c02b7, 0x7b47f409, 0xba1d3330])
    chk('splitmix64(0) first', _splitmix_stream(0, 1), [0xE220A8397B1DCDAF])
    chk('splitmix64(1234567) first two', _splitmix_stream(1234567, 2), [6457827717110365317, 3203168211198807973])
    chk('xorshift32(1) first two', xorshift32(1, 2)[0], [270369, 67634689])
    chk('java.util.Random(0).nextInt()', java_util_random(0, 1)[0], [(-1155484576 & M32) >> 1])
    chk('java.util.Random(42).nextInt()', java_util_random(42, 1)[0], [(-1170105035 & M32) >> 1])
    chk('glibc srand(1) first three', glibc_random(1, 3)[0], [1804289383, 846930886, 1681692777])
    chk('MSVC srand(1) first three', msvc_rand(1, 3)[0], [41, 18467, 6334])
    chk('MINSTD 16807 from 1, 10000th', minstd_16807(1, 10000)[0][-1], 1043618065)
    same = sum(dotnet(sd, 4)[0] == dotnet_nowrap(sd, 4)[0] for sd in range(1000003, 2147483647, 2147483))
    print('port without int32 wrap agrees with the wrapped port on', same, 'of 1000 seeds spread over [1e6, 2^31)')
    print('   wrapped   Random(2000000000) first three:', dotnet(2000000000, 3)[0], ' (no published vector checked for seeds > 161803398)')
    print('   unwrapped Random(2000000000) first three:', dotnet_nowrap(2000000000, 3)[0])
    print('Unity xorshift128 has no vector I can check offline; algorithm as in ../chain/chain_probe.py:',
          'InitState(1) first three raw =', unity(1, 3)[0])
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if selftest() else 1)
