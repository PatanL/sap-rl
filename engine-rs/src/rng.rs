//! NumPy legacy `RandomState` (MT19937) with exactly the draw shapes the
//! Python battle path uses. Seed-exact parity with the reference engine
//! depends on every call here consuming the same 32-bit words as NumPy.
//!
//! Shapes used by the reference (battle.py / effects.py):
//! - `rs.choice(a, (1,))` and `rs.choice(a, size=(1,))`, with replacement:
//!   `randint(0, len)` -> one masked-rejection draw, none when `len == 1`.
//! - `rs.choice(a, (1,), replace=False)`: `permutation(len)[:1]`, i.e. the
//!   whole index range is Fisher-Yates shuffled even for one draw.
//! - `rng.choice(np.arange(n), size=n, replace=False)`: `permutation(n)`.
//! - `rs.random_sample()`: 53-bit double from two words.

#[derive(Clone)]
pub struct Mt {
    words: [u32; 624],
    pos: usize,
}

impl Mt {
    /// `RandomState.seed(int)` -> `mt19937_seed`.
    pub fn new(seed: u32) -> Self {
        let mut words = [0u32; 624];
        words[0] = seed;
        for i in 1..624 {
            let prev = words[i - 1];
            words[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self { words, pos: 624 }
    }

    #[inline]
    fn word(&mut self) -> u32 {
        if self.pos == 624 {
            for i in 0..624 {
                let y = (self.words[i] & 0x8000_0000) | (self.words[(i + 1) % 624] & 0x7fff_ffff);
                self.words[i] = self.words[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ (if y & 1 != 0 { 0x9908_b0df } else { 0 });
            }
            self.pos = 0;
        }
        let mut y = self.words[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `rk_interval(max)`: uniform in `[0, max]` by masked rejection.
    #[inline]
    pub fn interval(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        let mask = u32::MAX >> (max as u32).leading_zeros();
        loop {
            let v = (self.word() & mask) as usize;
            if v <= max {
                return v;
            }
        }
    }

    /// `permutation(n)`: shuffled `arange(n)`.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut a: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.interval(i);
            a.swap(i, j);
        }
        a
    }

    /// `choice(arange(n), (1,))[0]` with replacement.
    #[inline]
    pub fn choice_replace(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        self.interval(n - 1)
    }

    /// `choice(arange(n), (1,), replace=False)[0]`.
    #[inline]
    pub fn choice_noreplace(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        // Only the element that ends at index 0 matters, but every swap
        // consumes draws, so run the full shuffle.
        let mut a: [usize; 64];
        if n <= 64 {
            a = [0; 64];
            for (i, v) in a.iter_mut().enumerate().take(n) {
                *v = i;
            }
            for i in (1..n).rev() {
                let j = self.interval(i);
                a.swap(i, j);
            }
            a[0]
        } else {
            self.permutation(n)[0]
        }
    }

    /// `get_state()` reduced to three numbers for the checker: the position
    /// in the block and the first and last key words. Does not advance.
    pub fn probe(&self) -> (usize, u32, u32) {
        (self.pos, self.words[0], self.words[623])
    }

    /// `random_sample()`.
    #[inline]
    pub fn random_sample(&mut self) -> f64 {
        let a = (self.word() >> 5) as f64;
        let b = (self.word() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values printed by NumPy for `RandomState(); seed(1234567)`:
    /// five `choice(arange(5),(1,))`, five `choice(arange(5),(1,),replace=False)`,
    /// `choice(arange(4), size=4, replace=False)`, `random_sample()`,
    /// `choice(['a','b','c'],(1,))`.
    #[test]
    fn matches_numpy() {
        let mut rs = Mt::new(1234567);
        let a: Vec<usize> = (0..5).map(|_| rs.choice_replace(5)).collect();
        assert_eq!(a, vec![3, 4, 2, 0, 0]);
        let b: Vec<usize> = (0..5).map(|_| rs.choice_noreplace(5)).collect();
        assert_eq!(b, vec![4, 0, 0, 0, 4]);
        assert_eq!(rs.permutation(4), vec![1, 2, 3, 0]);
        assert_eq!(rs.random_sample(), 0.8412098688686316);
        assert_eq!(rs.choice_replace(3), 0);
    }

    /// `SuperAutoPetsEnv.reset`: four `RandomState(seed).randint(0, 2**32 - 1)` child seeds,
    /// then, from `RandomState(child[1])` (a team's `seed_state`), five
    /// `choice(arange(10), (1,))` and one `random_sample()`. Values printed by NumPy.
    #[test]
    fn child_seeds_match_numpy() {
        let cases: [(u32, [u32; 4], [usize; 5], f64); 4] = [
            (881100000, [407604618, 2127920271, 52932538, 3981627331], [9, 8, 6, 7, 7], 0.04523447334008013),
            (0, [2357136044, 2546248239, 3071714933, 3626093760], [2, 4, 9, 5, 6], 0.9124847317821784),
            (4294967295, [419326371, 479346978, 3918654476, 2416749639], [2, 4, 6, 1, 5], 0.4345797990139785),
            (123456789, [2288500408, 4254805660, 2294099250, 56498137], [0, 1, 9, 5, 3], 0.5478766016476257),
        ];
        for (seed, children, picks, sample) in cases {
            let mut rs = Mt::new(seed);
            let got: Vec<u32> = (0..4).map(|_| rs.interval(4294967294) as u32).collect();
            assert_eq!(got, children.to_vec(), "seed {seed}");
            let mut team = Mt::new(children[1]);
            let drawn: Vec<usize> = (0..5).map(|_| team.choice_replace(10)).collect();
            assert_eq!(drawn, picks.to_vec(), "seed {seed}");
            assert_eq!(team.random_sample(), sample, "seed {seed}");
        }
    }
}
