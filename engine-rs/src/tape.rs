//! `sapai.paired_random_tape.RandomTape`: uniform draws keyed by world seed,
//! stream domain and index, hashed with BLAKE2b (8-byte digest).
//!
//! `uniform(domain)` is `(blake2b(f"{VERSION}|{seed}|{len(domain)}:{domain}|{index}") >> 11) * 2**-53`
//! with the digest read big-endian. Cursors are kept by the caller (one
//! counter per domain), exactly as the reference keeps them per tape object.

pub const VERSION: &str = "paired-random-tape-v1-20260911";

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for (i, word) in m.iter_mut().enumerate() {
        *word = u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for s in SIGMA.iter() {
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// `hashlib.blake2b(data, digest_size=8).digest()` read as a big-endian integer.
pub fn blake2b8(data: &[u8]) -> u64 {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 8;
    let mut offset = 0usize;
    let mut block = [0u8; 128];
    // Every block but the last is a full block; an empty message is one zero block.
    while data.len() - offset > 128 {
        block.copy_from_slice(&data[offset..offset + 128]);
        offset += 128;
        compress(&mut h, &block, offset as u128, false);
    }
    let rest = &data[offset..];
    block = [0u8; 128];
    block[..rest.len()].copy_from_slice(rest);
    compress(&mut h, &block, data.len() as u128, true);
    u64::from_be_bytes(h[0].to_le_bytes())
}

/// `RandomTape(seed).uniform(domain)` at cursor `index`.
pub fn uniform(seed: u64, domain: &str, index: u64) -> f64 {
    use std::io::Write;
    let mut buf = [0u8; 192];
    let mut cur = std::io::Cursor::new(&mut buf[..]);
    write!(cur, "{}|{}|{}:{}|{}", VERSION, seed, domain.len(), domain, index).expect("tape message fits");
    let n = cur.position() as usize;
    let bits = blake2b8(&buf[..n]);
    (bits >> 11) as f64 * (1.0 / 9007199254740992.0)
}

/// One tape domain with the constant part of its message prepared:
/// `uniform(index)` equals `tape::uniform(seed, domain, index)`.
#[derive(Clone)]
pub struct Stream {
    message: [u8; 128],
    prefix: usize,
}

impl Stream {
    pub fn new(seed: u64, domain: &str) -> Stream {
        let text = format!("{}|{}|{}:{}|", VERSION, seed, domain.len(), domain);
        // One BLAKE2b block holds the prefix and any 20-digit index.
        assert!(text.len() + 20 <= 128, "tape domain too long for the single-block fast path");
        let mut message = [0u8; 128];
        message[..text.len()].copy_from_slice(text.as_bytes());
        Stream { message, prefix: text.len() }
    }

    pub fn uniform(&mut self, index: u64) -> f64 {
        let mut digits = [0u8; 20];
        let mut n = 0;
        let mut value = index;
        loop {
            digits[n] = b'0' + (value % 10) as u8;
            n += 1;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        let mut len = self.prefix;
        for k in (0..n).rev() {
            self.message[len] = digits[k];
            len += 1;
        }
        let bits = blake2b8(&self.message[..len]);
        (bits >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

/// `RandomTape.sample(domain, items, weights=[w]*n)[0]`: one draw over `n`
/// equal weights `w`, with `total == math.fsum([w]*n)`.
#[inline]
pub fn pick_equal(u: f64, n: usize, w: f64, total: f64) -> usize {
    let threshold = u * total;
    let mut cumulative = 0.0f64;
    for i in 0..n {
        cumulative += w;
        if w > 0.0 && threshold < cumulative {
            return i;
        }
    }
    n - 1
}

/// `RandomTape.tier_up`: two distinct indices out of `n` with unit weights,
/// from two consecutive uniforms.
pub fn pick_two_distinct(u0: f64, u1: f64, n: usize) -> (usize, usize) {
    debug_assert!(n >= 2 && n <= 64);
    let mut w = [1.0f64; 64];
    let mut out = [0usize; 2];
    for (k, u) in [u0, u1].into_iter().enumerate() {
        let total: f64 = w[..n].iter().sum(); // small integers: exact, equals fsum
        let threshold = u * total;
        let mut cumulative = 0.0f64;
        let mut index = None;
        for (i, weight) in w[..n].iter().enumerate() {
            cumulative += *weight;
            if *weight > 0.0 && threshold < cumulative {
                index = Some(i);
                break;
            }
        }
        let index = index.unwrap_or_else(|| (0..n).rev().find(|&i| w[i] > 0.0).unwrap());
        out[k] = index;
        w[index] = 0.0;
    }
    (out[0], out[1])
}

/// SHA-256 (for observation hashes in the reference traces).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values printed by the reference:
    /// `RandomTape(881100000).uniform('player/0/shop/pet')` (index 0, 1),
    /// `RandomTape(2**63-1).uniform('battle/turn/7')`, and hashlib digests.
    #[test]
    fn matches_reference() {
        assert_eq!(blake2b8(b""), 0xe4a6a0577479b2b4);
        assert_eq!(blake2b8(b"abc"), 0xd8bb14d833d59559);
        let long = vec![b'x'; 300];
        assert_eq!(blake2b8(&long), TEST_LONG);
        assert_eq!(blake2b8(&vec![b'y'; 128]), 0x154f7ef96586c007);
        assert_eq!(blake2b8(&vec![b'z'; 256]), 0xfbd186ffb70d096c);
        assert_eq!(uniform(881100000, "player/0/shop/pet", 0), TEST_U0);
        assert_eq!(uniform(881100000, "player/0/shop/pet", 1), TEST_U1);
        assert_eq!(uniform(9223372036854775807, "battle/turn/7", 0), TEST_U2);
        let mut stream = Stream::new(881100000, "player/0/shop/pet");
        assert_eq!(stream.uniform(1), TEST_U1);
        assert_eq!(stream.uniform(0), TEST_U0);
        for index in [9u64, 10, 99, 100, 12345, u64::MAX] {
            assert_eq!(stream.uniform(index), uniform(881100000, "player/0/shop/pet", index));
        }
        let d = sha256(b"abc");
        assert_eq!(d[..4], [0xba, 0x78, 0x16, 0xbf]);
    }
    const TEST_LONG: u64 = 0x12b539c1cad94615;
    const TEST_U0: f64 = 0.39518722863231026;
    const TEST_U1: f64 = 0.09502611224612989;
    const TEST_U2: f64 = 0.7607629447160115;
}
