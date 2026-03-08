//! Adler-32 checksum implementation.

const MOD_ADLER: u64 = 65521;
const BLOCK: usize = 128;

// Maximum safe input length for a single `update` call. With u64 accumulators,
// worst-case `b` grows as ~255·n²/2. This limit keeps b below `u64::MAX`.
static_assertions::const_assert!((((MAX_UPDATE_LEN as u128).pow(2) * 255) / 2) < u64::MAX as u128);
const MAX_UPDATE_LEN: usize = 380_368_697;

/// Rolling Adler-32 checksum.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Adler32 {
    a: u32,
    b: u32,
    len: u64,
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Adler32 {
    /// Creates a new Adler-32 checksum with the initial value.
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            a: 1,
            b: 0,
            len: 0,
        }
    }

    // Adler-32 is defined as two running sums reduced modulo 65521:
    //
    //     a = 1 + d[0] + d[1] + ... + d[n-1]
    //     b = (1) + (1+d[0]) + (1+d[0]+d[1]) + ... + (1+d[0]+...+d[n-1])
    //
    // Two optimizations are applied here:
    //
    // 1. DEFERRED REDUCTION — Using u64 accumulators instead of u32 lets us
    //    defer the modulo reduction to the very end of the call, eliminating
    //    the periodic reduction loop that u32 requires every 5,552 bytes.
    //    This is safe for inputs up to MAX_UPDATE_LEN (~362 MiB).
    //
    // 2. BLOCK DECOMPOSITION — The naive inner loop has a serial dependency:
    //    each `b += a` depends on the just-updated a, which prevents SIMD
    //    vectorization. We can reformulate the update over a block of n bytes:
    //
    //        a_new = a + s1              where s1 = sum of bytes
    //        b_new = b + n*a + s2        where s2 = n*d[0] + (n-1)*d[1] + ... + 1*d[n-1]
    //
    //    Both s1 and s2 are independent per-element reductions with no loop-
    //    carried dependency, so LLVM can auto-vectorize them.
    /// Feeds `data` into the running checksum.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn update(&mut self, data: &[u8]) {
        debug_assert!(data.len() <= MAX_UPDATE_LEN, "Adler32::update - Buffer too large!");

        let (mut a, mut b) = (u64::from(self.a), u64::from(self.b));

        for block in data.chunks(BLOCK) {
            let n = block.len() as u32;
            let mut s1 = 0u32;
            let mut s2 = 0u32;

            for (i, &byte) in block.iter().enumerate() {
                let val = u32::from(byte);
                s1 += val;
                s2 += val * (n - i as u32);
            }

            b += a * u64::from(n) + u64::from(s2);
            a += u64::from(s1);
        }

        self.a = (a % MOD_ADLER) as u32;
        self.b = (b % MOD_ADLER) as u32;
        self.len += data.len() as u64;
    }

    // Combine two independently computed checksums for adjacent blocks.
    //
    // If block 1 has checksum (a1, b1) over len1 bytes, and block 2 has
    // checksum (a2, b2) over len2 bytes, the combined checksum is:
    //
    //     a = a1 + a2 - 1              (mod 65521)
    //     b = b1 + b2 + (a1 - 1) * len2   (mod 65521)
    //
    // The -1 adjustments account for the initial a=1 in each independent
    // computation.
    /// Combines two independently computed checksums for adjacent blocks.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn combine(self, other: &Self) -> Self {
        let a1 = u64::from(self.a);
        let b1 = u64::from(self.b);
        let a2 = u64::from(other.a);
        let b2 = u64::from(other.b);
        let len2 = other.len % MOD_ADLER;

        let a = (a1 + a2 - 1) % MOD_ADLER;
        let b = (b1 + b2 + (a1 - 1) * len2) % MOD_ADLER;

        Self {
            a: a as u32,
            b: b as u32,
            len: self.len + other.len,
        }
    }

    /// Returns the computed Adler-32 checksum.
    #[must_use]
    pub(crate) const fn checksum(&self) -> u32 {
        (self.b << 16) | self.a
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
impl Adler32 {
    // Naive per-byte implementation. Used as a reference for testing.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn update_naive(&mut self, data: &[u8]) {
        for byte in data {
            self.a = (self.a + u32::from(*byte)) % MOD_ADLER as u32;
            self.b = (self.b + self.a) % MOD_ADLER as u32;
        }
        self.len += data.len() as u64;
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod test {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    fn test_known_vector(data: &[u8], expected: u32) {
        let mut adler = super::Adler32::new();
        adler.update(data);
        assert_eq!(adler.checksum(), expected);
    }

    #[test]
    fn known_vectors() {
        test_known_vector(b"", 0x0000_0001_u32);
        test_known_vector(b"Hi", 0x00FB_00B2_u32);
        test_known_vector(b"Wikipedia", 0x11E6_0398_u32);
    }

    proptest! {
        #[test]
          fn test_random_vectors(data in prop::collection::vec(any::<u8>(), 0..65536)) {
              let mut naive = super::Adler32::new();
              naive.update_naive(&data);

              let mut optimized = super::Adler32::new();
              optimized.update(&data);

              prop_assert_eq!(optimized.checksum(), naive.checksum());
          }

        #[test]
          fn test_combine(data in prop::collection::vec(any::<u8>(), 2..65536)) {
              let split = data.len() / 2;

              let mut whole = super::Adler32::new();
              whole.update(&data);

              let mut first = super::Adler32::new();
              first.update(&data[..split]);
              let mut second = super::Adler32::new();
              second.update(&data[split..]);

              let combined = first.combine(&second);

              prop_assert_eq!(combined.checksum(), whole.checksum());
          }
    }
}
