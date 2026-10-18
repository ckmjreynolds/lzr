//! Adler-32 checksum implementation.
//!
//! See <https://en.wikipedia.org/wiki/Adler-32> for details on the algorithm.

// The largest prime number smaller than 2^16.
const MOD_ADLER: u32 = 65521;

// The largest N such that: 255·N·(N+1)/2 + (N+1)·(BASE-1) ≤ 2³² − 1
const NMAX: usize = 5552;

// "Chunk" size for auto-vectorization.
const FAST_NMAX: usize = 128;

const _: () = assert!(255 * NMAX * (NMAX + 1) / 2 + (NMAX + 1) * (MOD_ADLER as usize - 1) < u32::MAX as usize);
const _: () = assert!(FAST_NMAX < NMAX);

pub(crate) struct Adler32 {
    a: u32,
    b: u32,
}

impl Adler32 {
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            a: 1,
            b: 0,
        }
    }

    #[expect(clippy::cast_possible_truncation, reason = "assert! above prevents truncation.")]
    pub(crate) fn update(&mut self, data: &[u8]) {
        // AUTOVECTORIZED: Don't touch without benchmarking!
        //
        // FAST_NMAX represents the value I've found that performs the best. ~12GB/s vs ~7GB/s
        // checksum/compute        time:   [309.80 µs 310.32 µs 310.83 µs]
        //                         thrpt:  [12.127 GiB/s 12.147 GiB/s 12.167 GiB/s]
        for chunk in data.chunks(FAST_NMAX) {
            let mut a = 0u32;
            let mut b = 0u32;
            let n = chunk.len();

            for (i, &d) in chunk.iter().enumerate() {
                let d = u32::from(d);
                a += d;
                b += d * ((n - i) as u32);
            }

            self.b = (self.b + (self.a * (n as u32) + b)) % MOD_ADLER;
            self.a = (self.a + a) % MOD_ADLER;
        }
    }

    #[must_use]
    pub(crate) const fn checksum(&self) -> u32 {
        (self.b << 16) | self.a
    }

    #[must_use]
    pub(crate) const fn from_checksum(checksum: u32) -> Self {
        Self {
            a: checksum & 0xFFFF,
            b: checksum >> 16,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
impl Adler32 {
    /// Naive per-byte implementation. Used as a reference for testing.
    fn update_naive(&mut self, data: &[u8]) {
        for byte in data {
            self.a = (self.a + u32::from(*byte)) % MOD_ADLER;
            self.b = (self.b + self.a) % MOD_ADLER;
        }
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
        test_known_vector(b"Wikipedia", 0x11E6_0398_u32);
    }

    proptest! {
        #[test]
        fn test_random_vectors(data in prop::collection::vec(any::<u8>(), 0..8_192)) {
            let mut naive = super::Adler32::new();
            let mut optimized = super::Adler32::new();

            naive.update_naive(&data);
            optimized.update(&data);

            prop_assert_eq!(optimized.checksum(), naive.checksum());
        }

        #[test]
        fn from_checksum_roundtrip(data in prop::collection::vec(any::<u8>(), 0..8_192)) {
            let mut original = super::Adler32::new();
            original.update(&data);

            let mut restored = super::Adler32::from_checksum(original.checksum());
            prop_assert_eq!(restored.checksum(), original.checksum());

            original.update(b"extra");
            restored.update(b"extra");
            prop_assert_eq!(original.checksum(), restored.checksum());
        }
    }
}
