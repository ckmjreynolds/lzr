//! Adler-32 checksum implementation.
//!
//! See <https://en.wikipedia.org/wiki/Adler-32> for details on the algorithm.

// The largest prime number smaller than 2^16.
const MOD_ADLER: u32 = 65521;

// The largest N such that: 255·N·(N+1)/2 + (N+1)·(BASE-1) ≤ 2³² − 1
const NMAX: usize = 5552;

// Faster due to auto-vectorization.
const FAST_NMAX: usize = 128;

const _: () = assert!(255 * NMAX * (NMAX + 1) / 2 + (NMAX + 1) * (MOD_ADLER as usize - 1) < u32::MAX as usize);
const _: () = assert!(FAST_NMAX < NMAX);

/// Rolling Adler-32 checksum state.
///
/// # Examples
///
/// ```text
/// let mut ck = Adler32::new();
/// ck.update(b"Wikipedia");
/// assert_eq!(ck.checksum(), 0x11E6_0398);
/// ```
pub(crate) struct Adler32 {
    a: u32,
    b: u32,
}

impl Adler32 {
    /// Creates a new checksum with the Adler-32 initial value (`a = 1, b = 0`).
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            a: 1,
            b: 0,
        }
    }

    /// Restores a rolling checksum from a previously stored value.
    ///
    /// Used to resume hashing across Sonnet boundaries: load the cumulative
    /// checksum from a footer and continue updating from there.
    #[must_use]
    pub(crate) const fn from_checksum(checksum: u32) -> Self {
        Self {
            a: checksum & 0xFFFF,
            b: checksum >> 16,
        }
    }

    /// Feeds `data` into the running checksum.
    ///
    /// Can be called repeatedly to process data in chunks; the result is
    /// identical to a single call with the concatenated input.
    #[expect(clippy::cast_possible_truncation, reason = "const_assert! above prevents truncation.")]
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

    /// Returns the final 32-bit checksum (`b << 16 | a`).
    #[must_use]
    pub(crate) const fn checksum(&self) -> u32 {
        (self.b << 16) | self.a
    }
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
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
        test_known_vector(b"Hi", 0x00FB_00B2_u32);
        test_known_vector(b"Wikipedia", 0x11E6_0398_u32);
    }

    proptest! {
        #[test]
        fn from_checksum_roundtrip(data in prop::collection::vec(any::<u8>(), 0..8_192)) {
            let mut original = super::Adler32::default();
            original.update(&data);

            let restored = super::Adler32::from_checksum(original.checksum());
            prop_assert_eq!(restored.checksum(), original.checksum());

            // Continued updates on the restored state must match the original.
            let more = b"extra data";
            let mut from_original = original;
            from_original.update(more);
            let mut from_restored = restored;
            from_restored.update(more);
            prop_assert_eq!(from_restored.checksum(), from_original.checksum());
        }

        #[test]
        fn test_random_vectors(data in prop::collection::vec(any::<u8>(), 0..8_192)) {
          let mut naive = super::Adler32::new();
          naive.update_naive(&data);

          let mut optimized = super::Adler32::new();
          optimized.update(&data);

          prop_assert_eq!(optimized.checksum(), naive.checksum());
        }

    }
}
