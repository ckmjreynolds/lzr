//! Adler-32 checksum implementation.
//!
//! See <https://en.wikipedia.org/wiki/Adler-32> for details on the algorithm.

use static_assertions::const_assert;

// The largest prime number smaller than 2^16.
const MOD_ADLER: u32 = 65521;

// The largest N such that: 255·N·(N+1)/2 + (N+1)·(BASE-1) ≤ 2³² − 1
const NMAX: usize = 5552;

// Faster due to auto-vectorization.
const FAST_NMAX: usize = 128;

const_assert!(255 * NMAX * (NMAX + 1) / 2 + (NMAX + 1) * (MOD_ADLER as usize - 1) < u32::MAX as usize);
const_assert!(FAST_NMAX < NMAX);

/// Rolling Adler-32 checksum state.
///
/// Maintains the two 16-bit halves (`a` and `b`) and the total byte count,
/// supporting incremental updates and combination of independent checksums.
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
    len: usize,
}

impl Adler32 {
    /// Creates a new checksum with the Adler-32 initial value (`a = 1, b = 0`).
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            a: 1,
            b: 0,
            len: 0,
        }
    }

    /// Feeds `data` into the running checksum.
    ///
    /// Can be called repeatedly to process data in chunks; the result is
    /// identical to a single call with the concatenated input.
    #[allow(clippy::cast_lossless)]
    #[allow(clippy::cast_possible_truncation)]
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
                a += d as u32;
                b += d as u32 * ((n - i) as u32);
            }

            self.b = (self.b + (self.a * (n as u32) + b)) % MOD_ADLER;
            self.a = (self.a + a) % MOD_ADLER;
        }

        self.len += data.len();
    }

    /// Combines two independently computed checksums into one, as if the
    /// underlying data had been checksummed in a single pass.
    ///
    /// This enables parallel checksum computation: split the data, compute
    /// each part separately, then combine.
    ///
    /// # Examples
    ///
    /// ```text
    /// let data = b"Hello, world!";
    /// let (left, right) = data.split_at(5);
    ///
    /// let mut a = Adler32::new();
    /// a.update(left);
    /// let mut b = Adler32::new();
    /// b.update(right);
    ///
    /// let mut whole = Adler32::new();
    /// whole.update(data);
    ///
    /// assert_eq!(a.combine(&b).checksum(), whole.checksum());
    /// ```
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn combine(self, other: &Self) -> Self {
        let a1 = u64::from(self.a);
        let b1 = u64::from(self.b);
        let a2 = u64::from(other.a);
        let b2 = u64::from(other.b);
        let len2 = other.len as u64 % u64::from(MOD_ADLER);

        let m = u64::from(MOD_ADLER);
        let a = (a1 + a2 + m - 1) % m;
        let b = (b1 + b2 + (a1 + m - 1) * len2) % m;

        Self {
            a: a as u32,
            b: b as u32,
            len: self.len + other.len,
        }
    }

    /// Returns the final 32-bit checksum (`b << 16 | a`).
    #[must_use]
    pub(crate) const fn checksum(&self) -> u32 {
        (self.b << 16) | self.a
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
impl Adler32 {
    /// Naive per-byte implementation. Used as a reference for testing.
    #[allow(clippy::cast_lossless)]
    fn update_naive(&mut self, data: &[u8]) {
        for byte in data {
            self.a = (self.a + *byte as u32) % MOD_ADLER;
            self.b = (self.b + self.a) % MOD_ADLER;
        }
        self.len += data.len();
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
