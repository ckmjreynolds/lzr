//! Adler-32 checksum implementation.
//!
//! Adler-32 is a fast, non-cryptographic checksum used by zlib and other compression formats. It trades a small
//! amount of collision resistance compared to CRC-32 for significantly faster computation.
//!
//! # Examples
//!
//! ```
//! use lzr::adler32::Adler32;
//!
//! let mut checksum = Adler32::new();
//! checksum.update(b"Wikipedia");
//! assert_eq!(checksum.checksum(), 0x11E6_0398);
//! ```
//!
//! Multiple calls to [`Adler32::update`] are equivalent to a single call with the concatenated input:
//!
//! ```
//! use lzr::adler32::Adler32;
//!
//! let mut incremental = Adler32::new();
//! incremental.update(b"Wiki");
//! incremental.update(b"pedia");
//!
//! let mut whole = Adler32::new();
//! whole.update(b"Wikipedia");
//!
//! assert_eq!(incremental.checksum(), whole.checksum());
//! ```
//!
//! See <https://en.wikipedia.org/wiki/Adler-32> for details on the algorithm.

use static_assertions::const_assert;

const MOD_ADLER: u32 = 65521; // The largest prime number smaller than 2^16.
const NMAX: usize = 5552; // The largest N such that: 255·N·(N+1)/2 + (N+1)·(BASE-1) ≤ 2³² − 1
const FAST_NMAX: usize = 128; // Faster due to auto-vectorization.

const_assert!(255 * NMAX * (NMAX + 1) / 2 + (NMAX + 1) * (MOD_ADLER as usize - 1) < u32::MAX as usize);
const_assert!(FAST_NMAX < NMAX);

/// Rolling Adler-32 checksum.
///
/// Create a new instance with [`Adler32::new`], feed data with [`Adler32::update`], and retrieve the final value
/// with [`Adler32::checksum`].
///
/// # Examples
///
/// ```
/// use lzr::adler32::Adler32;
///
/// let mut adler = Adler32::new();
/// adler.update(b"Hello, world!");
/// let sum = adler.checksum();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Adler32 {
    a: u32,
    b: u32,
    len: usize,
}

impl Adler32 {
    /// Creates a new Adler-32 checksum initialized to `1` (the Adler-32 identity value).
    ///
    /// # Examples
    ///
    /// ```
    /// use lzr::adler32::Adler32;
    ///
    /// let adler = Adler32::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            a: 1,
            b: 0,
            len: 0,
        }
    }

    /// Feeds `data` into the running checksum.
    ///
    /// Can be called multiple times to incrementally process data. The result is identical to a single call with
    /// the concatenated input.
    ///
    /// The underlying sums are defined as:
    /// - `A = 1 + D1 + D2 + ... + Dn (mod 65521)`
    /// - `B = (1 + D1) + (1 + D1 + D2) + ... + (1 + D1 + D2 + ... + Dn) (mod 65521)`
    /// - `  = n×D1 + (n−1)×D2 + (n−2)×D3 + ... + Dn + n (mod 65521)`
    /// - `Adler-32(D) = B × 65536 + A`
    ///
    /// where D is the byte sequence being checksummed and n is its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use lzr::adler32::Adler32;
    ///
    /// let mut adler = Adler32::new();
    /// adler.update(b"hello ");
    /// adler.update(b"world");
    /// ```
    #[allow(clippy::cast_lossless)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn update(&mut self, data: &[u8]) {
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

    /// Combines two independently computed checksums for adjacent data blocks.
    ///
    /// If `self` is the checksum of block 1 and `other` is the checksum of block 2,
    /// the returned value is the checksum of block 1 concatenated with block 2.
    ///
    /// This is useful for parallel checksum computation: split the input, checksum each
    /// piece independently, then combine the results.
    ///
    /// # Examples
    ///
    /// ```
    /// use lzr::adler32::Adler32;
    ///
    /// let data = b"Hello, world!";
    /// let split = 5;
    ///
    /// let mut whole = Adler32::new();
    /// whole.update(data);
    ///
    /// let mut first = Adler32::new();
    /// first.update(&data[..split]);
    /// let mut second = Adler32::new();
    /// second.update(&data[split..]);
    ///
    /// assert_eq!(first.combine(&second).checksum(), whole.checksum());
    /// ```
    ///
    /// # Algorithm
    ///
    /// Given block 1 with checksum `(a1, b1)` over `len1` bytes, and block 2 with
    /// checksum `(a2, b2)` over `len2` bytes, the combined checksum is:
    ///
    /// - `a = a1 + a2 - 1 (mod 65521)`
    /// - `b = b1 + b2 + (a1 - 1) × len2 (mod 65521)`
    ///
    /// The `- 1` adjustments cancel the initial `a = 1` baked into each independent
    /// computation.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn combine(self, other: &Self) -> Self {
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

    /// Returns the computed Adler-32 checksum as a `u32`.
    ///
    /// The value is `(B << 16) | A` where A and B are the two running sums reduced modulo 65521.
    ///
    /// # Examples
    ///
    /// ```
    /// use lzr::adler32::Adler32;
    ///
    /// let mut adler = Adler32::new();
    /// adler.update(b"Wikipedia");
    /// assert_eq!(adler.checksum(), 0x11E6_0398);
    /// ```
    #[must_use]
    pub const fn checksum(&self) -> u32 {
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
