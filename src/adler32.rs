/// Adler-32 checksum as used in the LZR footer.
///
/// The checksum is computed over the uncompressed data and stored as a
/// little-endian `u32` in the stream footer.
///
/// Algorithm: two 16-bit accumulators (`a` starts at 1, `b` starts at 0).
/// For each byte, `a = (a + byte) mod 65521` and `b = (b + a) mod 65521`.
/// The final checksum is `(b << 16) | a`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Adler32 {
    a: u32,
    b: u32,
}

/// Largest prime smaller than 2^16, used as the Adler-32 modulus.
const MOD: u32 = 65521;

/// Maximum number of bytes that can be accumulated before `a` (or `b`)
/// could overflow a `u32`. With the worst-case input of all `0xFF` bytes:
///
/// - `a` grows by at most 255 per byte, starting below `MOD` (65 521).
///   After N bytes: `a_max = 65 520 + 255·N`.
/// - `b` grows by at most `a_max` per byte.
///
/// We need `b_max < 2^32`. A safe bound is `N = 5552`, which is the same
/// constant used by zlib.
const NMAX: usize = 5552;

#[allow(dead_code)]
impl Adler32 {
    /// Creates a new checksum initialized to the Adler-32 starting value (1).
    pub(crate) const fn new() -> Self {
        Self {
            a: 1,
            b: 0,
        }
    }

    /// Feeds a slice of bytes into the running checksum.
    pub(crate) fn update(&mut self, data: &[u8]) {
        for chunk in data.chunks(NMAX) {
            for &byte in chunk {
                self.a += u32::from(byte);
                self.b += self.a;
            }
            self.a %= MOD;
            self.b %= MOD;
        }
    }

    /// Returns the final 32-bit checksum value.
    pub(crate) const fn finish(self) -> u32 {
        (self.b << 16) | self.a
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn checksum(data: &[u8]) -> u32 {
        let mut h = Adler32::new();
        h.update(data);
        h.finish()
    }

    // ── Spec Worked Example ─────────────────────────────────────────────

    #[test]
    fn worked_example_hi() {
        // FORMAT.md: Adler-32("Hi") = 0x00FB00B2
        assert_eq!(checksum(b"Hi"), 0x00FB_00B2);
    }

    // ── Well-Known Values ───────────────────────────────────────────────

    #[test]
    fn empty() {
        // Initial state: a=1, b=0 → 0x0000_0001.
        assert_eq!(checksum(b""), 0x0000_0001);
    }

    #[test]
    fn wikipedia_example() {
        // Wikipedia: Adler-32("Wikipedia") = 0x11E6_0398
        assert_eq!(checksum(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn single_zero_byte() {
        // a = (1 + 0) = 1, b = (0 + 1) = 1 → 0x0001_0001.
        assert_eq!(checksum(&[0x00]), 0x0001_0001);
    }

    #[test]
    fn single_ff_byte() {
        // a = (1 + 255) = 256, b = (0 + 256) = 256 → 0x0100_0100.
        assert_eq!(checksum(&[0xFF]), 0x0100_0100);
    }

    // ── Incremental Update ──────────────────────────────────────────────

    #[test]
    fn incremental_matches_oneshot() {
        let mut incremental = Adler32::new();
        incremental.update(b"Wi");
        incremental.update(b"ki");
        incremental.update(b"pedia");
        assert_eq!(incremental.finish(), checksum(b"Wikipedia"));
    }

    #[test]
    fn byte_at_a_time() {
        let data = b"Hello, world!";
        let mut h = Adler32::new();
        for &byte in data.as_slice() {
            h.update(&[byte]);
        }
        assert_eq!(h.finish(), checksum(data));
    }

    // ── Chunk Boundary (NMAX) ───────────────────────────────────────────

    #[test]
    fn large_input_across_nmax() {
        // Verify the chunked modular reduction is correct for inputs
        // larger than NMAX.
        let data = vec![0xFF; NMAX * 3 + 17];
        let mut naive_a: u64 = 1;
        let mut naive_b: u64 = 0;
        for &byte in &data {
            naive_a = (naive_a + u64::from(byte)) % u64::from(MOD);
            naive_b = (naive_b + naive_a) % u64::from(MOD);
        }
        #[allow(clippy::cast_possible_truncation)]
        let expected = ((naive_b as u32) << 16) | naive_a as u32;
        assert_eq!(checksum(&data), expected);
    }

    // ── Property-Based ──────────────────────────────────────────────────

    proptest! {
        #[test]
        fn incremental_equals_oneshot(
            split in 0usize..256,
            data in prop::collection::vec(any::<u8>(), 0..512),
        ) {
            let split = split.min(data.len());
            let mut h = Adler32::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            prop_assert_eq!(h.finish(), checksum(&data));
        }

        #[test]
        fn never_zero_for_nonempty(data in prop::collection::vec(any::<u8>(), 1..256)) {
            // `a` starts at 1 and only increases (mod 65521), so the
            // low 16 bits can never be zero for non-empty input.
            let result = checksum(&data);
            prop_assert_ne!(result & 0xFFFF, 0);
        }
    }
}
