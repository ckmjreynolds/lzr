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

    fn checksum(data: &[u8]) -> u32 {
        let mut h = Adler32::new();
        h.update(data);
        h.finish()
    }

    #[test]
    fn worked_example_hi() {
        // FORMAT.md: Adler-32("Hi") = 0x00FB00B2
        assert_eq!(checksum(b"Hi"), 0x00FB_00B2);
    }

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
            let result = checksum(&data);
            prop_assert_ne!(result & 0xFFFF, 0);
        }
    }
}
