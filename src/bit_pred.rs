//! Bit-level adaptive predictor for PAQ-style codecs.
//!
//! For each context (an opaque 64-bit hash provided by the codec) we
//! maintain a small `(n0, n1)` count pair and report `P(bit = 0)`
//! quantized to the AC's `TOTAL` mass via Laplace `+1` smoothing.
//!
//! The context space is much bigger than the count table (e.g.,
//! Order-3 byte history × bit-position × partial-byte ≈ 2^33 logical
//! contexts), so the hash key is truncated to a power-of-two index.
//! Collisions are tolerated — distinct contexts that hash to the
//! same slot share counts, which is fine for the common case (sparse
//! contexts contribute little) and worth measuring at scale.
//!
//! Per-context counts use `u16` with an in-place halving step once
//! `n0 + n1` exceeds `2^14 - 2`. Keeps the model responsive to drift
//! without unbounded memory growth.

use crate::ac::TOTAL;

const RESCALE_THRESHOLD: u32 = 4094;

#[derive(Debug)]
pub(crate) struct BitPredictor {
    /// `[n0, n1]` count pairs, one per hashed context slot.
    counts: Vec<[u16; 2]>,
    mask: u64,
}

impl BitPredictor {
    /// Build a predictor whose context table has `1 << k_bits` slots.
    /// Memory is `4 * (1 << k_bits)` bytes (two `u16` per slot).
    pub(crate) fn new(k_bits: u32) -> Self {
        let n = 1usize << k_bits;
        Self {
            counts: vec![[0u16; 2]; n],
            mask: u64::try_from(n - 1).expect("table size fits u64"),
        }
    }

    /// Truncate the hashed context to a valid table slot.
    fn slot(ctx_hash: u64, mask: u64) -> usize {
        usize::try_from(ctx_hash & mask).expect("masked u64 fits usize on supported targets")
    }

    /// Probability that the next bit at this context is 0, scaled so
    /// `result + p1_scaled == TOTAL`. Always in `[1, TOTAL - 1]` so
    /// the AC's strictly-increasing-CDF contract holds.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn predict_p_zero(&self, ctx_hash: u64) -> u32 {
        let [n0, n1] = self.counts[Self::slot(ctx_hash, self.mask)];
        let n0 = u32::from(n0);
        let n1 = u32::from(n1);
        let total = n0 + n1 + 2;
        // p_zero = (n0 + 1) / total, scaled to TOTAL.
        let scaled = ((n0 + 1) * TOTAL) / total;
        scaled.clamp(1, TOTAL - 1)
    }

    /// Record an observed bit at this context.
    pub(crate) fn observe(&mut self, ctx_hash: u64, bit: u32) {
        let slot = Self::slot(ctx_hash, self.mask);
        let entry = &mut self.counts[slot];
        if bit == 0 {
            entry[0] = entry[0].saturating_add(1);
        } else {
            entry[1] = entry[1].saturating_add(1);
        }
        let total = u32::from(entry[0]) + u32::from(entry[1]);
        if total > RESCALE_THRESHOLD {
            entry[0] = (entry[0] >> 1).max(1);
            entry[1] = (entry[1] >> 1).max(1);
        }
    }
}

/// FNV-1a-style hash. Used to combine context features (recent
/// bytes, bit position, partial-byte-so-far) into a single 64-bit
/// key that the predictor truncates to its table-size mask.
pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
pub(crate) const fn fnv_mix(h: u64, v: u64) -> u64 {
    let h = h ^ v;
    h.wrapping_mul(FNV_PRIME)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_predictor_returns_balanced_probability() {
        let p = BitPredictor::new(10);
        let p_zero = p.predict_p_zero(12345);
        // With no observations: p_zero = 1/2, scaled to TOTAL.
        assert_eq!(p_zero, TOTAL / 2);
    }

    #[test]
    fn observed_bias_shifts_prediction() {
        let mut p = BitPredictor::new(10);
        let ctx = 12345_u64;
        for _ in 0..100 {
            p.observe(ctx, 0);
        }
        let p_zero = p.predict_p_zero(ctx);
        // After 100 zeros: p_zero = 101/102 of TOTAL, near max.
        assert!(p_zero > TOTAL * 9 / 10);
    }

    #[test]
    fn predictor_never_returns_extreme_values() {
        let mut p = BitPredictor::new(10);
        let ctx = 777_u64;
        // Saturate with 1s.
        for _ in 0..100_000 {
            p.observe(ctx, 1);
        }
        let p_zero = p.predict_p_zero(ctx);
        // Must stay in [1, TOTAL - 1] for the AC contract.
        assert!(p_zero >= 1);
        assert!(p_zero < TOTAL);
    }

    #[test]
    fn fnv_distinguishes_simple_inputs() {
        let a = fnv_mix(fnv_mix(FNV_OFFSET, 0x41), 0x42);
        let b = fnv_mix(fnv_mix(FNV_OFFSET, 0x42), 0x41);
        assert_ne!(a, b, "FNV-1a should distinguish input order");
    }
}
