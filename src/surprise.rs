//! Coding-surprise tracker: how badly the coder has been predicting lately, as a context.
//!
//! [`Surprise`] keeps two exponentially decayed averages of the *final* per-bit coding cost — a fast
//! one spanning roughly the last byte and a slow one spanning roughly the last four — and quantizes
//! them to small levels. Downstream stages key on those levels: the two-layer mixer's order-1
//! sub-mixer selects its weight set by (previous byte × fast level), its blend layer by the
//! (fast, slow) pair, and a third SSE stage refines the mixed probability per (fast level × previous
//! byte × bit-tree node) — "we are in an unpredictable stretch, be less confident". The signal is a
//! pure function of the already-coded `(probability, bit)` sequence, identical on encode and decode,
//! so it costs nothing on the wire and the stream still round-trips bit-for-bit. It is the model's own
//! recent loss used as a regime detector — a "confidence" context no byte-history context expresses.
//!
//! Swept on a 20 MB enwik8 slice: the *fast* timescale carries almost all of the gain (shift 3 ≈ shift
//! 2 > shift 4 ≫ shift 5), the slow timescale is nearly flat (shifts 5–8 within 0.0003 bpb), and
//! keying the order-1 sub-mixer and the order-1 APM on the fast level each earn ~−0.005 bpb and stack
//! to ~−0.0096 bpb; a fifth surprise-keyed sub-mixer and a 64-way (fast × slow) mixer key added <0.0005.

use std::sync::OnceLock;

/// 12-bit probability scale, matching the mixer/coder.
const PROB_ONE: i32 = 1 << 12;

/// Decay shift of the fast average (~2^3 bits = one byte of memory).
const FAST_SHIFT: u32 = 3;
/// Decay shift of the slow average (~2^5 bits = four bytes of memory).
const SLOW_SHIFT: u32 = 5;

/// Quantization levels per average (log2-spaced thresholds on the mean cost per bit); the range of
/// [`Surprise::fast_level`].
pub(crate) const LEVELS: usize = 8;
/// Number of distinct buckets [`Surprise::bucket`] returns (`LEVELS` fast × `LEVELS` slow).
pub(crate) const BUCKETS: usize = LEVELS * LEVELS;

/// Per-outcome coding cost in sixteenths of a bit: `cost16()[p]` = `round(-16·log2(p / 4096))` for
/// the probability `p` the coder assigned the outcome that occurred. `p = 0` never occurs (the coder
/// clamps to `1..=4095`) but is given the `p = 1` cost so the table is total.
fn cost16() -> &'static Vec<u32> {
    static LUT: OnceLock<Vec<u32>> = OnceLock::new();
    LUT.get_or_init(|| {
        (0..PROB_ONE)
            .map(|p| {
                let cost = -16.0 * (f64::from(p.max(1)) / f64::from(PROB_ONE)).log2();
                // Non-negative by construction; ≤ 16·12 = 192, so the truncation is exact.
                #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "0 ≤ cost ≤ 192")]
                let cost = cost.round() as u32;
                cost
            })
            .collect()
    })
}

/// Two-timescale running average of the final per-bit coding cost, quantized to a regime bucket.
#[derive(Debug)]
pub(crate) struct Surprise {
    /// Fast decayed cost sum, scaled by `2^FAST_SHIFT` (its mean is `fast >> FAST_SHIFT`).
    fast: u32,
    /// Slow decayed cost sum, scaled by `2^SLOW_SHIFT`.
    slow: u32,
}

impl Surprise {
    /// A fresh tracker primed at a typical text coding cost (~0.25 bit per bit) so the first bytes
    /// start in a mid bucket rather than the "perfectly predicted" corner.
    pub(crate) const fn new() -> Self {
        Self {
            fast: 4 << FAST_SHIFT,
            slow: 4 << SLOW_SHIFT,
        }
    }

    /// Fold in the coder's final 12-bit probability `p` (that the bit was 1) and the actual `bit`.
    pub(crate) fn update(&mut self, p: i32, bit: u8) {
        let p_correct = if bit == 1 {
            p
        } else {
            PROB_ONE - p
        };
        let cost = cost16()[usize::try_from(p_correct.clamp(1, PROB_ONE - 1)).unwrap_or(1)];
        self.fast = self.fast + cost - (self.fast >> FAST_SHIFT);
        self.slow = self.slow + cost - (self.slow >> SLOW_SHIFT);
    }

    /// The fast-average level alone, `0..LEVELS` — the timescale that carries nearly all the signal.
    pub(crate) fn fast_level(&self) -> usize {
        level(self.fast >> FAST_SHIFT)
    }

    /// The current regime bucket, `0..BUCKETS`: the fast level in the high bits, the slow in the low.
    pub(crate) fn bucket(&self) -> usize {
        level(self.fast >> FAST_SHIFT) * LEVELS + level(self.slow >> SLOW_SHIFT)
    }
}

/// Quantize a mean cost per bit (in sixteenths of a bit) onto `0..LEVELS` with log2-spaced edges at
/// 1, 2, 4, 8, 16, 32, 64 sixteenths — i.e. ~0.06, 0.13, 0.25, 0.5, 1, 2, 4 bits per bit.
fn level(mean_cost: u32) -> usize {
    // `ilog2(x + 1)` is 0 for x = 0, 1 for 1..=2, 2 for 3..=6, … — a log2 ladder starting at 1.
    (usize::try_from((mean_cost + 1).ilog2()).unwrap_or(0)).min(LEVELS - 1)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn cost_table_is_monotone_and_bounded() {
        let lut = cost16();
        assert_eq!(lut[PROB_ONE as usize - 1], 0);
        assert!(lut[1] <= 192);
        assert!(lut.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn confident_correct_predictions_drive_bucket_to_zero() {
        let mut s = Surprise::new();
        for _ in 0..2000 {
            s.update(4095, 1);
        }
        assert_eq!(s.bucket(), 0);
    }

    #[test]
    fn confident_wrong_predictions_drive_bucket_to_max() {
        let mut s = Surprise::new();
        for _ in 0..2000 {
            s.update(4095, 0);
        }
        assert_eq!(s.bucket(), BUCKETS - 1);
    }

    #[test]
    fn fast_reacts_before_slow() {
        let mut s = Surprise::new();
        for _ in 0..2000 {
            s.update(4095, 1);
        }
        for _ in 0..8 {
            s.update(4095, 0);
        }
        let b = s.bucket();
        assert!(b / LEVELS > b % LEVELS, "fast level should lead: bucket={b}");
    }
}
