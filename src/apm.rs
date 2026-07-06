//! Secondary symbol estimation (SSE) via an adaptive probability map (APM).
//!
//! An [`Apm`] refines the mixer's probability: it looks the (stretched) probability up in a small
//! per-context table, interpolates between the two straddling knots, and adapts them toward the
//! observed bit. This corrects systematic miscalibration of the mix at near-zero cost — a couple of
//! table lookups per bit, all cache-resident — and is the classic first refinement stage on top of a
//! logistic mixer. It is deterministic: encode and decode evolve the identical table from the
//! identical `(probability, context, bit)` sequence, so the stream still round-trips bit-for-bit.

use crate::mixer::{squash, stretch};

/// Interpolation knots across the stretched-probability axis: one every 128 logit units over the
/// `[-2048, 2048]` range (32 intervals, 33 knots).
const KNOTS: usize = 33;

/// Adaptation-rate shift for the APM knot update (higher = slower).
const RATE: i32 = 7;

/// An adaptive probability map: a `contexts × KNOTS` table of probability knots refined per bit.
#[derive(Debug)]
pub(crate) struct Apm {
    /// `contexts * KNOTS` probability knots, each scaled by 16 for sub-12-bit update resolution.
    t: Vec<i32>,
    /// Lower knot chosen by the last [`Apm::refine`], reused by [`Apm::update`].
    idx: usize,
    /// Interpolation weight (`0..=127`) of the last refine, reused by [`Apm::update`].
    weight: i32,
}

impl Apm {
    /// A fresh APM over `contexts` low-order contexts, initialized to the identity map — its refined
    /// output equals its input until adaptation moves the knots.
    pub(crate) fn new(contexts: usize) -> Self {
        let mut t = vec![0i32; contexts * KNOTS];
        for cx in 0..contexts {
            let base = cx * KNOTS;
            for j in 0..KNOTS {
                // Knot j sits at stretched value `(j - 16) * 128`; store its squashed prob, scaled 16.
                let knot = i32::try_from(j).unwrap_or(0) - 16;
                t[base + j] = squash(knot * 128) * 16;
            }
        }
        Self {
            t,
            idx: 0,
            weight: 0,
        }
    }

    /// Refine the 12-bit probability `p` under context `cx`, returning the interpolated 12-bit
    /// probability (clamped to `1..=4095`). Records the knot and weight for the paired [`Apm::update`].
    pub(crate) fn refine(&mut self, p: i32, cx: usize) -> i32 {
        let s = stretch(p) + 2048; // 1..=4095
        let bin = usize::try_from(s >> 7).unwrap_or(0); // 0..=31
        self.weight = s & 127;
        self.idx = cx * KNOTS + bin;
        let lo = self.t[self.idx];
        let hi = self.t[self.idx + 1];
        // >>11 unscales the ×16 knot storage (>>4) and the 0..128 interpolation weight (>>7).
        ((lo * (128 - self.weight) + hi * self.weight) >> 11).clamp(1, 4095)
    }

    /// Adapt the two knots straddling the last refine toward the observed `bit`.
    pub(crate) fn update(&mut self, bit: u8) {
        let b = i32::from(bit);
        let g = (b << 16) + (b << RATE) - b - b; // scaled target: ~65662 for 1, 0 for 0
        self.t[self.idx] += (g - self.t[self.idx]) >> RATE;
        self.t[self.idx + 1] += (g - self.t[self.idx + 1]) >> RATE;
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// A fresh APM is ~identity: refining a probability returns approximately itself (within the
    /// knot-interpolation quantization).
    #[test]
    fn fresh_apm_is_near_identity() {
        let mut apm = Apm::new(4);
        for p in [100, 1000, 2048, 3000, 4000] {
            let r = apm.refine(p, 1);
            assert!((r - p).abs() <= 40, "p={p} refined={r}");
        }
    }

    /// Repeatedly observing bit=1 in a context drives that context's refined probability upward.
    #[test]
    fn adapts_toward_observed_bit() {
        let mut apm = Apm::new(4);
        let start = apm.refine(2048, 2);
        for _ in 0..200 {
            let _ = apm.refine(2048, 2);
            apm.update(1);
        }
        let end = apm.refine(2048, 2);
        assert!(end > start, "expected upward adaptation: start={start} end={end}");
    }
}
