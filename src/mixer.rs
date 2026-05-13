//! Logit-space mixer for binary probability predictors.
//!
//! Given `N` predictors each producing `P(bit = 0)` (in AC's
//! `TOTAL`-scaled units), the mixer maintains per-context weights
//! `w_i` and reports a mixed probability via
//!
//! ```text
//!   z       = Σ w_i · stretch(P_i)
//!   P_mixed = squash(z)
//! ```
//!
//! where `stretch(p) = ln(p / (1 - p))` and `squash(z) = 1 / (1 + e^{-z})`.
//!
//! Weights are updated by online gradient descent on log-loss. Both
//! encoder and decoder follow the same observe schedule, so the
//! weight state stays in lock-step — no signaling bits are emitted.
//! Compared to the Phase-18 router (which paid `ceil(log2(N))` bits
//! per token plus a `BitPredictor` to compress the router decision),
//! the mixer pays *zero* signaling bits and can blend predictors per
//! bit instead of choosing one for the whole token.
//!
//! Determinism: the mixer's only non-deterministic ingredients are
//! `f32::ln`/`f32::exp`. On a single machine these are bit-exact
//! between encode and decode, which is all the codec needs. The
//! Hutter judging machine runs the same binary for both passes.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::ac::TOTAL;

/// Per-slot weight precision. `f32` is bit-exact for the basic ops
/// we use; the only transcendental is `ln`/`exp` in stretch/squash,
/// which is fine within one process.
pub(crate) type Weight = f32;

/// Maximum logit magnitude after stretching. Probabilities at the
/// edge of `[1, TOTAL - 1]` produce logits of about ±11.1 for
/// `TOTAL = 65536`; clamping at ±16 keeps the gradient updates from
/// exploding without measurably affecting precision.
const LOGIT_CLAMP: f32 = 16.0;

/// Convert AC-scale `P(bit = 0) ∈ [1, TOTAL - 1]` to a logit.
fn stretch(p_zero: u32) -> f32 {
    debug_assert!((1..TOTAL).contains(&p_zero));
    let p = (p_zero as f32) / (TOTAL as f32);
    let z = (p / (1.0 - p)).ln();
    z.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
}

/// Convert a logit back to AC-scale `P(bit = 0) ∈ [1, TOTAL - 1]`.
fn squash(z: f32) -> u32 {
    let p = 1.0 / (1.0 + (-z).exp());
    let scaled = (p * (TOTAL as f32)) as i32;
    // Clamp to the AC's strict-increasing-CDF contract.
    scaled.clamp(1, (TOTAL as i32) - 1) as u32
}

/// Logit-space mixer of `N` binary predictors.
///
/// `K_BITS` controls the per-context weight table size: `1 << K_BITS`
/// slots, each holding `N` `f32` weights (`4 N` bytes). Contexts
/// collide by hash truncation — fine for the common case (rare
/// contexts contribute little) and matches the `BitPredictor`
/// addressing convention.
pub(crate) struct LogitMixer<const N: usize> {
    weights: Vec<[Weight; N]>,
    mask: u64,
    learning_rate: f32,
}

impl<const N: usize> LogitMixer<N> {
    /// New mixer with `1 << k_bits` weight slots, all weights
    /// initialized to `1/N` so cold contexts return the unweighted
    /// average of the input predictors. Learning rate is the
    /// stepsize for the SGD weight update.
    #[must_use]
    pub(crate) fn new(k_bits: u32, learning_rate: f32) -> Self {
        assert!(N >= 1, "mixer needs at least one predictor");
        let n_slots = 1usize << k_bits;
        let init: Weight = 1.0 / (N as f32);
        Self {
            weights: vec![[init; N]; n_slots],
            mask: u64::try_from(n_slots - 1).expect("table size fits u64"),
            learning_rate,
        }
    }

    fn slot(&self, ctx_hash: u64) -> usize {
        usize::try_from(ctx_hash & self.mask).expect("masked u64 fits usize on supported targets")
    }

    /// Mixed `P(bit = 0)`. Pure read; does not mutate.
    pub(crate) fn predict(&self, ctx_hash: u64, p_zeros: &[u32; N]) -> u32 {
        let slot = self.slot(ctx_hash);
        let w = &self.weights[slot];
        let mut z = 0.0_f32;
        for i in 0..N {
            z += w[i] * stretch(p_zeros[i]);
        }
        squash(z)
    }

    /// Predict + observe in one pass. Returns the mixed `P(bit = 0)`
    /// (same value `predict` would have returned) so the caller can
    /// use it for AC encoding/decoding without re-computing.
    ///
    /// Updates per-slot weights by SGD on log-loss:
    ///   `w_i += lr · (target - mixed) · stretch(P_i)`
    /// where `target = 1` if `bit == 0`, else `0`.
    pub(crate) fn predict_and_observe(
        &mut self,
        ctx_hash: u64,
        p_zeros: &[u32; N],
        bit: u32,
    ) -> u32 {
        let slot = self.slot(ctx_hash);
        let mut logits = [0.0_f32; N];
        for i in 0..N {
            logits[i] = stretch(p_zeros[i]);
        }
        let w = &mut self.weights[slot];
        let mut z = 0.0_f32;
        for i in 0..N {
            z += w[i] * logits[i];
        }
        let p_mixed_f = 1.0 / (1.0 + (-z).exp());
        let target = if bit == 0 { 1.0 } else { 0.0 };
        let error = target - p_mixed_f;
        let step = self.learning_rate * error;
        for i in 0..N {
            w[i] += step * logits[i];
        }
        squash(z)
    }

    /// Observe-only variant for the prewarm path. Same weight
    /// update as `predict_and_observe` but no mixed probability is
    /// returned (the prewarm doesn't need it).
    pub(crate) fn observe(&mut self, ctx_hash: u64, p_zeros: &[u32; N], bit: u32) {
        let _ = self.predict_and_observe(ctx_hash, p_zeros, bit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_roundtrips_extremes() {
        let z_min = stretch(1);
        let z_max = stretch(TOTAL - 1);
        assert!(z_min <= -10.0);
        assert!(z_max >= 10.0);
        assert!(z_min >= -LOGIT_CLAMP);
        assert!(z_max <= LOGIT_CLAMP);
    }

    #[test]
    fn squash_stays_in_ac_range() {
        for z_i in -20..=20 {
            let z = z_i as f32;
            let p = squash(z);
            assert!(p >= 1, "z={z}, p={p}");
            assert!(p < TOTAL, "z={z}, p={p}");
        }
    }

    #[test]
    fn cold_mixer_returns_average_of_inputs() {
        let blender: LogitMixer<2> = LogitMixer::new(8, 0.01);
        // Two predictors: one says P(0) = 80%, other says P(0) = 20%.
        // Average in logit-space is logit(0.8)/2 + logit(0.2)/2 = 0,
        // so mixed P(0) = 0.5.
        let p_high = TOTAL * 8 / 10;
        let p_low = TOTAL * 2 / 10;
        let out = blender.predict(0, &[p_high, p_low]);
        let expected = TOTAL / 2;
        let diff = (out as i32 - expected as i32).abs();
        assert!(
            diff < (TOTAL / 100) as i32,
            "out={out}, expected={expected}"
        );
    }

    #[test]
    fn mixer_learns_to_prefer_better_predictor() {
        let mut blender: LogitMixer<2> = LogitMixer::new(4, 0.05);
        // Predictor 0 is right (always says bit=0 with high confidence)
        // Predictor 1 is wrong (always says bit=1 with high confidence)
        let p_right = TOTAL * 9 / 10;
        let p_wrong = TOTAL / 10;
        let ctx = 42u64;
        // Train: every observed bit is actually 0.
        for _ in 0..5000 {
            blender.predict_and_observe(ctx, &[p_right, p_wrong], 0);
        }
        // After training, the mixed prediction at this context should
        // strongly favor 0.
        let out = blender.predict(ctx, &[p_right, p_wrong]);
        assert!(
            out > TOTAL * 7 / 10,
            "mixer didn't learn: out P(0) = {out} (expected > {})",
            TOTAL * 7 / 10
        );
    }

    #[test]
    fn distinct_contexts_train_independently() {
        let mut blender: LogitMixer<2> = LogitMixer::new(8, 0.05);
        let p_a = TOTAL * 9 / 10;
        let p_b = TOTAL / 10;
        // Context 1: bit is always 0; predictor 0 is right.
        // Context 2: bit is always 1; predictor 1 is right.
        for _ in 0..5000 {
            blender.predict_and_observe(1, &[p_a, p_b], 0);
            blender.predict_and_observe(2, &[p_a, p_b], 1);
        }
        let pred_1 = blender.predict(1, &[p_a, p_b]);
        let pred_2 = blender.predict(2, &[p_a, p_b]);
        assert!(
            pred_1 > TOTAL * 6 / 10,
            "ctx 1 should favor predictor 0: {pred_1}"
        );
        assert!(
            pred_2 < TOTAL * 4 / 10,
            "ctx 2 should favor predictor 1: {pred_2}"
        );
    }
}
