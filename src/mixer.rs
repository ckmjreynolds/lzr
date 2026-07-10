//! Logistic context mixing.
//!
//! [`stretch`] / [`squash`] move between the 12-bit probability domain and the
//! signed logit domain in which predictions are mixed. [`Mixer`] takes the
//! stretched outputs of every model and forms a single probability via a
//! per-bit-position weighted sum, then adapts its weights toward the observed bit.
//!
//! This is the skeleton's deliberately minimal mixer: one weight set per bit
//! position over `n` model inputs. The prior attempt's richer forms — several
//! sub-mixers each selected by a different local context, secondary symbol
//! estimation (APM/SSE), and a neural refinement stage — are the Hutter-branch
//! growth path and are intentionally not built here.

use std::sync::OnceLock;

const PROB_BITS: i32 = 12;
const PROB_ONE: i32 = 1 << PROB_BITS; // 4096
const STRETCH_MAX: i32 = 2047;

/// Learning-rate shift for the mixer weight update (higher = slower adaptation).
const LR_SHIFT: i32 = 12;

/// Number of selector contexts for the mixer's weight sets: one per previous-byte value. The mix
/// blend can then specialize by local regime (markup, letters, digits, whitespace) instead of finding
/// one global blend. The selector is a pure function of already-coded bytes, so encode and decode
/// pick the same set and the stream still round-trips bit-for-bit. Standard lpaq-class technique.
pub(crate) const MIX_CONTEXTS: usize = 256;

#[expect(clippy::cast_possible_truncation, reason = "LUT values are clamped into `1..=4095`.")]
fn squash_lut() -> &'static Vec<i32> {
    static LUT: OnceLock<Vec<i32>> = OnceLock::new();
    LUT.get_or_init(|| {
        // index = x + STRETCH_MAX, for x in [-STRETCH_MAX, STRETCH_MAX]
        (0..=2 * STRETCH_MAX)
            .map(|i| {
                let x = f64::from(i - STRETCH_MAX);
                let p = f64::from(PROB_ONE) / (1.0 + (-x / 256.0).exp());
                (p.round() as i32).clamp(1, PROB_ONE - 1)
            })
            .collect()
    })
}

/// Map a stretched logit `x` to a 12-bit probability in `1..=4095`.
#[expect(clippy::cast_sign_loss, reason = "`x + STRETCH_MAX` is non-negative after the clamp.")]
pub(crate) fn squash(x: i32) -> i32 {
    let x = x.clamp(-STRETCH_MAX, STRETCH_MAX);
    squash_lut()[(x + STRETCH_MAX) as usize]
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "Loop indices are within the LUT bounds by construction."
)]
fn stretch_lut() -> &'static Vec<i32> {
    static LUT: OnceLock<Vec<i32>> = OnceLock::new();
    LUT.get_or_init(|| {
        let sq = squash_lut();
        let mut t = vec![0i32; PROB_ONE as usize];
        let mut x = -STRETCH_MAX;
        for (p, slot) in t.iter_mut().enumerate() {
            while x < STRETCH_MAX && sq[(x + STRETCH_MAX) as usize] < p as i32 {
                x += 1;
            }
            *slot = x;
        }
        t
    })
}

/// Inverse of [`squash`]: map a 12-bit probability to a logit in `[-2047, 2047]`.
#[expect(clippy::cast_sign_loss, reason = "`p` is clamped to `0..=4095`, a valid index.")]
pub(crate) fn stretch(p: i32) -> i32 {
    let p = p.clamp(0, PROB_ONE - 1);
    stretch_lut()[p as usize]
}

/// Adaptive logistic mixer: `contexts` × `bit_positions` independent weight sets over `n` stretched
/// model inputs, selected per bit by a `(context, bit position)` pair. `mix` forms the weighted logit
/// sum (16.16 fixed point) and squashes it to a 12-bit probability; `update` nudges the selected
/// weight set toward the observed bit.
#[derive(Debug)]
pub(crate) struct Mixer {
    w: Vec<i32>,          // [contexts * bit_positions * n] weights, 16.16 fixed point
    inputs: Vec<i32>,     // stretched inputs captured by the last `mix` (its length is the input count `n`)
    bit_positions: usize, // number of per-symbol bit-position weight sets within each context
    off: usize,           // selected weight-set offset from the last `mix`
    pr: i32,              // last squashed prediction (12-bit)
}

impl Mixer {
    /// New mixer over `n` model inputs, with `contexts` × `bit_positions` independent weight sets — one
    /// per (selector context, bit index within a coded symbol).
    pub(crate) fn new(n: usize, bit_positions: usize, contexts: usize) -> Self {
        Self {
            w: vec![0; contexts * bit_positions * n],
            inputs: vec![0; n],
            bit_positions,
            off: 0,
            pr: PROB_ONE / 2,
        }
    }

    /// Mix the `stretched` model outputs into a 12-bit probability using the weight set selected by
    /// `(sel, bpos)` — the selector context and the bit index within the current symbol.
    #[expect(clippy::cast_possible_truncation, reason = "`dot >> 16` stays within i32 for a small model set.")]
    pub(crate) fn mix(&mut self, stretched: &[i32], sel: usize, bpos: usize) -> i32 {
        let n = self.inputs.len();
        debug_assert_eq!(stretched.len(), n);
        self.inputs.copy_from_slice(stretched);
        self.off = (sel * self.bit_positions + bpos) * n;
        let row = &self.w[self.off..self.off + n];
        let dot: i64 = row.iter().zip(stretched).map(|(&w, &s)| i64::from(w) * i64::from(s)).sum();
        self.pr = squash((dot >> 16) as i32);
        self.pr
    }

    /// Adapt the selected weight set toward the observed `bit`.
    pub(crate) fn update(&mut self, bit: u8) {
        let err = (i32::from(bit) << PROB_BITS) - self.pr;
        for (w, &s) in self.w[self.off..self.off + self.inputs.len()].iter_mut().zip(&self.inputs) {
            *w += (s * err) >> LR_SHIFT;
        }
    }

    /// The current weight the mixer assigns each input, averaged over its per-bit-position weight sets
    /// and rescaled from 16.16 fixed point to a natural scale (`1.0` ≈ a typical strong weight). A
    /// near-zero average means the mixer has learned the input adds little *marginally* (its signal is
    /// already covered by the others) — the diagnostic the CLI reports per model.
    #[expect(clippy::cast_precision_loss, reason = "Diagnostic display; weight sums are small.")]
    pub(crate) fn input_weights(&self) -> Vec<f64> {
        let n = self.inputs.len();
        let positions = self.w.len().checked_div(n).unwrap_or(0);
        if positions == 0 {
            return vec![0.0; n];
        }
        (0..n)
            .map(|i| {
                let sum: i64 = (0..positions).map(|b| i64::from(self.w[b * n + i])).sum();
                sum as f64 / positions as f64 / 65536.0
            })
            .collect()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn stretch_inverts_squash() {
        // squash∘stretch is an approximate identity: the logit LUT quantizes to
        // integer steps, and in the steep middle of the logistic one step spans
        // ~4 probabilities, so the round-trip can differ by a few units. A tight
        // bound here would only be testing the quantization noise, not the maps.
        for p in 1..PROB_ONE {
            let round = squash(stretch(p));
            assert!((round - p).abs() <= 4, "p={p} round={round}");
        }
    }

    #[test]
    fn squash_is_monotone_and_bounded() {
        let mut prev = 0;
        for x in -STRETCH_MAX..=STRETCH_MAX {
            let p = squash(x);
            assert!((1..PROB_ONE).contains(&p));
            assert!(p >= prev, "not monotone at x={x}");
            prev = p;
        }
    }

    proptest! {
        // A two-input mixer fed two identical, confident logits must predict in
        // the same direction (a basic sanity check that mixing is wired up).
        #[test]
        fn mix_follows_confident_inputs(bit in any::<bool>()) {
            let mut m = Mixer::new(2, 1, 1);
            let s = if bit { STRETCH_MAX } else { -STRETCH_MAX };
            // Train a few steps so the weights move off zero.
            for _ in 0..64 {
                let _ = m.mix(&[s, s], 0, 0);
                m.update(u8::from(bit));
            }
            let p = m.mix(&[s, s], 0, 0);
            if bit {
                prop_assert!(p > PROB_ONE / 2);
            } else {
                prop_assert!(p < PROB_ONE / 2);
            }
        }
    }
}
