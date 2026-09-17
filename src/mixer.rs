//! Logistic context mixing.
//!
//! [`stretch`] / [`squash`] move between the 12-bit probability domain and the
//! signed logit domain in which predictions are mixed. [`Mixer`] takes the
//! stretched outputs of every model and forms a single probability via a
//! per-bit-position weighted sum, then adapts its weights toward the observed bit.
//!
//! [`Mixer`] is the minimal form — one weight set per (previous-byte selector × bit position) over
//! `n` model inputs. [`TwoLayerMixer`] is the richer form now in use by default: order-1..`L1`
//! sub-mixers each selected by a different local context, combined by a second layer. Secondary symbol
//! estimation (APM/SSE) refines the mixed probability downstream in [`crate::apm`].

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

/// Number of first-layer sub-mixers in the [`TwoLayerMixer`]. Each is an independent context-mixing
/// linear layer selected by a *different*-order local context (order-1..`L1`); the second layer learns
/// to blend their logits. Four is the measured sweet spot on enwik8 (adding order-5/6 sub-mixers
/// regressed — their contexts are too sparse to help); each adds one `n`-wide dot product per bit.
const L1: usize = 4;

/// Upper bound on the per-sub-mixer context-table width for the order-`>= 2` sub-mixers (the order-1
/// sub-mixer uses the dense 256 previous-byte table). Wider tables cut hash collisions on the deep
/// contexts; `2^14` was the knee of the curve on enwik8. The actual width is scaled down for smaller
/// inputs (see `TwoLayerMixer::new`), so at 20 MB / enwik8 sizing this cap is what is used.
const DEEP_CTX_BITS: u32 = 14;

/// Learning-rate shifts for the two layers (higher = slower). Layer 1 runs *faster* than the
/// single-layer mixer (`11`): its deep-order contexts are each visited less often, so a larger step per
/// visit converges them quicker. Layer 2 runs *slower* (`13`): it blends only `L1` logits over the
/// whole stream, so a small, stable step generalises best. Both tuned on enwik8.
const LR1_SHIFT: i32 = 11;
const LR2_SHIFT: i32 = 13;

/// Weight magnitude clamp (16.16 fixed point): a safety rail on both layers. With the decoupled
/// training below each layer is an ordinary stable mixer, so this only guards against pathological
/// self-similar inputs driving a weight to an extreme — it is never hit in normal operation.
const W_CLAMP: i32 = 1 << 19;

/// A two-layer logistic mixing network trained online by gradient descent. Layer 1 runs `L1`
/// independent linear mixers, each selecting its weight set by a distinct **order** of local context —
/// an order-1..`L1` progression (previous byte, then rolling hashes of the last 2, 3, 4 bytes) — so the
/// blend can specialise by *how deep* a context is currently predictive. Layer 2 blends the sub-mixers'
/// squashed-domain logits into the final prediction. It is a strict generalisation of the single-layer
/// [`Mixer`] — with the right weights layer 2 can defer entirely to the order-1 sub-mixer, recovering
/// the old behaviour — so it can only add expressive power. Fully deterministic (fixed-point, same code
/// on encode and decode), so the stream still round-trips bit-for-bit.
#[derive(Debug)]
pub(crate) struct TwoLayerMixer {
    /// Layer-1 weights: one table per sub-mixer, each `ctx1[j] * bit_positions * n`, 16.16 fixed point.
    w1: [Vec<i32>; L1],
    /// Context count per sub-mixer (the selector is reduced modulo this).
    ctx1: [usize; L1],
    /// Layer-2 weights: `ctx2 * bit_positions * L1`, 16.16 fixed point.
    w2: Vec<i32>,
    /// Layer-2 context count.
    ctx2: usize,
    n: usize,
    bit_positions: usize,
    /// Stretched inputs captured by the last `mix` (length `n`).
    inputs: Vec<i32>,
    /// Clamped layer-1 outputs captured by the last `mix`.
    y: [i32; L1],
    /// Selected layer-1 weight-set offsets from the last `mix`.
    off1: [usize; L1],
    /// Selected layer-2 weight-set offset from the last `mix`.
    off2: usize,
    /// Last squashed prediction (12-bit).
    pr: i32,
}

impl TwoLayerMixer {
    /// A fresh two-layer mixer over `n` model inputs with `bit_positions` per-symbol bit slots.
    /// Weights initialise so the untrained network outputs the mean of its model logits — already a
    /// reasonable predictor — while breaking the layer symmetry that a zero init would freeze.
    #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap, reason = "init constants are small")]
    pub(crate) fn new(n: usize, bit_positions: usize, contexts: usize, blend_contexts: usize, capacity: usize) -> Self {
        // Deep-context table width, scaled to the input so small inputs don't allocate an enwik8-sized
        // table (and so the deep contexts aren't wastefully sparse): ~log2(capacity), capped at
        // `DEEP_CTX_BITS` (the enwik8 knee) and floored so tiny inputs still have some resolution. A pure
        // function of the framed count, so encode and decode size identically and still round-trip.
        let deep_bits = (usize::BITS - capacity.max(1).leading_zeros()).clamp(9, DEEP_CTX_BITS);
        // Layer-1 contexts: an order-1..L1 progression (previous byte, then rolling k-byte hashes).
        let ctx1 = std::array::from_fn(|j| {
            if j == 0 {
                contexts.max(1)
            } else {
                1 << deep_bits
            }
        });
        // Each sub-mixer starts as a uniform average of its inputs (weight 1/n), a sane logit.
        let init1 = (65536 / n.max(1)) as i32;
        let w1 = std::array::from_fn(|j| vec![init1; ctx1[j] * bit_positions * n]);
        // Layer 2 starts as a uniform average of the sub-mixer logits (weight 1/L1). A single global
        // blend (per bit position) generalises best — a per-previous-byte layer-2 context overfits — so
        // callers pass `blend_contexts = 1` unless they have a low-cardinality regime selector (the
        // `surprise` bucket) to offer.
        let ctx2 = blend_contexts.max(1);
        let w2 = vec![(65536 / L1) as i32; ctx2 * bit_positions * L1];
        Self {
            w1,
            ctx1,
            w2,
            ctx2,
            n,
            bit_positions,
            inputs: vec![0; n],
            y: [0; L1],
            off1: [0; L1],
            off2: 0,
            pr: PROB_ONE / 2,
        }
    }

    /// Mix the `stretched` model outputs into a 12-bit probability. `layer1_sels` are the `L1` layer-1
    /// selector contexts (one per sub-mixer), `blend_sel` the layer-2 selector, and `bpos` the bit index.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::needless_range_loop,
        reason = "`dot >> 16` stays within i32 for a small model set; `j` indexes several parallel arrays."
    )]
    pub(crate) fn mix(&mut self, stretched: &[i32], layer1_sels: [usize; L1], blend_sel: usize, bpos: usize) -> i32 {
        debug_assert_eq!(stretched.len(), self.n);
        self.inputs.copy_from_slice(stretched);
        for j in 0..L1 {
            let s = layer1_sels[j] % self.ctx1[j];
            let off = (s * self.bit_positions + bpos) * self.n;
            self.off1[j] = off;
            let dot: i64 =
                self.w1[j][off..off + self.n].iter().zip(stretched).map(|(&w, &x)| i64::from(w) * i64::from(x)).sum();
            self.y[j] = ((dot >> 16) as i32).clamp(-STRETCH_MAX, STRETCH_MAX);
        }
        let s2 = blend_sel % self.ctx2;
        let off2 = (s2 * self.bit_positions + bpos) * L1;
        self.off2 = off2;
        let dot2: i64 = self.w2[off2..off2 + L1].iter().zip(&self.y).map(|(&w, &x)| i64::from(w) * i64::from(x)).sum();
        self.pr = squash((dot2 >> 16) as i32);
        self.pr
    }

    /// Learn from the observed `bit`. Layer 2 blends the sub-mixer logits toward the final target; each
    /// layer-1 sub-mixer is trained *independently* toward the bit from its own squashed output. The
    /// decoupled scheme keeps every sub-mixer as stable as the single-layer [`Mixer`] (no layer-2 gain
    /// multiplies the layer-1 step), while layer 2 learns which regime to trust — so the network cannot
    /// diverge over long inputs the way coupled backprop does.
    pub(crate) fn update(&mut self, bit: u8) {
        let target = i32::from(bit) << PROB_BITS;
        let final_err = target - self.pr;
        // Layer 2: nudge the blend toward the target given the sub-mixer logits.
        for j in 0..L1 {
            let w = &mut self.w2[self.off2 + j];
            *w = (*w + ((self.y[j] * final_err) >> LR2_SHIFT)).clamp(-W_CLAMP, W_CLAMP);
        }
        // Layer 1: each sub-mixer is its own standard logistic mixer, trained toward the actual bit.
        for j in 0..L1 {
            let err_j = target - squash(self.y[j]);
            let off = self.off1[j];
            for (w, &x) in self.w1[j][off..off + self.n].iter_mut().zip(&self.inputs) {
                *w = (*w + ((x * err_j) >> LR1_SHIFT)).clamp(-W_CLAMP, W_CLAMP);
            }
        }
    }

    /// Effective per-input weight for the scorecard: each model's layer-1 weight (averaged over
    /// sub-mixers and their contexts) scaled by the mean layer-2 gain, rescaled from 16.16. Diagnostic
    /// only — an approximation of the linearised end-to-end weight.
    #[expect(clippy::cast_precision_loss, reason = "Diagnostic display; weight sums are small.")]
    pub(crate) fn input_weights(&self) -> Vec<f64> {
        let l2_gain: f64 = {
            let sum: i64 = self.w2.iter().map(|&w| i64::from(w)).sum();
            sum as f64 / self.w2.len().max(1) as f64 / 65536.0
        };
        (0..self.n)
            .map(|i| {
                let mut acc = 0.0;
                for j in 0..L1 {
                    let positions = self.w1[j].len() / self.n;
                    let s: i64 = (0..positions).map(|b| i64::from(self.w1[j][b * self.n + i])).sum();
                    acc += s as f64 / positions.max(1) as f64 / 65536.0;
                }
                acc / L1 as f64 * l2_gain
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

        // The two-layer mixer must likewise learn to follow confident, consistent inputs — a basic
        // check that both layers train in the right direction and the network converges.
        #[test]
        fn two_layer_follows_confident_inputs(bit in any::<bool>()) {
            let mut m = TwoLayerMixer::new(2, 1, 4, 1, 4096);
            let s = if bit { STRETCH_MAX } else { -STRETCH_MAX };
            for _ in 0..256 {
                let _ = m.mix(&[s, s], [1, 2, 3, 0], 1, 0);
                m.update(u8::from(bit));
            }
            let p = m.mix(&[s, s], [1, 2, 3, 0], 1, 0);
            if bit {
                prop_assert!(p > PROB_ONE / 2, "p={p}");
            } else {
                prop_assert!(p < PROB_ONE / 2, "p={p}");
            }
        }
    }

    /// The two-layer mixer stays bounded and finite under a long adversarial run of a single confident
    /// input — the decoupled training and weight clamp must keep it from diverging.
    #[test]
    fn two_layer_stays_bounded() {
        let mut m = TwoLayerMixer::new(3, 8, 16, 1, 1 << 16);
        for i in 0..100_000 {
            let bpos = i % 8;
            let _ = m.mix(&[STRETCH_MAX, -STRETCH_MAX, 0], [7, 3, 11, 2], 5, bpos);
            m.update(u8::from(i % 3 == 0));
            let p = m.mix(&[STRETCH_MAX, STRETCH_MAX, STRETCH_MAX], [7, 3, 11, 2], 5, bpos);
            assert!((1..PROB_ONE).contains(&p), "prediction escaped valid range: {p}");
        }
    }
}
