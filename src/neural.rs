//! Sparse logistic regression for binary bit prediction.
//!
//! [`SparseLR<M>`] is the simplest "neural" predictor we plug into
//! the bit-stream stack — `M` categorical features, each backed by
//! its own hashed weight table of size `2^K`. Forward pass:
//!
//! ```text
//!   z      = bias + Σ_i w_i[ hash_i mod 2^K ]
//!   P(0)   = sigmoid(z)
//! ```
//!
//! Online SGD on log-loss after each observed bit:
//!
//! ```text
//!   target = 1 if bit == 0 else 0
//!   error  = target − P(0)
//!   w_i   += lr · error    (only the M active weights)
//!   bias  += lr · error
//! ```
//!
//! Why this and not a `BitPredictor`: count tables need *cells* per
//! distinct context to amortize Laplace smoothing; hashed SGD
//! degrades gracefully when cells collide and learns smoothly from
//! a handful of observations. This lets us cram Order-3-style word
//! context into a few MiB instead of the gigabytes a count table
//! would need.
//!
//! Determinism: pure scalar `f32` adds, one `f32::exp`, no atomics
//! or threads. Bit-exact between encode and decode on the same
//! machine, which is all the Hutter pipeline requires. CPU-only —
//! no matmul, no GPU primitives — and per-bit work is O(M) which
//! the runtime budget absorbs trivially at the call sites the codec
//! uses (M ≤ a handful).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use crate::ac::TOTAL;

/// Cap on the summed logit so a single runaway slot can't drag the
/// gradient out of its useful range. Matches the mixer's clamp.
const LOGIT_CLAMP: f32 = 16.0;

/// Sparse logistic regression with `M` categorical features. Each
/// feature has its own hashed weight table of size `1 << k_bits`.
pub(crate) struct SparseLR<const M: usize> {
    /// Flat backing storage. Feature `i`'s weight for slot `s` lives
    /// at index `(i << k_bits) | s`. A single allocation beats
    /// `[Vec<f32>; M]` for cache locality and is friendlier to the
    /// allocator.
    weights: Vec<f32>,
    bias: f32,
    k_bits: u32,
    mask: u64,
    learning_rate: f32,
}

impl<const M: usize> SparseLR<M> {
    /// New predictor with `1 << k_bits` slots per feature. Weights
    /// and bias start at zero so the cold prediction is `sigmoid(0)
    /// = 0.5` — a neutral arm the mixer can ignore until SGD warms
    /// it up.
    pub(crate) fn new(k_bits: u32, learning_rate: f32) -> Self {
        assert!(M >= 1, "SparseLR needs at least one feature");
        let n_slots: usize = 1 << k_bits;
        let total: usize = M.checked_mul(n_slots).expect("M * 2^k fits usize");
        Self {
            weights: vec![0.0_f32; total],
            bias: 0.0,
            k_bits,
            mask: u64::try_from(n_slots - 1).expect("table size fits u64"),
            learning_rate,
        }
    }

    fn slot(&self, feature_idx: usize, hash: u64) -> usize {
        let s = usize::try_from(hash & self.mask).expect("masked u64 fits usize");
        (feature_idx << self.k_bits) | s
    }

    fn logit(&self, feature_hashes: &[u64; M]) -> f32 {
        let mut z = self.bias;
        for (i, &h) in feature_hashes.iter().enumerate() {
            z += self.weights[self.slot(i, h)];
        }
        z.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
    }

    fn squash_to_ac(z: f32) -> u32 {
        let p = 1.0 / (1.0 + (-z).exp());
        let scaled = (p * (TOTAL as f32)) as i32;
        scaled.clamp(1, (TOTAL as i32) - 1) as u32
    }

    /// Pure read; does not mutate state.
    pub(crate) fn predict(&self, feature_hashes: &[u64; M]) -> u32 {
        Self::squash_to_ac(self.logit(feature_hashes))
    }

    /// SGD update from the observed bit. Returns the AC-scaled
    /// `P(bit = 0)` the predictor would have reported pre-update,
    /// so callers can compose this with a mixer in one pass.
    pub(crate) fn predict_and_observe(&mut self, feature_hashes: &[u64; M], bit: u32) -> u32 {
        let z = self.logit(feature_hashes);
        let p_zero_f = 1.0 / (1.0 + (-z).exp());
        let target: f32 = if bit == 0 { 1.0 } else { 0.0 };
        let error = target - p_zero_f;
        let step = self.learning_rate * error;
        for (i, &h) in feature_hashes.iter().enumerate() {
            let slot = self.slot(i, h);
            self.weights[slot] += step;
        }
        self.bias += step;
        Self::squash_to_ac(z)
    }

    /// Update only (prewarm path, where the AC stream doesn't emit).
    pub(crate) fn observe(&mut self, feature_hashes: &[u64; M], bit: u32) {
        let _ = self.predict_and_observe(feature_hashes, bit);
    }
}

/// Two-layer online MLP for binary bit prediction. `M` sparse
/// categorical features each map to an `H`-dim embedding vector;
/// the embeddings are summed, passed through `tanh`, then
/// projected to a scalar logit. SGD on log-loss flows through
/// the output layer and tanh derivative into the embedding tables.
///
/// Architecturally this is the smallest step up from
/// [`SparseLR`] that adds a nonlinearity and a learned hidden
/// dimension. cmix-style: per-feature embeddings replace per-feature
/// scalars, and a small dense layer (one set of output weights)
/// learns how to combine the hidden activations.
///
/// Per-bit work is `O(M · H)` for the forward pass and the same
/// for the backward pass — for the codec's typical `M=3, H=8`
/// that is ~100 scalar `f32` ops per emitted bit. Comfortably
/// inside the Hutter time budget.
///
/// Determinism: pure scalar `f32` adds and `mul_add`s, plus
/// `tanh` and `exp` for the output. Bit-exact across encode and
/// decode on the same machine; CPU-only, no matmul library, no
/// GPU primitives.
pub(crate) struct OnlineMLP<const M: usize, const H: usize> {
    /// Embedding tables. Feature `i`'s embedding for slot `s`
    /// occupies `embeddings[(i << k_bits + h_shift) | (s << h_shift) | h]`
    /// — laid out so a single slot's `H` floats are contiguous.
    /// The flat backing storage keeps the allocation simple and
    /// cache-friendly: each embedding lookup pulls one cache line
    /// (assuming `H` ≤ 16 with `f32`).
    embeddings: Vec<f32>,
    /// Bias added to the pre-`tanh` sum.
    hidden_bias: [f32; H],
    /// Final linear layer that produces the scalar logit.
    output_weight: [f32; H],
    output_bias: f32,
    k_bits: u32,
    /// `2^k_bits − 1`, used to mask the hashed feature key into a
    /// valid slot.
    slot_mask: u64,
    learning_rate: f32,
}

impl<const M: usize, const H: usize> OnlineMLP<M, H> {
    /// New MLP with `1 << k_bits` slots per feature. Output weights
    /// initialize at `1/H` (not zero) so the first observed bit
    /// produces a nonzero gradient into the embedding tables; if
    /// the output weights were zero the embeddings would never
    /// receive any signal (`tanh(0) = 0`, dead-tanh problem).
    pub(crate) fn new(k_bits: u32, learning_rate: f32) -> Self {
        assert!(M >= 1, "OnlineMLP needs at least one feature");
        assert!(H >= 1, "OnlineMLP needs at least one hidden unit");
        let n_slots: usize = 1 << k_bits;
        let total: usize = M
            .checked_mul(n_slots)
            .and_then(|x| x.checked_mul(H))
            .expect("M * 2^k * H fits usize");
        let init_output: f32 = 1.0 / (H as f32);
        Self {
            embeddings: vec![0.0_f32; total],
            hidden_bias: [0.0_f32; H],
            output_weight: [init_output; H],
            output_bias: 0.0,
            k_bits,
            slot_mask: u64::try_from(n_slots - 1).expect("table size fits u64"),
            learning_rate,
        }
    }

    /// Starting index of feature `i`'s embedding row at slot `s` in
    /// the flat `embeddings` vector. The row occupies the next `H`
    /// floats.
    fn embed_row(&self, feature_idx: usize, hash: u64) -> usize {
        let s = usize::try_from(hash & self.slot_mask).expect("masked u64 fits usize");
        let n_slots: usize = 1 << self.k_bits;
        ((feature_idx * n_slots) + s) * H
    }

    /// Forward pass — returns `(hidden_post, logit)`. `hidden_post`
    /// is the post-`tanh` hidden activation; the pre-tanh sum isn't
    /// needed downstream (the `tanh'` derivative is computed from
    /// `hidden_post` via the identity `1 − tanh²`).
    fn forward(&self, feature_hashes: &[u64; M]) -> ([f32; H], f32) {
        let mut hidden_pre = self.hidden_bias;
        for (i, &h) in feature_hashes.iter().enumerate() {
            let row = self.embed_row(i, h);
            for (k, hp) in hidden_pre.iter_mut().enumerate() {
                *hp += self.embeddings[row + k];
            }
        }
        let mut hidden_post = [0.0_f32; H];
        for (post, &pre) in hidden_post.iter_mut().zip(hidden_pre.iter()) {
            *post = pre.tanh();
        }
        let mut logit = self.output_bias;
        for (w, &post) in self.output_weight.iter().zip(hidden_post.iter()) {
            logit = w.mul_add(post, logit);
        }
        (hidden_post, logit)
    }

    fn squash_to_ac(logit: f32) -> u32 {
        // Match the SparseLR clamp so a runaway logit can't push the
        // mixer's `stretch` to ±∞.
        let z = logit.clamp(-LOGIT_CLAMP, LOGIT_CLAMP);
        let p = 1.0 / (1.0 + (-z).exp());
        let scaled = (p * (TOTAL as f32)) as i32;
        scaled.clamp(1, (TOTAL as i32) - 1) as u32
    }

    pub(crate) fn predict(&self, feature_hashes: &[u64; M]) -> u32 {
        let (_, logit) = self.forward(feature_hashes);
        Self::squash_to_ac(logit)
    }

    /// Predict and SGD-update from the observed bit. Returns the
    /// AC-scaled `P(bit = 0)` reported pre-update so callers can
    /// feed it into a mixer in one pass.
    pub(crate) fn predict_and_observe(&mut self, feature_hashes: &[u64; M], bit: u32) -> u32 {
        let (hidden_post, logit) = self.forward(feature_hashes);
        let p_zero_f = 1.0 / (1.0 + (-logit.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)).exp());
        let target: f32 = if bit == 0 { 1.0 } else { 0.0 };
        // `error` is `target − P(0)`. Standard sign convention used
        // throughout this codec; `+= lr · error · ...` ascends the
        // log-likelihood gradient.
        let error = target - p_zero_f;
        let step = self.learning_rate * error;

        // Hidden-layer gradients before applying the output update —
        // we still need the pre-update output weights to backprop.
        // tanh'(x) = 1 − tanh(x)² = (−hidden_post[k]).mul_add(hidden_post[k], 1).
        let mut hidden_pre_grad = [0.0_f32; H];
        for ((grad, &post), &w) in hidden_pre_grad
            .iter_mut()
            .zip(hidden_post.iter())
            .zip(self.output_weight.iter())
        {
            let tanh_deriv = (-post).mul_add(post, 1.0);
            *grad = step * w * tanh_deriv;
        }

        // Output-layer SGD.
        for (w, &post) in self.output_weight.iter_mut().zip(hidden_post.iter()) {
            *w = step.mul_add(post, *w);
        }
        self.output_bias += step;

        // Embedding-row SGD — each active row gets the same
        // `hidden_pre_grad` because the forward sum is symmetric in
        // the per-feature contributions.
        for (i, &h) in feature_hashes.iter().enumerate() {
            let row = self.embed_row(i, h);
            for (k, &grad) in hidden_pre_grad.iter().enumerate() {
                self.embeddings[row + k] += grad;
            }
        }
        for (b, &grad) in self.hidden_bias.iter_mut().zip(hidden_pre_grad.iter()) {
            *b += grad;
        }

        Self::squash_to_ac(logit)
    }

    /// Prewarm path — update only, no return value.
    pub(crate) fn observe(&mut self, feature_hashes: &[u64; M], bit: u32) {
        let _ = self.predict_and_observe(feature_hashes, bit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_predictor_returns_neutral_probability() {
        let lr: SparseLR<3> = SparseLR::new(8, 0.05);
        let p = lr.predict(&[1, 2, 3]);
        // sigmoid(0) = 0.5 → TOTAL/2.
        let diff = (p as i32 - (TOTAL / 2) as i32).abs();
        assert!(diff <= 1, "cold p_zero = {p}, expected ~{}", TOTAL / 2);
    }

    #[test]
    fn predictor_learns_constant_target() {
        let mut lr: SparseLR<3> = SparseLR::new(8, 0.05);
        // Same feature triple every time; bit is always 0 → P(0) → 1.
        let feats = [42u64, 17u64, 99u64];
        for _ in 0..5000 {
            lr.predict_and_observe(&feats, 0);
        }
        let p = lr.predict(&feats);
        assert!(
            p > TOTAL * 9 / 10,
            "predictor didn't learn: p_zero = {p} (expected > {})",
            TOTAL * 9 / 10,
        );
    }

    #[test]
    fn distinct_features_train_independently() {
        let mut lr: SparseLR<2> = SparseLR::new(10, 0.05);
        let feats_a = [1u64, 2u64];
        let feats_b = [3u64, 4u64];
        for _ in 0..5000 {
            lr.predict_and_observe(&feats_a, 0);
            lr.predict_and_observe(&feats_b, 1);
        }
        let p_a = lr.predict(&feats_a);
        let p_b = lr.predict(&feats_b);
        assert!(p_a > TOTAL * 6 / 10, "feats_a should favor 0: {p_a}");
        assert!(p_b < TOTAL * 4 / 10, "feats_b should favor 1: {p_b}");
    }

    #[test]
    fn squash_stays_in_ac_range() {
        let lr: SparseLR<1> = SparseLR::new(4, 0.05);
        // Even with extreme logits, output stays in [1, TOTAL-1].
        for z_i in -30..=30 {
            let z = z_i as f32;
            let p = SparseLR::<1>::squash_to_ac(z);
            assert!((1..TOTAL).contains(&p), "z={z}, p={p}");
        }
        // And the predict path agrees for a zero-weight predictor.
        let p = lr.predict(&[0]);
        assert!((1..TOTAL).contains(&p));
    }

    #[test]
    fn mlp_cold_predictor_returns_neutral_probability() {
        let mlp: OnlineMLP<3, 8> = OnlineMLP::new(8, 0.05);
        let p = mlp.predict(&[1, 2, 3]);
        // Output bias and embeddings are zero; output_weight is 1/H;
        // hidden = 0 → tanh(0) = 0 → logit = 0 → P(0) = 0.5.
        let diff = (p as i32 - (TOTAL / 2) as i32).abs();
        assert!(diff <= 1, "cold p_zero = {p}, expected ~{}", TOTAL / 2);
    }

    #[test]
    fn mlp_learns_constant_target() {
        let mut mlp: OnlineMLP<3, 8> = OnlineMLP::new(8, 0.05);
        let feats = [42u64, 17u64, 99u64];
        for _ in 0..5000 {
            mlp.predict_and_observe(&feats, 0);
        }
        let p = mlp.predict(&feats);
        assert!(
            p > TOTAL * 9 / 10,
            "MLP didn't learn: p_zero = {p} (expected > {})",
            TOTAL * 9 / 10,
        );
    }

    #[test]
    fn mlp_distinct_features_train_independently() {
        let mut mlp: OnlineMLP<2, 4> = OnlineMLP::new(10, 0.05);
        let feats_a = [1u64, 2u64];
        let feats_b = [3u64, 4u64];
        for _ in 0..5000 {
            mlp.predict_and_observe(&feats_a, 0);
            mlp.predict_and_observe(&feats_b, 1);
        }
        let p_a = mlp.predict(&feats_a);
        let p_b = mlp.predict(&feats_b);
        assert!(p_a > TOTAL * 6 / 10, "feats_a should favor 0: {p_a}");
        assert!(p_b < TOTAL * 4 / 10, "feats_b should favor 1: {p_b}");
    }
}
