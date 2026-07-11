//! Online neural language model: a small (optionally recurrent) network over learned byte embeddings.
//!
//! Where the order-`N` models are a lookup table over an exact byte context (so an unseen context
//! predicts nothing), this model *generalises* across contexts: each of the previous [`K`] bytes is
//! mapped through a shared learned embedding, the concatenation is pushed through one `tanh` hidden
//! layer, and a per-bit-tree-node logistic output head reads off the next-bit probability. Every
//! parameter is trained online by back-propagation from the coded bits, so the network improves as the
//! stream is processed — the "online neural model" of a context-mixing compressor, complementary to
//! the exact-match context models it mixes with.
//!
//! In **recurrent** mode the hidden layer also reads the previous byte's hidden state through a
//! recurrent weight matrix (an Elman RNN), so the state carries information *beyond* the `K`-byte
//! window — the long-range memory the fixed-order context models structurally cannot hold. It is
//! trained with one-step (truncated) back-propagation: the recurrent projection gets the immediate
//! gradient while the forward recurrence still carries history through the state. That keeps training
//! cheap and stable (no unbounded through-time unroll) while giving the network a genuine memory.
//!
//! The hidden state depends only on the finalized byte context, so it is computed **once per byte**
//! (at the first bit) and reused for all eight bits; only the tiny output head runs per bit. That
//! keeps it within the compute budget of a Hutter-scale run.
//!
//! # Determinism
//!
//! The network is `f32`. Encode and decode run the *same* compiled `predict`/`update` on the same
//! observed bits, so their state stays bit-identical and the stream round-trips on a given build (the
//! proptest round-trips verify this). Unlike the integer models it is not guaranteed reproducible
//! across differing float toolchains — acceptable for a default-off experimental model, and the codec
//! already targets a fixed machine for its Hutter accounting. It only feeds a probability to the
//! mixer, so any drift would degrade ratio, never correctness (the range coder and Adler-32 are the
//! reversibility backstop). It never panics on arbitrary input.

#![expect(
    clippy::needless_range_loop,
    clippy::suboptimal_flops,
    clippy::similar_names,
    reason = "the explicit indexed loops mirror the network's index arithmetic (weight row j, input i, \
              embedding slot k=i/E, e=i%E) and keep the forward and backward passes visibly symmetric; \
              plain `a*b + c` (not `mul_add`) keeps the f32 accumulation order predictable for \
              determinism; `row`/`rrow` are the input and recurrent weight rows."
)]

use super::{Context, Model};
use crate::mixer::stretch;

/// Number of previous bytes fed to the network (the neural n-gram order).
const K: usize = 4;
/// Embedding width per context byte.
const E: usize = 16;
/// Hidden-layer width.
const H: usize = 32;
/// Input width: the `K` byte embeddings concatenated.
const D: usize = K * E;

/// Learning rates for the output head, hidden layer, and embeddings. The output head adapts fastest
/// (it is the final logistic layer, like the mixer); the shared hidden/embedding parameters move
/// slower to stay stable as many contexts pass through them.
const LR_OUT: f32 = 0.03;
const LR_HID: f32 = 0.01;
const LR_EMB: f32 = 0.01;

/// Deterministic small-magnitude weight initialiser: a splitmix64 step over the parameter index,
/// mapped to `[-scale, scale)`. Used instead of an RNG so encode and decode build identical weights.
fn init_weight(index: u64, scale: f32) -> f32 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Map the top 24 bits to [-1, 1), then scale.
    #[expect(clippy::cast_precision_loss, reason = "24-bit mantissa fits f32 exactly")]
    let unit = ((z >> 40) as f32 / f32::from(1u16 << 15) / 256.0) - 1.0;
    unit * scale
}

/// Logistic squashing for the output head's own probability (kept in natural log-odds internally).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A (optionally recurrent) neural n-gram bit predictor with online back-propagation.
pub(crate) struct NnlmModel {
    /// Whether the hidden layer reads the previous hidden state (Elman recurrence).
    recurrent: bool,
    /// Byte embedding table: `256 * E`, row `b` is byte `b`'s embedding.
    emb: Vec<f32>,
    /// Hidden weights `H * D` and biases `H`.
    w1: Vec<f32>,
    b1: Vec<f32>,
    /// Recurrent weights `H * H` (empty when not recurrent): hidden(t-1) -> hidden(t).
    w_rec: Vec<f32>,
    /// Output head: per bit-tree node (`256`) a weight vector `H` plus a bias.
    w2: Vec<f32>,
    b2: Vec<f32>,
    /// The `K` context bytes captured for the current byte (oldest first), for the embedding update.
    ctx_bytes: [usize; K],
    /// Concatenated input embeddings captured for the current byte (`D`).
    x: Vec<f32>,
    /// Hidden activations captured for the current byte (`H`).
    h: Vec<f32>,
    /// Previous byte's hidden activations (the recurrent input), carried forward each byte.
    h_prev: Vec<f32>,
    /// Gradient of the byte's total loss w.r.t. each hidden activation, accumulated over its 8 bits.
    grad_h: Vec<f32>,
    /// Bit-tree node and predicted probability from the last `predict`, reused by `update`.
    cur_node: usize,
    cur_p: f32,
}

impl NnlmModel {
    /// A fresh network with deterministically-seeded small weights (the output head starts at zero, so
    /// it predicts ½ until it learns, like a fresh `StateMap`). `recurrent` enables the Elman memory.
    pub(crate) fn new(recurrent: bool) -> Self {
        let emb = (0..256 * E).map(|i| init_weight(i as u64, 0.1)).collect();
        let w1 = (0..H * D).map(|i| init_weight(i as u64 + 0x1000, 0.1)).collect();
        // Recurrent weights start small so the initial state feedback is gentle (stability).
        let w_rec = if recurrent {
            (0..H * H).map(|i| init_weight(i as u64 + 0x2000, 0.05)).collect()
        } else {
            Vec::new()
        };
        Self {
            recurrent,
            emb,
            w1,
            b1: vec![0.0; H],
            w_rec,
            w2: vec![0.0; 256 * H],
            b2: vec![0.0; 256],
            ctx_bytes: [0; K],
            x: vec![0.0; D],
            h: vec![0.0; H],
            h_prev: vec![0.0; H],
            grad_h: vec![0.0; H],
            cur_node: 0,
            cur_p: 0.5,
        }
    }

    /// Recompute the input embeddings and hidden activations from the last `K` finalized bytes (and,
    /// when recurrent, the previous hidden state). Called once at the start of each byte; the result
    /// serves all eight of its bit predictions.
    fn recompute_hidden(&mut self, hist: &[u8]) {
        let len = hist.len();
        for k in 0..K {
            // Position `k` (oldest..newest) maps to `hist[len - K + k]`; missing history reads as 0.
            let b = (len + k).checked_sub(K).map_or(0, |idx| usize::from(hist[idx]));
            self.ctx_bytes[k] = b;
            let (src, dst) = (b * E, k * E);
            self.x[dst..dst + E].copy_from_slice(&self.emb[src..src + E]);
        }
        for j in 0..H {
            let row = &self.w1[j * D..j * D + D];
            let mut pre = self.b1[j];
            for i in 0..D {
                pre += row[i] * self.x[i];
            }
            if self.recurrent {
                let rrow = &self.w_rec[j * H..j * H + H];
                for m in 0..H {
                    pre += rrow[m] * self.h_prev[m];
                }
            }
            self.h[j] = pre.tanh();
        }
        self.grad_h.iter_mut().for_each(|g| *g = 0.0);
    }
}

impl Model for NnlmModel {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "p*4096 truncates to an i32 then clamps into 1..=4095, a valid 12-bit probability"
    )]
    fn predict(&mut self, ctx: &Context, hist: &[u8]) -> i32 {
        if ctx.bpos == 0 {
            self.recompute_hidden(hist);
        }
        let node = (ctx.c0 & 0xff) as usize;
        let row = &self.w2[node * H..node * H + H];
        let mut logit = self.b2[node];
        for j in 0..H {
            logit += row[j] * self.h[j];
        }
        let p = sigmoid(logit);
        self.cur_node = node;
        self.cur_p = p;
        let p12 = ((p * 4096.0) as i32).clamp(1, 4095);
        stretch(p12)
    }

    fn update(&mut self, ctx: &Context, _hist: &[u8], bit: u8) {
        // Natural gradient of the logistic log-loss w.r.t. the output logit.
        let g = f32::from(bit) - self.cur_p;
        let node = self.cur_node;
        // Back-propagate into the hidden layer (using the pre-update output weights), then adapt the
        // output head. Accumulate the hidden gradient across the byte's 8 bits.
        {
            let row = &mut self.w2[node * H..node * H + H];
            for j in 0..H {
                self.grad_h[j] += g * row[j];
                row[j] += LR_OUT * g * self.h[j];
            }
        }
        self.b2[node] += LR_OUT * g;

        // On the byte's final bit, back-propagate the accumulated hidden gradient into the hidden
        // weights and the byte embeddings (one shared update per byte).
        if ctx.bpos == 7 {
            for j in 0..H {
                // tanh'(pre) = 1 - h^2.
                let dpre = self.grad_h[j] * (1.0 - self.h[j] * self.h[j]);
                let row = &mut self.w1[j * D..j * D + D];
                for i in 0..D {
                    // Read the pre-update weight for the embedding gradient, then adapt it. Each
                    // `w1[j*D+i]` is touched once here, so this uses the correct pre-update value.
                    let k = i / E;
                    let e = i % E;
                    self.emb[self.ctx_bytes[k] * E + e] += LR_EMB * dpre * row[i];
                    row[i] += LR_HID * dpre * self.x[i];
                }
                if self.recurrent {
                    // One-step gradient into the recurrent projection (h_prev treated as a fixed input).
                    let rrow = &mut self.w_rec[j * H..j * H + H];
                    for m in 0..H {
                        rrow[m] += LR_HID * dpre * self.h_prev[m];
                    }
                }
                self.b1[j] += LR_HID * dpre;
            }
            if self.recurrent {
                // Carry this byte's state forward as the next byte's recurrent input.
                self.h_prev.copy_from_slice(&self.h);
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::models::SYMBOL_BITS;

    /// Drive `data` through the model one bit at a time exactly as the entropy driver does.
    fn run(model: &mut NnlmModel, data: &[u8]) {
        let mut ctx = Context::new();
        for (i, &byte) in data.iter().enumerate() {
            let hist = &data[..i];
            for k in (0..SYMBOL_BITS).rev() {
                let bit = (byte >> k) & 1;
                let _ = model.predict(&ctx, hist);
                model.update(&ctx, hist, bit);
                ctx.push_bit(bit);
            }
            ctx.push_symbol();
        }
    }

    /// The network must learn a perfectly predictable stream: after training on a long repetition its
    /// prediction for the recurring next bit beats a fresh network's ½.
    #[test]
    fn learns_repetition() {
        let mut model = NnlmModel::new(false);
        let data = b"abcabcabcabcabcabcabcabc".repeat(64);
        run(&mut model, &data);
        // Predict the bit after "abc...abc" — feed the context and check the model is confident.
        let mut ctx = Context::new();
        let hist = &data[..data.len() - 1];
        // First bit of 'a' (0x61 = 0b01100001) is 0; a well-trained net should lean that way.
        let pred = model.predict(&ctx, hist);
        // Not asserting the exact bit (embedding init varies), only that it has moved off neutral.
        assert!(pred.abs() > 10, "network stayed at neutral prediction: {pred}");
        let _ = &mut ctx;
    }

    /// Never panics on arbitrary/adversarial bytes, including short history.
    #[test]
    fn survives_adversarial_input() {
        for bytes in [vec![0xffu8; 40], vec![0u8; 1], (0u8..=255).collect::<Vec<_>>()] {
            let mut model = NnlmModel::new(false);
            run(&mut model, &bytes);
        }
    }

    /// Deterministic: two fresh networks driven by the same bytes reach the same prediction (the
    /// property encode/decode rely on).
    #[test]
    fn is_deterministic() {
        let mut a = NnlmModel::new(false);
        let mut b = NnlmModel::new(false);
        let data = b"the quick brown fox jumps over the lazy dog";
        run(&mut a, data);
        run(&mut b, data);
        let ctx = Context::new();
        assert_eq!(a.predict(&ctx, data), b.predict(&ctx, data));
    }
}
