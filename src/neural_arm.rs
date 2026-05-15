// Phase 27 WIP: the wrapper is fully tested but no consumer wires it
// into the codec yet (that lands as the byte-aligned integration into
// `encode_oov_sep_bytes`). The module-level `dead_code` allow keeps
// the unused warnings out of `-Dwarnings` until then.
#![allow(dead_code)]

//! Streaming wrapper around [`crate::transformer::ByteTransformer`]
//! that the codec drives byte-by-byte during encode/decode.
//!
//! Lifecycle inside the codec:
//! 1. `NeuralArm::load(path)` once at the start of encode/decode.
//! 2. As the codec walks source bytes in order:
//!    - Before emitting a byte through a byte-aligned stream, call
//!      [`NeuralArm::predict_byte_cdf`] to mix the existing CDF with
//!      the transformer's next-byte distribution.
//!    - After the byte is known (encoded or decoded), call
//!      [`NeuralArm::feed`] to advance the cache state.
//! 3. Encode and decode see the same byte sequence in the same
//!    order, so the cache state is identical on both sides and
//!    the AC roundtrip is preserved.
//!
//! Context handling: when `cache.pos` reaches the trained context
//! length the cache resets to zero and the next prediction has no
//! prior history. The first byte of each chunk costs ~8 bits under
//! the neural prediction (close to uniform). On a 1 GB stream with
//! `ctx=256` this overhead is bounded above by 1 GB / 256 × 8 / 8e9
//! ≈ 0.004 bpb — negligible vs the per-stream gain target.

use std::path::Path;

use anyhow::{Context, Result};

use crate::ac::TOTAL;
use crate::transformer::{ByteTransformer, KvCache};

#[derive(Debug)]
pub(crate) struct NeuralArm {
    model: ByteTransformer,
    cache: KvCache,
    /// `Some(logits)` of shape `[vocab_size]` once at least one byte
    /// has been fed since the last cache reset; `None` at cold start.
    pending_logits: Option<Vec<f32>>,
    /// Source bytes fed total (across cache cycles). Useful for
    /// diagnostics; not used in the hot path.
    fed_count: usize,
}

impl NeuralArm {
    pub(crate) fn load(weights_path: &Path) -> Result<Self> {
        let bytes = std::fs::read(weights_path)
            .with_context(|| format!("reading neural-arm weights {}", weights_path.display()))?;
        let model = ByteTransformer::load_lzrn(&bytes)
            .with_context(|| "parsing .lzrn neural-arm weights")?;
        let cache = model.new_kv_cache();
        Ok(Self {
            model,
            cache,
            pending_logits: None,
            fed_count: 0,
        })
    }

    pub(crate) const fn cfg(&self) -> crate::transformer::TransformerConfig {
        self.model.cfg
    }

    pub(crate) const fn fed_count(&self) -> usize {
        self.fed_count
    }

    /// Feed one source byte. The cache cycles every `context` steps;
    /// callers don't need to track that.
    pub(crate) fn feed(&mut self, byte: u8) {
        if self.cache.pos >= self.model.cfg.context {
            self.cache.reset();
        }
        self.pending_logits = Some(self.model.forward_step(&mut self.cache, byte));
        self.fed_count += 1;
    }

    /// Reset both cache and pending logits; useful at panel boundaries
    /// where prior-window context would be misleading.
    pub(crate) fn reset(&mut self) {
        self.cache.reset();
        self.pending_logits = None;
    }

    /// Build a 257-entry AC CDF over bytes for the next emission,
    /// mixing the existing byte-level CDF (`existing_cdf`, 257 entries)
    /// with the transformer's softmax over its current pending logits.
    /// `weight` controls the neural-arm's contribution: 0.0 returns
    /// `existing_cdf` unchanged, 1.0 returns a pure neural CDF, 0.5
    /// is an equal-weight average.
    ///
    /// At cold start (no pending logits), returns `existing_cdf` so
    /// the AC bits are identical to the baseline emission — the
    /// neural arm only kicks in once it has been fed at least one
    /// byte.
    #[allow(
        clippy::needless_range_loop,
        clippy::suboptimal_flops,
        clippy::many_single_char_names
    )]
    pub(crate) fn mix_byte_cdf(
        &self,
        existing_cdf: &[u32; 257],
        weight: f32,
        out: &mut [u32; 257],
    ) {
        if weight == 0.0 || self.pending_logits.is_none() {
            *out = *existing_cdf;
            return;
        }
        let logits = self.pending_logits.as_ref().expect("non-None checked");
        let probs = softmax_256(logits);

        // Cumulative mix in f64 to avoid rounding drift across 256
        // adds; then scale back to u32 AC mass with strict
        // monotonicity guaranteed by floor-then-clamp.
        let total_f = f64::from(TOTAL);
        let w_e = f64::from(1.0 - weight);
        let w_n = f64::from(weight);
        // Existing CDF is already in u32 AC scale; per-byte mass is
        // `existing[i+1] - existing[i]`.
        let mut mixed_mass = [0f64; 256];
        for i in 0..256 {
            let e = f64::from(existing_cdf[i + 1] - existing_cdf[i]) / total_f;
            let n = f64::from(probs[i]);
            mixed_mass[i] = w_e * e + w_n * n;
        }
        // Normalize (defensive; mixed should already sum to ~1).
        let s: f64 = mixed_mass.iter().sum();
        let inv = if s > 0.0 { 1.0 / s } else { 1.0 };

        // Materialize the cumulative CDF, ensuring strict
        // monotonicity (every slot ≥ 1 unit of mass).
        out[0] = 0;
        let mut acc = 0f64;
        for i in 0..256 {
            acc += mixed_mass[i] * inv;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let scaled = (acc * total_f) as u32;
            out[i + 1] = scaled;
        }
        // Enforce min-mass floor: every byte must have ≥ 1 unit.
        out[256] = TOTAL;
        let mut prev = 0u32;
        for slot in out.iter_mut().take(257).skip(1) {
            if *slot <= prev {
                *slot = prev + 1;
            }
            prev = *slot;
        }
        // If the floor pushed us past TOTAL, scale back down to
        // preserve `out[256] == TOTAL`. The drift is bounded by 256
        // units and TOTAL ~2^16, so this scale-back is rare.
        if out[256] != TOTAL {
            // Renormalize gaps proportionally.
            let last = out[256];
            for slot in out.iter_mut().take(257).skip(1) {
                let scaled = u64::from(*slot) * u64::from(TOTAL) / u64::from(last);
                #[allow(clippy::cast_possible_truncation)]
                let s = scaled as u32;
                *slot = s.max(1);
            }
            out[256] = TOTAL;
            // Re-floor (paranoia for monotonic-after-renorm).
            let mut prev = 0u32;
            for slot in out.iter_mut().take(257).skip(1) {
                if *slot <= prev {
                    *slot = prev + 1;
                }
                prev = *slot;
            }
        }
    }
}

/// Numerically-stable softmax over the first 256 entries of `logits`.
/// (The model's vocab is exactly 256 — byte-level — so this matches
/// the full softmax.)
#[allow(clippy::cast_possible_truncation)]
fn softmax_256(logits: &[f32]) -> [f32; 256] {
    let mut out = [0f32; 256];
    let mut max = f32::NEG_INFINITY;
    for &v in logits.iter().take(256) {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0f32;
    for (i, &v) in logits.iter().take(256).enumerate() {
        let e = (v - max).exp();
        out[i] = e;
        sum += e;
    }
    let inv = sum.recip();
    for v in &mut out {
        *v *= inv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arm_loads_and_predicts() {
        let weights = "/Users/creynolds/Programming/lzr-neural/ckpts/nano_plus_42_step20000.lzrn";
        if !Path::new(weights).exists() {
            eprintln!("weights missing — skipping");
            return;
        }
        let mut arm = NeuralArm::load(Path::new(weights)).unwrap();
        assert!(arm.pending_logits.is_none());
        arm.feed(b'a');
        assert!(arm.pending_logits.is_some());
        let logits = arm.pending_logits.as_ref().unwrap();
        assert_eq!(logits.len(), arm.cfg().vocab_size);
    }

    #[test]
    #[allow(
        clippy::needless_range_loop,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn mix_cdf_zero_weight_returns_existing() {
        let weights = "/Users/creynolds/Programming/lzr-neural/ckpts/nano_plus_42_step20000.lzrn";
        if !Path::new(weights).exists() {
            eprintln!("weights missing — skipping");
            return;
        }
        let mut arm = NeuralArm::load(Path::new(weights)).unwrap();
        arm.feed(b'a');
        let mut existing = [0u32; 257];
        // Build a flat CDF.
        for i in 0..=256 {
            existing[i] = (u64::from(TOTAL) * i as u64 / 256) as u32;
        }
        existing[256] = TOTAL;
        let mut out = [0u32; 257];
        arm.mix_byte_cdf(&existing, 0.0, &mut out);
        assert_eq!(out, existing);
    }

    #[test]
    #[allow(
        clippy::needless_range_loop,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn mix_cdf_is_strictly_monotone() {
        let weights = "/Users/creynolds/Programming/lzr-neural/ckpts/nano_plus_42_step20000.lzrn";
        if !Path::new(weights).exists() {
            eprintln!("weights missing — skipping");
            return;
        }
        let mut arm = NeuralArm::load(Path::new(weights)).unwrap();
        arm.feed(b'a');
        // Flat existing CDF; mix at 0.5 should still be strictly increasing.
        let mut existing = [0u32; 257];
        for i in 0..=256 {
            existing[i] = (u64::from(TOTAL) * i as u64 / 256) as u32;
        }
        existing[256] = TOTAL;
        let mut out = [0u32; 257];
        arm.mix_byte_cdf(&existing, 0.5, &mut out);
        assert_eq!(out[0], 0);
        assert_eq!(out[256], TOTAL);
        for w in out.windows(2) {
            assert!(w[1] > w[0], "non-monotone CDF: {} <= {}", w[1], w[0]);
        }
    }
}
