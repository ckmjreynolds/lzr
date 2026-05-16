//! Streaming wrapper around [`crate::moe::MoeByteTransformer`] that
//! the v4 codec drives byte-by-byte during encode/decode.
//!
//! Lifecycle inside the codec:
//! 1. `MoeArm::load(path)` once at the start of encode/decode.
//! 2. Walk source bytes in order:
//!    - Call [`MoeArm::predict_byte_cdf`] to obtain a 257-entry AC
//!      CDF for the next byte's distribution.
//!    - After the byte is known (encoded or decoded), call
//!      [`MoeArm::feed`] to advance the cache state.
//! 3. Encode and decode see the same byte sequence, so cache state
//!    matches on both sides and the AC roundtrip is preserved bit-
//!    for-bit.
//!
//! Context handling: when `cache.pos` reaches the trained context
//! length, [`MoeArm::feed`] auto-resets the cache and the next
//! prediction has no prior history. The first byte of each
//! `context`-sized chunk costs ~8 bits under near-uniform prediction,
//! bounding total cold-start overhead at ~`8 / context` bpb (≈0.031
//! bpb at `context=256`). The codec amortizes this by warming
//! through real prefix bytes before scoring measure bytes.

#![allow(dead_code)]

use std::path::Path;

use anyhow::{Context, Result};

use crate::ac::TOTAL;
use crate::moe::{MoeByteTransformer, MoeConfig, MoeKvCache, Quantization};

/// Env var read at load time to choose a weight bit width. Values:
/// `"int8"`, `"int4"`. Anything else (including unset) keeps `f32`.
pub(crate) const QUANT_ENV: &str = "LZR_MOE_QUANT";

#[derive(Debug)]
pub(crate) struct MoeArm {
    model: MoeByteTransformer,
    cache: MoeKvCache,
    quant: Quantization,
    pending_logits: Option<Vec<f32>>,
    fed_count: usize,
}

impl MoeArm {
    pub(crate) fn load(weights_path: &Path) -> Result<Self> {
        Self::load_with_quantization(weights_path, Quantization::from_env(QUANT_ENV))
    }

    pub(crate) fn load_with_quantization(weights_path: &Path, q: Quantization) -> Result<Self> {
        let bytes = std::fs::read(weights_path)
            .with_context(|| format!("reading MoE arm weights {}", weights_path.display()))?;
        let mut model = MoeByteTransformer::load_lzrm(&bytes)
            .with_context(|| "parsing .lzrm MoE-arm weights")?;
        model.apply_quantization(q);
        let cache = model.new_kv_cache();
        Ok(Self {
            model,
            cache,
            quant: q,
            pending_logits: None,
            fed_count: 0,
        })
    }

    pub(crate) const fn quantization(&self) -> Quantization {
        self.quant
    }

    pub(crate) fn total_params(&self) -> usize {
        self.model.total_params()
    }

    pub(crate) const fn cfg(&self) -> MoeConfig {
        self.model.cfg
    }

    pub(crate) const fn fed_count(&self) -> usize {
        self.fed_count
    }

    /// Feed one source byte. Auto-resets the KV cache when it fills,
    /// so callers don't need to track context boundaries.
    pub(crate) fn feed(&mut self, byte: u8) {
        if self.cache.pos >= self.model.cfg.context {
            self.cache.reset();
        }
        self.pending_logits = Some(self.model.forward_step(&mut self.cache, byte));
        self.fed_count += 1;
    }

    /// Reset both KV cache and pending logits. Called by the codec
    /// at window boundaries where prior-window context is irrelevant.
    pub(crate) fn reset(&mut self) {
        self.cache.reset();
        self.pending_logits = None;
    }

    /// Build a 257-entry AC CDF over bytes for the next emission
    /// from the `MoE` arm's current state. At cold start (no logits
    /// pending) returns a uniform CDF — every byte gets ~1/256 mass.
    ///
    /// Strict monotonicity is enforced (every byte gets ≥ 1 unit of
    /// AC mass) so the AC encoder/decoder never see a zero-prob
    /// symbol, which would cause encode/decode failure.
    #[allow(
        clippy::needless_range_loop,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub(crate) fn predict_byte_cdf(&self, out: &mut [u32; 257]) {
        let probs = self
            .pending_logits
            .as_ref()
            .map_or([1.0_f32 / 256.0; 256], |logits| softmax_256(logits));

        // Materialize the cumulative CDF in f64 to avoid drift across
        // 256 adds, then enforce min-mass floor.
        let total_f = f64::from(TOTAL);
        out[0] = 0;
        let mut acc = 0_f64;
        for i in 0..256 {
            acc += f64::from(probs[i]);
            let scaled = (acc * total_f) as u32;
            out[i + 1] = scaled;
        }
        out[256] = TOTAL;

        // Strict-increasing floor: every slot ≥ 1 unit of mass.
        let mut prev = 0_u32;
        for slot in out.iter_mut().take(257).skip(1) {
            if *slot <= prev {
                *slot = prev + 1;
            }
            prev = *slot;
        }
        if out[256] != TOTAL {
            // Renormalize gaps proportionally back to TOTAL endpoint.
            let last = out[256];
            for slot in out.iter_mut().take(257).skip(1) {
                let scaled = u64::from(*slot) * u64::from(TOTAL) / u64::from(last);
                *slot = (scaled as u32).max(1);
            }
            out[256] = TOTAL;
            let mut prev = 0_u32;
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
/// (The model's vocab is exactly 256.)
#[allow(clippy::cast_possible_truncation)]
fn softmax_256(logits: &[f32]) -> [f32; 256] {
    let mut out = [0_f32; 256];
    let mut max = f32::NEG_INFINITY;
    for &v in logits.iter().take(256) {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0_f32;
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
