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

/// Scalar temperature applied to logits before softmax. `T < 1`
/// sharpens. Fit on 1 MB and 10 MB sweeps in Phase 48 — see journal
/// entry; the `MoE` was systematically ~6% under-confident and `T=0.94`
/// recovers ~0.0046 bpb at zero shipped-binary cost.
const LOGIT_TEMP: f32 = 0.94;

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
        self.pending_logits = Some(self.model.forward_step(&mut self.cache, u32::from(byte)));
        self.fed_count += 1;
    }

    /// Same as [`feed`] but accepts a token id directly (for non-byte
    /// vocabs, where the token id may exceed `u8` range).
    pub(crate) fn feed_token(&mut self, token: u32) {
        if self.cache.pos >= self.model.cfg.context {
            self.cache.reset();
        }
        self.pending_logits = Some(self.model.forward_step(&mut self.cache, token));
        self.fed_count += 1;
    }

    /// Reset both KV cache and pending logits. Called by the codec
    /// at window boundaries where prior-window context is irrelevant.
    pub(crate) fn reset(&mut self) {
        self.cache.reset();
        self.pending_logits = None;
    }

    /// Build a CDF over the model's full vocab for the next emission.
    /// `out.len()` must equal `vocab_size + 1`. At cold start
    /// (no logits pending) returns a uniform CDF. Strict monotonicity
    /// is enforced so the AC encoder/decoder never see zero-prob
    /// symbols.
    #[allow(
        clippy::needless_range_loop,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub(crate) fn predict_cdf(&self, out: &mut [u32]) {
        self.predict_cdf_with_bias(out, &[]);
    }

    /// Same as [`MoeArm::predict_cdf`] but adds `bias[i]` to each
    /// logit before the (temperature-scaled) softmax. `bias` may be
    /// empty, in which case it is treated as all zeros. Used by the
    /// `moe-tok` codec to apply a structural mask (Phase 49).
    #[allow(
        clippy::needless_range_loop,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub(crate) fn predict_cdf_with_bias(&self, out: &mut [u32], bias: &[f32]) {
        let vocab = self.model.cfg.vocab_size;
        assert_eq!(
            out.len(),
            vocab + 1,
            "predict_cdf: out must have vocab+1 entries"
        );
        assert!(
            bias.is_empty() || bias.len() == vocab,
            "bias must be empty or vocab-sized"
        );
        #[allow(clippy::cast_precision_loss)]
        let mut probs = vec![1.0_f32 / vocab as f32; vocab];
        if let Some(logits) = self.pending_logits.as_ref() {
            softmax_into_with_bias(logits, bias, &mut probs);
        }
        let total_f = f64::from(TOTAL);
        out[0] = 0;
        let mut acc = 0_f64;
        for i in 0..vocab {
            acc += f64::from(probs[i]);
            let scaled = (acc * total_f) as u32;
            out[i + 1] = scaled;
        }
        out[vocab] = TOTAL;
        enforce_monotonic(out, vocab);
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

/// Enforce strict monotonicity on a length-`(vocab+1)` CDF in place,
/// ending with `out[vocab] == TOTAL`. Used by [`MoeArm::predict_cdf`]
/// and shared with [`MoeArm::predict_byte_cdf`].
fn enforce_monotonic(out: &mut [u32], vocab: usize) {
    debug_assert_eq!(out.len(), vocab + 1);
    let mut prev = 0_u32;
    for slot in out.iter_mut().take(vocab + 1).skip(1) {
        if *slot <= prev {
            *slot = prev + 1;
        }
        prev = *slot;
    }
    if out[vocab] != TOTAL {
        let last = out[vocab];
        for slot in out.iter_mut().take(vocab + 1).skip(1) {
            let scaled = u64::from(*slot) * u64::from(TOTAL) / u64::from(last);
            #[allow(clippy::cast_possible_truncation)]
            let s = scaled as u32;
            *slot = s.max(1);
        }
        out[vocab] = TOTAL;
        let mut prev = 0_u32;
        for slot in out.iter_mut().take(vocab + 1).skip(1) {
            if *slot <= prev {
                *slot = prev + 1;
            }
            prev = *slot;
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

/// Variable-length numerically-stable softmax — writes into `out`,
/// reading `out.len()` entries from `logits`. Used for token vocabs
/// where the size isn't known at compile time.
fn softmax_into(logits: &[f32], out: &mut [f32]) {
    softmax_into_with_bias(logits, &[], out);
}

fn softmax_into_with_bias(logits: &[f32], bias: &[f32], out: &mut [f32]) {
    debug_assert!(logits.len() >= out.len());
    debug_assert!(bias.is_empty() || bias.len() >= out.len());
    let inv_t = LOGIT_TEMP.recip();
    let use_bias = !bias.is_empty();
    let mut max = f32::NEG_INFINITY;
    for i in 0..out.len() {
        let v = if use_bias {
            logits[i] + bias[i]
        } else {
            logits[i]
        };
        if v > max {
            max = v;
        }
    }
    let mut sum = 0_f32;
    for i in 0..out.len() {
        let v = if use_bias {
            logits[i] + bias[i]
        } else {
            logits[i]
        };
        let e = ((v - max) * inv_t).exp();
        out[i] = e;
        sum += e;
    }
    let inv = sum.recip();
    for v in out.iter_mut() {
        *v *= inv;
    }
}
