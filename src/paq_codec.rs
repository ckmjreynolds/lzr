//! Phase-11 codec: bit-level arithmetic coding with multi-context
//! predictor ensemble and an adaptive logistic mixer.
//!
//! Each byte is decomposed into 8 binary decisions. For each bit:
//!
//! 1. Several independent context predictors estimate `P(bit=1)`,
//!    each conditioned on a different feature of the byte history
//!    (Order-0, Order-1, Order-2, Order-3 over the previous bytes,
//!    plus bit-position-within-byte and the bits already emitted of
//!    the current byte).
//! 2. Their predictions are mapped through the logit
//!    `stretch(p) = log(p / (1 - p))` and combined as a weighted
//!    sum.
//! 3. The combined logit is squashed back to a probability, scaled to
//!    `TOTAL`, and used as a 2-entry CDF through the existing AC.
//! 4. After each bit observation, weights update by stochastic
//!    gradient descent on cross-entropy loss
//!    (`dL/dw_i = (p1 - bit) * logit_i`), and each predictor's count
//!    table is incremented at the observed bit.
//!
//! This is the PAQ skeleton: single global mixer, no per-context
//! mixing weights, no Secondary Symbol Estimation, no match model,
//! no neural network in the mixer. The architecture is shaped to
//! accept all of those as additional context features and additional
//! mixer inputs without changing the framing.
//!
//! All contexts include `bit_pos` and `partial_byte` so the predictor
//! sees intra-byte structure — once the MSBs of a byte are emitted,
//! the prediction for the remaining bits sharpens dramatically (an
//! advantage byte-level codecs structurally can't have).

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bit_pred::{BitPredictor, FNV_OFFSET, fnv_mix};
use crate::bits::{BitReader, BitWriter};
use crate::codec::{Codec, Decomposition};
use crate::match_model::MatchModel;

/// Predictor table sizes per order. Higher orders need more slots to
/// hold their richer context space; orders 0/1 fit comfortably in a
/// few hundred K slots.
const K_BITS: [u32; 4] = [11, 17, 20, 20];

/// Total number of mixer inputs: four context-predictor logits +
/// one match-model logit.
const N_MIXERS: usize = 5;

/// Mixer learning rate. Tuned conservatively — gradient magnitudes on
/// English text are small (per-bit logits ~0-5), so 0.01 keeps
/// weights from oscillating while still converging on the panel's
/// 256 KiB window.
const MIXER_LR: f32 = 0.02;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PaqCodec;

/// Per-order context hash for the next bit. `n` is the byte-history
/// depth. The hash mixes in `bit_pos` and `partial_byte` so each bit
/// within a byte gets a distinct context (and the predictor sees the
/// intra-byte refinement structure).
#[inline]
fn ctx_hash(history: &[u8], n: usize, bit_pos: u8, partial: u8) -> u64 {
    let mut h = FNV_OFFSET;
    let start = history.len().saturating_sub(n);
    for &b in &history[start..] {
        h = fnv_mix(h, u64::from(b));
    }
    h = fnv_mix(h, u64::from(bit_pos));
    fnv_mix(h, u64::from(partial))
}

/// Compute all 4 per-order context hashes for the current bit.
#[inline]
fn all_contexts(history: &[u8], bit_pos: u8, partial: u8) -> [u64; 4] {
    [
        ctx_hash(history, 0, bit_pos, partial),
        ctx_hash(history, 1, bit_pos, partial),
        ctx_hash(history, 2, bit_pos, partial),
        ctx_hash(history, 3, bit_pos, partial),
    ]
}

/// `logit(p1)` where `p1 = (TOTAL - p_zero_scaled) / TOTAL`.
/// Saturating around `p1 = 0` or `p1 = 1` to keep `log` finite.
#[inline]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn stretch_p1(p_zero_scaled: u32) -> f32 {
    let p1 = f64::from(TOTAL - p_zero_scaled) / f64::from(TOTAL);
    let p1 = p1.clamp(1e-6, 1.0 - 1e-6);
    (p1 / (1.0 - p1)).ln() as f32
}

/// Inverse of `stretch_p1`: convert a logit-of-`p1` back to
/// `p_zero` scaled to `TOTAL`. Clamped to `[1, TOTAL - 1]` for the
/// AC's strictly-increasing-CDF contract.
#[inline]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn squash_to_p_zero(logit_p1: f32) -> u32 {
    let p1 = 1.0 / (1.0 + (-f64::from(logit_p1)).exp());
    let p_zero = 1.0 - p1;
    let scaled = (p_zero * f64::from(TOTAL)) as u32;
    scaled.clamp(1, TOTAL - 1)
}

/// Adaptive logistic mixer. One scalar weight per predictor; updated
/// online via cross-entropy gradient after every observed bit.
#[derive(Debug)]
struct Mixer {
    weights: [f32; N_MIXERS],
    lr: f32,
}

impl Mixer {
    const fn new(lr: f32) -> Self {
        // Equal weights at init so each predictor contributes 1/N
        // before the gradient updates differentiate them.
        Self {
            weights: [0.2_f32; N_MIXERS],
            lr,
        }
    }

    /// Mixed logit = weighted sum of stretched per-predictor logits.
    fn mix(&self, logits: &[f32; N_MIXERS]) -> f32 {
        let mut sum = 0.0_f32;
        for (w, l) in self.weights.iter().zip(logits.iter()) {
            sum = w.mul_add(*l, sum);
        }
        sum
    }

    /// SGD step on cross-entropy: `dL/dw_i = (p1 - bit) * logit_i`.
    /// `bit` is 0 or 1; `combined_logit` is the output of `mix`; the
    /// gradient term uses the squashed `p1`.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn update(&mut self, logits: &[f32; N_MIXERS], combined_logit: f32, bit: u32) {
        let p1 = 1.0_f64 / (1.0 + (-f64::from(combined_logit)).exp());
        let err = (bit as f32) - p1 as f32;
        for (w, l) in self.weights.iter_mut().zip(logits.iter()) {
            *w = (self.lr * err).mul_add(*l, *w);
        }
    }
}

impl Codec for PaqCodec {
    fn name(&self) -> &'static str {
        "paq"
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let mut writer = BitWriter::new();
        let mut predictors: [BitPredictor; 4] = [
            BitPredictor::new(K_BITS[0]),
            BitPredictor::new(K_BITS[1]),
            BitPredictor::new(K_BITS[2]),
            BitPredictor::new(K_BITS[3]),
        ];
        let mut mixer = Mixer::new(MIXER_LR);
        let mut match_model = MatchModel::new();
        let mut hist = [0u8; 3];
        if warm.len() >= 3 {
            hist.copy_from_slice(&warm[warm.len() - 3..]);
        }
        prewarm_predictors(&mut predictors, &mut mixer, &mut match_model, warm);

        let bits_before = writer.bits_written();
        {
            let mut enc = AcEncoder::new(&mut writer);
            for &byte in measure {
                match_model.enter_byte();
                let mut partial: u8 = 0;
                for bit_pos in (0..8u8).rev() {
                    let bit = u32::from((byte >> bit_pos) & 1);
                    let ctxs = all_contexts(&hist, bit_pos, partial);
                    let logits = [
                        stretch_p1(predictors[0].predict_p_zero(ctxs[0])),
                        stretch_p1(predictors[1].predict_p_zero(ctxs[1])),
                        stretch_p1(predictors[2].predict_p_zero(ctxs[2])),
                        stretch_p1(predictors[3].predict_p_zero(ctxs[3])),
                        stretch_p1(match_model.predict_p_zero(bit_pos, partial)),
                    ];
                    let combined_logit = mixer.mix(&logits);
                    let p_zero = squash_to_p_zero(combined_logit);
                    let cdf = [0u32, p_zero, TOTAL];
                    enc.encode(&cdf, bit as usize);
                    for (pred, &ctx) in predictors.iter_mut().zip(ctxs.iter()) {
                        pred.observe(ctx, bit);
                    }
                    mixer.update(&logits, combined_logit, bit);
                    partial = (partial << 1) | (bit as u8);
                }
                match_model.commit_byte(byte);
                hist[0] = hist[1];
                hist[1] = hist[2];
                hist[2] = byte;
            }
            enc.finish();
        }
        let payload_bits = writer.bits_written() - bits_before;
        let (mut payload, pad_bits) = writer.finish();

        let measure_len =
            u32::try_from(measure.len()).context("measure window > 4 GiB — unsupported")?;
        let mut archive = Vec::with_capacity(4 + payload.len());
        archive.extend_from_slice(&measure_len.to_le_bytes());
        archive.append(&mut payload);

        let mut decomp = Decomposition::new();
        decomp.add("paq_bits", payload_bits);
        decomp.add("framing", 32);
        decomp.add("padding", u64::from(pad_bits));
        Ok((archive, decomp))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        if archive.len() < 4 {
            bail!(
                "archive too short for length prefix ({} bytes)",
                archive.len()
            );
        }
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&archive[..4]);
        let measure_len = usize::try_from(u32::from_le_bytes(len_bytes))
            .expect("u32 measure_len fits usize on supported targets");

        let mut predictors: [BitPredictor; 4] = [
            BitPredictor::new(K_BITS[0]),
            BitPredictor::new(K_BITS[1]),
            BitPredictor::new(K_BITS[2]),
            BitPredictor::new(K_BITS[3]),
        ];
        let mut mixer = Mixer::new(MIXER_LR);
        let mut match_model = MatchModel::new();
        let mut hist = [0u8; 3];
        if warm.len() >= 3 {
            hist.copy_from_slice(&warm[warm.len() - 3..]);
        }
        prewarm_predictors(&mut predictors, &mut mixer, &mut match_model, warm);

        let mut reader = BitReader::new(&archive[4..]);
        let mut dec = AcDecoder::new(&mut reader);
        let mut out = Vec::with_capacity(measure_len);
        for _ in 0..measure_len {
            match_model.enter_byte();
            let mut byte: u8 = 0;
            let mut partial: u8 = 0;
            for bit_pos in (0..8u8).rev() {
                let ctxs = all_contexts(&hist, bit_pos, partial);
                let logits = [
                    stretch_p1(predictors[0].predict_p_zero(ctxs[0])),
                    stretch_p1(predictors[1].predict_p_zero(ctxs[1])),
                    stretch_p1(predictors[2].predict_p_zero(ctxs[2])),
                    stretch_p1(predictors[3].predict_p_zero(ctxs[3])),
                    stretch_p1(match_model.predict_p_zero(bit_pos, partial)),
                ];
                let combined_logit = mixer.mix(&logits);
                let p_zero = squash_to_p_zero(combined_logit);
                let cdf = [0u32, p_zero, TOTAL];
                let bit_sym = dec.decode(&cdf)?;
                let bit_u8 = u8::try_from(bit_sym).expect("bit-AC symbol ∈ {0,1}");
                let bit = u32::from(bit_u8);
                for (pred, &ctx) in predictors.iter_mut().zip(ctxs.iter()) {
                    pred.observe(ctx, bit);
                }
                mixer.update(&logits, combined_logit, bit);
                byte = (byte << 1) | bit_u8;
                partial = (partial << 1) | bit_u8;
            }
            match_model.commit_byte(byte);
            out.push(byte);
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = byte;
        }
        Ok(out)
    }
}

/// Walk the warm prefix bit-by-bit, updating predictor counts, mixer
/// weights, AND match-model buffer+hash without emitting through the
/// AC. Identical paths on both sides so the measure-region state
/// matches exactly.
#[allow(clippy::cast_possible_truncation)]
fn prewarm_predictors(
    predictors: &mut [BitPredictor; 4],
    mixer: &mut Mixer,
    match_model: &mut MatchModel,
    warm: &[u8],
) {
    let mut hist = [0u8; 3];
    for (i, &byte) in warm.iter().enumerate() {
        match_model.enter_byte();
        let mut partial: u8 = 0;
        for bit_pos in (0..8u8).rev() {
            let bit = u32::from((byte >> bit_pos) & 1);
            let ctxs = all_contexts(&hist, bit_pos, partial);
            let logits = [
                stretch_p1(predictors[0].predict_p_zero(ctxs[0])),
                stretch_p1(predictors[1].predict_p_zero(ctxs[1])),
                stretch_p1(predictors[2].predict_p_zero(ctxs[2])),
                stretch_p1(predictors[3].predict_p_zero(ctxs[3])),
                stretch_p1(match_model.predict_p_zero(bit_pos, partial)),
            ];
            let combined_logit = mixer.mix(&logits);
            for (pred, &ctx) in predictors.iter_mut().zip(ctxs.iter()) {
                pred.observe(ctx, bit);
            }
            mixer.update(&logits, combined_logit, bit);
            partial = (partial << 1) | (bit as u8);
        }
        match_model.commit_byte(byte);
        if i >= 2 {
            hist[0] = hist[1];
            hist[1] = hist[2];
            hist[2] = byte;
        } else {
            hist[i] = byte;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(warm: &[u8], measure: &[u8]) {
        let codec = PaqCodec;
        let (archive, _decomp) = codec.encode_window(warm, measure).unwrap();
        let decoded = codec.decode_window(warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn paq_codec_roundtrips_short_text() {
        roundtrip(b"", b"The quick brown fox jumps over the lazy dog.");
    }

    #[test]
    fn paq_codec_roundtrips_with_warm() {
        let warm = b"<page><title>Foo</title></page>";
        let measure = b"some content following the warm prefix.";
        roundtrip(warm, measure);
    }

    #[test]
    fn paq_codec_roundtrips_long_repetition() {
        let mut s = Vec::new();
        for _ in 0..200 {
            s.extend_from_slice(b"abcdefghijklmnop");
        }
        roundtrip(b"", &s);
    }

    #[test]
    fn paq_codec_roundtrips_real_enwik8_window() {
        let Ok(bytes) = std::fs::read("assets/enwik8") else {
            return;
        };
        let warm_start = 4_194_304 - 4 * 1024 * 1024;
        let warm = &bytes[warm_start..4_194_304];
        let measure = &bytes[4_194_304..4_194_304 + 8192];
        roundtrip(warm, measure);
    }

    #[test]
    fn stretch_squash_roundtrips_at_neutral() {
        let p_zero = TOTAL / 2;
        let l = stretch_p1(p_zero);
        let back = squash_to_p_zero(l);
        assert!((i64::from(back) - i64::from(p_zero)).abs() <= 1);
    }
}
