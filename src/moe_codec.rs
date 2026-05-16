//! V4 neural-first codec: AC over a sparse-MoE byte distribution.
//!
//! The simplest possible codec built on top of the trained `MoE` arm:
//!
//! 1. For each byte in `measure`, ask the [`MoeArm`] for a 257-entry
//!    AC CDF over bytes (built from the model's softmax over its
//!    next-byte logits across the 256-element vocab).
//! 2. Encode the actual byte against that CDF using the shared bit-
//!    level [`AcEncoder`].
//! 3. Feed the byte to the arm to advance the KV cache for the next
//!    prediction.
//!
//! Decoding is symmetric: the arm runs deterministically on both
//! sides, so the AC stream is bit-identical and the decoder
//! reconstructs the bytes exactly.
//!
//! Warm bytes are fed through the arm before any AC emission, so the
//! arm's KV cache is primed identically on both sides without
//! contributing to the bit count. The cache auto-resets at every
//! `context`-byte boundary (handled inside [`MoeArm::feed`]); the
//! first byte after each reset costs ~8 bits under near-uniform
//! prediction. At `context=256` this caps cold-start overhead at
//! `8 / 256 ≈ 0.031` bpb.
//!
//! Archive layout:
//!
//! ```text
//!   bytes 0..4   : measure_len as u32 LE (so decode knows when to stop)
//!   bytes 4..    : AC stream, bit-packed via BitWriter
//! ```
//!
//! Decomposition components:
//! - `moe_ac`: bits emitted by the AC encoder for the byte stream.
//! - `framing`: the 4-byte length prefix and any AC tail-padding bits.
//!
//! Weights are loaded lazily from the path in `LZR_MOE_WEIGHTS`. If
//! the env var is unset, both `encode_window` and `decode_window`
//! return an error — the v4 codec has no fallback predictor.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bits::{BitReader, BitWriter};
use crate::codec::{Codec, Decomposition};
use crate::moe_arm::MoeArm;
use crate::ngram_arm::NgramArm;

const WEIGHTS_ENV: &str = "LZR_MOE_WEIGHTS";
/// Weight given to the runtime n-gram arm when mixing with the `MoE`
/// arm in probability space. Default 0.0 means MoE-only (the
/// Phase 30 baseline). Override via `LZR_NGRAM_WEIGHT` env var in
/// `[0.0, 1.0]` to sweep without recompiling.
const NGRAM_WEIGHT_ENV: &str = "LZR_NGRAM_WEIGHT";

#[derive(Debug, Default)]
pub(crate) struct MoeCodec {
    /// Lazy-init the arms on first encode/decode call so the codec
    /// can be constructed without weights present (matches the v3
    /// pattern where `make_codec` is called eagerly during CLI
    /// dispatch but actual weight access can be deferred).
    arms: Mutex<Option<Arms>>,
}

#[derive(Debug)]
struct Arms {
    moe: MoeArm,
    ngram: NgramArm,
    /// Mixing weight on the n-gram CDF; `1.0 - ngram_weight` goes
    /// to the `MoE`. Set once at load time from `LZR_NGRAM_WEIGHT`
    /// env var. At 0.0 the n-gram arm is still fed (deterministic
    /// state) but contributes no mass to the AC distribution.
    ngram_weight: f64,
}

impl MoeCodec {
    pub(crate) const fn new() -> Self {
        Self {
            arms: Mutex::new(None),
        }
    }

    /// Run `op` against the (lazily-loaded) arms. The mutex guard is
    /// held for the entire encode/decode pass so no other thread can
    /// touch the cache mid-emission. Single-threaded by design
    /// (submission binary invariant), but the mutex is required
    /// because [`Codec`] is `&self`.
    #[allow(clippy::significant_drop_tightening)]
    fn with_arms<R>(&self, op: impl FnOnce(&mut Arms) -> Result<R>) -> Result<R> {
        let mut guard = self.arms.lock().expect("MoE codec arms mutex poisoned");
        if guard.is_none() {
            let path: PathBuf = std::env::var_os(WEIGHTS_ENV)
                .map(PathBuf::from)
                .with_context(|| {
                    format!("v4 MoE codec requires {WEIGHTS_ENV} env var pointing at .lzrm weights")
                })?;
            let moe = MoeArm::load(&path)
                .with_context(|| format!("loading MoE arm from {}", path.display()))?;
            let ngram_weight = std::env::var(NGRAM_WEIGHT_ENV)
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            *guard = Some(Arms {
                moe,
                ngram: NgramArm::new(),
                ngram_weight,
            });
        }
        let arms = guard.as_mut().expect("loaded above");
        arms.moe.reset();
        arms.ngram.reset();
        op(arms)
    }
}

/// Mix `moe` and `ngram` 257-entry AC CDFs in probability space:
/// `mixed_mass = (1-w) * moe_mass + w * ngram_mass`, then enforce
/// strict monotonicity. Writes to `out`.
#[allow(
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn mix_cdfs(moe: &[u32; 257], ngram: &[u32; 257], w_ngram: f64, out: &mut [u32; 257]) {
    let total_f = f64::from(TOTAL);
    let w_moe = 1.0 - w_ngram;
    let mut acc = 0_f64;
    out[0] = 0;
    for i in 0..256 {
        let moe_mass = f64::from(moe[i + 1] - moe[i]) / total_f;
        let ng_mass = f64::from(ngram[i + 1] - ngram[i]) / total_f;
        acc += w_ngram.mul_add(ng_mass, w_moe * moe_mass);
        out[i + 1] = (acc * total_f) as u32;
    }
    out[256] = TOTAL;

    let mut prev = 0_u32;
    for slot in out.iter_mut().take(257).skip(1) {
        if *slot <= prev {
            *slot = prev + 1;
        }
        prev = *slot;
    }
    if out[256] != TOTAL {
        let last = out[256];
        for slot in out.iter_mut().take(257).skip(1) {
            let scaled = u64::from(*slot) * u64::from(TOTAL) / u64::from(last);
            *slot = u32::try_from(scaled).unwrap_or(u32::MAX).max(1);
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

impl Codec for MoeCodec {
    fn name(&self) -> &'static str {
        "moe"
    }

    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        if measure.len() > u32::MAX as usize {
            bail!(
                "measure too large for u32 length prefix ({} > {})",
                measure.len(),
                u32::MAX
            );
        }
        let measure_len = u32::try_from(measure.len()).expect("checked above");

        let (archive, ac_bits, total_bits) = self.with_arms(|arms| {
            for &b in warm {
                arms.moe.feed(b);
                arms.ngram.feed(b);
            }

            let mut bw = BitWriter::new();
            let mut moe_cdf = [0_u32; 257];
            let mut ngram_cdf = [0_u32; 257];
            let mut mixed_cdf = [0_u32; 257];
            let bits_before;
            let bits_after;
            {
                let mut enc = AcEncoder::new(&mut bw);
                bits_before = enc.bits_written();
                let w = arms.ngram_weight;
                for &b in measure {
                    arms.moe.predict_byte_cdf(&mut moe_cdf);
                    let cdf_to_use: &[u32; 257] = if w == 0.0 {
                        &moe_cdf
                    } else {
                        arms.ngram.predict_byte_cdf(&mut ngram_cdf);
                        mix_cdfs(&moe_cdf, &ngram_cdf, w, &mut mixed_cdf);
                        &mixed_cdf
                    };
                    enc.encode(cdf_to_use, b as usize);
                    arms.moe.feed(b);
                    arms.ngram.feed(b);
                }
                bits_after = enc.bits_written();
                enc.finish();
            }
            let (ac_bytes, _trailing) = bw.finish();
            let ac_bits = bits_after - bits_before;

            let mut archive = Vec::with_capacity(4 + ac_bytes.len());
            archive.extend_from_slice(&measure_len.to_le_bytes());
            archive.extend_from_slice(&ac_bytes);
            let total_bits = 8 * archive.len() as u64;
            Ok((archive, ac_bits, total_bits))
        })?;

        let mut decomp = Decomposition::new();
        decomp.add("moe_ac", ac_bits);
        // Framing absorbs everything else (length prefix + AC tail
        // padding) so the decomposition sums to 8 * archive_bytes.
        debug_assert!(total_bits >= ac_bits);
        decomp.add("framing", total_bits - ac_bits);
        Ok((archive, decomp))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        if archive.len() < 4 {
            bail!(
                "archive too short for length prefix: {} bytes",
                archive.len()
            );
        }
        let measure_len = u32::from_le_bytes([archive[0], archive[1], archive[2], archive[3]]);
        let measure_len = measure_len as usize;
        let ac_bytes = &archive[4..];

        self.with_arms(|arms| {
            for &b in warm {
                arms.moe.feed(b);
                arms.ngram.feed(b);
            }

            let mut br = BitReader::new(ac_bytes);
            let mut dec = AcDecoder::new(&mut br);
            let mut moe_cdf = [0_u32; 257];
            let mut ngram_cdf = [0_u32; 257];
            let mut mixed_cdf = [0_u32; 257];
            let mut out = Vec::with_capacity(measure_len);
            let w = arms.ngram_weight;
            for _ in 0..measure_len {
                arms.moe.predict_byte_cdf(&mut moe_cdf);
                let cdf_to_use: &[u32; 257] = if w == 0.0 {
                    &moe_cdf
                } else {
                    arms.ngram.predict_byte_cdf(&mut ngram_cdf);
                    mix_cdfs(&moe_cdf, &ngram_cdf, w, &mut mixed_cdf);
                    &mixed_cdf
                };
                let b = dec.decode(cdf_to_use)?;
                let byte =
                    u8::try_from(b).with_context(|| format!("AC produced non-byte symbol {b}"))?;
                out.push(byte);
                arms.moe.feed(byte);
                arms.ngram.feed(byte);
            }
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end roundtrip on a small enwik8 prefix using the
    /// committed `nano_plus` `MoE` checkpoint. Requires
    /// `LZR_MOE_WEIGHTS` to be set externally (via the cargo test
    /// command line) and the corpus to be present. Skips cleanly
    /// otherwise so the test is safe on machines without the trained
    /// `.lzrm` artifact.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn moe_codec_roundtrips_with_committed_weights() {
        if std::env::var_os(WEIGHTS_ENV).is_none() {
            eprintln!("{WEIGHTS_ENV} unset — skipping codec roundtrip test");
            return;
        }
        // Read first 1 KB of enwik8 as the warm + measure split.
        let corpus_path = "/Users/creynolds/Programming/lzr/assets/enwik8";
        if !std::path::Path::new(corpus_path).exists() {
            eprintln!("corpus missing — skipping codec roundtrip test");
            return;
        }
        let corpus = std::fs::read(corpus_path).unwrap();
        let warm = &corpus[0..512];
        let measure = &corpus[512..1024];

        let codec = MoeCodec::new();
        let (archive, decomp) = codec.encode_window(warm, measure).unwrap();
        let decoded = codec.decode_window(warm, &archive).unwrap();
        assert_eq!(decoded, measure, "MoE codec must roundtrip");
        // Decomposition must sum to bits in the archive.
        assert_eq!(decomp.total(), 8 * archive.len() as u64);

        let bpb = 8.0 * archive.len() as f64 / measure.len() as f64;
        eprintln!(
            "moe_codec 512-byte window: archive={} bpb={:.3}",
            archive.len(),
            bpb
        );
    }
}
