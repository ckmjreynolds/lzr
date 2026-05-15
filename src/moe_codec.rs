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

use crate::ac::{AcDecoder, AcEncoder};
use crate::bits::{BitReader, BitWriter};
use crate::codec::{Codec, Decomposition};
use crate::moe_arm::MoeArm;

const WEIGHTS_ENV: &str = "LZR_MOE_WEIGHTS";

#[derive(Debug, Default)]
pub(crate) struct MoeCodec {
    /// Lazy-init the arm on first encode/decode call so the codec
    /// can be constructed without weights present (matches the v3
    /// pattern where `make_codec` is called eagerly during CLI
    /// dispatch but actual weight access can be deferred).
    arm: Mutex<Option<MoeArm>>,
}

impl MoeCodec {
    pub(crate) const fn new() -> Self {
        Self {
            arm: Mutex::new(None),
        }
    }

    /// Run `op` against the (lazily-loaded) `MoeArm`. The mutex
    /// guard is held for the entire encode/decode pass so no other
    /// thread can touch the cache mid-emission. Single-threaded by
    /// design (submission binary invariant), but the mutex is
    /// required because [`Codec`] is `&self`.
    #[allow(clippy::significant_drop_tightening)]
    fn with_arm<R>(&self, op: impl FnOnce(&mut MoeArm) -> Result<R>) -> Result<R> {
        let mut guard = self.arm.lock().expect("MoE arm mutex poisoned");
        if guard.is_none() {
            let path: PathBuf = std::env::var_os(WEIGHTS_ENV)
                .map(PathBuf::from)
                .with_context(|| {
                    format!("v4 MoE codec requires {WEIGHTS_ENV} env var pointing at .lzrm weights")
                })?;
            let arm = MoeArm::load(&path)
                .with_context(|| format!("loading MoE arm from {}", path.display()))?;
            *guard = Some(arm);
        }
        let arm = guard.as_mut().expect("loaded above");
        arm.reset();
        op(arm)
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

        let (archive, ac_bits, total_bits) = self.with_arm(|arm| {
            for &b in warm {
                arm.feed(b);
            }

            let mut bw = BitWriter::new();
            let mut cdf = [0_u32; 257];
            let bits_before;
            let bits_after;
            {
                let mut enc = AcEncoder::new(&mut bw);
                bits_before = enc.bits_written();
                for &b in measure {
                    arm.predict_byte_cdf(&mut cdf);
                    enc.encode(&cdf, b as usize);
                    arm.feed(b);
                }
                bits_after = enc.bits_written();
                enc.finish();
            }
            let (ac_bytes, _trailing) = bw.finish();
            let ac_bits = bits_after - bits_before;

            // Archive = 4-byte LE length || AC bytes.
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

        self.with_arm(|arm| {
            for &b in warm {
                arm.feed(b);
            }

            let mut br = BitReader::new(ac_bytes);
            let mut dec = AcDecoder::new(&mut br);
            let mut cdf = [0_u32; 257];
            let mut out = Vec::with_capacity(measure_len);
            for _ in 0..measure_len {
                arm.predict_byte_cdf(&mut cdf);
                let b = dec.decode(&cdf)?;
                let byte =
                    u8::try_from(b).with_context(|| format!("AC produced non-byte symbol {b}"))?;
                out.push(byte);
                arm.feed(byte);
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
