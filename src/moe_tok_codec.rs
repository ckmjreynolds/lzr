//! Token-level v4 codec: BPE-tokenize bytes, AC-encode tokens against
//! a `MoE` arm's vocab-wide distribution.
//!
//! Pipeline (encode):
//! 1. Load BPE table from `LZR_BPE_TABLE` env var (`.bin` format).
//! 2. Load `MoE` arm from `LZR_MOE_WEIGHTS`; `vocab_size` must
//!    match the BPE table.
//! 3. BPE-encode `warm` bytes; feed tokens through the arm to prime
//!    the KV cache (no AC emission for warm).
//! 4. BPE-encode `measure` bytes. For each token:
//!    a. Ask the arm for a `vocab+1`-entry CDF.
//!    b. AC-encode the token id against the CDF.
//!    c. Feed the token to advance the arm's cache.
//! 5. Write `(u32 LE measure_byte_len)(u32 LE n_tokens)(AC bytes)`.
//!
//! Decode is symmetric: read header, AC-decode `n_tokens` token IDs
//! (using the same arm state), BPE-decode tokens to bytes, validate
//! byte length matches.
//!
//! The token count is shipped in the archive so the decoder knows
//! when to stop AC-decoding. The byte count is shipped so the
//! decoder can sanity-check the BPE round-trip.
//!
//! Weights load lazily on first encode/decode call (same pattern as
//! `moe_codec.rs`). Both `LZR_MOE_WEIGHTS` and `LZR_BPE_TABLE` must
//! be set; otherwise the codec errors at first use.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder};
use crate::bits::{BitReader, BitWriter};
use crate::bpe::Bpe;
use crate::codec::{Codec, Decomposition};
use crate::moe_arm::MoeArm;

const WEIGHTS_ENV: &str = "LZR_MOE_WEIGHTS";
const BPE_TABLE_ENV: &str = "LZR_BPE_TABLE";

#[derive(Debug, Default)]
pub(crate) struct MoeTokCodec {
    arms: Mutex<Option<TokArms>>,
}

#[derive(Debug)]
struct TokArms {
    moe: MoeArm,
    bpe: Bpe,
}

impl MoeTokCodec {
    pub(crate) const fn new() -> Self {
        Self {
            arms: Mutex::new(None),
        }
    }

    #[allow(clippy::significant_drop_tightening)]
    fn with_arms<R>(&self, op: impl FnOnce(&mut TokArms) -> Result<R>) -> Result<R> {
        let mut guard = self.arms.lock().expect("moe-tok arms mutex poisoned");
        if guard.is_none() {
            let weights_path: PathBuf = std::env::var_os(WEIGHTS_ENV)
                .map(PathBuf::from)
                .with_context(|| format!("v4 moe-tok codec requires {WEIGHTS_ENV} env var"))?;
            let bpe_path: PathBuf = std::env::var_os(BPE_TABLE_ENV)
                .map(PathBuf::from)
                .with_context(|| format!("v4 moe-tok codec requires {BPE_TABLE_ENV} env var"))?;
            let moe = MoeArm::load(&weights_path)
                .with_context(|| format!("loading MoE arm from {}", weights_path.display()))?;
            let bpe = Bpe::load(&bpe_path)
                .with_context(|| format!("loading BPE table from {}", bpe_path.display()))?;
            if bpe.vocab_size != moe.cfg().vocab_size {
                bail!(
                    "BPE vocab {} != MoE vocab {}",
                    bpe.vocab_size,
                    moe.cfg().vocab_size
                );
            }
            *guard = Some(TokArms { moe, bpe });
        }
        let arms = guard.as_mut().expect("loaded above");
        arms.moe.reset();
        op(arms)
    }
}

impl Codec for MoeTokCodec {
    fn name(&self) -> &'static str {
        "moe-tok"
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
            // Prime the arm with BPE-encoded warm bytes (no AC emit).
            let warm_tokens = arms.bpe.encode(warm);
            for &tok in &warm_tokens {
                arms.moe.feed_token(tok);
            }

            // BPE-encode measure once; we AC-encode the resulting
            // token sequence in order.
            let measure_tokens = arms.bpe.encode(measure);
            let n_tokens =
                u32::try_from(measure_tokens.len()).context("token count exceeds u32 capacity")?;
            let vocab = arms.moe.cfg().vocab_size;

            let mut bw = BitWriter::new();
            let mut cdf = vec![0_u32; vocab + 1];
            let bits_before;
            let bits_after;
            {
                let mut enc = AcEncoder::new(&mut bw);
                bits_before = enc.bits_written();
                for &tok in &measure_tokens {
                    arms.moe.predict_cdf(&mut cdf);
                    enc.encode(&cdf, tok as usize);
                    arms.moe.feed_token(tok);
                }
                bits_after = enc.bits_written();
                enc.finish();
            }
            let (ac_bytes, _trailing) = bw.finish();
            let ac_bits = bits_after - bits_before;

            // Archive = 4-byte LE measure_byte_len, 4-byte LE n_tokens, AC bytes.
            let mut archive = Vec::with_capacity(8 + ac_bytes.len());
            archive.extend_from_slice(&measure_len.to_le_bytes());
            archive.extend_from_slice(&n_tokens.to_le_bytes());
            archive.extend_from_slice(&ac_bytes);
            let total_bits = 8 * archive.len() as u64;
            Ok((archive, ac_bits, total_bits))
        })?;

        let mut decomp = Decomposition::new();
        decomp.add("moe_tok_ac", ac_bits);
        debug_assert!(total_bits >= ac_bits);
        decomp.add("framing", total_bits - ac_bits);
        Ok((archive, decomp))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        if archive.len() < 8 {
            bail!("archive too short for header: {} bytes", archive.len());
        }
        let measure_len =
            u32::from_le_bytes([archive[0], archive[1], archive[2], archive[3]]) as usize;
        let n_tokens =
            u32::from_le_bytes([archive[4], archive[5], archive[6], archive[7]]) as usize;
        let ac_bytes = &archive[8..];

        self.with_arms(|arms| {
            let warm_tokens = arms.bpe.encode(warm);
            for &tok in &warm_tokens {
                arms.moe.feed_token(tok);
            }

            let vocab = arms.moe.cfg().vocab_size;
            let mut br = BitReader::new(ac_bytes);
            let mut dec = AcDecoder::new(&mut br);
            let mut cdf = vec![0_u32; vocab + 1];
            let mut tokens = Vec::with_capacity(n_tokens);
            for _ in 0..n_tokens {
                arms.moe.predict_cdf(&mut cdf);
                let sym = dec.decode(&cdf)?;
                let tok = u32::try_from(sym)
                    .with_context(|| format!("AC produced non-vocab symbol {sym}"))?;
                tokens.push(tok);
                arms.moe.feed_token(tok);
            }

            let bytes = arms.bpe.decode(&tokens);
            if bytes.len() != measure_len {
                bail!(
                    "BPE decode mismatch: got {} bytes, archive header says {}",
                    bytes.len(),
                    measure_len
                );
            }
            Ok(bytes)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagnostic: encode 4 KB enwik8, then decode and compare TOKENS
    /// (not bytes). Catches AC↔model state divergence at the token
    /// level so we can see exactly where it goes wrong if it does.
    #[test]
    fn moe_tok_codec_token_level_diagnostic() {
        if std::env::var_os(WEIGHTS_ENV).is_none() || std::env::var_os(BPE_TABLE_ENV).is_none() {
            eprintln!("env unset — skipping");
            return;
        }
        let corpus_path = "/Users/creynolds/Programming/lzr/assets/enwik8";
        if !std::path::Path::new(corpus_path).exists() {
            eprintln!("corpus missing — skipping");
            return;
        }
        let corpus = std::fs::read(corpus_path).unwrap();
        let warm: &[u8] = b"";
        let measure = &corpus[0..64 * 1024];

        let codec = MoeTokCodec::new();
        // Encode normally.
        let (archive, _) = codec.encode_window(warm, measure).unwrap();

        // Re-compute the expected token sequence using just the BPE.
        // We can't reach the encoded tokens directly without exposing
        // internals, so we BPE-encode again here for the reference.
        let bpe_path: PathBuf = std::env::var_os(BPE_TABLE_ENV).unwrap().into();
        let bpe = Bpe::load(&bpe_path).unwrap();
        let expected_tokens = bpe.encode(measure);

        // Manually AC-decode to see what tokens we get.
        let measure_bytes_in_archive =
            u32::from_le_bytes([archive[0], archive[1], archive[2], archive[3]]) as usize;
        let n_tokens_in_archive =
            u32::from_le_bytes([archive[4], archive[5], archive[6], archive[7]]) as usize;
        let ac_bytes = &archive[8..];

        eprintln!(
            "header: measure_bytes={measure_bytes_in_archive}  n_tokens={n_tokens_in_archive}  expected_n_tokens={}",
            expected_tokens.len()
        );
        assert_eq!(measure_bytes_in_archive, measure.len());
        assert_eq!(
            n_tokens_in_archive,
            expected_tokens.len(),
            "n_tokens in archive doesn't match BPE encode"
        );

        // Now drive the decode manually so we can compare.
        let weights_path: PathBuf = std::env::var_os(WEIGHTS_ENV).unwrap().into();
        let mut arm = MoeArm::load(&weights_path).unwrap();
        let vocab = arm.cfg().vocab_size;
        // (warm is empty)
        let mut br = BitReader::new(ac_bytes);
        let mut dec = AcDecoder::new(&mut br);
        let mut cdf = vec![0_u32; vocab + 1];
        let mut decoded_tokens = Vec::with_capacity(n_tokens_in_archive);
        for i in 0..n_tokens_in_archive {
            arm.predict_cdf(&mut cdf);
            let sym = dec.decode(&cdf).unwrap();
            let tok = u32::try_from(sym).expect("symbol fits u32");
            if i < expected_tokens.len() && tok != expected_tokens[i] {
                eprintln!(
                    "TOKEN MISMATCH at position {i}: decoded={tok} expected={}",
                    expected_tokens[i]
                );
                eprintln!(
                    "  last 4 expected: {:?}",
                    &expected_tokens[i.saturating_sub(4)..i]
                );
                eprintln!(
                    "  last 4 decoded:  {:?}",
                    &decoded_tokens[decoded_tokens.len().saturating_sub(4)..]
                );
                panic!("token-level mismatch at position {i}");
            }
            decoded_tokens.push(tok);
            arm.feed_token(tok);
        }
        eprintln!("all {n_tokens_in_archive} tokens round-trip OK");
    }

    /// Full codec round-trip on a substantial 1 MB enwik9 sample —
    /// the same scale as the `compress` CLI exercises. Catches any
    /// state-leak between `encode_window` and `decode_window` calls
    /// on the same codec instance.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn moe_tok_codec_roundtrips_1mb() {
        if std::env::var_os(WEIGHTS_ENV).is_none() || std::env::var_os(BPE_TABLE_ENV).is_none() {
            eprintln!("env unset — skipping");
            return;
        }
        let corpus_path = "/Users/creynolds/Programming/lzr/assets/enwik9";
        if !std::path::Path::new(corpus_path).exists() {
            eprintln!("corpus missing — skipping");
            return;
        }
        let mut data = std::fs::read(corpus_path).unwrap();
        data.truncate(1_000_000);

        let codec = MoeTokCodec::new();
        let (archive, _) = codec.encode_window(b"", &data).unwrap();
        eprintln!(
            "encoded {} bytes into {}-byte archive",
            data.len(),
            archive.len()
        );
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded.len(), data.len(), "byte count mismatch");
        assert_eq!(decoded, data, "byte content mismatch");
        eprintln!("1MB round-trip OK");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn moe_tok_codec_roundtrips() {
        if std::env::var_os(WEIGHTS_ENV).is_none() || std::env::var_os(BPE_TABLE_ENV).is_none() {
            eprintln!("{WEIGHTS_ENV} or {BPE_TABLE_ENV} unset — skipping");
            return;
        }
        let corpus_path = "/Users/creynolds/Programming/lzr/assets/enwik8";
        if !std::path::Path::new(corpus_path).exists() {
            eprintln!("corpus missing — skipping");
            return;
        }
        let corpus = std::fs::read(corpus_path).unwrap();
        let warm = &corpus[0..512];
        let measure = &corpus[512..1024];

        let codec = MoeTokCodec::new();
        let (archive, decomp) = codec.encode_window(warm, measure).unwrap();
        let decoded = codec.decode_window(warm, &archive).unwrap();
        assert_eq!(decoded, measure, "moe-tok codec must roundtrip");
        assert_eq!(decomp.total(), 8 * archive.len() as u64);

        let bpb = 8.0 * archive.len() as f64 / measure.len() as f64;
        eprintln!(
            "moe-tok 512-byte window: archive={} bpb={:.3}",
            archive.len(),
            bpb
        );
    }
}
