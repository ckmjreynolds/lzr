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

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bits::{BitReader, BitWriter};
use crate::bpe::Bpe;
use crate::codec::{Codec, Decomposition};
use crate::match_model::{self, MatchModel};
use crate::moe_arm::MoeArm;
use crate::struct_mask;

const WEIGHTS_ENV: &str = "LZR_MOE_WEIGHTS";
const BPE_TABLE_ENV: &str = "LZR_BPE_TABLE";
/// Online mixer learning rate for the match arm's per-bucket logit
/// boost, and the cap on that boost (in logits). η swept on 1 MB
/// enwik8 (0.05→−0.0081, 0.2→−0.0089, saturating by 0.4); 0.2 warms the
/// per-bucket weights faster and tracks non-stationary content. `β_max`
/// is non-binding (weights self-calibrate below it). Both overridable
/// via `LZR_MATCH_ETA` / `LZR_MATCH_BETAMAX`.
const MATCH_ETA: f32 = 0.2;
const MATCH_BETA_MAX: f32 = 12.0;
/// Measure tokens per batched-encode window. Bounds the precomputed
/// logit slab to `ENCODE_WINDOW * vocab` f32 (~0.5 GB at vocab 16384)
/// instead of `n_tokens * vocab`, which OOMs on large corpora.
const ENCODE_WINDOW: usize = 8192;
/// When set to a writable path, the encode loop dumps one CSV row per
/// emitted token so we can analyze where the predictor loses bits.
/// Diagnostic only — does not affect bits emitted to the archive.
const ENTROPY_DUMP_ENV: &str = "LZR_ENTROPY_DUMP";

/// Soft mask penalty as additive logit bias = ln(0.01). Applied to
/// tokens flagged "never seen after this byte class" in the embedded
/// `struct_mask::MASK`. Fit on 1 MB / 10 MB / 100 MB sweeps in Phase
/// 49; the gain saturates at penalty ≤ 0.01.
const MASK_PENALTY_LOG: f32 = -4.605_17;

#[inline]
fn fill_bias(bias: &mut [f32], class: usize) {
    for (i, slot) in bias.iter_mut().enumerate() {
        let allowed = struct_mask::mask_bit(class, u32::try_from(i).expect("vocab fits u32"));
        *slot = if allowed { 0.0 } else { MASK_PENALTY_LOG };
    }
}

/// Apply the PPM match arm's follower distribution as a per-token logit
/// boost: `bias[f] += beta[Lb] * w_f` for each follower `f`. Fills
/// `followers` with the `(token, weight)` pairs and returns the bucket
/// `Lb`, so the mixer update can reuse them. Shared by encode and
/// decode so both sides apply the identical boost.
#[inline]
fn apply_match_bias(
    matcher: Option<&MatchModel>,
    beta: &[f32],
    bias: &mut [f32],
    followers: &mut Vec<(u32, f32)>,
) -> Option<usize> {
    let l = matcher?.predict(followers)?;
    let lb = l.min(match_model::MAX_L);
    let b = beta[lb];
    for &(f, w) in followers.iter() {
        let fi = f as usize;
        bias[fi] = b.mul_add(w, bias[fi]);
    }
    Some(lb)
}

/// Online log-loss update of the match mixer weight for the bucket that
/// fired. For the multi-token boost `bias[f] += beta * w_f`, the
/// gradient of `−log p'(tok)` w.r.t. `beta` is `Σ_f w_f p'(f) − w_tok`;
/// the descent step `beta += η (w_tok − Σ_f w_f p'(f))` is computed from
/// the coding CDF and the true token — identical on both sides, so
/// encode and decode stay in lockstep.
#[inline]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn update_match_mixer(
    lb: Option<usize>,
    followers: &[(u32, f32)],
    beta: &mut [f32],
    cdf: &[u32],
    tok: u32,
    eta: f32,
    beta_max: f32,
) {
    if let Some(lb) = lb {
        let total = f64::from(TOTAL);
        let mut sum_wp = 0.0_f64;
        let mut w_tok = 0.0_f64;
        for &(f, w) in followers {
            let fi = f as usize;
            let pf = f64::from(cdf[fi + 1] - cdf[fi]) / total;
            sum_wp = f64::from(w).mul_add(pf, sum_wp);
            if f == tok {
                w_tok = f64::from(w);
            }
        }
        let next = f64::from(eta).mul_add(w_tok - sum_wp, f64::from(beta[lb]));
        beta[lb] = (next as f32).clamp(0.0, beta_max);
    }
}

/// Parse an `f32` tuning override from the environment (for sweeps);
/// falls back to the shipped default.
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Diagnostic stats for one literal token: ideal bits paid, `MoE`
/// distribution entropy, top-1 probability, rank of the actual token.
struct LitStats {
    ideal_bits: f64,
    entropy_bits: f64,
    top1_prob: f64,
    rank: u32,
}

#[allow(clippy::cast_precision_loss)]
fn lit_stats(cdf: &[u32], t: u32) -> LitStats {
    let vocab = cdf.len() - 1;
    let total_f = f64::from(TOTAL);
    let span_t = u64::from(cdf[t as usize + 1] - cdf[t as usize]);
    let p_t = span_t as f64 / total_f;
    let ideal_bits = if p_t > 0.0 { -p_t.log2() } else { 0.0 };

    let mut entropy = 0.0_f64;
    let mut top1 = 0.0_f64;
    let mut rank = 0_u32;
    for i in 0..vocab {
        let span = u64::from(cdf[i + 1] - cdf[i]);
        let p = span as f64 / total_f;
        if p > 0.0 {
            entropy = p.mul_add(-p.log2(), entropy);
        }
        if p > top1 {
            top1 = p;
        }
        if p > p_t {
            rank += 1;
        }
    }
    LitStats {
        ideal_bits,
        entropy_bits: entropy,
        top1_prob: top1,
        rank,
    }
}

fn open_entropy_dump() -> Result<Option<BufWriter<File>>> {
    let Some(path) = std::env::var_os(ENTROPY_DUMP_ENV) else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let f = File::create(&path)
        .with_context(|| format!("creating entropy dump at {}", path.display()))?;
    let mut w = BufWriter::new(f);
    writeln!(
        w,
        "src_pos,tok_id,tok_bytes,ideal_bits,entropy,top1_prob,rank"
    )?;
    Ok(Some(w))
}

#[derive(Debug, Default)]
pub(crate) struct MoeTokCodec {
    arms: Mutex<Option<TokArms>>,
    /// When set, mix the token-level longest-match arm into the `MoE`
    /// distribution via an online per-length-bucket logit boost.
    match_arm: bool,
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
            match_arm: false,
        }
    }

    /// Variant that mixes the token longest-match arm into the `MoE`
    /// distribution (Proposal 1, slice 1).
    pub(crate) const fn new_with_match() -> Self {
        Self {
            arms: Mutex::new(None),
            match_arm: true,
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
        if self.match_arm {
            "moe-tok-match"
        } else {
            "moe-tok"
        }
    }

    #[allow(clippy::too_many_lines)]
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
            let timing = std::env::var("LZR_TIMING").is_ok();
            let t_bpe_start = std::time::Instant::now();
            // BPE-encode warm + measure up-front.
            let warm_tokens = arms.bpe.encode(warm);
            // BPE-encode measure once; we AC-encode the resulting
            // token sequence in order.
            let measure_tokens = arms.bpe.encode(measure);
            if timing {
                eprintln!(
                    "[timing] BPE encode: {:.2}s",
                    t_bpe_start.elapsed().as_secs_f64()
                );
            }

            let t_forward_start = std::time::Instant::now();
            // Phase 50C Day-4 Phase-4: if the int batched-encode path
            // is available, precompute every prediction in one
            // (chunked) batched forward instead of feeding warm
            // tokens one-by-one through the per-step path. The codec
            // loop below then becomes pure CDF/AC work with no
            // per-token forward dispatch. Bit-identical archive.
            // Batched encode now runs windowed (see the AC loop below):
            // logits are precomputed one bounded window at a time so the
            // slab stays small on large corpora. Only the per-step path
            // needs warm priming here.
            let batched = arms.moe.can_batched_encode();
            if !batched {
                for &tok in &warm_tokens {
                    arms.moe.feed_token(tok);
                }
                if timing {
                    eprintln!(
                        "[timing] per-step warm feed: {:.2}s",
                        t_forward_start.elapsed().as_secs_f64()
                    );
                }
            }
            let n_tokens =
                u32::try_from(measure_tokens.len()).context("token count exceeds u32 capacity")?;
            let vocab = arms.moe.cfg().vocab_size;

            let mut bias = vec![0_f32; vocab];

            let mut bw = BitWriter::new();
            let mut cdf = vec![0_u32; vocab + 1];
            let mut dump = open_entropy_dump()?;
            let bits_before;
            let bits_after;
            {
                let mut enc = AcEncoder::new(&mut bw);
                bits_before = enc.bits_written();
                let mut src_pos: usize = 0;
                let mut last_byte = if warm.is_empty() {
                    b'\n'
                } else {
                    warm[warm.len() - 1]
                };
                let mut matcher = self.match_arm.then(|| {
                    let mut m = MatchModel::new();
                    for &t in &warm_tokens {
                        m.push(t);
                    }
                    m
                });
                let mut beta = [0_f32; match_model::MAX_L + 1];
                let match_eta = env_f32("LZR_MATCH_ETA", MATCH_ETA);
                let match_beta_max = env_f32("LZR_MATCH_BETAMAX", MATCH_BETA_MAX);
                let mut followers: Vec<(u32, f32)> = Vec::new();
                let t_ac_start = std::time::Instant::now();
                // Process measure tokens in bounded windows: in batched
                // mode, precompute one window's logits at a time so the
                // slab is `ENCODE_WINDOW * vocab`, not `n_tokens * vocab`.
                // Per-step mode uses a single window (no precompute).
                let mut wlo = 0usize;
                while wlo < measure_tokens.len() {
                    let whi = if batched {
                        (wlo + ENCODE_WINDOW).min(measure_tokens.len())
                    } else {
                        measure_tokens.len()
                    };
                    if batched {
                        arms.moe.precompute_for_encode_range(
                            &warm_tokens,
                            &measure_tokens,
                            wlo,
                            whi,
                        );
                    }
                    for &tok in &measure_tokens[wlo..whi] {
                        let class = struct_mask::byte_class(last_byte);
                        fill_bias(&mut bias, class);
                        let lb =
                            apply_match_bias(matcher.as_ref(), &beta, &mut bias, &mut followers);
                        arms.moe.predict_cdf_with_bias(&mut cdf, &bias);
                        let tok_bytes_view = arms.bpe.token_bytes(tok);
                        let tok_bytes = tok_bytes_view.len();
                        if let Some(w) = dump.as_mut() {
                            let s = lit_stats(&cdf, tok);
                            writeln!(
                                w,
                                "{src_pos},{tok},{tok_bytes},{:.6},{:.6},{:.6},{}",
                                s.ideal_bits, s.entropy_bits, s.top1_prob, s.rank
                            )?;
                        }
                        if let Some(&b) = tok_bytes_view.last() {
                            last_byte = b;
                        }
                        src_pos += tok_bytes;
                        enc.encode(&cdf, tok as usize);
                        update_match_mixer(
                            lb,
                            &followers,
                            &mut beta,
                            &cdf,
                            tok,
                            match_eta,
                            match_beta_max,
                        );
                        arms.moe.feed_token(tok);
                        if let Some(m) = matcher.as_mut() {
                            m.push(tok);
                        }
                    }
                    wlo = whi;
                }
                bits_after = enc.bits_written();
                if timing {
                    eprintln!(
                        "[timing] AC encode loop ({} tokens): {:.2}s",
                        measure_tokens.len(),
                        t_ac_start.elapsed().as_secs_f64()
                    );
                }
                enc.finish();
            }
            if let Some(mut w) = dump {
                w.flush()?;
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
            let mut bias = vec![0_f32; vocab];

            let mut br = BitReader::new(ac_bytes);
            let mut dec = AcDecoder::new(&mut br);
            let mut cdf = vec![0_u32; vocab + 1];
            let mut tokens = Vec::with_capacity(n_tokens);
            let mut last_byte = if warm.is_empty() {
                b'\n'
            } else {
                warm[warm.len() - 1]
            };
            let mut matcher = self.match_arm.then(|| {
                let mut m = MatchModel::new();
                for &t in &warm_tokens {
                    m.push(t);
                }
                m
            });
            let mut beta = [0_f32; match_model::MAX_L + 1];
            let match_eta = env_f32("LZR_MATCH_ETA", MATCH_ETA);
            let match_beta_max = env_f32("LZR_MATCH_BETAMAX", MATCH_BETA_MAX);
            let mut followers: Vec<(u32, f32)> = Vec::new();
            for _ in 0..n_tokens {
                let class = struct_mask::byte_class(last_byte);
                fill_bias(&mut bias, class);
                let lb = apply_match_bias(matcher.as_ref(), &beta, &mut bias, &mut followers);
                arms.moe.predict_cdf_with_bias(&mut cdf, &bias);
                let sym = dec.decode(&cdf)?;
                let tok = u32::try_from(sym)
                    .with_context(|| format!("AC produced non-vocab symbol {sym}"))?;
                let tok_bytes_view = arms.bpe.token_bytes(tok);
                if let Some(&b) = tok_bytes_view.last() {
                    last_byte = b;
                }
                tokens.push(tok);
                update_match_mixer(
                    lb,
                    &followers,
                    &mut beta,
                    &cdf,
                    tok,
                    match_eta,
                    match_beta_max,
                );
                arms.moe.feed_token(tok);
                if let Some(m) = matcher.as_mut() {
                    m.push(tok);
                }
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
