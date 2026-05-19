//! Token-level v4 codec with LZ77 absorption layer (`moe-tok-lz`).
//!
//! Same `MoE` + BPE + AC backbone as `moe-tok`, plus a token-level LZ77
//! stage that catches long exact repeats that escape the `MoE`'s
//! attention context. For each output position we emit a 2-symbol
//! adaptive AC flag (literal vs LZ); on LZ we follow with raw-bit
//! uniform-CDF offset/length pairs, and on literal we follow with the
//! existing `MoE`-CDF token-id encoding.
//!
//! Decoder mirrors encode exactly: it maintains its own LZ window and
//! literal/match flag counters, so encoder and decoder stay in lockstep.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result, bail};

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bits::{BitReader, BitWriter};
use crate::bpe::Bpe;
use crate::codec::{Codec, Decomposition};
use crate::lz77::{DEFAULT_MIN_OFFSET, Lz77, MAX_MATCH, MIN_MATCH, WINDOW};
use crate::moe_arm::MoeArm;

const WEIGHTS_ENV: &str = "LZR_MOE_WEIGHTS";
const BPE_TABLE_ENV: &str = "LZR_BPE_TABLE";

#[derive(Debug, Default)]
pub(crate) struct MoeTokLzCodec {
    arms: Mutex<Option<TokArms>>,
}

#[derive(Debug)]
struct TokArms {
    moe: MoeArm,
    bpe: Bpe,
}

impl MoeTokLzCodec {
    pub(crate) const fn new() -> Self {
        Self {
            arms: Mutex::new(None),
        }
    }

    #[allow(clippy::significant_drop_tightening)]
    fn with_arms<R>(&self, op: impl FnOnce(&mut TokArms) -> Result<R>) -> Result<R> {
        let mut guard = self.arms.lock().expect("moe-tok-lz arms mutex poisoned");
        if guard.is_none() {
            let weights_path: PathBuf = std::env::var_os(WEIGHTS_ENV)
                .map(PathBuf::from)
                .with_context(|| format!("v4 moe-tok-lz codec requires {WEIGHTS_ENV} env var"))?;
            let bpe_path: PathBuf = std::env::var_os(BPE_TABLE_ENV)
                .map(PathBuf::from)
                .with_context(|| format!("v4 moe-tok-lz codec requires {BPE_TABLE_ENV} env var"))?;
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

/// Krichevsky-Trofimov-ish 2-symbol CDF for the LZ/literal flag.
/// Starts at 50/50 (both counters 0); updates per observation on both
/// encoder and decoder so they stay in lockstep.
#[allow(clippy::cast_possible_truncation)]
fn flag_cdf(n_lit: u64, n_lz: u64) -> [u32; 3] {
    let lit_q = 2 * n_lit + 1;
    let denom = 2 * (n_lit + n_lz) + 2;
    let mass = ((u128::from(lit_q) * u128::from(TOTAL)) / u128::from(denom)) as u32;
    let lit_mass = mass.clamp(1, TOTAL - 1);
    [0, lit_mass, TOTAL]
}

/// Uniform CDF over `n` symbols. `n` must divide TOTAL or close to it;
/// the floor on per-symbol mass at our 24-bit precision is 1 unit, and
/// for the two callers here (`WINDOW=16384`, `256`) every symbol gets
/// thousands of units of mass.
#[allow(clippy::cast_possible_truncation)]
fn uniform_cdf(n: usize) -> Vec<u32> {
    debug_assert!(n >= 2);
    let mut cdf = vec![0_u32; n + 1];
    let total = u64::from(TOTAL);
    let n_u64 = n as u64;
    for (i, slot) in cdf.iter_mut().enumerate().take(n) {
        *slot = ((i as u64 * total) / n_u64) as u32;
    }
    cdf[n] = TOTAL;
    cdf
}

impl Codec for MoeTokLzCodec {
    fn name(&self) -> &'static str {
        "moe-tok-lz"
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

        let (archive, ac_breakdown, total_bits) = self.with_arms(|arms| {
            // Prime: BPE the warm bytes, feed both MoE and LZ window.
            let warm_tokens = arms.bpe.encode(warm);
            let mut lz = Lz77::new(DEFAULT_MIN_OFFSET);
            for &tok in &warm_tokens {
                arms.moe.feed_token(tok);
                lz.push(tok);
            }

            let measure_tokens = arms.bpe.encode(measure);
            let n_tokens =
                u32::try_from(measure_tokens.len()).context("token count exceeds u32 capacity")?;
            let vocab = arms.moe.cfg().vocab_size;

            let offset_cdf = uniform_cdf(WINDOW);
            let length_cdf = uniform_cdf(MAX_MATCH - MIN_MATCH + 1);

            let mut bw = BitWriter::new();
            let mut tok_cdf = vec![0_u32; vocab + 1];
            let mut n_lit: u64 = 0;
            let mut n_lz: u64 = 0;
            let (flag_bits, lz_payload_bits, tok_bits);
            {
                let mut enc = AcEncoder::new(&mut bw);

                let mut i = 0_usize;
                let mut bits_at = enc.bits_written();
                let mut tally_flag = 0_u64;
                let mut tally_lz = 0_u64;
                let mut tally_tok = 0_u64;

                while i < measure_tokens.len() {
                    let lookahead_end = (i + MAX_MATCH).min(measure_tokens.len());
                    let lookahead = &measure_tokens[i..lookahead_end];

                    let cdf = flag_cdf(n_lit, n_lz);
                    if let Some((back, len)) = lz.longest_match(lookahead) {
                        enc.encode(&cdf, 1);
                        let after_flag = enc.bits_written();
                        tally_flag += after_flag - bits_at;
                        bits_at = after_flag;

                        enc.encode(&offset_cdf, back - 1);
                        enc.encode(&length_cdf, len - MIN_MATCH);
                        let after_payload = enc.bits_written();
                        tally_lz += after_payload - bits_at;
                        bits_at = after_payload;

                        for &t in &lookahead[..len] {
                            arms.moe.feed_token(t);
                            lz.push(t);
                        }
                        n_lz += 1;
                        i += len;
                    } else {
                        enc.encode(&cdf, 0);
                        let after_flag = enc.bits_written();
                        tally_flag += after_flag - bits_at;
                        bits_at = after_flag;

                        let t = measure_tokens[i];
                        arms.moe.predict_cdf(&mut tok_cdf);
                        enc.encode(&tok_cdf, t as usize);
                        let after_tok = enc.bits_written();
                        tally_tok += after_tok - bits_at;
                        bits_at = after_tok;

                        arms.moe.feed_token(t);
                        lz.push(t);
                        n_lit += 1;
                        i += 1;
                    }
                }
                flag_bits = tally_flag;
                lz_payload_bits = tally_lz;
                tok_bits = tally_tok;
                enc.finish();
            }
            let (ac_bytes, _trailing) = bw.finish();

            let mut archive = Vec::with_capacity(8 + ac_bytes.len());
            archive.extend_from_slice(&measure_len.to_le_bytes());
            archive.extend_from_slice(&n_tokens.to_le_bytes());
            archive.extend_from_slice(&ac_bytes);
            let total_bits = 8 * archive.len() as u64;
            Ok((archive, (flag_bits, lz_payload_bits, tok_bits), total_bits))
        })?;

        let (flag_bits, lz_payload_bits, tok_bits) = ac_breakdown;
        let mut decomp = Decomposition::new();
        decomp.add("moe_tok_lz_flag", flag_bits);
        decomp.add("moe_tok_lz_lzpayload", lz_payload_bits);
        decomp.add("moe_tok_lz_tok", tok_bits);
        let ac_bits = flag_bits + lz_payload_bits + tok_bits;
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
            let mut lz = Lz77::new(DEFAULT_MIN_OFFSET);
            for &tok in &warm_tokens {
                arms.moe.feed_token(tok);
                lz.push(tok);
            }

            let vocab = arms.moe.cfg().vocab_size;
            let offset_cdf = uniform_cdf(WINDOW);
            let length_cdf = uniform_cdf(MAX_MATCH - MIN_MATCH + 1);

            let mut br = BitReader::new(ac_bytes);
            let mut dec = AcDecoder::new(&mut br);
            let mut tok_cdf = vec![0_u32; vocab + 1];
            let mut out = Vec::with_capacity(n_tokens);
            let mut n_lit: u64 = 0;
            let mut n_lz: u64 = 0;

            while out.len() < n_tokens {
                let cdf = flag_cdf(n_lit, n_lz);
                let flag = dec.decode(&cdf)?;
                if flag == 1 {
                    let off_sym = dec.decode(&offset_cdf)?;
                    let len_sym = dec.decode(&length_cdf)?;
                    let back = off_sym + 1;
                    let len = len_sym + MIN_MATCH;
                    if back > lz.len() {
                        bail!("LZ offset {back} exceeds window length {}", lz.len());
                    }
                    if len > back {
                        bail!("LZ length {len} > offset {back} (overlapping copy not supported)");
                    }
                    let cur = lz.len();
                    let start = cur - back;
                    // Copy by re-pushing each token through feed_token + lz.push.
                    // We have to materialize the tokens first since lz.push mutates.
                    let copy: Vec<u32> = {
                        let mut v = Vec::with_capacity(len);
                        for k in 0..len {
                            // Safe because we asserted back >= len; reading from
                            // [start, start+len) doesn't overlap [cur, cur+len).
                            // We read via a method since lz.tokens is private.
                            v.push(lz_read(&lz, start + k));
                        }
                        v
                    };
                    for &t in &copy {
                        out.push(t);
                        arms.moe.feed_token(t);
                        lz.push(t);
                        if out.len() >= n_tokens {
                            break;
                        }
                    }
                    n_lz += 1;
                } else {
                    arms.moe.predict_cdf(&mut tok_cdf);
                    let sym = dec.decode(&tok_cdf)?;
                    let t = u32::try_from(sym)
                        .with_context(|| format!("AC produced non-vocab symbol {sym}"))?;
                    out.push(t);
                    arms.moe.feed_token(t);
                    lz.push(t);
                    n_lit += 1;
                }
            }

            let bytes = arms.bpe.decode(&out);
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

/// Read a token from the LZ window at absolute position `pos`.
/// Defined here so we don't have to make `Lz77::tokens` public for the
/// decoder's copy loop.
fn lz_read(lz: &Lz77, pos: usize) -> u32 {
    lz.token_at(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn moe_tok_lz_codec_roundtrips_1mb() {
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

        let codec = MoeTokLzCodec::new();
        let (archive, decomp) = codec.encode_window(b"", &data).unwrap();
        eprintln!(
            "encoded {} bytes into {}-byte archive  ({:.4} bpb)",
            data.len(),
            archive.len(),
            8.0 * archive.len() as f64 / data.len() as f64
        );
        for (k, v) in &decomp.by_component {
            eprintln!(
                "  {k}: {v} bits  ({:.4} bpb)",
                *v as f64 / data.len() as f64
            );
        }
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded.len(), data.len(), "byte count mismatch");
        assert_eq!(decoded, data, "byte content mismatch");
        eprintln!("1MB round-trip OK");
    }

    #[test]
    fn flag_cdf_starts_50_50() {
        let c = flag_cdf(0, 0);
        assert_eq!(c[0], 0);
        assert_eq!(c[2], TOTAL);
        // 0.5 mass each, within rounding tolerance
        assert!(
            c[1] >= TOTAL / 2 - 4 && c[1] <= TOTAL / 2 + 4,
            "got {}",
            c[1]
        );
    }

    #[test]
    fn flag_cdf_skews_toward_majority() {
        let c = flag_cdf(99, 1);
        // P(lit) ≈ 199/202 ≈ 0.985
        let p = f64::from(c[1]) / f64::from(TOTAL);
        assert!(p > 0.97, "got p={p}");
    }
}
