//! Multi-order context-mixing byte codec — the v7 clean base.
//!
//! A deliberately small, deterministic foundation: an online ensemble of
//! order-0..=`MAX_ORDER` adaptive byte models, blended by a log-loss-weighted
//! mixer, driving the shared arithmetic coder. No neural net, no LZ/PPM/match
//! arms, no shipped weights — encoder and decoder rebuild byte-identical model
//! state by replaying the same stream, so `L(D) = 0`.
//!
//! Determinism: every model lookup is keyed on the full packed context (a
//! `u64`), so `HashMap`'s per-process random seed is irrelevant to results
//! (it only moves buckets, never changes a lookup's value). The CDF is built
//! from each context's symbol list in first-seen order — identical on both
//! sides — and all float work runs the same code path in both directions, so
//! the round-trip is bit-exact.
//!
//! Archive layout: a 64-bit little-end-first length prefix (so the decoder
//! knows how many symbols to pull), then the AC bitstream.
//!
//! This is intentionally a base to extend — a frequency-ranked sub-128
//! alphabet, compact count storage, logistic mixing, and additional context
//! models (match, word, sparse) are follow-on steps, not part of this floor.

use std::collections::HashMap;

use anyhow::Result;

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bits::{BitReader, BitWriter};
use crate::codec::{Codec, Decomposition};

/// Highest context order, in bytes. Models orders `0..=MAX_ORDER`.
const MAX_ORDER: usize = 6;
/// Number of order models.
const N_ORDERS: usize = MAX_ORDER + 1;
/// Symbol alphabet — raw bytes.
const ALPHABET: usize = 256;
/// Dirichlet smoothing prior per symbol. Small values favor sparse high-order
/// contexts; the enwik8 adaptive sweep bottomed out near `alpha = 0.02`.
const ALPHA: f64 = 0.02;
/// Per-order log-likelihood scores decay by this each byte, so the mixer
/// tracks which orders predict well *locally* rather than over all history.
const MIX_DECAY: f64 = 0.97;

/// Sparse next-symbol distribution for one context: running total plus the
/// per-symbol counts in first-seen order (deterministic on both sides).
#[derive(Debug, Default)]
struct Stats {
    tot: u32,
    syms: Vec<(u8, u32)>,
}

/// Online multi-order count model with a log-loss-weighted mixer.
#[derive(Debug)]
struct Model {
    /// Order-0 is dense — one context, effectively all symbols populated.
    o0: [u32; ALPHABET],
    o0_tot: u32,
    /// Orders `1..=MAX_ORDER`, sparse. `maps[k - 1]` is keyed on the packed
    /// low `8 * k` bits of the rolling context.
    maps: Vec<HashMap<u64, Stats>>,
    /// Decayed per-order log-likelihood; mixer weights are `softmax(score)`.
    score: [f64; N_ORDERS],
    /// Rolling context — the last up-to-8 bytes, most recent in the low byte.
    ctx: u64,
}

impl Model {
    fn new() -> Self {
        Self {
            o0: [0; ALPHABET],
            o0_tot: 0,
            maps: (0..MAX_ORDER).map(|_| HashMap::new()).collect(),
            score: [0.0; N_ORDERS],
            ctx: 0,
        }
    }

    /// Packed key for order `k` (`1..=MAX_ORDER`): the last `k` bytes.
    const fn key(&self, k: usize) -> u64 {
        if k >= 8 {
            self.ctx
        } else {
            self.ctx & ((1u64 << (8 * k)) - 1)
        }
    }

    /// Current mixer weights — `softmax` of the decayed per-order scores.
    fn weights(&self) -> [f64; N_ORDERS] {
        let max = self.score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut w = [0.0; N_ORDERS];
        let mut sum = 0.0;
        for (wk, &sk) in w.iter_mut().zip(self.score.iter()) {
            *wk = (sk - max).exp();
            sum += *wk;
        }
        for wk in &mut w {
            *wk /= sum;
        }
        w
    }

    /// Mixed next-symbol distribution over the alphabet (sums to 1).
    #[allow(
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        clippy::needless_range_loop,
        clippy::similar_names
    )]
    fn predict(&self) -> [f64; ALPHABET] {
        let af = ALPHABET as f64 * ALPHA;
        let w = self.weights();

        // Per-order count multiplier `w_k / (tot_k + af)` and the shared
        // smoothing floor `sum_k w_k * alpha / (tot_k + af)`.
        let mut mult = [0.0; N_ORDERS];
        let mut floor = 0.0;
        let inv0 = 1.0 / (f64::from(self.o0_tot) + af);
        mult[0] = w[0] * inv0;
        floor += w[0] * ALPHA * inv0;
        for k in 1..=MAX_ORDER {
            if let Some(st) = self.maps[k - 1].get(&self.key(k)) {
                let invk = 1.0 / (f64::from(st.tot) + af);
                mult[k] = w[k] * invk;
                floor += w[k] * ALPHA * invk;
            } else {
                // Novel context: tot = 0, so the floor term is w_k / ALPHABET.
                floor += w[k] / ALPHABET as f64;
            }
        }

        let mut acc = [floor; ALPHABET];
        for (a, &c) in acc.iter_mut().zip(self.o0.iter()) {
            *a += f64::from(c) * mult[0];
        }
        for k in 1..=MAX_ORDER {
            if mult[k] > 0.0 {
                if let Some(st) = self.maps[k - 1].get(&self.key(k)) {
                    for &(sym, cnt) in &st.syms {
                        acc[sym as usize] += f64::from(cnt) * mult[k];
                    }
                }
            }
        }
        acc
    }

    /// After symbol `s` is coded: credit each order by how well it predicted
    /// `s` (decayed log-loss), bump counts, and advance the context.
    #[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
    fn update(&mut self, s: u8) {
        let af = ALPHABET as f64 * ALPHA;
        let si = s as usize;

        let p0 = (f64::from(self.o0[si]) + ALPHA) / (f64::from(self.o0_tot) + af);
        self.score[0] = MIX_DECAY * self.score[0] + p0.ln();
        self.o0[si] += 1;
        self.o0_tot += 1;

        for k in 1..=MAX_ORDER {
            let key = self.key(k);
            let st = self.maps[k - 1].entry(key).or_default();
            let mut cnt = 0u32;
            let mut at = None;
            for (i, &(sym, c)) in st.syms.iter().enumerate() {
                if sym == s {
                    cnt = c;
                    at = Some(i);
                    break;
                }
            }
            let pk = (f64::from(cnt) + ALPHA) / (f64::from(st.tot) + af);
            self.score[k] = MIX_DECAY * self.score[k] + pk.ln();
            if let Some(i) = at {
                st.syms[i].1 += 1;
            } else {
                st.syms.push((s, 1));
            }
            st.tot += 1;
        }

        self.ctx = (self.ctx << 8) | u64::from(s);
    }
}

/// Quantize a probability distribution into an AC-ready CDF: length
/// `ALPHABET + 1`, `cdf[0] = 0`, `cdf[ALPHABET] = TOTAL`, every gap `>= 1`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn build_cdf(acc: &[f64; ALPHABET], cdf: &mut [u32]) {
    let scale = f64::from(TOTAL - ALPHABET as u32);
    let mut freq = [1u32; ALPHABET];
    let mut sum = ALPHABET as u32;
    let mut argmax = 0usize;
    let mut maxf = 0u32;
    for (s, (f, &a)) in freq.iter_mut().zip(acc.iter()).enumerate() {
        let extra = (a * scale) as u32;
        *f += extra;
        sum += extra;
        if *f > maxf {
            maxf = *f;
            argmax = s;
        }
    }
    // Reserved-mass rounding leaves `sum <= TOTAL`; give the slack to the mode.
    freq[argmax] += TOTAL - sum;
    cdf[0] = 0;
    for (i, &f) in freq.iter().enumerate() {
        cdf[i + 1] = cdf[i] + f;
    }
}

/// The v7 clean base codec: multi-order context mixing + AC, nothing else.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CmixCodec;

impl Codec for CmixCodec {
    fn name(&self) -> &'static str {
        "cmix"
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        let mut writer = BitWriter::new();
        writer.write_bits(measure.len() as u64, 64);
        let mut model = Model::new();
        for &b in warm {
            model.update(b);
        }
        let mut cdf = vec![0u32; ALPHABET + 1];
        {
            let mut enc = AcEncoder::new(&mut writer);
            for &b in measure {
                let acc = model.predict();
                build_cdf(&acc, &mut cdf);
                enc.encode(&cdf, b as usize);
                model.update(b);
            }
            enc.finish();
        }
        let (buf, _pad) = writer.finish();
        let mut decomp = Decomposition::new();
        decomp.add("cmix", 8 * buf.len() as u64);
        Ok((buf, decomp))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        let mut reader = BitReader::new(archive);
        let n = usize::try_from(reader.read_bits(64)?).expect("length prefix fits usize");
        let mut model = Model::new();
        for &b in warm {
            model.update(b);
        }
        let mut cdf = vec![0u32; ALPHABET + 1];
        let mut out = Vec::with_capacity(n);
        let mut dec = AcDecoder::new(&mut reader);
        for _ in 0..n {
            let acc = model.predict();
            build_cdf(&acc, &mut cdf);
            let s = dec.decode(&cdf)?;
            let b = u8::try_from(s).expect("AC symbol < ALPHABET fits u8");
            out.push(b);
            model.update(b);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmix_roundtrips_text() {
        let measure = b"the quick brown fox jumps over the lazy dog. \
                        the quick brown fox jumps over the lazy dog again."
            .repeat(40);
        let codec = CmixCodec;
        let (archive, decomp) = codec.encode_window(b"", &measure).unwrap();
        assert_eq!(decomp.total(), 8 * archive.len() as u64);
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
        // Repetitive text must beat the 8 bpb raw floor handily.
        assert!(
            archive.len() < measure.len(),
            "archive {} not smaller than input {}",
            archive.len(),
            measure.len()
        );
    }

    #[test]
    fn cmix_roundtrips_with_warm_prefix() {
        let warm = b"warm priming context that both sides share verbatim. ".repeat(8);
        let measure = b"warm priming context that both sides share, then new tail.".to_vec();
        let codec = CmixCodec;
        let (archive, _) = codec.encode_window(&warm, &measure).unwrap();
        let decoded = codec.decode_window(&warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn cmix_roundtrips_all_byte_values() {
        let measure: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let codec = CmixCodec;
        let (archive, _) = codec.encode_window(b"", &measure).unwrap();
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    #[test]
    fn cmix_roundtrips_empty() {
        let codec = CmixCodec;
        let (archive, _) = codec.encode_window(b"", b"").unwrap();
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert!(decoded.is_empty());
    }
}
