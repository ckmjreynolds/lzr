//! Bitwise logistic context-mixing byte codec — the v7 logit-domain mixer.
//!
//! Where [`crate::cmix`] blends per-order count distributions *linearly*, this
//! codec mixes in the **logit (stretch) domain**, the PAQ/lpaq design. Each
//! byte is coded as eight binary decisions, MSB-first. For each bit, every
//! model emits a probability that the bit is 1; those are mapped through
//! `stretch(p) = ln(p / (1-p))`, combined by a learned weighted sum, and mapped
//! back with `squash`. The mixer weights are trained online by gradient descent
//! on coding loss (`w += lr * (y - p) * stretch(p_k)`).
//!
//! Why logit-domain: a linear blend of agreeing predictors cannot exceed the
//! most confident input — if two models both say 0.9, the blend stays ~0.9.
//! In the logit domain their evidence *adds*, so agreement sharpens the
//! prediction past either input. That sharpening is where the bits are.
//!
//! Models: orders `0..=MAX_ORDER` (byte-history count contexts) plus an
//! optional **match model** (long-range exact repeats). The match model is
//! what separates the `lmatch` codec from the pure-mixer `lmix` codec — both
//! share this module; a `use_match` flag selects whether the match input is
//! live. When off, the match mixer input is held at a neutral `0`, so `lmix`
//! is bit-identical to the no-match design.
//!
//! Per-bit context: within a byte we track a tree node — start at 1, and after
//! each coded bit `node = (node << 1) | bit`, so `node` walks `1..=255` over
//! the eight decisions. Each order model is keyed on `(context_k, node)`.
//!
//! Determinism / `L(D) = 0`: every lookup is keyed on content (packed
//! `(context, node)`, or a verified history match), so the `HashMap` seed is
//! irrelevant; all float work runs the same code path on both sides; encoder
//! and decoder replay the same stream and rebuild byte-identical state. No
//! weights are shipped.
//!
//! Archive layout: a 64-bit length prefix, then the AC bitstream — each of the
//! `8 * len` bits coded against a two-symbol CDF `[0, c0, TOTAL]`.

use std::collections::HashMap;

use anyhow::Result;

use crate::ac::{AcDecoder, AcEncoder, TOTAL};
use crate::bits::{BitReader, BitWriter};
use crate::codec::{Codec, Decomposition};

/// Highest context order, in bytes. Models orders `0..=MAX_ORDER`.
const MAX_ORDER: usize = 6;
/// Number of order models.
const N_ORDERS: usize = MAX_ORDER + 1;
/// Mixer inputs: one per order, plus the match model.
const N_INPUTS: usize = N_ORDERS + 1;
/// Index of the match model's mixer input.
const MATCH_IN: usize = N_ORDERS;
/// Mixer gradient-descent step size on coding loss. An enwik8 quick-panel
/// sweep bottomed out near 0.002 (0.05 → 2.059, 0.02 → 1.971, 0.004 → 1.936,
/// 0.002 → 1.933); below that the curve is flat.
const MIX_LR: f64 = 0.002;
/// Initial per-input mixer weight (before any online training).
const INIT_W: f64 = 0.3;
/// Floor on a bit-predictor's adaptive learning rate, so a well-observed
/// context still tracks local drift instead of freezing.
const RATE_FLOOR: f64 = 1.0 / 256.0;
/// Cap on a bit-predictor's observation count (caps the slowest rate).
const N_CAP: u32 = 255;
/// Clamp bounds on a probability before `stretch`, so the logit stays finite
/// (and the mixer never sees an infinite input → `NaN` weight).
const P_LO: f64 = 1e-6;
const P_HI: f64 = 1.0 - 1e-6;

/// Bytes of matching context required to seed a match. An enwik8 quick-panel
/// sweep favored short seeds — 8 → 1.839, 6 → 1.839, 5 → 1.831, 4 → 1.817,
/// 3 → 1.805, 2 → 1.807 — bottoming at 3 (earlier seeding catches repeats
/// sooner; the adaptive length-bucketed confidence map handles the higher
/// false-match rate a short seed brings).
const MIN_MATCH: usize = 3;
/// Log2 of the match hash-table size (entries map context-hash → last
/// position). 2^22 entries × 4 B = 16 MiB.
const MATCH_BITS: u32 = 22;
/// Match-length buckets for the adaptive confidence map (a longer verified
/// match predicts more reliably; the map learns how reliable each bucket is).
const MATCH_BUCKETS: usize = 64;
/// Empty slot sentinel in the match table. enwik9 (1e9) < `u32::MAX`, so a
/// real position never collides with it.
const MATCH_EMPTY: u32 = u32::MAX;

/// Logit of a probability: `ln(p / (1-p))`, clamped to keep it finite.
#[inline]
fn stretch(p: f64) -> f64 {
    let p = p.clamp(P_LO, P_HI);
    (p / (1.0 - p)).ln()
}

/// Inverse of [`stretch`]: the logistic squashing function.
#[inline]
fn squash(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One adaptive bit predictor: a probability of "next bit is 1" plus an
/// observation count that schedules the learning rate (fast while young,
/// floored once mature so it keeps tracking drift).
#[derive(Clone, Copy, Debug)]
struct BitModel {
    p: f64,
    n: u32,
}

impl Default for BitModel {
    fn default() -> Self {
        Self { p: 0.5, n: 0 }
    }
}

impl BitModel {
    /// Move the probability toward the observed bit and age the counter.
    #[inline]
    #[allow(clippy::suboptimal_flops)]
    fn update(&mut self, y: f64) {
        let rate = (1.0 / (f64::from(self.n) + 1.5)).max(RATE_FLOOR);
        self.p += rate * (y - self.p);
        if self.n < N_CAP {
            self.n += 1;
        }
    }
}

/// Online multi-order bit model with a logistic (logit-domain) mixer and an
/// optional long-range match model.
#[derive(Debug)]
struct Model {
    /// Order-0: one predictor per within-byte tree node (`1..=255`).
    o0: Vec<BitModel>,
    /// Orders `1..=MAX_ORDER`, sparse. `maps[k - 1]` is keyed on the packed
    /// `(context_k, node)` — see [`Model::key`].
    maps: Vec<HashMap<u64, BitModel>>,
    /// Mixer weights, trained online by gradient descent on coding loss.
    w: [f64; N_INPUTS],
    /// Rolling context — the last up-to-8 bytes, most recent in the low byte.
    ctx: u64,

    /// Whether the match model contributes (selects `lmatch` over `lmix`).
    use_match: bool,
    /// All bytes seen so far (warm + coded), indexed by the match pointer.
    hist: Vec<u8>,
    /// Match table: hash of the last `MIN_MATCH` bytes → the position that
    /// context last ended at. `MATCH_EMPTY` marks an unused slot.
    mtable: Vec<u32>,
    /// Adaptive confidence per match-length bucket: P(actual bit == predicted).
    mmap: Vec<BitModel>,
    /// Index in `hist` of the byte the match model predicts next (valid iff
    /// `mlen > 0`).
    mptr: usize,
    /// Current verified match length, in bytes.
    mlen: u32,
}

impl Model {
    fn new(use_match: bool) -> Self {
        let (hist, mtable, mmap) = if use_match {
            (
                Vec::new(),
                vec![MATCH_EMPTY; 1usize << MATCH_BITS],
                vec![BitModel::default(); MATCH_BUCKETS],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        Self {
            o0: vec![BitModel::default(); 256],
            maps: (0..MAX_ORDER).map(|_| HashMap::new()).collect(),
            w: [INIT_W; N_INPUTS],
            ctx: 0,
            use_match,
            hist,
            mtable,
            mmap,
            mptr: 0,
            mlen: 0,
        }
    }

    /// Low `8 * k` bits of the rolling context (order-`k` byte history).
    const fn ctx_k(&self, k: usize) -> u64 {
        if k >= 8 {
            self.ctx
        } else {
            self.ctx & ((1u64 << (8 * k)) - 1)
        }
    }

    /// Packed lookup key for order `k` at within-byte tree `node`: the order's
    /// byte history shifted up by 9 bits, with `node` (`< 512`) in the low
    /// bits. For `k <= 6` the history is `<= 48` bits, so the whole key fits a
    /// `u64` losslessly — no hashing, no collisions.
    #[inline]
    const fn key(&self, k: usize, node: usize) -> u64 {
        (self.ctx_k(k) << 9) | node as u64
    }

    /// The match model's prediction at `node`, or `None` if there is no active
    /// match or the match's predicted byte has already diverged from the bits
    /// coded so far in this byte. Returns `(length bucket, predicted bit)`.
    #[inline]
    fn match_pred(&self, node: usize) -> Option<(usize, u8)> {
        if !self.use_match || self.mlen == 0 {
            return None;
        }
        let pb = self.hist[self.mptr];
        let j = node.ilog2();
        // The bits coded so far (`node`) must be the top-`j` bits of the
        // predicted byte, else this candidate no longer applies this byte.
        let prefix = usize::from(pb) >> (8 - j);
        if node != (1usize << j) | prefix {
            return None;
        }
        let pred_bit = (pb >> (7 - j)) & 1;
        let bucket = (self.mlen as usize).min(MATCH_BUCKETS - 1);
        Some((bucket, pred_bit))
    }

    /// Predict P(next bit = 1) at `node`, filling `x` with each model's logit
    /// (a novel/inactive model contributes a neutral `stretch(0.5) = 0`).
    #[allow(clippy::needless_range_loop)]
    fn predict_bit(&self, node: usize, x: &mut [f64; N_INPUTS]) -> f64 {
        x[0] = stretch(self.o0[node].p);
        for k in 1..=MAX_ORDER {
            x[k] = self.maps[k - 1]
                .get(&self.key(k, node))
                .map_or(0.0, |bm| stretch(bm.p));
        }
        x[MATCH_IN] = match self.match_pred(node) {
            Some((bucket, pred_bit)) => {
                let pc = self.mmap[bucket].p;
                let p1 = if pred_bit == 1 { pc } else { 1.0 - pc };
                stretch(p1)
            }
            None => 0.0,
        };
        let s: f64 = self.w.iter().zip(x.iter()).map(|(wk, xk)| wk * xk).sum();
        squash(s)
    }

    /// After bit `y` is coded at `node`: step the mixer weights down the
    /// coding-loss gradient, then update each model's bit predictor.
    #[allow(clippy::suboptimal_flops)]
    fn update_bit(&mut self, node: usize, x: &[f64; N_INPUTS], p1: f64, y: u8) {
        let err = f64::from(y) - p1;
        for (wk, &xk) in self.w.iter_mut().zip(x.iter()) {
            *wk += MIX_LR * err * xk;
        }
        let yf = f64::from(y);
        self.o0[node].update(yf);
        for k in 1..=MAX_ORDER {
            let key = self.key(k, node);
            self.maps[k - 1].entry(key).or_default().update(yf);
        }
        if let Some((bucket, pred_bit)) = self.match_pred(node) {
            self.mmap[bucket].update(f64::from(u8::from(y == pred_bit)));
        }
    }

    /// Hash of the `MIN_MATCH` bytes ending at `pos` (caller guarantees
    /// `pos + 1 >= MIN_MATCH`).
    #[inline]
    fn match_hash_at(&self, pos: usize) -> usize {
        let mut h = 0x9E37_79B9_7F4A_7C15u64;
        for &byte in &self.hist[pos + 1 - MIN_MATCH..=pos] {
            h = h
                .wrapping_mul(0x0100_0000_01B3)
                .wrapping_add(u64::from(byte));
        }
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h >> (64 - MATCH_BITS)) as usize
    }

    /// True if the `MIN_MATCH` bytes ending at `ci` equal those ending at
    /// `pos` — a real context match, not a hash collision.
    #[inline]
    fn verify_match(&self, ci: usize, pos: usize) -> bool {
        (0..MIN_MATCH).all(|i| self.hist[ci - i] == self.hist[pos - i])
    }

    /// Fold byte `b` into the match state: extend or break the current match,
    /// append to history, and (if no match is active) seed a new one from the
    /// table, then record this position.
    #[allow(clippy::cast_possible_truncation)]
    fn advance_match(&mut self, b: u8) {
        if !self.use_match {
            return;
        }
        if self.mlen > 0 {
            if self.hist[self.mptr] == b {
                self.mptr += 1;
                self.mlen = self.mlen.saturating_add(1);
            } else {
                self.mlen = 0;
            }
        }
        self.hist.push(b);
        if self.hist.len() < MIN_MATCH {
            return;
        }
        let pos = self.hist.len() - 1;
        let h = self.match_hash_at(pos);
        if self.mlen == 0 {
            let cand = self.mtable[h];
            if cand != MATCH_EMPTY {
                let ci = cand as usize;
                if ci < pos && self.verify_match(ci, pos) {
                    self.mptr = ci + 1;
                    self.mlen = MIN_MATCH as u32;
                }
            }
        }
        self.mtable[h] = pos as u32;
    }

    /// Replay a byte through predict+update without coding it — used to prime
    /// model state from the `warm` prefix on both sides identically.
    fn learn_byte(&mut self, b: u8) {
        let mut node = 1usize;
        let mut x = [0.0; N_INPUTS];
        for i in (0..8).rev() {
            let y = (b >> i) & 1;
            let p1 = self.predict_bit(node, &mut x);
            self.update_bit(node, &x, p1, y);
            node = (node << 1) | usize::from(y);
        }
        self.advance_match(b);
        self.ctx = (self.ctx << 8) | u64::from(b);
    }
}

/// Fill a two-symbol CDF `[0, c0, TOTAL]` where symbol 0 (bit = 0) holds mass
/// `1 - p1`. `c0` is clamped to `[1, TOTAL - 1]` to satisfy the AC's
/// strictly-increasing-CDF contract.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn fill_bit_cdf(p1: f64, cdf: &mut [u32; 3]) {
    let c0 = ((1.0 - p1) * f64::from(TOTAL)).round() as u32;
    cdf[1] = c0.clamp(1, TOTAL - 1);
}

/// Encode `measure` (preceded by `warm` priming) with or without the match
/// model. Shared by [`LmixCodec`] and [`LmatchCodec`].
#[allow(clippy::cast_possible_truncation)]
fn encode_impl(
    use_match: bool,
    component: &str,
    warm: &[u8],
    measure: &[u8],
) -> (Vec<u8>, Decomposition) {
    let mut writer = BitWriter::new();
    writer.write_bits(measure.len() as u64, 64);
    let mut model = Model::new(use_match);
    for &b in warm {
        model.learn_byte(b);
    }
    {
        let mut enc = AcEncoder::new(&mut writer);
        let mut x = [0.0; N_INPUTS];
        let mut cdf = [0u32, 0, TOTAL];
        for &b in measure {
            let mut node = 1usize;
            for i in (0..8).rev() {
                let y = (b >> i) & 1;
                let p1 = model.predict_bit(node, &mut x);
                fill_bit_cdf(p1, &mut cdf);
                enc.encode(&cdf, usize::from(y));
                model.update_bit(node, &x, p1, y);
                node = (node << 1) | usize::from(y);
            }
            model.advance_match(b);
            model.ctx = (model.ctx << 8) | u64::from(b);
        }
        enc.finish();
    }
    let (buf, _pad) = writer.finish();
    let mut decomp = Decomposition::new();
    decomp.add(component, 8 * buf.len() as u64);
    (buf, decomp)
}

/// Decode an archive produced by [`encode_impl`] with the same `use_match`.
fn decode_impl(use_match: bool, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
    let mut reader = BitReader::new(archive);
    let n = usize::try_from(reader.read_bits(64)?).expect("length prefix fits usize");
    let mut model = Model::new(use_match);
    for &b in warm {
        model.learn_byte(b);
    }
    let mut out = Vec::with_capacity(n);
    let mut dec = AcDecoder::new(&mut reader);
    let mut x = [0.0; N_INPUTS];
    let mut cdf = [0u32, 0, TOTAL];
    for _ in 0..n {
        let mut node = 1usize;
        let mut b = 0u8;
        for _ in 0..8 {
            let p1 = model.predict_bit(node, &mut x);
            fill_bit_cdf(p1, &mut cdf);
            let y = u8::try_from(dec.decode(&cdf)?).expect("bit symbol 0/1 fits u8");
            model.update_bit(node, &x, p1, y);
            b = (b << 1) | y;
            node = (node << 1) | usize::from(y);
        }
        model.advance_match(b);
        model.ctx = (model.ctx << 8) | u64::from(b);
        out.push(b);
    }
    Ok(out)
}

/// The v7 logistic-mixing codec: bitwise multi-order context mixing in the
/// logit domain + AC, nothing else (no match model).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LmixCodec;

impl Codec for LmixCodec {
    fn name(&self) -> &'static str {
        "lmix"
    }

    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        Ok(encode_impl(false, "lmix", warm, measure))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        decode_impl(false, warm, archive)
    }
}

/// [`LmixCodec`] plus a long-range match model as an extra mixer input.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LmatchCodec;

impl Codec for LmatchCodec {
    fn name(&self) -> &'static str {
        "lmatch"
    }

    fn encode_window(&self, warm: &[u8], measure: &[u8]) -> Result<(Vec<u8>, Decomposition)> {
        Ok(encode_impl(true, "lmatch", warm, measure))
    }

    fn decode_window(&self, warm: &[u8], archive: &[u8]) -> Result<Vec<u8>> {
        decode_impl(true, warm, archive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrips_text(codec: &dyn Codec) {
        let measure = b"the quick brown fox jumps over the lazy dog. \
                        the quick brown fox jumps over the lazy dog again."
            .repeat(40);
        let (archive, decomp) = codec.encode_window(b"", &measure).unwrap();
        assert_eq!(decomp.total(), 8 * archive.len() as u64);
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
        assert!(
            archive.len() < measure.len(),
            "archive {} not smaller than input {}",
            archive.len(),
            measure.len()
        );
    }

    fn roundtrips_with_warm(codec: &dyn Codec) {
        let warm = b"warm priming context that both sides share verbatim. ".repeat(8);
        let measure = b"warm priming context that both sides share, then new tail.".to_vec();
        let (archive, _) = codec.encode_window(&warm, &measure).unwrap();
        let decoded = codec.decode_window(&warm, &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    fn roundtrips_all_bytes(codec: &dyn Codec) {
        let measure: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let (archive, _) = codec.encode_window(b"", &measure).unwrap();
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert_eq!(decoded, measure);
    }

    fn roundtrips_empty(codec: &dyn Codec) {
        let (archive, _) = codec.encode_window(b"", b"").unwrap();
        let decoded = codec.decode_window(b"", &archive).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn lmix_roundtrips() {
        roundtrips_text(&LmixCodec);
        roundtrips_with_warm(&LmixCodec);
        roundtrips_all_bytes(&LmixCodec);
        roundtrips_empty(&LmixCodec);
    }

    #[test]
    fn lmatch_roundtrips() {
        roundtrips_text(&LmatchCodec);
        roundtrips_with_warm(&LmatchCodec);
        roundtrips_all_bytes(&LmatchCodec);
        roundtrips_empty(&LmatchCodec);
    }

    #[test]
    fn lmatch_beats_lmix_on_long_repeat() {
        // A long verbatim repeat is exactly what the match model exists for:
        // after the first copy, the second should cost almost nothing.
        let unit = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
        let measure = unit.repeat(200);
        let (a_mix, _) = LmixCodec.encode_window(b"", &measure).unwrap();
        let (a_match, _) = LmatchCodec.encode_window(b"", &measure).unwrap();
        assert!(
            a_match.len() < a_mix.len(),
            "match archive {} should beat mixer-only {}",
            a_match.len(),
            a_mix.len()
        );
    }
}
