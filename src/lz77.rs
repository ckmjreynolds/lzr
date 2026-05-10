//! LZ77 pre-pass layered on top of the type-routed PPM codec.
//!
//! For each input position the encoder asks the matcher whether a long
//! match exists in the prior window. If yes, an AC-encoded match record
//! (binary flag + bucketed offset + adaptive length code) replaces what
//! would otherwise be a chunk of routed-PPM literal bytes. Otherwise
//! the byte falls through to the routed codec unchanged.
//!
//! ## Design choices
//!
//! - **Window**: 1 MiB sliding. Covers the bench's 1 MiB pre-warm with
//!   headroom; for the full 1 GB submission the matcher's working set
//!   stays inside Hutter's budget at 1 MiB chain plus the input buffer.
//! - **Matcher**: hash-chain, chain bound 32. zlib's "fast" preset
//!   territory — not optimal-parsing but the order-of-magnitude
//!   question is answerable in seconds per bench offset.
//! - **Offset encoding**: log-bucket via an adaptive predictor (most
//!   offsets fall in a few buckets, so AC compression of the bucket
//!   index is tight) plus the bucket-relative bits as a uniform AC
//!   event over a power-of-2 alphabet sized to the exact bit count.
//! - **Length encoding**: Order-0 adaptive on `length - MIN_MATCH`. The
//!   length distribution is heavily skewed toward short matches so the
//!   adaptive predictor lands at ~3–4 bits/length vs 8 bits raw.
//! - **`MIN_MATCH = 6`**: with the bucketed offset and adaptive length
//!   encoding, match cost drops to ~16–20 bits average, well below the
//!   ~2.6 bpb routed-PPM literal cost for any 6+ byte match.
//! - **Greedy parse**: at each position, take the longest match the
//!   matcher reports if it clears `MIN_MATCH`; otherwise emit literal.
//!   Lazy / optimal parsing is a follow-up.
//!
//! ## Insertion timing
//!
//! `hash3` reads three bytes; the decoder can't insert position `p`
//! into the matcher until it has decoded byte `p+2`. To stay in
//! lockstep with the decoder, the encoder uses the same lazy rule:
//! before each event, insert all positions `q` with `q + 3 ≤ pos`.
//! Both sides advance an internal `next_insert` cursor identically;
//! state carries from `prewarm` through `encode_range` / `decode_into`.

use std::io::{self, BufWriter, Read, Write};

use crate::ac::{Decoder, Encoder, TOTAL};
use crate::arch::{CDF_LEN, VOCAB};
use crate::codec::{ProbSource, read_header, write_header};
use crate::model::ByteTransformer;
use crate::predict::{ByteClass, Order0Adaptive};
use crate::probs::{logits_to_cdf, uniform_cdf};
use crate::routed::RoutedProbs;
use crate::tokenizer::Token;
use crate::weights::Weights;

const WINDOW_LOG: usize = 22;
const WINDOW_SIZE: usize = 1 << WINDOW_LOG;
const WINDOW_MASK: usize = WINDOW_SIZE - 1;

const HASH_LOG: usize = 16;
const HASH_SIZE: usize = 1 << HASH_LOG;

const MIN_MATCH: usize = 6;
const MAX_MATCH: usize = MIN_MATCH + 255; // length-MIN_MATCH fits in one Order0 token
const CHAIN_DEPTH: usize = 32;

const NIL: u32 = u32::MAX;

/// Hash-chain LZ77 longest-match finder.
pub(crate) struct Matcher {
    hash: Vec<u32>,
    prev: Vec<u32>,
}

impl Matcher {
    pub(crate) fn new() -> Self {
        Self {
            hash: vec![NIL; HASH_SIZE],
            prev: vec![NIL; WINDOW_SIZE],
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn hash3(buf: &[u8], pos: usize) -> usize {
        let h =
            u32::from(buf[pos]) | (u32::from(buf[pos + 1]) << 8) | (u32::from(buf[pos + 2]) << 16);
        let h = h.wrapping_mul(2_654_435_761);
        (h >> (32 - HASH_LOG)) as usize
    }

    /// Insert `pos` into the hash chain. Caller guarantees
    /// `pos + 3 <= buf.len()`.
    #[allow(clippy::cast_possible_truncation)]
    fn insert(&mut self, buf: &[u8], pos: usize) {
        let h = Self::hash3(buf, pos);
        self.prev[pos & WINDOW_MASK] = self.hash[h];
        self.hash[h] = pos as u32;
    }

    /// Return `(offset, length)` of the longest in-window match for
    /// `buf[pos..]` clearing `MIN_MATCH`, else `None`.
    #[allow(clippy::cast_possible_truncation)]
    fn find_match(&self, buf: &[u8], pos: usize) -> Option<(u32, u32)> {
        if pos + MIN_MATCH > buf.len() {
            return None;
        }
        let max_len = (buf.len() - pos).min(MAX_MATCH);
        let h = Self::hash3(buf, pos);

        let mut candidate = self.hash[h];
        let mut best_offset: u32 = 0;
        let mut best_length: usize = 0;

        for _ in 0..CHAIN_DEPTH {
            if candidate == NIL {
                break;
            }
            // Once we've already matched up to `max_len`, no candidate
            // can do better — and accessing `buf[pos + best_length]` in
            // the quick-reject below would index past the end.
            if best_length >= max_len {
                break;
            }
            let cpos = candidate as usize;
            if cpos >= pos || pos - cpos > WINDOW_SIZE {
                break;
            }
            // Quick reject: candidate's byte at `best_length` must
            // match to have a chance of being a new best. Bounds:
            // best_length < max_len ≤ buf.len() - pos, so
            // pos + best_length < buf.len(); cpos < pos, so
            // cpos + best_length < buf.len() too.
            if best_length > 0 && buf[cpos + best_length] != buf[pos + best_length] {
                candidate = self.prev[cpos & WINDOW_MASK];
                continue;
            }
            let mut len = 0;
            while len < max_len && buf[cpos + len] == buf[pos + len] {
                len += 1;
            }
            if len >= MIN_MATCH && len > best_length {
                best_length = len;
                best_offset = (pos - cpos) as u32;
            }
            candidate = self.prev[cpos & WINDOW_MASK];
        }

        if best_length >= MIN_MATCH {
            Some((best_offset, best_length as u32))
        } else {
            None
        }
    }
}

impl Default for Matcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Tiny adaptive binary predictor for the literal/match flag.
struct FlagProb {
    counts: [u32; 2],
    total: u64,
}

const FLAG_RESCALE: u64 = (TOTAL / 2) as u64;

impl FlagProb {
    const fn new() -> Self {
        Self {
            counts: [1, 1],
            total: 2,
        }
    }

    /// 3-entry CDF: `[0, mass(literal), TOTAL]`. Symbol 0 = literal,
    /// symbol 1 = match.
    #[allow(clippy::cast_possible_truncation)]
    fn cdf(&self) -> [u32; 3] {
        let m0 = ((u64::from(self.counts[0]) * u64::from(TOTAL)) / self.total) as u32;
        let m0 = m0.clamp(1, TOTAL - 1);
        [0, m0, TOTAL]
    }

    fn observe(&mut self, sym: usize) {
        self.counts[sym] += 1;
        self.total += 1;
        if self.total > FLAG_RESCALE {
            self.counts[0] = (self.counts[0] >> 1).max(1);
            self.counts[1] = (self.counts[1] >> 1).max(1);
            self.total = u64::from(self.counts[0]) + u64::from(self.counts[1]);
        }
    }
}

/// Precomputed uniform CDFs for power-of-2 alphabet sizes.
/// `cdfs[k]` covers an alphabet of `2^k` symbols (CDF length `2^k + 1`).
/// Used by the LZ-routed offset encoder to emit bucket-relative bits
/// without padding to byte boundaries.
struct UniformCdfTable {
    /// `cdfs[k]` for k = 1..=8 (alphabet sizes 2, 4, ..., 256).
    cdfs: [Vec<u32>; 9],
}

impl UniformCdfTable {
    #[allow(clippy::cast_possible_truncation)]
    fn new() -> Self {
        let cdfs: [Vec<u32>; 9] = std::array::from_fn(|k| {
            if k == 0 {
                // Degenerate; never used. Filler so the index is the
                // bit count.
                vec![0, TOTAL]
            } else {
                let n: usize = 1 << k;
                let mut cdf = vec![0u32; n + 1];
                let n_u64 = n as u64;
                for (i, slot) in cdf.iter_mut().enumerate().take(n) {
                    *slot = (((i as u64) * u64::from(TOTAL)) / n_u64) as u32;
                }
                cdf[n] = TOTAL;
                cdf
            }
        });
        Self { cdfs }
    }

    fn encode_bits<W: Write>(
        &self,
        enc: &mut Encoder<W>,
        value: u32,
        n_bits: u32,
    ) -> io::Result<()> {
        let mut remaining = n_bits;
        let mut v = value;
        while remaining > 0 {
            let chunk = remaining.min(8);
            let mask = (1u32 << chunk) - 1;
            // chunk ≤ 8 so (v & mask) ≤ 255, fits Token (u16) trivially.
            let chunk_value = Token::try_from(v & mask).expect("chunk fits u16");
            v >>= chunk;
            enc.encode(&self.cdfs[chunk as usize], chunk_value)?;
            remaining -= chunk;
        }
        Ok(())
    }

    fn decode_bits<R: Read>(&self, dec: &mut Decoder<R>, n_bits: u32) -> io::Result<u32> {
        let mut value = 0u32;
        let mut shift = 0u32;
        let mut remaining = n_bits;
        while remaining > 0 {
            let chunk = remaining.min(8);
            let chunk_value = u32::from(dec.decode(&self.cdfs[chunk as usize])?);
            value |= chunk_value << shift;
            shift += chunk;
            remaining -= chunk;
        }
        Ok(value)
    }
}

/// Optional neural arm: a byte-level model whose per-position CDF is
/// linearly mixed with the routed-class predictor at literal positions.
/// Match bytes don't go through the AC, so the neural is only queried
/// at literals; its state is advanced through every output byte (literal
/// or match-copied) so the model's context tracks the actual byte
/// sequence, not just the literals.
struct NeuralState {
    model: ByteTransformer,
    /// CDF for the next byte conditioned on the model's current state.
    /// Refreshed after every `model.step` call.
    cdf: [u32; CDF_LEN],
    /// Mixing weight for the neural CDF in `[0, 1]`. The routed CDF
    /// gets `1 - mix_weight`.
    mix_weight: f32,
    /// Linear or logistic combination of routed and neural CDFs.
    mix_mode: MixMode,
}

impl NeuralState {
    fn new(weights: Weights, mix_weight: f32, mix_mode: MixMode) -> Self {
        let mut model = ByteTransformer::new(weights);
        model.reset();
        Self {
            model,
            cdf: uniform_cdf(),
            mix_weight,
            mix_mode,
        }
    }

    /// Consume `byte`, advance state, refresh `cdf` for the next position.
    fn advance(&mut self, byte: u8) {
        let logits = self.model.step(Token::from(byte));
        self.cdf = logits_to_cdf(logits);
    }

    /// Reset model state and CDF to the cold-start configuration.
    fn reset(&mut self) {
        self.model.reset();
        self.cdf = uniform_cdf();
    }
}

/// Mixing scheme for combining two CDFs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MixMode {
    /// `p_mix[s] = w · p_a[s] + (1−w) · p_b[s]`. Simple, fast, but
    /// dilutes both predictors where they disagree.
    Linear,
    /// `p_mix[s] ∝ p_a[s]^w · p_b[s]^(1−w)`. PAQ-canonical: favors
    /// symbols where both agree, sharpens the combined prediction
    /// where one is confident.
    Logistic,
}

/// Linear count-space mix of two CDFs:
/// `count_mix[i] = round(w · count_a[i] + (1 − w) · count_b[i])`.
/// Both inputs and the output have `cdf[VOCAB] == TOTAL` and every gap
/// strictly positive.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn mix_cdfs_linear(a: &[u32; CDF_LEN], b: &[u32; CDF_LEN], w_a: f32) -> [u32; CDF_LEN] {
    // Per-symbol gaps are at most TOTAL = 65536 (= 1 << 16), so they fit
    // u16 → f32 conversion exactly.
    let w_a = w_a.clamp(0.0, 1.0);
    let w_b = 1.0 - w_a;
    let mut counts = [1u32; VOCAB];
    let mut allocated: u32 = VOCAB as u32;
    let mut frac_idx: Vec<(f32, usize)> = Vec::with_capacity(VOCAB);
    let total_f = f32::from(u16::try_from(TOTAL).unwrap_or(u16::MAX));
    let extra = f32::from(u16::try_from(TOTAL - VOCAB as u32).expect("TOTAL - VOCAB fits u16"));
    for i in 0..VOCAB {
        let gap_a = u16::try_from(a[i + 1] - a[i]).expect("CDF gap fits u16");
        let gap_b = u16::try_from(b[i + 1] - b[i]).expect("CDF gap fits u16");
        let pa = f32::from(gap_a) / total_f;
        let pb = f32::from(gap_b) / total_f;
        let p = w_a.mul_add(pa, w_b * pb);
        let prop = (p * extra).floor();
        let prop_u = prop as u32;
        counts[i] += prop_u;
        allocated += prop_u;
        let frac = p.mul_add(extra, -prop);
        frac_idx.push((frac, i));
    }
    frac_idx.sort_by(|x, y| {
        y.0.partial_cmp(&x.0)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(x.1.cmp(&y.1))
    });
    let mut residual = TOTAL - allocated;
    let mut idx = 0;
    while residual > 0 && idx < frac_idx.len() {
        counts[frac_idx[idx].1] += 1;
        residual -= 1;
        idx += 1;
    }
    let mut cdf = [0u32; CDF_LEN];
    let mut acc = 0u32;
    for i in 0..VOCAB {
        cdf[i] = acc;
        acc += counts[i];
    }
    cdf[VOCAB] = TOTAL;
    cdf
}

/// Logistic mix:
/// `p_mix[s] ∝ p_a[s]^w · p_b[s]^(1−w)`, which is
/// `log p_mix[s] = w · log p_a[s] + (1−w) · log p_b[s] − log Z`.
/// Computed in f64 for numerical headroom (log of 1/65536 = −11.1, so
/// f32 would suffice but f64 is cheap).
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn mix_cdfs_logistic(a: &[u32; CDF_LEN], b: &[u32; CDF_LEN], w_a: f32) -> [u32; CDF_LEN] {
    let w_a = f64::from(w_a.clamp(0.0, 1.0));
    let w_b = 1.0 - w_a;
    // 1/TOTAL guards log of 0 (gaps are guaranteed >= 1, so this is
    // just a precaution).
    let inv_total = 1.0 / f64::from(TOTAL);
    let mut log_p = [0.0_f64; VOCAB];
    let mut max_lp = f64::NEG_INFINITY;
    for i in 0..VOCAB {
        let gap_a = u16::try_from(a[i + 1] - a[i]).expect("CDF gap fits u16");
        let gap_b = u16::try_from(b[i + 1] - b[i]).expect("CDF gap fits u16");
        let pa = (f64::from(gap_a)).max(1.0) * inv_total;
        let pb = (f64::from(gap_b)).max(1.0) * inv_total;
        let lp = w_a.mul_add(pa.ln(), w_b * pb.ln());
        log_p[i] = lp;
        if lp > max_lp {
            max_lp = lp;
        }
    }
    // Exponentiate (subtract max for stability) and normalize.
    let mut probs = [0.0_f64; VOCAB];
    let mut sum = 0.0_f64;
    for i in 0..VOCAB {
        let p = (log_p[i] - max_lp).exp();
        probs[i] = p;
        sum += p;
    }
    let inv_sum = 1.0 / sum;

    // Quantize to integer CDF with each gap >= 1.
    let mut counts = [1u32; VOCAB];
    let mut allocated: u32 = VOCAB as u32;
    let mut frac_idx: Vec<(f64, usize)> = Vec::with_capacity(VOCAB);
    let extra = f64::from(TOTAL - VOCAB as u32);
    for i in 0..VOCAB {
        let p = probs[i] * inv_sum;
        let prop = (p * extra).floor();
        let prop_u = prop as u32;
        counts[i] += prop_u;
        allocated += prop_u;
        let frac = p.mul_add(extra, -prop);
        frac_idx.push((frac, i));
    }
    frac_idx.sort_by(|x, y| {
        y.0.partial_cmp(&x.0)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(x.1.cmp(&y.1))
    });
    let mut residual = TOTAL - allocated;
    let mut idx = 0;
    while residual > 0 && idx < frac_idx.len() {
        counts[frac_idx[idx].1] += 1;
        residual -= 1;
        idx += 1;
    }
    let mut cdf = [0u32; CDF_LEN];
    let mut acc = 0u32;
    for i in 0..VOCAB {
        cdf[i] = acc;
        acc += counts[i];
    }
    cdf[VOCAB] = TOTAL;
    cdf
}

fn mix_cdfs(a: &[u32; CDF_LEN], b: &[u32; CDF_LEN], w_a: f32, mode: MixMode) -> [u32; CDF_LEN] {
    match mode {
        MixMode::Linear => mix_cdfs_linear(a, b, w_a),
        MixMode::Logistic => mix_cdfs_logistic(a, b, w_a),
    }
}

/// Fold a raw-byte neural CDF into the routed letter encoding's
/// alphabet at Upper-class positions: A-Z mass moves to the
/// corresponding a-z slot, A-Z slots are zeroed out (then bumped back
/// to 1 for AC validity, with the excess subtracted from the largest
/// gap to keep the total exactly `TOTAL`).
fn fold_neural_cdf_for_upper(neural_cdf: &[u32; CDF_LEN]) -> [u32; CDF_LEN] {
    let mut gaps = [0u32; VOCAB];
    for i in 0..VOCAB {
        gaps[i] = neural_cdf[i + 1] - neural_cdf[i];
    }
    for c in b'A'..=b'Z' {
        let upper_i = c as usize;
        let lower_i = (c | 0x20) as usize;
        gaps[lower_i] += gaps[upper_i];
        gaps[upper_i] = 0;
    }
    // Bump zeros to 1 so every CDF gap is positive; track shortfall to
    // subtract back from the dominant slot.
    let mut total: u64 = 0;
    let mut shortfall: u32 = 0;
    for g in &mut gaps {
        if *g == 0 {
            *g = 1;
            shortfall += 1;
        }
        total += u64::from(*g);
    }
    let target = u64::from(TOTAL);
    if total > target {
        let excess = total - target;
        let (max_idx, _) = gaps
            .iter()
            .enumerate()
            .max_by_key(|(_, g)| **g)
            .expect("non-empty gaps");
        // The excess bound is `shortfall + (mass folded - 0)` ≤ 26 + ~few hundred,
        // far smaller than any peaky neural CDF's max gap.
        gaps[max_idx] = gaps[max_idx].saturating_sub(u32::try_from(excess).unwrap_or(u32::MAX));
        gaps[max_idx] = gaps[max_idx].max(1);
    } else if total < target {
        let deficit = target - total;
        let (max_idx, _) = gaps
            .iter()
            .enumerate()
            .max_by_key(|(_, g)| **g)
            .expect("non-empty gaps");
        gaps[max_idx] = gaps[max_idx].saturating_add(u32::try_from(deficit).unwrap_or(0));
    }
    let _ = shortfall;
    let mut cdf = [0u32; CDF_LEN];
    let mut acc: u32 = 0;
    for i in 0..VOCAB {
        cdf[i] = acc;
        acc += gaps[i];
    }
    cdf[VOCAB] = TOTAL;
    cdf
}

/// LZ77-on-routed codec state: routed predictors, matcher, flag prob,
/// offset/length predictors, uniform-CDF table, insert cursor, and
/// optional neural arm.
#[allow(clippy::struct_field_names)]
pub(crate) struct LzRouted {
    pub(crate) routed: RoutedProbs,
    matcher: Matcher,
    flag: FlagProb,
    offset_bucket_pred: Order0Adaptive,
    offset_bucket_cdf: [u32; CDF_LEN],
    length_pred: Order0Adaptive,
    length_cdf: [u32; CDF_LEN],
    uniform: UniformCdfTable,
    next_insert: usize,
    neural: Option<NeuralState>,
}

impl LzRouted {
    pub(crate) fn new() -> Self {
        Self::with_neural(None)
    }

    pub(crate) fn with_neural(neural: Option<(Weights, f32, MixMode)>) -> Self {
        let mut offset_bucket_pred = Order0Adaptive::new();
        let mut length_pred = Order0Adaptive::new();
        let offset_bucket_cdf = offset_bucket_pred.initial_cdf();
        let length_cdf = length_pred.initial_cdf();
        Self {
            routed: RoutedProbs::new(),
            matcher: Matcher::new(),
            flag: FlagProb::new(),
            offset_bucket_pred,
            offset_bucket_cdf,
            length_pred,
            length_cdf,
            uniform: UniformCdfTable::new(),
            next_insert: 0,
            neural: neural.map(|(w, mix, mode)| NeuralState::new(w, mix, mode)),
        }
    }

    /// Compute the literal byte CDF for the given class. With neural
    /// active and class is `Lower` or `NonLetter`, this is the linear
    /// mix of routed + neural. `Upper` class skips neural mixing
    /// because the case-folding scheme makes the two CDFs index
    /// different things; at `Upper` positions the routed letter CDF
    /// (over folded a-z) is used alone.
    fn literal_byte_cdf(&self, class: ByteClass) -> [u32; CDF_LEN] {
        let routed_base = *self.routed.cdf_for_class(class);
        match (&self.neural, class) {
            (Some(n), ByteClass::Lower | ByteClass::NonLetter) => {
                mix_cdfs(&n.cdf, &routed_base, n.mix_weight, n.mix_mode)
            }
            (Some(n), ByteClass::Upper) => {
                // Routed encodes the folded byte (a-z), but the neural
                // model was trained on raw bytes (which include A-Z).
                // Fold neural's A-Z mass into the corresponding a-z slot
                // so both inputs to `mix_cdfs` are on the same alphabet.
                let folded = fold_neural_cdf_for_upper(&n.cdf);
                mix_cdfs(&folded, &routed_base, n.mix_weight, n.mix_mode)
            }
            _ => routed_base,
        }
    }

    fn advance_neural(&mut self, byte: u8) {
        if let Some(n) = self.neural.as_mut() {
            n.advance(byte);
        }
    }

    fn catch_up_inserts(&mut self, buf: &[u8], up_to: usize) {
        let bound = up_to.min(buf.len());
        while self.next_insert + 3 <= bound {
            self.matcher.insert(buf, self.next_insert);
            self.next_insert += 1;
        }
    }

    /// Walk `buf` (entire warm prefix) through the LZ pre-pass without
    /// emitting to AC. Caller passes the prefix bytes the real
    /// submission encoder would have seen up to this corpus position.
    /// The matcher, routed predictors, and (if present) neural state
    /// converge to the configuration the encoder would be in at byte
    /// `buf.len()`. The neural sees every byte regardless of literal-
    /// vs-match status — its job is to track the byte sequence, while
    /// match bytes don't go through the AC.
    pub(crate) fn prewarm(&mut self, buf: &[u8]) {
        if let Some(n) = self.neural.as_mut() {
            n.reset();
        }
        let mut pos = 0;
        while pos < buf.len() {
            self.catch_up_inserts(buf, pos);
            let m = if pos + MIN_MATCH <= buf.len() {
                self.matcher.find_match(buf, pos)
            } else {
                None
            };
            match m {
                Some((_, length)) if pos + (length as usize) <= buf.len() => {
                    self.flag.observe(1);
                    let len = length as usize;
                    if let Some(n) = self.neural.as_mut() {
                        for &b in &buf[pos..pos + len] {
                            n.advance(b);
                        }
                    }
                    pos += len;
                }
                _ => {
                    self.flag.observe(0);
                    let b = buf[pos];
                    self.routed.observe(b);
                    if let Some(n) = self.neural.as_mut() {
                        n.advance(b);
                    }
                    pos += 1;
                }
            }
        }
        self.catch_up_inserts(buf, buf.len());
        self.routed.refresh_cdfs();
    }
}

impl Default for LzRouted {
    fn default() -> Self {
        Self::new()
    }
}

const fn log2_floor(x: u32) -> u32 {
    x.ilog2()
}

/// Encode `buf[start..]` into a complete archive (header + payload).
/// Caller has already populated `state` via `prewarm` over `buf[..start]`.
/// `buf` must remain identical to the prewarm contents in `..start` so
/// the matcher's hash positions resolve correctly.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn encode_range<W: Write>(
    buf: &[u8],
    start: usize,
    state: &mut LzRouted,
    out: W,
) -> io::Result<()> {
    let payload_len = buf.len() - start;
    let mut w = BufWriter::new(out);
    write_header(
        &mut w,
        u64::try_from(payload_len).expect("payload fits in u64"),
    )?;
    let mut enc = Encoder::new(&mut w);

    let mut pos = start;
    while pos < buf.len() {
        state.catch_up_inserts(buf, pos);

        let m_p = if pos + MIN_MATCH <= buf.len() {
            state.matcher.find_match(buf, pos)
        } else {
            None
        };

        // Lazy parse: if a strictly longer match starts at `pos + 1`,
        // defer to it (emit a literal at `pos`, take the better match
        // next iteration). The peek queries the matcher in its current
        // state, which lags the would-be next-iteration state by a few
        // positions due to the lazy `catch_up_inserts` rule — tiny
        // accuracy loss, ~3–7% match-quality win typical of zlib's
        // lazy mode.
        let take_match = match m_p {
            Some((_, len_p)) if pos + (len_p as usize) <= buf.len() => {
                let m_next = if pos + 1 + MIN_MATCH <= buf.len() {
                    state.matcher.find_match(buf, pos + 1)
                } else {
                    None
                };
                !matches!(m_next, Some((_, len_next)) if len_next > len_p)
            }
            _ => false,
        };

        if take_match {
            let (offset, length) = m_p.unwrap();
            let flag_cdf = state.flag.cdf();
            enc.encode(&flag_cdf, 1)?;
            state.flag.observe(1);

            // Offset: log-bucket via adaptive predictor, then
            // bucket-relative bits via uniform CDF of size 2^bucket.
            let bucket = log2_floor(offset);
            let bucket_tok = bucket as Token;
            enc.encode(&state.offset_bucket_cdf, bucket_tok)?;
            state.offset_bucket_cdf = state.offset_bucket_pred.advance(bucket_tok);
            if bucket > 0 {
                let rel = offset - (1u32 << bucket);
                state.uniform.encode_bits(&mut enc, rel, bucket)?;
            }

            // Length-MIN_MATCH via Order-0 adaptive (fits in one
            // Token; distribution skewed strongly toward short).
            let len_tok = (length as usize - MIN_MATCH) as Token;
            enc.encode(&state.length_cdf, len_tok)?;
            state.length_cdf = state.length_pred.advance(len_tok);

            // Push matched bytes into the routed cross-stream history
            // (without counting in PPM) so the next literal's context
            // is correct. Neural sees every byte too, regardless of
            // literal/match.
            let len_us = length as usize;
            state.routed.note_match_bytes(&buf[pos..pos + len_us]);
            for &b in &buf[pos..pos + len_us] {
                state.advance_neural(b);
            }
            pos += len_us;
        } else {
            let flag_cdf = state.flag.cdf();
            enc.encode(&flag_cdf, 0)?;
            state.flag.observe(0);
            let byte = buf[pos];
            let class = state.routed.encode_class(&mut enc, byte)?;
            let byte_cdf = state.literal_byte_cdf(class);
            state
                .routed
                .encode_byte_value(&mut enc, byte, class, &byte_cdf)?;
            state.advance_neural(byte);
            pos += 1;
        }
    }
    state.catch_up_inserts(buf, buf.len());

    enc.finish()?;
    w.flush()?;
    Ok(())
}

/// Decode an LZ-routed archive into bytes appended to `out`. `out` is
/// pre-loaded with the same prewarm prefix the encoder saw, so back-
/// references resolve against it.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn decode_into<R: Read>(
    inp: &mut R,
    state: &mut LzRouted,
    out: &mut Vec<u8>,
) -> io::Result<()> {
    let payload_len = read_header(inp)?;
    let mut dec = Decoder::new(inp)?;

    let mut produced: u64 = 0;
    while produced < payload_len {
        state.catch_up_inserts(out, out.len());

        let flag_cdf = state.flag.cdf();
        let flag = dec.decode(&flag_cdf)?;
        state.flag.observe(flag as usize);

        if flag == 1 {
            let bucket_tok = dec.decode(&state.offset_bucket_cdf)?;
            state.offset_bucket_cdf = state.offset_bucket_pred.advance(bucket_tok);
            let bucket = u32::from(bucket_tok);
            let offset = if bucket == 0 {
                1
            } else {
                let rel = state.uniform.decode_bits(&mut dec, bucket)?;
                (1u32 << bucket) + rel
            };

            let len_tok = dec.decode(&state.length_cdf)?;
            state.length_cdf = state.length_pred.advance(len_tok);
            let length = MIN_MATCH + usize::from(len_tok);

            let pos = out.len();
            let src_start = pos - offset as usize;
            for i in 0..length {
                let b = out[src_start + i];
                out.push(b);
            }
            // Mirror encoder: matched bytes inform routed cross-stream
            // history without counting in PPM, and advance neural too.
            state.routed.note_match_bytes(&out[pos..pos + length]);
            for i in 0..length {
                let b = out[pos + i];
                state.advance_neural(b);
            }
            produced += length as u64;
        } else {
            let class = state.routed.decode_class(&mut dec)?;
            let byte_cdf = state.literal_byte_cdf(class);
            let byte = state.routed.decode_byte_value(&mut dec, class, &byte_cdf)?;
            out.push(byte);
            state.advance_neural(byte);
            produced += 1;
        }
    }
    state.catch_up_inserts(out, out.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_short_no_matches() {
        let buf = b"The quick brown fox jumps.".to_vec();
        let mut e = LzRouted::new();
        let mut archive = Vec::new();
        encode_range(&buf, 0, &mut e, &mut archive).unwrap();

        let mut d = LzRouted::new();
        let mut out = Vec::new();
        let mut cur = &archive[..];
        decode_into(&mut cur, &mut d, &mut out).unwrap();
        assert_eq!(out, buf);
    }

    #[test]
    fn roundtrip_long_repetition() {
        let head = b"This is the prefix that we are about to repeat right after.\n";
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(head);
        buf.extend_from_slice(head);
        buf.extend_from_slice(head);

        let mut e = LzRouted::new();
        let mut archive = Vec::new();
        encode_range(&buf, 0, &mut e, &mut archive).unwrap();

        let mut d = LzRouted::new();
        let mut out = Vec::new();
        let mut cur = &archive[..];
        decode_into(&mut cur, &mut d, &mut out).unwrap();
        assert_eq!(out, buf);
    }

    #[test]
    fn roundtrip_with_prewarm() {
        let warm: &[u8] = b"warmup buffer text that primes the matcher and routed predictors.\n";
        let measure: &[u8] =
            b"warmup buffer text that primes the matcher and routed predictors AGAIN.\n";

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(warm);
        let warm_end = buf.len();
        buf.extend_from_slice(measure);

        let mut e = LzRouted::new();
        e.prewarm(warm);
        let mut archive = Vec::new();
        encode_range(&buf, warm_end, &mut e, &mut archive).unwrap();

        let mut d = LzRouted::new();
        d.prewarm(warm);
        let mut decoder_buf: Vec<u8> = warm.to_vec();
        let mut cur = &archive[..];
        decode_into(&mut cur, &mut d, &mut decoder_buf).unwrap();
        assert_eq!(&decoder_buf[warm_end..], measure);
    }
}
