//! PPM-C adaptive multi-order context model with escape.
//!
//! Used by Phase-8 codec for Content literals — replaces dense
//! Order-N tables with sparse hash-keyed contexts at orders 0..=M.
//! The "C" variant emits an explicit escape symbol with mass equal
//! to the number of distinct seen symbols in the current context;
//! when the symbol-to-encode hasn't been seen in this context, we
//! emit the escape and fall back to order M-1. Unseen contexts emit
//! nothing (implicit escape, zero bits) and fall through similarly.
//! At order -1 we emit uniformly over the full alphabet.
//!
//! No exclusion mechanism (PPM-D adds that) — code stays tight at
//! the cost of some double-counting when a low-order context covers
//! a symbol already covered by a higher-order one. The bench will
//! tell us how much that costs.
//!
//! Per-context count rescaling: when a context's total mass exceeds
//! `TOTAL/2`, all counts are halved (with `max(1)` floor) — same
//! discipline as the dense `Order0` / `Order1Ctx` models.

use std::collections::HashMap;

use anyhow::Result;

use crate::ac::{AcDecoder, AcEncoder, TOTAL};

const RESCALE_THRESHOLD: u32 = TOTAL / 2;

/// A single context's symbol-count distribution.
#[derive(Debug, Default)]
struct PpmCtx {
    /// `(symbol, count)` entries in insertion order. Lookup is linear
    /// scan; most contexts have ≤ 8 entries so this is a wash with a
    /// `HashMap` and avoids the per-context hashmap overhead.
    counts: Vec<(u16, u32)>,
    total: u32,
}

impl PpmCtx {
    fn position_of(&self, sym: u16) -> Option<usize> {
        self.counts.iter().position(|e| e.0 == sym)
    }

    fn observe(&mut self, sym: u16) {
        if let Some(idx) = self.position_of(sym) {
            self.counts[idx].1 += 1;
        } else {
            self.counts.push((sym, 1));
        }
        self.total += 1;
        if self.total > RESCALE_THRESHOLD {
            let mut new_total: u32 = 0;
            self.counts.retain_mut(|(_, c)| {
                *c = (*c >> 1).max(1);
                new_total += *c;
                true
            });
            self.total = new_total;
        }
    }
}

/// Multi-order PPM-C predictor over a fixed alphabet.
#[derive(Debug)]
pub(crate) struct PpmC {
    alphabet: u32,
    max_order: usize,
    /// `tables[k]` holds order-k contexts keyed by an FNV-1a hash of
    /// the last `k` symbols.
    tables: Vec<HashMap<u64, PpmCtx>>,
    /// History of the last `max_order` symbols (newest at the back).
    history: Vec<u16>,
}

impl PpmC {
    pub(crate) fn new(alphabet: u32, max_order: usize) -> Self {
        Self {
            alphabet,
            max_order,
            tables: (0..=max_order).map(|_| HashMap::new()).collect(),
            history: Vec::with_capacity(max_order),
        }
    }

    /// Encode `sym` through the AC, navigating the escape chain and
    /// updating all order tables. Used by the encoder.
    pub(crate) fn encode(&mut self, enc: &mut AcEncoder<'_>, sym: u16) {
        debug_assert!(u32::from(sym) < self.alphabet, "symbol out of alphabet");

        // Walk orders from max_order down to 0, emitting escape until
        // we find a context that contains `sym`.
        let mut emit_done = false;
        for order in (0..=self.max_order).rev() {
            let Some(ctx) = self.tables[order].get(&ctx_key(&self.history, order)) else {
                continue;
            };
            let (cdf, escape_idx) = build_cdf(ctx);
            if let Some(idx) = ctx.position_of(sym) {
                enc.encode(&cdf, idx);
                emit_done = true;
                break;
            }
            // Emit escape, continue to lower order.
            enc.encode(&cdf, escape_idx);
        }
        if !emit_done {
            // Order -1 fallback: uniform over the full alphabet.
            let cdf = uniform_cdf(self.alphabet);
            enc.encode(&cdf, sym as usize);
        }
        self.observe(sym);
    }

    /// Decode the next symbol from the AC, navigating the same
    /// escape chain. Used by the decoder.
    pub(crate) fn decode(&mut self, dec: &mut AcDecoder<'_, '_>) -> Result<u16> {
        let mut decoded: Option<u16> = None;
        for order in (0..=self.max_order).rev() {
            let Some(ctx) = self.tables[order].get(&ctx_key(&self.history, order)) else {
                continue;
            };
            let (cdf, escape_idx) = build_cdf(ctx);
            let idx = dec.decode(&cdf)?;
            if idx == escape_idx {
                continue;
            }
            decoded = Some(ctx.counts[idx].0);
            break;
        }
        let sym = if let Some(s) = decoded {
            s
        } else {
            let cdf = uniform_cdf(self.alphabet);
            let sym_usize = dec.decode(&cdf)?;
            u16::try_from(sym_usize).expect("symbol fits u16")
        };
        self.observe(sym);
        Ok(sym)
    }

    /// Update the per-order count tables and slide the history.
    /// Used both for encode/decode (called by `encode`/`decode`) and
    /// during prewarm when no AC traffic happens.
    pub(crate) fn observe(&mut self, sym: u16) {
        for order in 0..=self.max_order {
            let key = ctx_key(&self.history, order);
            self.tables[order].entry(key).or_default().observe(sym);
        }
        self.history.push(sym);
        if self.history.len() > self.max_order {
            self.history.remove(0);
        }
    }

    /// Slide a symbol into history WITHOUT touching the count
    /// tables. Used for LZ match-copied bytes whose context should
    /// inform future predictions but whose emission was handled by
    /// the match record, not this model. (Same role as v1's
    /// `note_match_bytes`.)
    pub(crate) fn slide_history(&mut self, sym: u16) {
        self.history.push(sym);
        if self.history.len() > self.max_order {
            self.history.remove(0);
        }
    }

    /// Reset history without dropping the learned tables — used when
    /// the codec wants to interleave with another symbol stream that
    /// shouldn't pollute this predictor's context.
    #[allow(dead_code)]
    pub(crate) fn reset_history(&mut self) {
        self.history.clear();
    }
}

/// FNV-1a hash of the last `order` symbols in `history`. Most-recent
/// symbol contributes last so prefix-shared contexts (`abc`/`xbc`)
/// produce different hashes.
fn ctx_key(history: &[u16], order: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let start = history.len().saturating_sub(order);
    for &sym in &history[start..] {
        h ^= u64::from(sym);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Build a quantized CDF over `ctx.counts.len() + 1` symbols where
/// the last entry is the escape symbol. Returns `(cdf, escape_idx)`.
/// Total mass `cdf[len-1] == TOTAL`; every gap is ≥ 1.
#[allow(clippy::cast_possible_truncation)]
fn build_cdf(ctx: &PpmCtx) -> (Vec<u32>, usize) {
    let k = ctx.counts.len();
    // Escape symbol's natural mass is `k` (PPM-C). Total = T + K.
    let escape_mass = k as u32;
    let total_mass = ctx.total + escape_mass;
    let mut cdf = vec![0u32; k + 2];
    // First pass: integer scale, accumulate.
    let mut acc: u32 = 0;
    for (i, (_, count)) in ctx.counts.iter().enumerate() {
        cdf[i] = (u64::from(acc) * u64::from(TOTAL) / u64::from(total_mass)) as u32;
        acc += count;
    }
    cdf[k] = (u64::from(acc) * u64::from(TOTAL) / u64::from(total_mass)) as u32;
    cdf[k + 1] = TOTAL;
    // Second pass: bump every zero-width gap to width 1, stealing
    // from the largest gap (the dominant symbol).
    let mut largest_idx = 0;
    let mut largest_gap: u32 = 0;
    for i in 0..=k {
        let gap = cdf[i + 1] - cdf[i];
        if gap > largest_gap {
            largest_gap = gap;
            largest_idx = i;
        }
    }
    for i in 0..=k {
        if cdf[i + 1] == cdf[i] && i != largest_idx {
            // Steal one unit from the largest gap. K ≤ alphabet and
            // TOTAL = 65536; in the worst case the largest_gap is
            // still huge relative to the number of zero-width entries
            // so saturating_sub is just a safety belt.
            for slot in cdf
                .iter_mut()
                .skip(largest_idx + 1)
                .take(k + 1 - largest_idx)
            {
                *slot = slot.saturating_sub(1);
            }
            for slot in cdf.iter_mut().skip(i + 1).take(k + 1 - i) {
                *slot += 1;
            }
        }
    }
    (cdf, k)
}

/// Uniform CDF over `n` symbols totaling `TOTAL`.
#[allow(clippy::cast_possible_truncation)]
fn uniform_cdf(n: u32) -> Vec<u32> {
    let n_us = n as usize;
    let mut cdf = vec![0u32; n_us + 1];
    let n_u64 = u64::from(n);
    for (i, slot) in cdf.iter_mut().enumerate().take(n_us) {
        *slot = ((i as u64 * u64::from(TOTAL)) / n_u64) as u32;
    }
    cdf[n_us] = TOTAL;
    cdf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{BitReader, BitWriter};

    fn roundtrip_symbols(alphabet: u32, max_order: usize, syms: &[u16]) {
        let mut writer = BitWriter::new();
        {
            let mut enc = AcEncoder::new(&mut writer);
            let mut model = PpmC::new(alphabet, max_order);
            for &s in syms {
                model.encode(&mut enc, s);
            }
            enc.finish();
        }
        let (buf, _) = writer.finish();
        let mut reader = BitReader::new(&buf);
        let mut dec = AcDecoder::new(&mut reader);
        let mut model = PpmC::new(alphabet, max_order);
        let decoded: Vec<u16> = (0..syms.len())
            .map(|_| model.decode(&mut dec).unwrap())
            .collect();
        assert_eq!(decoded, syms);
    }

    #[test]
    fn ppmc_order0_roundtrips() {
        let syms: Vec<u16> = (0u32..256)
            .map(|i| u16::try_from((i * 17) % 52).unwrap())
            .collect();
        roundtrip_symbols(52, 0, &syms);
    }

    #[test]
    fn ppmc_order3_roundtrips_skewed() {
        // Heavily repeated phrase — order-3 should learn the bigram
        // structure and compress well, but the test just checks
        // roundtrip correctness.
        let phrase = b"the quick brown fox jumps over the lazy dog. ";
        let mut syms = Vec::new();
        for _ in 0..50 {
            for &b in phrase {
                syms.push(u16::from(b));
            }
        }
        roundtrip_symbols(256, 3, &syms);
    }

    #[test]
    fn ppmc_handles_novel_symbols() {
        // Symbols outside the seen-so-far set must roundtrip via the
        // escape chain — this exercises the order-(-1) uniform path.
        let syms: Vec<u16> = (0u16..1000).map(|i| (i.wrapping_mul(257)) % 256).collect();
        roundtrip_symbols(256, 5, &syms);
    }
}
