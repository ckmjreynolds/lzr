//! Token-level LZ77 longest-match finder.
//!
//! Maintains a window of recently-seen tokens plus a 3-gram hash index
//! pointing into that window. Used by the v4 `moe-tok-lz` codec to
//! catch repeated runs that exceed the `MoE` arm's attention context.
//!
//! `MIN_MATCH` / `MAX_MATCH` constants are calibrated on Phase 43's
//! 1 MB enwik9 token stream (see
//! `lzr-neural/scripts/measure_lz77_potential.py`).

use std::collections::HashMap;

/// Window in tokens. 14-bit offsets address [1, 16384] back.
pub(crate) const WINDOW_LOG: u32 = 14;
pub(crate) const WINDOW: usize = 1 << WINDOW_LOG;

/// Default minimum offset (in tokens). Empirically we tested
/// `min_offset=512` (only matches outside `MoE` attention) and
/// `min_offset=1` (any distance); both shipped a net win at
/// `MIN_MATCH=16`, with no-gate slightly better (-0.0049 vs -0.0037
/// bpb on 1 MB enwik9). The `MoE`+BPE arm is good enough at predicting
/// in-attention repeats that LZ rarely catches what attention missed;
/// what helps is just rejecting matches shorter than the LZ overhead.
pub(crate) const DEFAULT_MIN_OFFSET: usize = 1;

/// Default minimum match length. Each emitted match pays
/// `flag + 14-bit offset + 8-bit length ≈ 26` bits, so the absorbed
/// tokens must collectively beat that to be a win. Empirically the
/// `MoE` arm predicts far-back repeats at ~2.5 bits/token (distributional
/// learning, not just attention memory), so `MIN_MATCH` must be high
/// enough that 26 bits beats `len * 2.5`: length 16 needed to break
/// even with comfortable margin.
pub(crate) const MIN_MATCH: usize = 16;

/// Maximum match length emitted in one LZ token. Length is encoded as
/// `(length - MIN_MATCH)` in 8 raw bits, so the inclusive upper bound
/// is `MIN_MATCH + 255 = 260`.
pub(crate) const MAX_MATCH: usize = MIN_MATCH + 255;

/// Per-key candidate cap (rough cap on search work per position).
const CHAIN_CAP: usize = 32;

#[derive(Debug)]
pub(crate) struct Lz77 {
    tokens: Vec<u32>,
    index: HashMap<(u32, u32, u32), Vec<usize>>,
    min_offset: usize,
}

impl Lz77 {
    pub(crate) fn new(min_offset: usize) -> Self {
        Self {
            tokens: Vec::new(),
            index: HashMap::new(),
            min_offset,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn token_at(&self, pos: usize) -> u32 {
        self.tokens[pos]
    }

    /// Append one token to the window and update the 3-gram index.
    pub(crate) fn push(&mut self, t: u32) {
        let pos = self.tokens.len();
        self.tokens.push(t);
        if pos >= 2 {
            let key = (self.tokens[pos - 2], self.tokens[pos - 1], t);
            let triple_pos = pos - 2;
            let entry = self.index.entry(key).or_default();
            entry.push(triple_pos);
            if entry.len() > CHAIN_CAP {
                let drop = entry.len() - CHAIN_CAP;
                entry.drain(0..drop);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn push_many(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.push(t);
        }
    }

    /// Find the longest match at the next-to-be-pushed position. The
    /// lookahead slice carries the upcoming tokens (`lookahead[0]` is
    /// the next token to encode, `lookahead[1]` follows it, etc.).
    /// Returns `Some((offset, length))` where `offset` is how many
    /// tokens back the match begins (1 = previous token) and
    /// `length ∈ [MIN_MATCH, MAX_MATCH]`. `None` if no qualifying
    /// match exists.
    pub(crate) fn longest_match(&self, lookahead: &[u32]) -> Option<(usize, usize)> {
        if lookahead.len() < 3 {
            return None;
        }
        let cur = self.tokens.len();
        if cur < 3 {
            return None;
        }
        let key = (lookahead[0], lookahead[1], lookahead[2]);
        let cands = self.index.get(&key)?;
        let mut best: Option<(usize, usize)> = None;
        for &cand_pos in cands.iter().rev() {
            let back = cur - cand_pos;
            if back > WINDOW {
                break;
            }
            if back < self.min_offset {
                continue;
            }
            let mut len = 3;
            let max = MAX_MATCH.min(lookahead.len()).min(back);
            while len < max && self.tokens[cand_pos + len] == lookahead[len] {
                len += 1;
            }
            if len >= MIN_MATCH {
                let take = best.is_none_or(|(_, blen)| len > blen);
                if take {
                    best = Some((back, len));
                    if len >= MAX_MATCH {
                        break;
                    }
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_simple_repeat() {
        let mut lz = Lz77::new(1);
        // Prefix has 11 tokens so back=11 for the (10,20,30) triple at pos 0;
        // extension matches 8 tokens (10..80). MIN_MATCH=8 is the threshold.
        let prefix = [10_u32, 20, 30, 40, 50, 60, 70, 80, 91, 92, 93];
        lz.push_many(&prefix);
        let m = lz.longest_match(&[10, 20, 30, 40, 50, 60, 70, 80, 99]);
        assert_eq!(m, Some((11, 8)));
    }

    #[test]
    fn rejects_short_matches() {
        let mut lz = Lz77::new(1);
        lz.push_many(&[1, 2, 3, 4, 5]);
        // 3-gram (1,2,3) matches but extension is only length 3 < MIN_MATCH.
        assert_eq!(lz.longest_match(&[1, 2, 3, 99, 100]), None);
    }

    #[test]
    fn respects_min_offset() {
        // Same data, but min_offset=100 prevents the back=8 match.
        let mut lz = Lz77::new(100);
        let mut block = vec![10_u32, 20, 30];
        block.extend(std::iter::repeat_n(0_u32, 50));
        lz.push_many(&block);
        assert_eq!(
            lz.longest_match(&[10, 20, 30, 0, 0, 0, 0, 0, 0, 0, 0]),
            None
        );
    }

    #[test]
    fn caps_at_max_match() {
        // Source: unique 3-gram (1,2,3) once, then a long run of 0s.
        // Only one chain entry exists for key (1,2,3), so CHAIN_CAP
        // doesn't truncate the match; extension caps only at MAX_MATCH.
        let mut lz = Lz77::new(1);
        let mut block = vec![1_u32, 2, 3];
        block.extend(std::iter::repeat_n(0_u32, MAX_MATCH + 50));
        lz.push_many(&block);
        let mut look = vec![1_u32, 2, 3];
        look.extend(std::iter::repeat_n(0_u32, MAX_MATCH + 50));
        let m = lz.longest_match(&look);
        let (_off, len) = m.expect("match");
        assert_eq!(len, MAX_MATCH);
    }

    #[test]
    fn returns_none_for_no_candidates() {
        let mut lz = Lz77::new(1);
        lz.push_many(&[1, 2, 3, 4, 5]);
        assert_eq!(lz.longest_match(&[100, 101, 102, 103, 104]), None);
    }
}
