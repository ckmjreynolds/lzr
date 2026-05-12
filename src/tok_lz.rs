//! Token-level LZ77 matcher for the Phase-16 xml-tok extension.
//!
//! Mirrors `src/lz.rs`'s structure but indexes on 16-bit token IDs
//! rather than bytes. Hash3 → 256 K buckets, hash-chain prev links
//! over a 1 M-token sliding window. Min match is 3 tokens (the cost
//! threshold for the match record to amortize: ~17 bits of flag +
//! bucket + length + raw-bits vs. ~7-8 bits per Order-1 token hit).
//!
//! The matcher only sees tokens that have a dictionary id — both
//! hits and OOVs that successfully inserted. Tokens that didn't
//! make it into the dictionary (capacity-full corner case) break
//! any match passing through that position.

use crate::bit_pred::{FNV_OFFSET, fnv_mix};

const WINDOW_LOG: u32 = 20;
/// 1 M tokens of lookback; enough to span the entire 4 MiB warm
/// prefix plus a 256 KiB measure window in token units.
pub(crate) const WINDOW_SIZE: usize = 1 << WINDOW_LOG;
const WINDOW_MASK: usize = WINDOW_SIZE - 1;

const HASH_LOG: u32 = 18;
const HASH_SIZE: usize = 1 << HASH_LOG;
const HASH_MASK: u64 = (HASH_SIZE - 1) as u64;

/// Min match length in tokens. Below 3, the match record cost
/// (~17 bits: 1-bit flag + 5-bit bucket + log₂(offset) raw bits +
/// length token) exceeds 3 tokens' worth of Order-1 hit cost.
pub(crate) const MIN_MATCH: usize = 3;
pub(crate) const MAX_MATCH: usize = MIN_MATCH + 255;
const CHAIN_DEPTH: usize = 32;

const NIL: u32 = u32::MAX;

pub(crate) struct TokenMatcher {
    /// Per-bucket head of the hash chain; points to the most-recent
    /// position whose 3-token prefix hashes here, or `NIL` if empty.
    hash: Vec<u32>,
    /// Ring buffer of previous-same-hash positions, indexed by
    /// `pos & WINDOW_MASK`. Walks back through the chain.
    prev: Vec<u32>,
    /// All token ids emitted to the Content stream so far, in
    /// emission order. Both warm and measure tokens land here; the
    /// matcher walks back up to `WINDOW_SIZE` positions.
    stream: Vec<u32>,
}

impl TokenMatcher {
    pub(crate) fn new() -> Self {
        Self {
            hash: vec![NIL; HASH_SIZE],
            prev: vec![NIL; WINDOW_SIZE],
            stream: Vec::new(),
        }
    }

    /// Number of tokens currently in the stream.
    pub(crate) fn stream_len(&self) -> usize {
        self.stream.len()
    }

    /// Token id at stream position `pos`.
    pub(crate) fn at(&self, pos: usize) -> u32 {
        self.stream[pos]
    }

    /// Append a token id to the stream. After the third token,
    /// every appended token closes a `MIN_MATCH`-sized window whose
    /// hash is inserted at the head of its bucket's chain.
    pub(crate) fn push(&mut self, id: u32) {
        self.stream.push(id);
        let n = self.stream.len();
        if n >= MIN_MATCH {
            let start = n - MIN_MATCH;
            let h = self.hash_at(start);
            let bucket = (h & HASH_MASK) as usize;
            let prev_pos = self.hash[bucket];
            self.prev[start & WINDOW_MASK] = prev_pos;
            self.hash[bucket] = u32::try_from(start).expect("stream position fits u32");
        }
    }

    fn hash_at(&self, pos: usize) -> u64 {
        let mut h = FNV_OFFSET;
        for i in 0..MIN_MATCH {
            h = fnv_mix(h, u64::from(self.stream[pos + i]));
        }
        h
    }

    fn hash_slice(slice: &[u32]) -> u64 {
        debug_assert!(slice.len() >= MIN_MATCH);
        let mut h = FNV_OFFSET;
        for &t in &slice[..MIN_MATCH] {
            h = fnv_mix(h, u64::from(t));
        }
        h
    }

    /// Find the longest match for `upcoming[0..MAX_MATCH]` against
    /// the stream's history. Returns `(offset_in_tokens, length)`
    /// or `None` when no `MIN_MATCH`-clearing match exists within
    /// the `WINDOW_SIZE` lookback.
    pub(crate) fn find_match(&self, upcoming: &[u32]) -> Option<(u32, u32)> {
        if upcoming.len() < MIN_MATCH {
            return None;
        }
        let h = Self::hash_slice(upcoming);
        let bucket = (h & HASH_MASK) as usize;
        let cur_pos = self.stream.len();
        let mut cand = self.hash[bucket];
        let mut best: Option<(u32, u32)> = None;
        let mut depth = 0;
        while cand != NIL && depth < CHAIN_DEPTH {
            let cand_pos = cand as usize;
            if cur_pos - cand_pos > WINDOW_SIZE {
                break;
            }
            if cand_pos + MIN_MATCH <= self.stream.len() {
                // Verify the MIN_MATCH prefix (defends against hash
                // collisions across distinct token triplets).
                let matches = self.stream[cand_pos..cand_pos + MIN_MATCH] == upcoming[..MIN_MATCH];
                if matches {
                    // Extend up to MAX_MATCH or until streams diverge,
                    // bounded by both lookahead size AND remaining
                    // stream from the candidate position.
                    let mut len = MIN_MATCH;
                    let cap = upcoming
                        .len()
                        .min(MAX_MATCH)
                        .min(self.stream.len() - cand_pos);
                    while len < cap && self.stream[cand_pos + len] == upcoming[len] {
                        len += 1;
                    }
                    let offset = u32::try_from(cur_pos - cand_pos).expect("offset fits u32");
                    let length = u32::try_from(len).expect("length fits u32");
                    if best.is_none_or(|(_, bl)| length > bl) {
                        best = Some((offset, length));
                    }
                }
            }
            cand = self.prev[cand_pos & WINDOW_MASK];
            depth += 1;
        }
        best
    }
}

impl Default for TokenMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stream_no_match() {
        let m = TokenMatcher::new();
        assert!(m.find_match(&[1, 2, 3]).is_none());
    }

    #[test]
    fn finds_simple_repeat() {
        let mut m = TokenMatcher::new();
        for id in [1u32, 2, 3, 4, 5] {
            m.push(id);
        }
        // Looking for "1 2 3 4 ..." should find offset 5 (back to the start).
        let r = m.find_match(&[1, 2, 3, 4, 9]);
        let (offset, length) = r.expect("match");
        assert_eq!(offset, 5);
        assert_eq!(length, 4);
    }

    #[test]
    fn no_match_below_min() {
        let mut m = TokenMatcher::new();
        for id in [1u32, 2, 3, 4, 5] {
            m.push(id);
        }
        // "1 2 9 ..." has only 2 matching tokens, below MIN_MATCH=3.
        assert!(m.find_match(&[1, 2, 9, 8]).is_none());
    }

    #[test]
    fn longer_match_preferred() {
        let mut m = TokenMatcher::new();
        // Two earlier matches of "1 2 3": one extends 3 tokens, one extends 5.
        for id in [1u32, 2, 3, 99, 99] {
            m.push(id);
        }
        for id in [1u32, 2, 3, 4, 5] {
            m.push(id);
        }
        for id in [77u32, 77, 77] {
            m.push(id);
        }
        // Now look for "1 2 3 4 5 ..." — the second occurrence is 5 long.
        let (offset, length) = m.find_match(&[1, 2, 3, 4, 5, 88]).expect("match");
        assert_eq!(length, 5);
        assert_eq!(offset, 8); // 3 (77 77 77) + 5 (1 2 3 4 5)
    }
}
