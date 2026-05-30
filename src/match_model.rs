//! Token-level PPM-style match predictor arm for the `moe-tok` codec.
//!
//! For the last [`K`] tokens of history (the context), it gathers the
//! tokens that followed the most recent [`CHAIN_CAP`] prior occurrences
//! of that context — an LZ-style hash chain — and returns them as a
//! count-normalized distribution, plus the true match length `L` of the
//! most recent occurrence as a confidence signal. The codec mixes this
//! distribution into the `MoE` CDF via an online per-`L`-bucket logit
//! boost.
//!
//! A count *distribution* over followers (vs a single most-recent
//! follower) is the lever: vetting on the BPE-16K stream showed the
//! distribution pays ~7.37 vs 8.45 bits per match — when the top
//! follower is wrong, the actual token still usually has mass, so the
//! boost helps instead of hurting.
//!
//! Ships **no state** (L(D) = 0): both `comp9a` and `decomp9` rebuild
//! the identical chains from the identical token stream. The context
//! hash is a fixed FNV-1a polynomial (not `RandomState`) so the two
//! separate program runs agree on every lookup.

use std::collections::HashMap;

/// Hash-context length: predictions key off the last `K` tokens.
pub(crate) const K: usize = 4;
/// Confidence buckets cap at this true match length.
pub(crate) const MAX_L: usize = 16;
/// Cap on backward-extension work per prediction.
const MAX_EXT: usize = 64;
/// Prior occurrences of the context walked per prediction.
const CHAIN_CAP: usize = 16;

const SENTINEL: u32 = u32::MAX;

pub(crate) struct MatchModel {
    history: Vec<u32>,
    /// Context hash → most recent position whose preceding `K`-context
    /// hashes here (the follower is `history[pos]`).
    head: HashMap<u64, u32>,
    /// `prev[pos]` = previous position with the same context hash, or
    /// `SENTINEL`. Together with `head` this is an LZ-style hash chain.
    prev: Vec<u32>,
}

impl MatchModel {
    pub(crate) fn new() -> Self {
        Self {
            history: Vec::new(),
            head: HashMap::new(),
            prev: Vec::new(),
        }
    }

    /// Fixed FNV-1a hash of a token slice — deterministic across
    /// separate process runs (unlike `HashMap`'s `RandomState`).
    fn key(slice: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in slice {
            h ^= u64::from(t);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Fill `out` with the follower distribution `(token, weight)` for
    /// the current context (weights sum to 1), and return the true
    /// match length `L` of the most recent occurrence. `None` (with
    /// `out` cleared) when there is no `K`-context match.
    pub(crate) fn predict(&self, out: &mut Vec<(u32, f32)>) -> Option<usize> {
        out.clear();
        let n = self.history.len();
        if n < K {
            return None;
        }
        let key = Self::key(&self.history[n - K..n]);
        let head_pos = *self.head.get(&key)?;

        // Confidence: true match length of the most recent occurrence.
        let (mut ai, mut bi, mut mlen) = (n, head_pos as usize, 0usize);
        while ai > 0 && bi > 0 && mlen < MAX_EXT && self.history[ai - 1] == self.history[bi - 1] {
            mlen += 1;
            ai -= 1;
            bi -= 1;
        }

        // Gather followers over up to CHAIN_CAP prior occurrences.
        let mut q = head_pos;
        let mut steps = 0u32;
        while q != SENTINEL && (steps as usize) < CHAIN_CAP {
            let f = self.history[q as usize];
            if let Some(e) = out.iter_mut().find(|(t, _)| *t == f) {
                e.1 += 1.0;
            } else {
                out.push((f, 1.0));
            }
            q = self.prev[q as usize];
            steps += 1;
        }
        let inv = 1.0 / f32::from(u16::try_from(steps).unwrap_or(u16::MAX));
        for e in out.iter_mut() {
            e.1 *= inv;
        }
        Some(mlen)
    }

    /// Append an observed token and extend the chain for the context
    /// preceding it.
    pub(crate) fn push(&mut self, tok: u32) {
        self.history.push(tok);
        self.prev.push(SENTINEL);
        let pos = self.history.len() - 1;
        if pos >= K {
            let key = Self::key(&self.history[pos - K..pos]);
            let old = self.head.get(&key).copied().unwrap_or(SENTINEL);
            let pos_u32 = u32::try_from(pos).expect("position fits u32");
            self.prev[pos] = old;
            self.head.insert(key, pos_u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(m: &MatchModel) -> Option<(Vec<(u32, f32)>, usize)> {
        let mut out = Vec::new();
        m.predict(&mut out).map(|l| (out, l))
    }

    #[test]
    fn predicts_follower_distribution_of_a_repeated_context() {
        let mut m = MatchModel::new();
        for &t in &[10u32, 20, 30, 40, 99, 7, 8, 9, 10, 20, 30, 40] {
            m.push(t);
        }
        let (out, l) = dist(&m).expect("a K-context match exists");
        // Only one prior occurrence of "10 20 30 40", followed by 99.
        assert_eq!(out, vec![(99, 1.0)]);
        assert!(l >= K, "match length must be at least K, got {l}");
    }

    #[test]
    fn aggregates_followers_across_occurrences() {
        // context "1 2 3 4" occurs twice, followed by 7 then 8.
        let mut m = MatchModel::new();
        for &t in &[1u32, 2, 3, 4, 7, 9, 1, 2, 3, 4, 8, 9, 1, 2, 3, 4] {
            m.push(t);
        }
        let (mut out, _) = dist(&m).expect("match exists");
        out.sort_by_key(|&(t, _)| t);
        // Followers were 8 (more recent) and 7; each weight 0.5.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 7);
        assert_eq!(out[1].0, 8);
        assert!((out[0].1 - 0.5).abs() < 1e-6 && (out[1].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn no_prediction_without_a_match() {
        let mut m = MatchModel::new();
        for &t in &[1u32, 2, 3, 4, 5, 6, 7, 8] {
            m.push(t);
        }
        let mut out = vec![(123, 1.0)];
        assert!(m.predict(&mut out).is_none());
        assert!(out.is_empty(), "out must be cleared on no-match");
    }

    #[test]
    fn weights_sum_to_one() {
        let mut m = MatchModel::new();
        let pat = [5u32, 6, 7, 8];
        for rep in 0..5u32 {
            for &t in &pat {
                m.push(t);
            }
            m.push(100 + rep); // distinct follower each time
        }
        for &t in &pat {
            m.push(t);
        }
        let (out, _) = dist(&m).expect("match exists");
        let s: f32 = out.iter().map(|&(_, w)| w).sum();
        assert!((s - 1.0).abs() < 1e-5, "weights should sum to 1, got {s}");
    }
}
