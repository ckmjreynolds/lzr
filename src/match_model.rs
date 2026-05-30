//! Token-level variable-order PPM match predictor arm for the `moe-tok`
//! codec.
//!
//! For each order in [`ORDERS`] (longest first), it keeps an LZ-style
//! hash chain over the last `k` tokens. At prediction it uses the
//! **longest** context that has a prior match (PPM\* back-off), gathers
//! the tokens that followed its most recent [`CHAIN_CAP`] occurrences as
//! a count-normalized distribution, and reports the true match length
//! `L` (backward extension) as confidence. The codec mixes this
//! distribution into the `MoE` CDF via an online per-`L`-bucket logit
//! boost — so short back-off matches (L=2,3) get their own low-weight
//! bucket that the mixer drives toward 0 if they don't help, while long
//! matches (L≥4) get strongly boosted.
//!
//! A count *distribution* over followers (not a single most-recent one)
//! is the key lever: vetting on the BPE-16K stream showed it pays ~7.37
//! vs 8.45 bits per match — when the top follower is wrong, the actual
//! token usually still has mass, so the boost helps instead of hurting.
//!
//! Ships **no state** (L(D) = 0): both `comp9a` and `decomp9` rebuild
//! the identical chains from the identical token stream. The context
//! hash is a fixed FNV-1a polynomial (not `RandomState`) so the two
//! separate program runs agree on every lookup.

use std::collections::HashMap;

/// Context lengths tried, longest first (PPM\* back-off order).
pub(crate) const ORDERS: [usize; 3] = [8, 4, 2];
const N_ORDERS: usize = ORDERS.len();
/// Confidence buckets cap at this true match length.
pub(crate) const MAX_L: usize = 16;
/// Cap on backward-extension work per prediction.
const MAX_EXT: usize = 64;
/// Prior occurrences of the context walked per prediction.
const CHAIN_CAP: usize = 16;

const SENTINEL: u32 = u32::MAX;

/// One LZ-style hash chain: `head[context_hash]` = most recent position
/// with that context, `prev[pos]` = the previous such position.
struct Chain {
    head: HashMap<u64, u32>,
    prev: Vec<u32>,
}

impl Chain {
    fn new() -> Self {
        Self {
            head: HashMap::new(),
            prev: Vec::new(),
        }
    }
}

pub(crate) struct MatchModel {
    history: Vec<u32>,
    chains: [Chain; N_ORDERS],
}

impl MatchModel {
    pub(crate) fn new() -> Self {
        Self {
            history: Vec::new(),
            chains: std::array::from_fn(|_| Chain::new()),
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

    /// True match length of the most recent occurrence at `head_pos`:
    /// the matching suffix ending at `n-1` vs `head_pos-1`.
    fn match_len(&self, n: usize, head_pos: usize) -> usize {
        let (mut ai, mut bi, mut mlen) = (n, head_pos, 0usize);
        while ai > 0 && bi > 0 && mlen < MAX_EXT && self.history[ai - 1] == self.history[bi - 1] {
            mlen += 1;
            ai -= 1;
            bi -= 1;
        }
        mlen
    }

    /// Walk chain `ci` from `head_pos` and fill `out` with the
    /// count-normalized follower distribution.
    fn gather(&self, ci: usize, head_pos: u32, out: &mut Vec<(u32, f32)>) {
        let mut q = head_pos;
        let mut steps = 0u32;
        while q != SENTINEL && (steps as usize) < CHAIN_CAP {
            let f = self.history[q as usize];
            if let Some(e) = out.iter_mut().find(|(t, _)| *t == f) {
                e.1 += 1.0;
            } else {
                out.push((f, 1.0));
            }
            q = self.chains[ci].prev[q as usize];
            steps += 1;
        }
        let inv = 1.0 / f32::from(u16::try_from(steps).unwrap_or(u16::MAX));
        for e in out.iter_mut() {
            e.1 *= inv;
        }
    }

    /// Fill `out` with the follower distribution `(token, weight)` for
    /// the longest matching context (weights sum to 1), and return its
    /// true match length `L`. `None` (with `out` cleared) when no order
    /// has a match.
    pub(crate) fn predict(&self, out: &mut Vec<(u32, f32)>) -> Option<usize> {
        out.clear();
        let n = self.history.len();
        for (ci, &k) in ORDERS.iter().enumerate() {
            if n < k {
                continue;
            }
            let key = Self::key(&self.history[n - k..n]);
            if let Some(&head_pos) = self.chains[ci].head.get(&key) {
                let l = self.match_len(n, head_pos as usize);
                self.gather(ci, head_pos, out);
                return Some(l);
            }
        }
        None
    }

    /// Append an observed token and extend every order's chain for the
    /// context preceding it.
    pub(crate) fn push(&mut self, tok: u32) {
        self.history.push(tok);
        let pos = self.history.len() - 1;
        let pos_u32 = u32::try_from(pos).expect("position fits u32");
        for (ci, &k) in ORDERS.iter().enumerate() {
            self.chains[ci].prev.push(SENTINEL);
            if pos >= k {
                let key = Self::key(&self.history[pos - k..pos]);
                let old = self.chains[ci].head.get(&key).copied().unwrap_or(SENTINEL);
                self.chains[ci].prev[pos] = old;
                self.chains[ci].head.insert(key, pos_u32);
            }
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
    fn uses_longest_context_match() {
        let mut m = MatchModel::new();
        for &t in &[10u32, 20, 30, 40, 99, 7, 8, 9, 10, 20, 30, 40] {
            m.push(t);
        }
        let (out, l) = dist(&m).expect("a context match exists");
        assert_eq!(out, vec![(99, 1.0)]);
        assert!(l >= 4, "longest match (order-4) should give L>=4, got {l}");
    }

    #[test]
    fn aggregates_followers_across_occurrences() {
        let mut m = MatchModel::new();
        for &t in &[1u32, 2, 3, 4, 7, 9, 1, 2, 3, 4, 8, 9, 1, 2, 3, 4] {
            m.push(t);
        }
        let (mut out, _) = dist(&m).expect("match exists");
        out.sort_by_key(|&(t, _)| t);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, 7);
        assert_eq!(out[1].0, 8);
        assert!((out[0].1 - 0.5).abs() < 1e-6 && (out[1].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn backs_off_to_shorter_order() {
        // No 4-context repeats, but the 2-context "50 60" recurs (once
        // followed by 70). Order-4 misses; order-2 catches it.
        let mut m = MatchModel::new();
        for &t in &[50u32, 60, 70, 1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 60] {
            m.push(t);
        }
        let (out, l) = dist(&m).expect("order-2 back-off match exists");
        assert_eq!(out, vec![(70, 1.0)]);
        assert!(l < 4, "back-off match length should be < 4, got {l}");
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
            m.push(100 + rep);
        }
        for &t in &pat {
            m.push(t);
        }
        let (out, _) = dist(&m).expect("match exists");
        let s: f32 = out.iter().map(|&(_, w)| w).sum();
        assert!((s - 1.0).abs() < 1e-5, "weights should sum to 1, got {s}");
    }
}
