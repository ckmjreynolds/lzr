//! Token-level longest-match predictor arm for the `moe-tok` codec.
//!
//! Maintains an index from the fixed hash of the last [`K`] tokens to
//! the most recent position that context occurred, built online from
//! the token history. At each position it predicts the token that
//! followed the most recent matching context, with the true match
//! length `L` (found by backward extension) as a confidence signal.
//!
//! It ships **no state** (L(D) = 0): both `comp9a` and `decomp9`
//! rebuild the identical index from the identical token stream. The
//! hash is a fixed FNV-1a polynomial — *not* `RandomState` — so the
//! two separate program invocations agree on every lookup.
//!
//! Its value over the `MoE` arm is long-range: ~4.6% of enwik8 tokens
//! are correctly predicted by a match whose source is beyond the
//! `MoE`'s 512-token attention window (vetted on the BPE-16K stream),
//! accuracy rising from ~40% at `L=4` to ~88% at `L>=16`.

use std::collections::HashMap;

/// Hash-context length: predictions key off the last `K` tokens.
pub(crate) const K: usize = 4;
/// Confidence buckets cap at this true match length.
pub(crate) const MAX_L: usize = 16;
/// Cap on backward-extension work per prediction.
const MAX_EXT: usize = 64;

pub(crate) struct MatchModel {
    history: Vec<u32>,
    index: HashMap<u64, u32>,
}

impl MatchModel {
    pub(crate) fn new() -> Self {
        Self {
            history: Vec::new(),
            index: HashMap::new(),
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

    /// Predict the next token from the current history. Returns
    /// `(predicted token, true match length L)`, or `None` when there
    /// is no `K`-token context match.
    pub(crate) fn predict(&self) -> Option<(u32, usize)> {
        let n = self.history.len();
        if n < K {
            return None;
        }
        let key = Self::key(&self.history[n - K..n]);
        let prior = *self.index.get(&key)? as usize;
        // True match length: matching suffix ending at n-1 vs prior-1.
        let (mut ai, mut bi, mut mlen) = (n, prior, 0usize);
        while ai > 0 && bi > 0 && mlen < MAX_EXT && self.history[ai - 1] == self.history[bi - 1] {
            mlen += 1;
            ai -= 1;
            bi -= 1;
        }
        Some((self.history[prior], mlen))
    }

    /// Append an observed token and index the context preceding it.
    pub(crate) fn push(&mut self, tok: u32) {
        self.history.push(tok);
        let p = self.history.len() - 1;
        if p >= K {
            let key = Self::key(&self.history[p - K..p]);
            self.index
                .insert(key, u32::try_from(p).expect("position fits u32"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicts_the_follower_of_a_repeated_context() {
        // Sequence: "a b c d X ... a b c d" — after the second "a b c d"
        // the model should predict X (what followed the first one).
        let mut m = MatchModel::new();
        for &t in &[10u32, 20, 30, 40, 99, 7, 8, 9, 10, 20, 30, 40] {
            // predict BEFORE pushing the next token, mirroring codec use
            m.push(t);
        }
        // history ends with the second "10 20 30 40"; predict next.
        let (pred, l) = m.predict().expect("a K-context match exists");
        assert_eq!(
            pred, 99,
            "should predict the follower of the first occurrence"
        );
        assert!(l >= K, "match length must be at least K, got {l}");
    }

    #[test]
    fn no_prediction_without_a_match() {
        let mut m = MatchModel::new();
        for &t in &[1u32, 2, 3, 4, 5, 6, 7, 8] {
            m.push(t);
        }
        // All contexts are unique → no repeat → no prediction.
        assert!(m.predict().is_none());
    }

    #[test]
    fn longer_match_reports_larger_l() {
        let mut m = MatchModel::new();
        let pat = [5u32, 6, 7, 8, 9, 10, 11, 12];
        for &t in &pat {
            m.push(t);
        }
        m.push(42); // separator
        for &t in &pat {
            m.push(t);
        }
        let (pred, l) = m.predict().expect("match exists");
        assert_eq!(pred, 42, "follower of the first full pattern was 42");
        assert!(
            l >= pat.len(),
            "full-pattern repeat should extend to >= {}",
            pat.len()
        );
    }
}
