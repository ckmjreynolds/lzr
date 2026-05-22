//! Token-level LZ77 longest-match finder.
//!
//! Maintains a window of recently-seen tokens plus a 3-gram hash index
//! pointing into that window. Used by the v4 `moe-tok-lz` codec to
//! catch repeated runs that exceed the `MoE` arm's attention context.
//!
//! `MIN_MATCH` / `MAX_MATCH` constants are calibrated on Phase 43's
//! 1 MB enwik9 token stream (see
//! `lzr-neural/scripts/measure_lz77_potential.py`).
//!
//! Storage is a ring buffer of `WINDOW` tokens plus a 3-gram hash
//! index whose chains are capped at `CHAIN_CAP` entries and pruned
//! periodically to drop empty keys. Memory use is bounded at ~1 MB
//! regardless of how many tokens have been pushed, vs the unbounded
//! `Vec<u32>` + `HashMap` of the original implementation that grew to
//! ~5 GB on a full enwik9 encode.

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

/// How often (in pushes) to run the index-key-eviction sweep. Each
/// sweep is `O(n_keys)`, so we don't want to do it on every push — but
/// the longer we wait the more dead-key memory accumulates. Setting
/// this to `WINDOW` means the sweep amortizes to `O(1)` per push and
/// peak dead-key memory is bounded at ~one window's worth.
const CLEANUP_INTERVAL: usize = WINDOW;

#[derive(Debug)]
pub(crate) struct Lz77 {
    /// Ring buffer of the last `WINDOW` tokens. Position `p` lives at
    /// `ring[p % WINDOW]` and is only valid while `cur_pos - p <= WINDOW`.
    ring: Vec<u32>,
    /// Monotonic count of tokens pushed since `new`. Never wraps.
    cur_pos: usize,
    /// Maps a 3-gram to absolute positions where it appears. Chains
    /// are capped at `CHAIN_CAP` (newest kept) and stale entries
    /// (older than `cur_pos - WINDOW`) are dropped by the periodic
    /// sweep in `push`.
    index: HashMap<(u32, u32, u32), Vec<usize>>,
    min_offset: usize,
}

impl Lz77 {
    pub(crate) fn new(min_offset: usize) -> Self {
        Self {
            ring: vec![0; WINDOW],
            cur_pos: 0,
            index: HashMap::new(),
            min_offset,
        }
    }

    /// Returns the current ring + index size in bytes (rough estimate
    /// for ad-hoc memory checks; counts allocated capacity, not just
    /// occupied slots).
    #[cfg(test)]
    pub(crate) fn approx_bytes(&self) -> usize {
        let ring_bytes = self.ring.capacity() * size_of::<u32>();
        let mut chain_bytes = 0;
        for chain in self.index.values() {
            chain_bytes += chain.capacity() * size_of::<usize>();
        }
        let map_overhead = self.index.capacity() * 64;
        ring_bytes + chain_bytes + map_overhead
    }

    /// Total tokens ever pushed. The ring only holds the last
    /// `WINDOW` of them; older positions are not addressable.
    pub(crate) const fn len(&self) -> usize {
        self.cur_pos
    }

    /// Read a token at absolute position `pos`. The caller must keep
    /// `pos` inside `[cur_pos - WINDOW, cur_pos)` — readers outside
    /// that range read whatever stale value the ring currently holds
    /// at that slot, which would be a silent correctness bug. The
    /// `moe-tok-lz` codec is the only caller and clamps its reads via
    /// the encoder's `back <= WINDOW` invariant.
    pub(crate) fn token_at(&self, pos: usize) -> u32 {
        debug_assert!(pos < self.cur_pos, "read above cur_pos");
        debug_assert!(
            self.cur_pos - pos <= WINDOW,
            "read outside ring window: cur_pos={} pos={}",
            self.cur_pos,
            pos
        );
        self.ring[pos % WINDOW]
    }

    /// Append one token to the ring and update the 3-gram index.
    /// Runs an index-key sweep every `CLEANUP_INTERVAL` pushes.
    pub(crate) fn push(&mut self, t: u32) {
        self.ring[self.cur_pos % WINDOW] = t;
        self.cur_pos += 1;

        if self.cur_pos >= 3 {
            let triple_start = self.cur_pos - 3;
            let key = (
                self.ring[triple_start % WINDOW],
                self.ring[(triple_start + 1) % WINDOW],
                self.ring[(triple_start + 2) % WINDOW],
            );
            let entry = self.index.entry(key).or_default();
            entry.push(triple_start);
            if entry.len() > CHAIN_CAP {
                let drop = entry.len() - CHAIN_CAP;
                entry.drain(0..drop);
            }
        }

        if self.cur_pos % CLEANUP_INTERVAL == 0 {
            self.sweep_stale_entries();
        }
    }

    /// Drop chain entries whose absolute position is older than
    /// `cur_pos - WINDOW` (unreachable by `longest_match` regardless),
    /// then remove any keys whose chain became empty. Bounds the
    /// `HashMap` key count at roughly the number of distinct 3-grams
    /// seen in the last window.
    fn sweep_stale_entries(&mut self) {
        let cutoff = self.cur_pos.saturating_sub(WINDOW);
        self.index.retain(|_, chain| {
            while let Some(&head) = chain.first() {
                if head < cutoff {
                    chain.remove(0);
                } else {
                    break;
                }
            }
            !chain.is_empty()
        });
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
        let cur = self.cur_pos;
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
            while len < max && self.token_at(cand_pos + len) == lookahead[len] {
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
        // Prefix length must exceed MIN_MATCH so the extension can reach it.
        // We lay down the canary [100..(100+MIN_MATCH+1)] then a sentinel,
        // then look for the canary again: extension matches MIN_MATCH+1 tokens
        // (everything up to but not including the sentinel mismatch).
        let len_canary = MIN_MATCH + 1;
        let mut prefix: Vec<u32> = (100..100 + u32::try_from(len_canary).unwrap()).collect();
        prefix.push(7_777); // sentinel terminator
        lz.push_many(&prefix);
        let mut look: Vec<u32> = (100..100 + u32::try_from(len_canary).unwrap()).collect();
        look.push(9_999); // different sentinel so the match stops at len_canary
        let m = lz.longest_match(&look);
        let expected_back = prefix.len();
        assert_eq!(m, Some((expected_back, len_canary)));
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

    /// Push more than 4×`WINDOW` tokens to exercise the ring rollover
    /// plus several index-sweep cycles, then verify a match recorded
    /// inside the most recent window is still findable. Catches off-by-
    /// one errors in the sweep cutoff or ring-modulo arithmetic.
    #[test]
    fn ring_rolls_over_and_finds_recent_match() {
        let mut lz = Lz77::new(1);
        // Phase 1: fill with WINDOW + 100 distinct triples so the ring
        // and index roll over at least once.
        for i in 0..(u32::try_from(WINDOW).unwrap() + 100) {
            lz.push(i);
            lz.push(i.wrapping_add(1));
            lz.push(i.wrapping_add(2));
        }
        // Phase 2: lay down a known long-match pattern at a recent
        // position, then push more filler.
        let pattern: Vec<u32> =
            (1_000_000..1_000_000 + u32::try_from(MIN_MATCH).unwrap() + 4).collect();
        for &t in &pattern {
            lz.push(t);
        }
        // A small amount of filler so the pattern is still inside the window.
        for j in 0..50u32 {
            lz.push(2_000_000 + j);
        }
        // Looking up the pattern should still find it.
        let m = lz.longest_match(&pattern);
        let (off, len) = m.expect("match within window");
        assert!(len >= MIN_MATCH, "expected len>=MIN_MATCH, got {len}");
        // back = how many tokens between the pattern's start and cur.
        assert!(off <= WINDOW, "back must stay within WINDOW, got {off}");
    }

    /// Push enough tokens to trigger many cleanup sweeps and confirm
    /// the index size stays bounded. Without the sweep, the `HashMap`
    /// would grow to ~3× `WINDOW` keys here.
    #[test]
    fn index_stays_bounded_under_many_pushes() {
        let mut lz = Lz77::new(1);
        // 8 × WINDOW pushes with mostly-distinct 3-grams.
        for i in 0..(8 * u32::try_from(WINDOW).unwrap()) {
            lz.push(i);
            lz.push(i ^ 0xdead_beef);
            lz.push(i.wrapping_mul(2_654_435_761));
        }
        // After many sweeps, the index should hold at most ~one
        // window's worth of distinct keys. Allow 2× headroom to
        // tolerate the dead-key buildup between sweeps.
        assert!(
            lz.index.len() <= 2 * WINDOW,
            "index unbounded: {} keys (window={WINDOW})",
            lz.index.len()
        );
        // Approx memory should be well under 4 MB — without the
        // sweep + ring this would be ~750 MB for 24×WINDOW pushes.
        let bytes = lz.approx_bytes();
        assert!(bytes < 4 * 1024 * 1024, "Lz77 memory unbounded: {bytes} bytes");
    }
}
