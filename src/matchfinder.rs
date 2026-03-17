//! LZ77 match finders.

use std::collections::{BTreeMap, HashMap};

use smallvec::SmallVec;

use crate::WINDOW_SIZE;
use crate::frame::Frame;
use crate::options::EncodeOptions;
use crate::ringbuf::RingBuf;

/// Level-dependent tuning parameters.
#[derive(Debug, Clone, Copy)]
struct LevelParams {
    hash_bits: u32,
    max_chain: usize,
    min_match_len: u16,
    short_scan_dist: u16,
    try_reverse: bool,
    lazy_threshold: isize,
}

const LEVEL_PARAMS: [LevelParams; 10] = [
    // [0] unused sentinel
    LevelParams {
        hash_bits: 13,
        max_chain: 64,
        min_match_len: 3,
        short_scan_dist: 0,
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [1]
    LevelParams {
        hash_bits: 13,
        max_chain: 64,
        min_match_len: 3,
        short_scan_dist: 0,
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [2]
    LevelParams {
        hash_bits: 14,
        max_chain: 128,
        min_match_len: 3,
        short_scan_dist: 0,
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [3]
    LevelParams {
        hash_bits: 14,
        max_chain: 256,
        min_match_len: 3,
        short_scan_dist: 0,
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [4]
    LevelParams {
        hash_bits: 15,
        max_chain: 512,
        min_match_len: 2,
        short_scan_dist: 64,
        try_reverse: false,
        lazy_threshold: 1,
    },
    // [5]
    LevelParams {
        hash_bits: 15,
        max_chain: 1024,
        min_match_len: 2,
        short_scan_dist: 64,
        try_reverse: false,
        lazy_threshold: 1,
    },
    // [6]
    LevelParams {
        hash_bits: 16,
        max_chain: 2048,
        min_match_len: 2,
        short_scan_dist: 128,
        try_reverse: false,
        lazy_threshold: 2,
    },
    // [7]
    LevelParams {
        hash_bits: 16,
        max_chain: 4096,
        min_match_len: 1,
        short_scan_dist: 128,
        try_reverse: true,
        lazy_threshold: 2,
    },
    // [8]
    LevelParams {
        hash_bits: 16,
        max_chain: 8192,
        min_match_len: 1,
        short_scan_dist: 256,
        try_reverse: true,
        lazy_threshold: 3,
    },
    // [9]
    LevelParams {
        hash_bits: 16,
        max_chain: usize::MAX,
        min_match_len: 1,
        short_scan_dist: 512,
        try_reverse: true,
        lazy_threshold: 4,
    },
];

const WINDOW_MASK: usize = WINDOW_SIZE - 1;
const EMPTY: u32 = u32::MAX;

/// Multiplicative hash on 3 bytes (Knuth).
#[inline]
fn hash3(b0: u8, b1: u8, b2: u8, hash_bits: u32) -> usize {
    let h = u32::from(b0) | (u32::from(b1) << 8) | (u32::from(b2) << 16);
    (h.wrapping_mul(2_654_435_761) >> (32 - hash_bits)) as usize
}

/// Hash-chain based LZ77 match finder.
pub(crate) struct MatchFinder {
    params: LevelParams,
    head: Vec<u32>,
    prev: Vec<u32>,
    rev_head: Vec<u32>,
    rev_prev: Vec<u32>,
    hash_mask: usize,
}

impl MatchFinder {
    /// Construct from encoder options.
    pub(crate) fn new(options: &EncodeOptions) -> Self {
        let level = options.get_level().clamp(1, 9);
        let params = LEVEL_PARAMS[level];
        let hash_size = 1 << params.hash_bits;
        let hash_mask = hash_size - 1;

        let (rev_head, rev_prev) = if params.try_reverse {
            (vec![EMPTY; hash_size], vec![EMPTY; WINDOW_SIZE])
        } else {
            (Vec::new(), Vec::new())
        };

        Self {
            params,
            head: vec![EMPTY; hash_size],
            prev: vec![EMPTY; WINDOW_SIZE],
            rev_head,
            rev_prev,
            hash_mask,
        }
    }

    /// Index positions `start..end` into hash chains.
    ///
    /// Clears and rebuilds all chains. Call once per batch after loading data.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn build(&mut self, window: &RingBuf, start: isize, end: isize) {
        self.head.fill(EMPTY);
        self.prev.fill(EMPTY);
        if self.params.try_reverse {
            self.rev_head.fill(EMPTY);
            self.rev_prev.fill(EMPTY);
        }
        for pos in start..end {
            self.insert(window, pos);
        }
    }

    /// Insert position into forward (and optionally reverse) hash chains.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn insert(&mut self, window: &RingBuf, pos: isize) {
        // Forward chain: hash on bytes at pos, pos+1, pos+2.
        // We need at least 3 bytes ahead to hash (but we always insert even at boundaries;
        // the search side handles short remaining).
        if pos >= 0 {
            let b0 = window[pos];
            let b1 = window[pos + 1];
            let b2 = window[pos + 2];
            let h = hash3(b0, b1, b2, self.params.hash_bits) & self.hash_mask;
            let slot = pos as usize & WINDOW_MASK;
            self.prev[slot] = self.head[h];
            self.head[h] = pos as u32;

            // Reverse chain: hash on bytes at pos, pos-1, pos-2.
            if self.params.try_reverse && pos >= 2 {
                let rb0 = window[pos];
                let rb1 = window[pos - 1];
                let rb2 = window[pos - 2];
                let rh = hash3(rb0, rb1, rb2, self.params.hash_bits) & self.hash_mask;
                self.rev_prev[slot] = self.rev_head[rh];
                self.rev_head[rh] = pos as u32;
            }
        }
    }

    /// Insert positions `start..end` without searching (after emitting a match).
    pub(crate) fn bulk_insert(&mut self, window: &RingBuf, start: isize, end: isize) {
        for p in start..end {
            self.insert(window, p);
        }
    }

    /// Lazy matching threshold from `LevelParams`.
    pub(crate) const fn lazy_threshold(&self) -> isize {
        self.params.lazy_threshold
    }

    /// Returns `(0, 0)` — hash-chain finder doesn't track `SmallVec` lengths.
    #[allow(clippy::unused_self)]
    pub(crate) const fn max_positions_used(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Find the best profitable match at `pos`. Returns `(distance, length)` or `None`.
    ///
    /// Uses `window.write_pos()` to reject hash chain candidates whose
    /// slots have been overwritten by newer data in the ring buffer.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    pub(crate) fn find_best_match(&self, window: &RingBuf, pos: isize, bytes_remaining: usize) -> Option<(u16, i16)> {
        if bytes_remaining == 0 {
            return None;
        }

        let mut best_dist: u16 = 0;
        let mut best_len: i16 = 0;
        let mut best_gain: isize = -1;

        let max_forward = bytes_remaining.min(i16::MAX as usize);
        let wp = window.write_pos() as u32;

        // 1. Forward hash chain (3+ bytes) — searched first since it dominates compression.
        if max_forward >= 3 && pos >= 0 {
            let b0 = window[pos];
            let b1 = window[pos + 1];
            let b2 = window[pos + 2];
            let h = hash3(b0, b1, b2, self.params.hash_bits) & self.hash_mask;

            let mut candidate = self.head[h];
            let pos_u32 = pos as u32;
            let mut steps = 0;
            let mut best_fwd_len: usize = 2; // Track best forward length for early rejection.

            while candidate != EMPTY && steps < self.params.max_chain {
                let dist_u32 = pos_u32.wrapping_sub(candidate);
                // Reject if same position, too far for u16, or slot overwritten by ring buffer.
                if dist_u32 == 0 || dist_u32 > WINDOW_SIZE as u32 || wp.wrapping_sub(candidate) >= WINDOW_SIZE as u32 {
                    candidate = self.prev[candidate as usize & WINDOW_MASK];
                    steps += 1;
                    continue;
                }

                let cand_pos = pos - dist_u32 as isize;
                let dist = dist_u32 as u16;

                // Quick reject: check byte at current best length first.
                // If it doesn't match, this candidate can't beat the current best.
                if window[cand_pos + best_fwd_len as isize] != window[pos + best_fwd_len as isize] {
                    candidate = self.prev[candidate as usize & WINDOW_MASK];
                    steps += 1;
                    continue;
                }

                // Verify first 3 bytes (hash collision check).
                if window[cand_pos] == b0 && window[cand_pos + 1] == b1 && window[cand_pos + 2] == b2 {
                    // Extend forward using bulk comparison.
                    let len = 3 + window.match_length(cand_pos + 3, pos + 3, max_forward - 3);

                    let fwd_len = len.min(i16::MAX as usize) as i16;
                    let gain = Frame::match_gain(dist, fwd_len);
                    if gain > best_gain || (gain == best_gain && dist < best_dist) {
                        best_dist = dist;
                        best_len = fwd_len;
                        best_gain = gain;
                        best_fwd_len = len;
                    }

                    if len >= 128 {
                        break; // Good enough.
                    }
                }

                candidate = self.prev[candidate as usize & WINDOW_MASK];
                steps += 1;
            }
        }

        // 2. Reverse hash chain (3+ bytes).
        if self.params.try_reverse && max_forward >= 3 && pos >= 0 {
            let b0 = window[pos];
            let b1 = window[pos + 1];
            let b2 = window[pos + 2];
            let rh = hash3(b0, b1, b2, self.params.hash_bits) & self.hash_mask;

            let mut candidate = self.rev_head[rh];
            let pos_u32 = pos as u32;
            let mut steps = 0;
            // Track best absolute length (from any source) for early rejection.
            let mut best_abs_len = best_len.unsigned_abs() as usize;

            while candidate != EMPTY && steps < self.params.max_chain {
                let dist_u32 = pos_u32.wrapping_sub(candidate);
                if dist_u32 == 0 || dist_u32 > WINDOW_SIZE as u32 || wp.wrapping_sub(candidate) >= WINDOW_SIZE as u32 {
                    candidate = self.rev_prev[candidate as usize & WINDOW_MASK];
                    steps += 1;
                    continue;
                }

                let cand_pos = pos - dist_u32 as isize;
                let dist = dist_u32 as u16;

                // Quick reject: check byte at current best length.
                // For reverse: source goes backward (cand_pos - N), target goes forward (pos + N).
                if best_abs_len >= 3
                    && (cand_pos - best_abs_len as isize) >= 0
                    && window[cand_pos - best_abs_len as isize] != window[pos + best_abs_len as isize]
                {
                    candidate = self.rev_prev[candidate as usize & WINDOW_MASK];
                    steps += 1;
                    continue;
                }

                // Verify first 3 reverse bytes + live-window checks.
                if window[cand_pos] == b0
                    && cand_pos >= 1
                    && wp.wrapping_sub((cand_pos - 1) as u32) < WINDOW_SIZE as u32
                    && window[cand_pos - 1] == b1
                    && cand_pos >= 2
                    && wp.wrapping_sub((cand_pos - 2) as u32) < WINDOW_SIZE as u32
                    && window[cand_pos - 2] == b2
                {
                    // Extend backward, keeping within the live window.
                    let mut len: usize = 3;
                    while len < max_forward
                        && (cand_pos - len as isize) >= 0
                        && wp.wrapping_sub((cand_pos - len as isize) as u32) < WINDOW_SIZE as u32
                        && window[cand_pos - len as isize] == window[pos + len as isize]
                    {
                        len += 1;
                    }

                    let rev_len = -(len.min(i16::MAX as usize) as i16);
                    let gain = Frame::match_gain(dist, rev_len);
                    if gain > best_gain || (gain == best_gain && dist < best_dist) {
                        best_dist = dist;
                        best_len = rev_len;
                        best_gain = gain;
                        best_abs_len = len;
                    }
                }

                candidate = self.rev_prev[candidate as usize & WINDOW_MASK];
                steps += 1;
            }
        }

        // 3. Short-range scan for 1–2 byte matches (only if hash chains didn't find anything good).
        if best_gain <= 0 && self.params.short_scan_dist > 0 && self.params.min_match_len <= 2 {
            let scan_limit = (self.params.short_scan_dist as isize).min(pos);
            let target = window[pos];
            for d in 1..=scan_limit {
                let cand = pos - d;
                if cand < 0 || wp.wrapping_sub(cand as u32) >= WINDOW_SIZE as u32 {
                    break;
                }
                if window[cand] == target {
                    let dist = d as u16;
                    if max_forward >= 2 && window[cand + 1] == window[pos + 1] {
                        let gain = Frame::match_gain(dist, 2);
                        if gain > best_gain || (gain == best_gain && dist < best_dist) {
                            best_dist = dist;
                            best_len = 2;
                            best_gain = gain;
                        }
                    }
                    if self.params.min_match_len <= 1 {
                        let gain = Frame::match_gain(dist, 1);
                        if gain > best_gain || (gain == best_gain && dist < best_dist) {
                            best_dist = dist;
                            best_len = 1;
                            best_gain = gain;
                        }
                    }
                }
            }
        }

        if best_gain >= 0 {
            Some((best_dist, best_len))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// BTreeMap-based match finder
// ---------------------------------------------------------------------------

/// Maximum number of neighbors to check on each side of the `BTreeMap` lookup.
const BTREE_NEIGHBORS: usize = 4;

/// Maximum positions stored per key.
const POSITIONS_PER_KEY: usize = 4;

/// Number of bytes packed into the `BTreeMap` key.
const KEY_BYTES: usize = 8;

/// Build a `u64` key from up to 8 bytes at `pos` (big-endian so numeric order = lexicographic).
#[inline]
fn make_key(window: &RingBuf, pos: isize, bytes_available: usize) -> u64 {
    window.read_u64_be(pos, bytes_available)
}

/// Number of matching leading bytes between two big-endian `u64` keys.
#[inline]
const fn common_prefix_len(a: u64, b: u64) -> usize {
    let xor = a ^ b;
    if xor == 0 {
        KEY_BYTES
    } else {
        (xor.leading_zeros() / 8) as usize
    }
}

/// `BTreeMap`-based match finder. Uses a `u64` key (8 bytes) so that `BTree` ordering
/// naturally groups entries by longest common prefix. `O(log n)` lookup instead of
/// `O(chain_length)`.
///
/// Designed for batch operation: call [`build`](Self::build) to index a range of
/// positions up front, then [`find_best_match`](Self::find_best_match) for each
/// position. The distance check naturally rejects "future" positions, so building
/// the index ahead of the encoding cursor is safe.
pub(crate) struct BTreeMatchFinder {
    tree: BTreeMap<u64, SmallVec<[u32; POSITIONS_PER_KEY]>>,
    params: LevelParams,
}

impl BTreeMatchFinder {
    /// Construct from encoder options.
    pub(crate) fn new(options: &EncodeOptions) -> Self {
        let level = options.get_level().clamp(1, 9);
        Self {
            tree: BTreeMap::new(),
            params: LEVEL_PARAMS[level],
        }
    }

    /// Insert a position into the entry for `key`, keeping at most `POSITIONS_PER_KEY`.
    #[inline]
    fn insert_pos(entry: &mut SmallVec<[u32; POSITIONS_PER_KEY]>, pos_u32: u32) {
        if entry.len() >= POSITIONS_PER_KEY {
            // Drop the oldest (first) to make room for the newest.
            entry.remove(0);
        }
        entry.push(pos_u32);
    }

    /// Index positions `start..end` into the tree.
    ///
    /// Clears and rebuilds. Call once per batch after loading data.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn build(&mut self, window: &RingBuf, start: isize, end: isize) {
        self.tree.clear();
        let wp = window.write_pos();
        for pos in start..end {
            let bytes_avail = (wp - pos) as usize;
            if bytes_avail == 0 {
                break;
            }
            let key = make_key(window, pos, bytes_avail);
            Self::insert_pos(self.tree.entry(key).or_default(), pos as u32);
        }
    }

    /// Insert a single position (used by lazy matching for pos+1).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn insert(&mut self, window: &RingBuf, pos: isize) {
        if pos < 0 {
            return;
        }
        let wp = window.write_pos();
        let bytes_avail = (wp - pos) as usize;
        if bytes_avail == 0 {
            return;
        }
        let key = make_key(window, pos, bytes_avail);
        Self::insert_pos(self.tree.entry(key).or_default(), pos as u32);
    }

    /// No-op — the batch [`build`](Self::build) already indexed all positions.
    #[allow(clippy::needless_pass_by_ref_mut, clippy::missing_const_for_fn, clippy::unused_self)]
    pub(crate) fn bulk_insert(&mut self, _window: &RingBuf, _start: isize, _end: isize) {}

    /// Lazy matching threshold.
    pub(crate) const fn lazy_threshold(&self) -> isize {
        self.params.lazy_threshold
    }

    /// Returns `(0, 0)` — `BTree` finder doesn't track `SmallVec` lengths.
    #[allow(clippy::unused_self)]
    pub(crate) const fn max_positions_used(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Find the best profitable match at `pos`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    pub(crate) fn find_best_match(&self, window: &RingBuf, pos: isize, bytes_remaining: usize) -> Option<(u16, i16)> {
        if bytes_remaining == 0 {
            return None;
        }

        let max_forward = bytes_remaining.min(i16::MAX as usize);
        let key = make_key(window, pos, bytes_remaining);
        let pos_u32 = pos as u32;

        let mut best_dist: u16 = 0;
        let mut best_len: i16 = 0;
        let mut best_gain: isize = -1;

        // Check candidates from the BTree: neighbors below and above (including exact match).
        // BTree ordering groups entries by longest common prefix.

        // Lower neighbors (keys <= ours).
        for (&cand_key, positions) in self.tree.range(..=key).rev().take(BTREE_NEIGHBORS) {
            for &cand_pos in positions {
                Self::evaluate_candidate(
                    window,
                    pos,
                    pos_u32,
                    max_forward,
                    key,
                    cand_key,
                    cand_pos,
                    &mut best_dist,
                    &mut best_len,
                    &mut best_gain,
                );
            }
        }

        // Upper neighbors (keys > ours).
        for (&cand_key, positions) in
            self.tree.range((std::ops::Bound::Excluded(key), std::ops::Bound::Unbounded)).take(BTREE_NEIGHBORS)
        {
            for &cand_pos in positions {
                Self::evaluate_candidate(
                    window,
                    pos,
                    pos_u32,
                    max_forward,
                    key,
                    cand_key,
                    cand_pos,
                    &mut best_dist,
                    &mut best_len,
                    &mut best_gain,
                );
            }
        }

        if best_gain >= 0 {
            Some((best_dist, best_len))
        } else {
            None
        }
    }

    /// Evaluate a single `BTree` candidate.
    #[inline]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_arguments
    )]
    fn evaluate_candidate(
        window: &RingBuf,
        pos: isize,
        pos_u32: u32,
        max_forward: usize,
        key: u64,
        cand_key: u64,
        cand_pos: u32,
        best_dist: &mut u16,
        best_len: &mut i16,
        best_gain: &mut isize,
    ) {
        let dist_u32 = pos_u32.wrapping_sub(cand_pos);
        if dist_u32 == 0 || dist_u32 > WINDOW_SIZE as u32 {
            return;
        }
        let dist = dist_u32 as u16;

        // Common prefix from keys gives us a guaranteed match length (up to KEY_BYTES).
        let prefix_len = common_prefix_len(key, cand_key);
        if prefix_len == 0 {
            return;
        }

        // Always extend via the ring buffer beyond the prefix.
        // The prefix guarantees the first `prefix_len` bytes match; the ring buffer
        // may find more (keys are built from different positions, so bytes beyond the
        // prefix can still match at the actual positions).
        let cand_isize = pos - dist_u32 as isize;
        let match_len = if prefix_len < max_forward {
            prefix_len
                + window.match_length(
                    cand_isize + prefix_len as isize,
                    pos + prefix_len as isize,
                    max_forward - prefix_len,
                )
        } else {
            prefix_len.min(max_forward)
        };

        let fwd_len = match_len.min(i16::MAX as usize) as i16;
        let gain = Frame::match_gain(dist, fwd_len);
        if gain > *best_gain || (gain == *best_gain && dist < *best_dist) {
            *best_dist = dist;
            *best_len = fwd_len;
            *best_gain = gain;
        }
    }
}

// ---------------------------------------------------------------------------
// HashMap-based match finder
// ---------------------------------------------------------------------------

/// Pack 3 bytes into a `u32` key (big-endian, top byte zero).
/// 16M possible keys — zero hash collisions.
#[inline]
const fn pack3(b0: u8, b1: u8, b2: u8) -> u32 {
    (b0 as u32) << 16 | (b1 as u32) << 8 | b2 as u32
}

/// `HashMap`-based match finder with exact 3-byte prefix grouping.
///
/// Uses two maps (forward and reverse) keyed on raw 3-byte prefixes packed
/// into `u32`. No hash collisions — every entry sharing a key truly shares the
/// same 3-byte prefix. Position lists are sorted ascending (build order).
///
/// Designed for batch operation: call [`build`](Self::build) to index a range,
/// then [`find_best_match`](Self::find_best_match) for each position.
pub(crate) struct HashMapMatchFinder {
    forward: HashMap<u32, SmallVec<[u32; 8]>>,
    reverse: HashMap<u32, SmallVec<[u32; 8]>>,
    params: LevelParams,
    max_forward_len: usize,
    max_reverse_len: usize,
}

impl HashMapMatchFinder {
    /// Construct from encoder options.
    pub(crate) fn new(options: &EncodeOptions) -> Self {
        let level = options.get_level().clamp(1, 9);
        Self {
            forward: HashMap::with_capacity(60_000),
            reverse: HashMap::new(),
            params: LEVEL_PARAMS[level],
            max_forward_len: 0,
            max_reverse_len: 0,
        }
    }

    /// Index positions `start..end` into both maps.
    ///
    /// Clears and rebuilds. Call once per batch after loading data.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn build(&mut self, window: &RingBuf, start: isize, end: isize) {
        self.forward.clear();
        self.reverse.clear();
        self.max_forward_len = 0;
        self.max_reverse_len = 0;

        let wp = window.write_pos();
        for pos in start..end {
            let bytes_avail = (wp - pos) as usize;
            if bytes_avail < 3 {
                break;
            }
            let key = pack3(window[pos], window[pos + 1], window[pos + 2]);
            let fwd_entry = self.forward.entry(key).or_default();
            fwd_entry.push(pos as u32);
            if fwd_entry.len() > self.max_forward_len {
                self.max_forward_len = fwd_entry.len();
            }

            if self.params.try_reverse && pos - start >= 2 {
                let rkey = pack3(window[pos], window[pos - 1], window[pos - 2]);
                let rev_entry = self.reverse.entry(rkey).or_default();
                rev_entry.push(pos as u32);
                if rev_entry.len() > self.max_reverse_len {
                    self.max_reverse_len = rev_entry.len();
                }
            }
        }
    }

    /// Insert a single position (used by lazy matching for pos+1).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn insert(&mut self, window: &RingBuf, pos: isize) {
        if pos < 0 {
            return;
        }
        let wp = window.write_pos();
        let bytes_avail = (wp - pos) as usize;
        if bytes_avail < 3 {
            return;
        }
        let key = pack3(window[pos], window[pos + 1], window[pos + 2]);
        let fwd_entry = self.forward.entry(key).or_default();
        fwd_entry.push(pos as u32);
        if fwd_entry.len() > self.max_forward_len {
            self.max_forward_len = fwd_entry.len();
        }

        if self.params.try_reverse && pos >= 2 {
            let rkey = pack3(window[pos], window[pos - 1], window[pos - 2]);
            let rev_entry = self.reverse.entry(rkey).or_default();
            rev_entry.push(pos as u32);
            if rev_entry.len() > self.max_reverse_len {
                self.max_reverse_len = rev_entry.len();
            }
        }
    }

    /// No-op — the batch [`build`](Self::build) already indexed all positions.
    #[allow(clippy::needless_pass_by_ref_mut, clippy::missing_const_for_fn, clippy::unused_self)]
    pub(crate) fn bulk_insert(&mut self, _window: &RingBuf, _start: isize, _end: isize) {}

    /// Lazy matching threshold.
    pub(crate) const fn lazy_threshold(&self) -> isize {
        self.params.lazy_threshold
    }

    /// Returns `(max_forward_len, max_reverse_len)` — peak `SmallVec` lengths observed.
    pub(crate) const fn max_positions_used(&self) -> (usize, usize) {
        (self.max_forward_len, self.max_reverse_len)
    }

    /// Find the best profitable match at `pos`. Returns `(distance, length)` or `None`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    pub(crate) fn find_best_match(&self, window: &RingBuf, pos: isize, bytes_remaining: usize) -> Option<(u16, i16)> {
        if bytes_remaining == 0 {
            return None;
        }

        let mut best_dist: u16 = 0;
        let mut best_len: i16 = 0;
        let mut best_gain: isize = -1;

        let max_forward = bytes_remaining.min(i16::MAX as usize);
        let pos_u32 = pos as u32;
        let wp = window.write_pos() as u32;

        // 1. Forward lookup: 3+ byte matches.
        if max_forward >= 3 && pos >= 0 {
            let b0 = window[pos];
            let b1 = window[pos + 1];
            let b2 = window[pos + 2];
            let key = pack3(b0, b1, b2);

            if let Some(positions) = self.forward.get(&key) {
                let mut steps = 0;
                let mut best_fwd_len: usize = 2;

                // Iterate from end (most recent = shortest distance = usually best).
                for &cand in positions.iter().rev() {
                    if steps >= self.params.max_chain {
                        break;
                    }

                    let dist_u32 = pos_u32.wrapping_sub(cand);
                    if dist_u32 == 0 || dist_u32 > WINDOW_SIZE as u32 {
                        steps += 1;
                        continue;
                    }

                    let cand_pos = pos - dist_u32 as isize;
                    let dist = dist_u32 as u16;

                    // Quick reject: check byte at current best length first.
                    if best_fwd_len < max_forward
                        && window[cand_pos + best_fwd_len as isize] != window[pos + best_fwd_len as isize]
                    {
                        steps += 1;
                        continue;
                    }

                    // First 3 bytes guaranteed to match (same key = same prefix).
                    // Extend forward beyond the prefix.
                    let len = 3 + window.match_length(cand_pos + 3, pos + 3, max_forward - 3);

                    let fwd_len = len.min(i16::MAX as usize) as i16;
                    let gain = Frame::match_gain(dist, fwd_len);
                    if gain > best_gain || (gain == best_gain && dist < best_dist) {
                        best_dist = dist;
                        best_len = fwd_len;
                        best_gain = gain;
                        best_fwd_len = len;
                    }

                    if len >= 128 {
                        break;
                    }

                    steps += 1;
                }
            }
        }

        // 2. Reverse lookup: 3+ byte matches (source goes backward, target goes forward).
        if self.params.try_reverse && max_forward >= 3 && pos >= 0 {
            let b0 = window[pos];
            let b1 = window[pos + 1];
            let b2 = window[pos + 2];
            let rkey = pack3(b0, b1, b2);

            if let Some(positions) = self.reverse.get(&rkey) {
                let mut steps = 0;
                let mut best_abs_len = best_len.unsigned_abs() as usize;

                for &cand in positions.iter().rev() {
                    if steps >= self.params.max_chain {
                        break;
                    }

                    let dist_u32 = pos_u32.wrapping_sub(cand);
                    if dist_u32 == 0 || dist_u32 > WINDOW_SIZE as u32 {
                        steps += 1;
                        continue;
                    }

                    let cand_pos = pos - dist_u32 as isize;
                    let dist = dist_u32 as u16;

                    // Quick reject: check byte at current best length.
                    if best_abs_len >= 3
                        && (cand_pos - best_abs_len as isize) >= 0
                        && wp.wrapping_sub((cand_pos - best_abs_len as isize) as u32) < WINDOW_SIZE as u32
                        && window[cand_pos - best_abs_len as isize] != window[pos + best_abs_len as isize]
                    {
                        steps += 1;
                        continue;
                    }

                    // First 3 bytes guaranteed by key: window[cand]==b0, window[cand-1]==b1, window[cand-2]==b2.
                    // Extend backward, keeping within the live window.
                    let mut len: usize = 3;
                    while len < max_forward
                        && (cand_pos - len as isize) >= 0
                        && wp.wrapping_sub((cand_pos - len as isize) as u32) < WINDOW_SIZE as u32
                        && window[cand_pos - len as isize] == window[pos + len as isize]
                    {
                        len += 1;
                    }

                    let rev_len = -(len.min(i16::MAX as usize) as i16);
                    let gain = Frame::match_gain(dist, rev_len);
                    if gain > best_gain || (gain == best_gain && dist < best_dist) {
                        best_dist = dist;
                        best_len = rev_len;
                        best_gain = gain;
                        best_abs_len = len;
                    }

                    steps += 1;
                }
            }
        }

        // 3. Short-range scan for 1–2 byte matches.
        if best_gain <= 0 && self.params.short_scan_dist > 0 && self.params.min_match_len <= 2 {
            let scan_limit = (self.params.short_scan_dist as isize).min(pos);
            let target = window[pos];
            for d in 1..=scan_limit {
                let cand = pos - d;
                if cand < 0 {
                    break;
                }
                if window[cand] == target {
                    let dist = d as u16;
                    if max_forward >= 2 && window[cand + 1] == window[pos + 1] {
                        let gain = Frame::match_gain(dist, 2);
                        if gain > best_gain || (gain == best_gain && dist < best_dist) {
                            best_dist = dist;
                            best_len = 2;
                            best_gain = gain;
                        }
                    }
                    if self.params.min_match_len <= 1 {
                        let gain = Frame::match_gain(dist, 1);
                        if gain > best_gain || (gain == best_gain && dist < best_dist) {
                            best_dist = dist;
                            best_len = 1;
                            best_gain = gain;
                        }
                    }
                }
            }
        }

        if best_gain >= 0 {
            Some((best_dist, best_len))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Brute-force (exhaustive) match finder
// ---------------------------------------------------------------------------

/// Exhaustive match finder that checks every position in the window.
///
/// `O(n × window_size)` — far too slow for production, but establishes the
/// theoretical compression ceiling for the LZR format with greedy + lazy
/// parsing. Useful for validating that faster finders approach optimal.
pub(crate) struct BruteForceMatchFinder {
    params: LevelParams,
}

impl BruteForceMatchFinder {
    /// Construct from encoder options.
    pub(crate) fn new(options: &EncodeOptions) -> Self {
        let level = options.get_level().clamp(1, 9);
        Self {
            params: LEVEL_PARAMS[level],
        }
    }

    /// No-op — brute force needs no index.
    #[allow(clippy::needless_pass_by_ref_mut, clippy::missing_const_for_fn, clippy::unused_self)]
    pub(crate) fn build(&mut self, _window: &RingBuf, _start: isize, _end: isize) {}

    /// No-op.
    #[allow(clippy::needless_pass_by_ref_mut, clippy::missing_const_for_fn, clippy::unused_self)]
    pub(crate) fn insert(&mut self, _window: &RingBuf, _pos: isize) {}

    /// No-op.
    #[allow(clippy::needless_pass_by_ref_mut, clippy::missing_const_for_fn, clippy::unused_self)]
    pub(crate) fn bulk_insert(&mut self, _window: &RingBuf, _start: isize, _end: isize) {}

    /// Lazy matching threshold.
    pub(crate) const fn lazy_threshold(&self) -> isize {
        self.params.lazy_threshold
    }

    /// Returns `(0, 0)`.
    #[allow(clippy::unused_self)]
    pub(crate) const fn max_positions_used(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Find the best match at `pos` by checking every candidate in the window.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    pub(crate) fn find_best_match(&self, window: &RingBuf, pos: isize, bytes_remaining: usize) -> Option<(u16, i16)> {
        if bytes_remaining == 0 || pos <= 0 {
            return None;
        }

        let max_forward = bytes_remaining.min(i16::MAX as usize);
        let wp = window.write_pos();
        // Oldest position whose data is still valid in the ring buffer.
        let min_cand = 0isize
            .max(pos - WINDOW_SIZE as isize + 1) // distance must fit in u16 (max 65535)
            .max(wp - WINDOW_SIZE as isize); // ring buffer liveness

        let mut best_dist: u16 = 0;
        let mut best_len: i16 = 0;
        let mut best_gain: isize = -1;

        let target_byte = window[pos];

        for cand_pos in (min_cand..pos).rev() {
            // First-byte gate: both forward and reverse matches require window[cand] == window[pos].
            if window[cand_pos] != target_byte {
                continue;
            }

            let dist = (pos - cand_pos) as u16;

            // Forward match.
            let fwd_len = 1 + window.match_length(cand_pos + 1, pos + 1, max_forward - 1);
            let fwd_len_i16 = fwd_len.min(i16::MAX as usize) as i16;
            let fwd_gain = Frame::match_gain(dist, fwd_len_i16);
            if fwd_gain > best_gain || (fwd_gain == best_gain && dist < best_dist) {
                best_dist = dist;
                best_len = fwd_len_i16;
                best_gain = fwd_gain;
            }

            // Reverse match.
            if self.params.try_reverse {
                let mut rev_len: usize = 1;
                while rev_len < max_forward
                    && (cand_pos - rev_len as isize) >= min_cand
                    && window[cand_pos - rev_len as isize] == window[pos + rev_len as isize]
                {
                    rev_len += 1;
                }

                let rev_len_i16 = -(rev_len.min(i16::MAX as usize) as i16);
                let rev_gain = Frame::match_gain(dist, rev_len_i16);
                if rev_gain > best_gain || (rev_gain == best_gain && dist < best_dist) {
                    best_dist = dist;
                    best_len = rev_len_i16;
                    best_gain = rev_gain;
                }
            }
        }

        if best_gain >= 0 {
            Some((best_dist, best_len))
        } else {
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::options::EncodeOptions;

    fn make_window(data: &[u8]) -> RingBuf {
        let mut w = RingBuf::new(WINDOW_SIZE);
        for &b in data {
            w.push(b);
        }
        w
    }

    fn make_finder(level: usize) -> MatchFinder {
        MatchFinder::new(&EncodeOptions::new().level(level))
    }

    #[test]
    fn basic_forward_match() {
        // "abcabc" → at pos 3, should find match (3, 3)
        let data = b"abcabc";
        let window = make_window(data);
        let mut mf = make_finder(1);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        let result = mf.find_best_match(&window, 3, 3);
        assert!(result.is_some(), "should find a match");
        let (dist, len) = result.unwrap();
        assert_eq!(dist, 3);
        assert_eq!(len, 3);
    }

    #[test]
    fn short_match_close_distance() {
        // "aba" at level 9 → should find (2, 1) at pos 2
        let data = b"aba";
        let window = make_window(data);
        let mut mf = make_finder(9);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        let result = mf.find_best_match(&window, 2, 1);
        assert!(result.is_some(), "should find short match at level 9");
        let (dist, len) = result.unwrap();
        assert_eq!(dist, 2);
        assert_eq!(len, 1);
    }

    #[test]
    fn unprofitable_match_rejected() {
        // 1-byte match at D=8 → gain = -1, should return None
        // Create data where only a 1-byte match at dist 8 is possible.
        let data = b"a1234567a";
        let window = make_window(data);
        // Use level 9 so short scan is enabled, but dist 8 for a 1-byte match is unprofitable.
        let mut mf = make_finder(9);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        // At pos 8, 'a' matches pos 0 at distance 8. gain(8,1)=-1.
        // There's only 1 byte remaining, so no longer match possible.
        let result = mf.find_best_match(&window, 8, 1);
        assert!(result.is_none(), "1-byte match at D=8 should be rejected");
    }

    #[test]
    fn chain_depth_respected() {
        // Many positions with the same hash prefix, low level shouldn't find distant ones.
        let mut data = Vec::new();
        // Repeat the same 3-byte prefix many times with different suffixes.
        for i in 0u8..64 {
            data.extend_from_slice(&[0xAA, 0xBB, 0xCC, i]);
        }
        // Add one more "AABBCC" that should match.
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0x00]);

        let window = make_window(&data);
        let mut mf_low = make_finder(1); // max_chain=4
        let mut mf_high = make_finder(9); // max_chain=1024

        for i in 0..data.len() as isize {
            mf_low.insert(&window, i);
            mf_high.insert(&window, i);
        }

        let pos = data.len() as isize - 4;
        let remaining = 4;

        // High level should find a match.
        let high_result = mf_high.find_best_match(&window, pos, remaining);
        assert!(high_result.is_some(), "high level should find match");

        // Both should find something (chains aren't that deep here), but let's just verify
        // they don't crash and high level finds at least as good.
        let low_result = mf_low.find_best_match(&window, pos, remaining);
        if let (Some((_, l_len)), Some((_, h_len))) = (low_result, high_result) {
            assert!(h_len >= l_len, "high level should find at least as long a match");
        }
    }

    #[test]
    fn reverse_match() {
        // "abccba" at level 9 → at pos 3, "cba" is the reverse of "abc" at pos 0.
        // Distance = 3, the candidate at pos 2 has window[2]='c', window[1]='b', window[0]='a'.
        // We're looking at pos 3: window[3]='c', window[4]='b', window[5]='a'.
        let data = b"abccba";
        let window = make_window(data);
        let mut mf = make_finder(9);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        let result = mf.find_best_match(&window, 3, 3);
        assert!(result.is_some(), "should find a reverse match");
        let (dist, len) = result.unwrap();
        // Could be forward or reverse depending on gain.
        // The reverse match: at pos 3, cand=2, dist=1. window[2]='c'=window[3], window[1]='b'=window[4], window[0]='a'=window[5].
        // That's a reverse match of length 3 at distance 1: gain(1, -3) = 6-1-1 = 4.
        // Forward: "cba" at pos 3 doesn't match "abc" at pos 0 forward, so no 3-byte forward match.
        assert!(len == -3 || len == 3, "should find length 3 match, got {len}");
        assert!(dist > 0);
    }

    #[test]
    fn end_of_input_boundary() {
        // Fewer than 3 bytes remaining → should not crash.
        let data = b"ab";
        let window = make_window(data);
        let mut mf = make_finder(9);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        // Only 1 byte remaining at pos 1.
        let result = mf.find_best_match(&window, 1, 1);
        // Might be None (no profitable match) or Some if short scan finds something.
        // Either way, it shouldn't panic.
        if let Some((dist, len)) = result {
            assert!(dist > 0);
            assert!(len.unsigned_abs() >= 1);
        }
    }

    #[test]
    fn window_wrap() {
        // Push more than 65536 bytes, matches should still work across wrap.
        let mut window = RingBuf::new(WINDOW_SIZE);
        let mut mf = make_finder(5);

        // Fill with ~66000 bytes of unique data so no hash collisions with our pattern.
        for i in 0..66_000u32 {
            window.push((i.wrapping_mul(131) & 0xFF) as u8);
        }
        // Insert all positions (need at least 3 lookahead bytes in window for correct hash).
        // In a real encoder, bytes are pushed then positions inserted sequentially.
        // Here we batch-insert after filling so hashes are computed on correct data.
        for i in 0..65_997u32 {
            mf.insert(&window, i as isize);
        }

        // Push "XYZXYZ" — first occurrence + second occurrence.
        let base = window.write_pos(); // 66000
        for &b in b"XYZXYZ" {
            window.push(b);
        }
        // Insert first "XYZ" positions (66000, 66001, 66002).
        for i in base..base + 3 {
            mf.insert(&window, i);
        }

        let search_pos = base + 3; // 66003
        let result = mf.find_best_match(&window, search_pos, 3);
        assert!(result.is_some(), "should find match after window wrap");
        let (dist, len) = result.unwrap();
        assert_eq!(dist, 3);
        assert_eq!(len, 3);
    }
}
