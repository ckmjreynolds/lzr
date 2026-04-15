//! LZ77 encoder and decoder.
//!
//! A [`Sequence`] describes zero or more literal bytes followed by an optional
//! back-reference (match). The encoder uses a `wabi_tree` `OSBTreeMap` with
//! u128 prefix keys and range queries for match finding within a 64 KB sliding
//! window.
//!
//! Compression levels 1–4 control how many candidate positions are checked
//! when an exact 16-byte prefix match is found (1 = newest only, 4 = all four
//! stored positions).
//!
//! # Pipeline
//!
//! ```text
//! Encode:  Bytes → LZ77 Encoder  → Arithmetic Coder
//! Decode:  Arithmetic Decoder    → LZ77 Decoder → Bytes
//! ```

use static_assertions::assert_eq_size;
use wabi_tree::OSBTreeMap;

/// Sliding-window size in bytes (matches the `u16` distance range).
const WINDOW_SIZE: usize = 65_536;

/// Minimum useful match length.
const MIN_MATCH: usize = 3;

/// Maximum match length (constrained by `match_len: u8`).
const MAX_MATCH: usize = 255;

/// Maximum literal run per sequence (constrained by `literal_len: u8`).
pub(crate) const MAX_LITERALS: usize = 255;

/// Minimum lookahead required in streaming mode before emitting tokens.
const LOOKAHEAD_MIN: usize = MAX_MATCH + MIN_MATCH;

/// Maximum candidates checked per scan direction during range queries.
const MAX_SCAN: usize = 16;

/// Default compression level.
pub(crate) const DEFAULT_LEVEL: u8 = 4;

/// Minimum compression level.
pub(crate) const MIN_LEVEL: u8 = 1;

/// Maximum compression level.
pub(crate) const MAX_LEVEL: u8 = 4;

// ===========================================================================
// Types
// ===========================================================================

/// One LZ77 token: zero or more literal bytes followed by an optional
/// back-reference.
///
/// * `literal_len` — number of literal bytes that precede the match (0..=255).
/// * `match_len`   — length of the back-reference copy. **0** means no match;
///   valid matches are in the range `3..=255`.
/// * `match_distance` — how far back to look (1..=65 535). Ignored when
///   `match_len == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sequence {
    pub(crate) literal_len: u8,
    pub(crate) match_len: u8,
    pub(crate) match_distance: u16,
}

assert_eq_size!(Sequence, u32);

/// A [`Sequence`] together with the literal bytes it owns.
///
/// The valid literal region is `&self.literals[..self.seq.literal_len as usize]`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Token {
    pub(crate) seq: Sequence,
    pub(crate) literals: [u8; 255],
}

// ===========================================================================
// Match finder
// ===========================================================================

/// Returns the length of the common prefix of `a` and `b`, up to `max`.
fn common_prefix_len(a: &[u8], b: &[u8], max: usize) -> usize {
    a.iter().zip(b.iter()).take(max).take_while(|(x, y)| x == y).count()
}

/// Builds a big-endian u128 key from the first 16 bytes at `buf[offset..]`.
///
/// Big-endian ensures numeric order equals lexicographic order, so range
/// queries on the B+tree naturally find lexicographically adjacent suffixes.
fn make_key(buf: &[u8], offset: usize) -> u128 {
    let avail = buf.len() - offset;
    let take = avail.min(16);
    let mut bytes = [0u8; 16];
    bytes[..take].copy_from_slice(&buf[offset..offset + take]);
    u128::from_be_bytes(bytes)
}

/// Returns the number of leading bytes two u128 keys share.
const fn common_prefix_u128(a: u128, b: u128) -> usize {
    let xor = a ^ b;
    if xor == 0 {
        return 16;
    }
    (xor.leading_zeros() / 8) as usize
}

/// Extracts up to 4 packed u32 positions from a u128 value.
///
/// Positions are packed newest-at-LSB: on each insert the value is shifted
/// left by 32 and the new position is `OR`ed into the low 32 bits. To read
/// back, position 0 (newest) is `value as u32`, position 1 is
/// `(value >> 32) as u32`, etc.
#[allow(clippy::cast_possible_truncation)]
const fn unpack_position(packed: u128, index: usize) -> u32 {
    (packed >> (index * 32)) as u32
}

/// Packs a new position into a u128 value, shifting older positions up.
const fn pack_position(packed: u128, pos: u32) -> u128 {
    (packed << 32) | pos as u128
}

/// B+tree match finder using u128 prefix keys.
///
/// Each key maps to a u128 packing up to 4 recent absolute positions.
/// Stale entries (outside the sliding window) are not evicted — they are
/// filtered at match time by the distance check. The tree is cleared at
/// Sonnet boundaries.
struct MatchFinder {
    index: OSBTreeMap<u128, u128>,
    /// How many packed positions to check on exact-key match (1–4).
    depth: usize,
}

impl MatchFinder {
    const fn new(level: u8) -> Self {
        Self {
            index: OSBTreeMap::new(),
            depth: level as usize,
        }
    }

    /// Records `pos` in the index under its 16-byte prefix key.
    #[allow(clippy::cast_possible_truncation)]
    fn insert(&mut self, pos: usize, buf: &[u8], base: usize) {
        let buf_pos = pos - base;
        if buf_pos + 2 >= buf.len() {
            return;
        }

        let key = make_key(buf, buf_pos);
        let pos32 = pos as u32;

        let packed = self.index.get(&key).copied().unwrap_or(0);
        self.index.insert(key, pack_position(packed, pos32));
    }

    /// Finds the best match at `pos`.
    fn find_match(&self, pos: usize, buf: &[u8], base: usize) -> (usize, usize) {
        let available = base + buf.len();
        let remaining = available - pos;
        let max_len = MAX_MATCH.min(remaining);

        if max_len < MIN_MATCH {
            return (0, 0);
        }

        let buf_pos = pos - base;
        let min_pos = pos.saturating_sub(u16::MAX as usize);
        let key = make_key(buf, buf_pos);

        let mut best_len = MIN_MATCH - 1;
        let mut best_dist = 0;

        // Check exact key match — up to `depth` packed positions.
        if let Some(&packed) = self.index.get(&key) {
            for i in 0..self.depth {
                let cand = unpack_position(packed, i) as usize;
                if cand < min_pos || cand >= pos || cand < base {
                    continue;
                }
                let ci = cand - base;
                let len = common_prefix_len(&buf[ci..], &buf[buf_pos..], max_len);
                if len > best_len {
                    best_len = len;
                    best_dist = pos - cand;
                    if best_len == max_len {
                        return (best_dist, best_len);
                    }
                }
            }
        }

        // Range scan: check neighbors with different prefixes (newest position only).
        let mut checked = 0;

        // Forward (keys > our key).
        for (&candidate_key, &packed) in self.index.range((std::ops::Bound::Excluded(key), std::ops::Bound::Unbounded))
        {
            if checked >= MAX_SCAN {
                break;
            }
            if common_prefix_u128(key, candidate_key) < MIN_MATCH {
                break;
            }
            let cand = unpack_position(packed, 0) as usize;
            if cand >= min_pos && cand < pos && cand >= base {
                let ci = cand - base;
                let len = common_prefix_len(&buf[ci..], &buf[buf_pos..], max_len);
                if len > best_len {
                    best_len = len;
                    best_dist = pos - cand;
                    if best_len == max_len {
                        return (best_dist, best_len);
                    }
                }
            }
            checked += 1;
        }

        // Backward (keys < our key).
        checked = 0;
        for (&candidate_key, &packed) in self.index.range(..key).rev() {
            if checked >= MAX_SCAN {
                break;
            }
            if common_prefix_u128(key, candidate_key) < MIN_MATCH {
                break;
            }
            let cand = unpack_position(packed, 0) as usize;
            if cand >= min_pos && cand < pos && cand >= base {
                let ci = cand - base;
                let len = common_prefix_len(&buf[ci..], &buf[buf_pos..], max_len);
                if len > best_len {
                    best_len = len;
                    best_dist = pos - cand;
                    if best_len == max_len {
                        return (best_dist, best_len);
                    }
                }
            }
            checked += 1;
        }

        if best_len >= MIN_MATCH {
            (best_dist, best_len)
        } else {
            (0, 0)
        }
    }

    fn clear(&mut self) {
        self.index.clear();
    }
}

// ===========================================================================
// Encoder
// ===========================================================================

/// Greedy streaming LZ77 encoder with B+tree match finding.
///
/// Data is fed incrementally via [`feed`](Self::feed). Call
/// [`next`](Self::next) to pull tokens. Call [`finish`](Self::finish)
/// to signal end of input, then drain remaining tokens with `next`.
pub(crate) struct Encoder {
    buf: Vec<u8>,
    base: usize,
    pos: usize,
    lit_start: usize,
    lit_len: usize,
    finder: MatchFinder,
    finished: bool,
}

impl Encoder {
    /// Creates a new encoder at the given compression level (1–4).
    ///
    /// # Panics
    ///
    /// Panics if `level` is not in `1..=4`.
    pub(crate) fn new(level: u8) -> Self {
        assert!((MIN_LEVEL..=MAX_LEVEL).contains(&level), "compression level must be {MIN_LEVEL}–{MAX_LEVEL}");
        Self {
            buf: Vec::with_capacity(2 * WINDOW_SIZE + MAX_MATCH),
            base: 0,
            pos: 0,
            lit_start: 0,
            lit_len: 0,
            finder: MatchFinder::new(level),
            finished: false,
        }
    }

    /// Appends input data to the encoder's internal buffer.
    ///
    /// # Panics
    ///
    /// Panics if called after [`finish`](Self::finish).
    pub(crate) fn feed(&mut self, data: &[u8]) {
        assert!(!self.finished, "cannot feed after finish");
        self.maybe_compact();
        self.buf.extend_from_slice(data);
    }

    /// Signals that no more input will be fed.
    pub(crate) const fn finish(&mut self) {
        self.finished = true;
    }

    /// Emits any pending literals as a literal-only token.
    pub(crate) fn drain(&mut self) -> Option<Token> {
        if self.lit_len == 0 {
            return None;
        }
        Some(self.emit_literals())
    }

    /// Processes all remaining buffered data and returns the resulting tokens.
    pub(crate) fn flush(&mut self) -> Vec<Token> {
        self.finished = true;
        let mut tokens = Vec::new();
        while let Some(token) = self.next() {
            tokens.push(token);
        }
        self.finished = false;
        tokens
    }

    /// Resets encoder state for a new Sonnet (preserves level).
    pub(crate) fn reset(&mut self) {
        self.buf.clear();
        self.base = 0;
        self.pos = 0;
        self.lit_start = 0;
        self.lit_len = 0;
        self.finder.clear();
        self.finished = false;
    }

    /// Returns `true` if there are pending literal bytes not yet emitted.
    pub(crate) const fn has_pending_literals(&self) -> bool {
        self.lit_len > 0
    }

    /// Returns input bytes that have been fed but not yet emitted as tokens.
    pub(crate) fn unconsumed_input(&self) -> Vec<u8> {
        let start = self.lit_start - self.base;
        self.buf[start..].to_vec()
    }

    /// Returns the next token, or `None` if more data is needed.
    pub(crate) fn next(&mut self) -> Option<Token> {
        self.next_inner(MAX_LITERALS)
    }

    /// Like [`next`](Self::next), but caps the literal run at `max_lits`.
    pub(crate) fn next_capped(&mut self, max_lits: u8) -> Option<Token> {
        let cap = max_lits as usize;

        if self.lit_len > 0 && cap == 0 {
            return None;
        }

        if self.lit_len > cap && cap > 0 {
            return Some(self.emit_partial_literals(cap));
        }

        self.next_inner(cap)
    }

    fn next_inner(&mut self, max_lits: usize) -> Option<Token> {
        let available = self.base + self.buf.len();

        loop {
            if self.lit_len >= max_lits && self.lit_len > 0 {
                return Some(self.emit_literals());
            }

            if self.pos >= available {
                if self.finished {
                    break;
                }
                return None;
            }

            if !self.finished && available - self.pos < LOOKAHEAD_MIN {
                return None;
            }

            let (dist, mlen) = self.finder.find_match(self.pos, &self.buf, self.base);

            if mlen >= MIN_MATCH {
                self.finder.insert(self.pos, &self.buf, self.base);
                for i in 1..mlen {
                    self.finder.insert(self.pos + i, &self.buf, self.base);
                }
                self.pos += mlen;
                return Some(self.emit_match(dist, mlen));
            }

            self.finder.insert(self.pos, &self.buf, self.base);
            self.pos += 1;
            self.lit_len += 1;
        }

        if self.lit_len > 0 {
            Some(self.emit_literals())
        } else {
            None
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn emit_literals(&mut self) -> Token {
        debug_assert!(self.lit_len > 0 && self.lit_len <= MAX_LITERALS);
        let mut literals = [0u8; 255];
        let start = self.lit_start - self.base;
        literals[..self.lit_len].copy_from_slice(&self.buf[start..start + self.lit_len]);

        let token = Token {
            seq: Sequence {
                literal_len: self.lit_len as u8,
                match_len: 0,
                match_distance: 0,
            },
            literals,
        };
        self.lit_start = self.pos;
        self.lit_len = 0;
        token
    }

    #[allow(clippy::cast_possible_truncation)]
    fn emit_partial_literals(&mut self, count: usize) -> Token {
        debug_assert!(count > 0 && count <= self.lit_len);
        let mut literals = [0u8; 255];
        let start = self.lit_start - self.base;
        literals[..count].copy_from_slice(&self.buf[start..start + count]);

        let token = Token {
            seq: Sequence {
                literal_len: count as u8,
                match_len: 0,
                match_distance: 0,
            },
            literals,
        };
        self.lit_start += count;
        self.lit_len -= count;
        token
    }

    #[allow(clippy::cast_possible_truncation)]
    fn emit_match(&mut self, dist: usize, mlen: usize) -> Token {
        let mut literals = [0u8; 255];
        if self.lit_len > 0 {
            let start = self.lit_start - self.base;
            literals[..self.lit_len].copy_from_slice(&self.buf[start..start + self.lit_len]);
        }

        let token = Token {
            seq: Sequence {
                literal_len: self.lit_len as u8,
                match_len: mlen as u8,
                match_distance: dist as u16,
            },
            literals,
        };
        self.lit_start = self.pos;
        self.lit_len = 0;
        token
    }

    fn maybe_compact(&mut self) {
        if self.buf.len() <= 2 * WINDOW_SIZE {
            return;
        }
        let abs_keep_from = self.pos.saturating_sub(WINDOW_SIZE);
        if abs_keep_from <= self.base {
            return;
        }
        let keep_from = abs_keep_from - self.base;
        self.buf.drain(..keep_from);
        self.base += keep_from;
    }
}

// ===========================================================================
// Decoder
// ===========================================================================

/// LZ77 decoder that reconstructs raw bytes from a stream of [`Sequence`]
/// tokens.
pub(crate) struct Decoder {
    window: Vec<u8>,
    pos: usize,
}

impl Decoder {
    pub(crate) fn new() -> Self {
        Self {
            window: vec![0; WINDOW_SIZE],
            pos: 0,
        }
    }

    pub(crate) fn decode(&mut self, seq: Sequence, literals: &[u8], output: &mut Vec<u8>) {
        debug_assert_eq!(literals.len(), seq.literal_len as usize);

        for &b in literals {
            output.push(b);
            self.window[self.pos % WINDOW_SIZE] = b;
            self.pos += 1;
        }

        let match_len = seq.match_len as usize;
        if match_len > 0 {
            let dist = seq.match_distance as usize;
            debug_assert!(dist > 0, "match_distance must be > 0 when match_len > 0");
            for _ in 0..match_len {
                let src = self.pos.wrapping_sub(dist) % WINDOW_SIZE;
                let b = self.window[src];
                output.push(b);
                self.window[self.pos % WINDOW_SIZE] = b;
                self.pos += 1;
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::bench::lipsum_bytes;

    fn token_literals(token: &Token) -> &[u8] {
        &token.literals[..token.seq.literal_len as usize]
    }

    fn roundtrip_codec(data: &[u8], level: u8) -> Vec<u8> {
        let mut enc = Encoder::new(level);
        enc.feed(data);
        enc.finish();

        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        while let Some(token) = enc.next() {
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        assert_eq!(data, &decoded[..]);
        decoded
    }

    #[test]
    fn decode_empty() {
        roundtrip_codec(b"", DEFAULT_LEVEL);
    }

    #[test]
    fn lipsum_roundtrip_compresses() {
        let data = lipsum_bytes(65_536);

        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(&data);
        enc.finish();

        let mut dec = Decoder::new();
        let mut decoded = Vec::with_capacity(data.len());
        let mut seq_count = 0_usize;
        let mut total_literal_bytes = 0_usize;

        while let Some(token) = enc.next() {
            seq_count += 1;
            total_literal_bytes += token.seq.literal_len as usize;
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        let encoded_size = seq_count * size_of::<Sequence>() + total_literal_bytes;

        assert_eq!(data, &decoded[..]);
        assert!(encoded_size < data.len());
    }

    #[test]
    fn larger_than_window() {
        let data = lipsum_bytes(128 * 1024);
        roundtrip_codec(&data, DEFAULT_LEVEL);
    }

    #[test]
    fn multi_feed_roundtrip() {
        let data = lipsum_bytes(4096);

        let mut enc = Encoder::new(DEFAULT_LEVEL);
        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        for chunk in data.chunks(100) {
            enc.feed(chunk);
            while let Some(token) = enc.next() {
                dec.decode(token.seq, token_literals(&token), &mut decoded);
            }
        }
        enc.finish();
        while let Some(token) = enc.next() {
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        assert_eq!(data, &decoded[..]);
    }

    #[test]
    fn drain_without_finish() {
        let data = lipsum_bytes(1024);

        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(&data);

        let mut tokens_before_drain = Vec::new();
        while let Some(token) = enc.next() {
            tokens_before_drain.push(token);
        }

        if let Some(token) = enc.drain() {
            tokens_before_drain.push(token);
        }

        let more = lipsum_bytes(512);
        enc.feed(&more);
        enc.finish();

        let mut all_tokens = tokens_before_drain;
        while let Some(token) = enc.next() {
            all_tokens.push(token);
        }

        let mut dec = Decoder::new();
        let mut decoded = Vec::new();
        for token in &all_tokens {
            dec.decode(token.seq, token_literals(token), &mut decoded);
        }

        let mut expected = data;
        expected.extend_from_slice(&more);
        assert_eq!(expected, decoded);
    }

    #[test]
    fn next_capped_basic() {
        let data = lipsum_bytes(1024);
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(&data);
        enc.finish();

        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        while let Some(token) = enc.next_capped(10) {
            assert!(token.seq.literal_len <= 10);
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        assert_eq!(data, decoded);
    }

    #[test]
    fn next_capped_zero_with_pending() {
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(b"abc");
        enc.finish();

        if enc.has_pending_literals() {
            assert!(enc.next_capped(0).is_none());
        }
    }

    #[test]
    fn unconsumed_input_basic() {
        let data = lipsum_bytes(2048);
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        enc.feed(&data);

        let mut consumed = 0;
        while let Some(token) = enc.next() {
            consumed += token.seq.literal_len as usize + token.seq.match_len as usize;
        }

        let unconsumed = enc.unconsumed_input();
        assert_eq!(consumed + unconsumed.len(), data.len());
    }

    proptest! {
        #[test]
        fn roundtrip(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            roundtrip_codec(&data, DEFAULT_LEVEL);
        }

        #[test]
        fn all_levels_roundtrip(
            data in prop::collection::vec(any::<u8>(), 1..1_024),
            level in 1u8..=4,
        ) {
            roundtrip_codec(&data, level);
        }

        #[test]
        fn sequence_invariants(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            let mut enc = Encoder::new(DEFAULT_LEVEL);
            enc.feed(&data);
            enc.finish();

            let mut total_bytes = 0_usize;

            while let Some(token) = enc.next() {
                prop_assert_eq!(token_literals(&token).len(), token.seq.literal_len as usize);
                prop_assert!(
                    token.seq.match_len == 0 || token.seq.match_len as usize >= MIN_MATCH,
                    "bad match_len: {}", token.seq.match_len,
                );
                if token.seq.match_len > 0 {
                    prop_assert!(token.seq.match_distance > 0);
                }
                let consumed = token.seq.literal_len as usize + token.seq.match_len as usize;
                prop_assert!(consumed > 0, "zero-byte token");
                total_bytes += consumed;
            }

            prop_assert_eq!(total_bytes, data.len());
        }

        #[test]
        fn streaming_roundtrip(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            let mut enc = Encoder::new(DEFAULT_LEVEL);
            let mut dec = Decoder::new();
            let mut decoded = Vec::new();

            let chunk_size = 100;
            for chunk in data.chunks(chunk_size) {
                enc.feed(chunk);
                while let Some(token) = enc.next() {
                    dec.decode(token.seq, token_literals(&token), &mut decoded);
                }
            }
            enc.finish();
            while let Some(token) = enc.next() {
                dec.decode(token.seq, token_literals(&token), &mut decoded);
            }

            prop_assert_eq!(&data[..], &decoded[..]);
        }

        #[test]
        fn capped_roundtrip(
            data in prop::collection::vec(any::<u8>(), 1..4_096),
            cap in 1u8..=255,
        ) {
            let mut enc = Encoder::new(DEFAULT_LEVEL);
            enc.feed(&data);
            enc.finish();

            let mut dec = Decoder::new();
            let mut decoded = Vec::new();

            while let Some(token) = enc.next_capped(cap) {
                prop_assert!(token.seq.literal_len <= cap);
                dec.decode(token.seq, token_literals(&token), &mut decoded);
            }

            prop_assert_eq!(&data[..], &decoded[..]);
        }
    }

    #[test]
    fn compact_triggers_on_large_incremental_feed() {
        // Feed data in chunks totaling >128 KiB to trigger maybe_compact().
        let data = lipsum_bytes(200 * 1024);
        let mut enc = Encoder::new(DEFAULT_LEVEL);
        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        for chunk in data.chunks(4096) {
            enc.feed(chunk);
            while let Some(token) = enc.next() {
                dec.decode(token.seq, token_literals(&token), &mut decoded);
            }
        }
        enc.finish();
        while let Some(token) = enc.next() {
            dec.decode(token.seq, token_literals(&token), &mut decoded);
        }

        assert_eq!(data, &decoded[..]);
    }
}
