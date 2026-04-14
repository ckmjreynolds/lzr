//! Greedy LZ77 encoder and decoder.
//!
//! A [`Sequence`] describes zero or more literal bytes followed by an optional
//! back-reference (match). The encoder uses a hash-chain match finder with a
//! 64 KB sliding window. The decoder maintains the same window to resolve
//! back-references.
//!
//! The encoder is streaming: data is fed via [`Encoder::feed`], and tokens are
//! pulled out via [`Encoder::next`]. Call [`Encoder::finish`] to signal end of
//! input, then drain remaining tokens.
//!
//! # Pipeline
//!
//! ```text
//! Encode:  Bytes → LZ77 Encoder  → Arithmetic Coder
//! Decode:  Arithmetic Decoder    → LZ77 Decoder → Bytes
//! ```

use static_assertions::assert_eq_size;

/// Sliding-window size in bytes (matches the `u16` distance range).
const WINDOW_SIZE: usize = 65_536;

/// Minimum useful match length.  Shorter matches cost more to encode than
/// the literal bytes they replace.
const MIN_MATCH: usize = 3;

/// Maximum match length (constrained by `match_len: u8`).
const MAX_MATCH: usize = 255;

/// Maximum literal run per sequence (constrained by `literal_len: u8`).
const MAX_LITERALS: usize = 255;

/// Number of bits in the hash (2^15 = 32 768 buckets).
const HASH_BITS: u32 = 15;

/// Hash-table size.
const HASH_SIZE: usize = 1 << HASH_BITS;

/// Maximum number of hash-chain steps before giving up.
const MAX_CHAIN: usize = 32;

/// Sentinel value meaning "no entry" in the hash tables.
const NIL: u32 = u32::MAX;

/// Minimum lookahead required in streaming mode before emitting tokens.
/// Ensures matches are not truncated at the buffer boundary.
const LOOKAHEAD_MIN: usize = MAX_MATCH + MIN_MATCH;

/// One LZ77 token: zero or more literal bytes followed by an optional
/// back-reference.
///
/// * `literal_len` — number of literal bytes that precede the match (0..=255).
/// * `match_len`   — length of the back-reference copy.  **0** means no match;
///   valid matches are in the range `3..=255`.
/// * `match_distance` — how far back to look (1..=65 535).  Ignored when
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

/// Multiply-shift hash over the three bytes at `data[pos..pos+3]`.
#[allow(clippy::cast_possible_truncation)]
fn hash3(data: &[u8], pos: usize) -> usize {
    let h = u32::from(data[pos]) | (u32::from(data[pos + 1]) << 8) | (u32::from(data[pos + 2]) << 16);
    (h.wrapping_mul(0x1E35_A7BD) >> (32 - HASH_BITS)) as usize
}

/// Returns the length of the common prefix of `a` and `b`, up to `max`.
fn common_prefix_len(a: &[u8], b: &[u8], max: usize) -> usize {
    a.iter().zip(b.iter()).take(max).take_while(|(x, y)| x == y).count()
}

/// Greedy streaming LZ77 encoder with hash-chain match finding.
///
/// Data is fed incrementally via [`feed`](Self::feed). Call
/// [`next`](Self::next) to pull tokens. Call [`finish`](Self::finish)
/// to signal end of input, then drain remaining tokens with `next`.
pub(crate) struct Encoder {
    /// Compacting flat buffer holding recent + unprocessed input.
    buf: Vec<u8>,
    /// Absolute stream offset of `buf[0]`.
    base: usize,
    /// Current encode position (absolute).
    pos: usize,
    /// Absolute position of the first pending literal byte.
    lit_start: usize,
    /// Number of pending literal bytes.
    lit_len: usize,
    /// Hash-chain heads: `head[hash] = most recent absolute position`.
    head: Vec<u32>,
    /// Hash-chain links: `prev[pos % WINDOW_SIZE] = previous position with
    /// the same hash`.
    prev: Vec<u32>,
    /// True once [`finish`](Self::finish) has been called.
    finished: bool,
}

impl Encoder {
    /// Creates a new empty encoder.
    pub(crate) fn new() -> Self {
        Self {
            buf: Vec::with_capacity(2 * WINDOW_SIZE + MAX_MATCH),
            base: 0,
            pos: 0,
            lit_start: 0,
            lit_len: 0,
            head: vec![NIL; HASH_SIZE],
            prev: vec![NIL; WINDOW_SIZE],
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
    ///
    /// After this call, [`next`](Self::next) will drain all remaining data.
    pub(crate) const fn finish(&mut self) {
        self.finished = true;
    }

    /// Emits any pending literals as a literal-only token without signaling
    /// end of input.
    pub(crate) fn drain(&mut self) -> Option<Token> {
        if self.lit_len == 0 {
            return None;
        }
        Some(self.emit_literals())
    }

    /// Processes all remaining buffered data and returns the resulting tokens.
    ///
    /// Used at frame boundaries and seal. Unlike [`finish`](Self::finish),
    /// the encoder can continue receiving [`feed`](Self::feed) calls after
    /// this. Hash chains and window state are preserved.
    pub(crate) fn flush(&mut self) -> Vec<Token> {
        self.finished = true;
        let mut tokens = Vec::new();
        while let Some(token) = self.next() {
            tokens.push(token);
        }
        self.finished = false;
        tokens
    }

    /// Resets all encoder state for a new block.
    pub(crate) fn reset(&mut self) {
        self.buf.clear();
        self.base = 0;
        self.pos = 0;
        self.lit_start = 0;
        self.lit_len = 0;
        self.head.fill(NIL);
        self.prev.fill(NIL);
        self.finished = false;
    }

    /// Returns the next token, or `None` if more data is needed.
    ///
    /// In streaming mode (before [`finish`](Self::finish)), returns `None`
    /// when insufficient lookahead remains. After `finish`, drains all
    /// remaining data.
    pub(crate) fn next(&mut self) -> Option<Token> {
        let available = self.base + self.buf.len();

        loop {
            // Max literals reached — flush them without a match.
            if self.lit_len == MAX_LITERALS {
                return Some(self.emit_literals());
            }

            // No more data at current position.
            if self.pos >= available {
                if self.finished {
                    break;
                }
                return None;
            }

            // In streaming mode, require enough lookahead.
            if !self.finished && available - self.pos < LOOKAHEAD_MIN {
                return None;
            }

            // Try to find a match at the current position.
            let (dist, mlen) = self.find_match();

            if mlen >= MIN_MATCH {
                self.insert_hash(self.pos);
                for i in 1..mlen {
                    self.insert_hash(self.pos + i);
                }
                self.pos += mlen;
                return Some(self.emit_match(dist, mlen));
            }

            // No match — accumulate as a literal.
            self.insert_hash(self.pos);
            self.pos += 1;
            self.lit_len += 1;
        }

        // Trailing literals (end of input).
        if self.lit_len > 0 {
            Some(self.emit_literals())
        } else {
            None
        }
    }

    /// Builds a literal-only token from pending literals and resets them.
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

    /// Builds a token with pending literals and a match, then resets literals.
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

    /// Compacts the buffer when it grows too large, keeping the match window
    /// and unprocessed data.
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

    /// Inserts `pos` into the hash chain (if ≥ 3 bytes remain at that
    /// position).
    #[allow(clippy::cast_possible_truncation)]
    fn insert_hash(&mut self, pos: usize) {
        // Guard against u32 overflow for extremely long streams.
        if pos > u32::MAX as usize - WINDOW_SIZE {
            self.head.fill(NIL);
            self.prev.fill(NIL);
            return;
        }
        let buf_pos = pos - self.base;
        if buf_pos + 2 >= self.buf.len() {
            return;
        }
        let h = hash3(&self.buf, buf_pos);
        self.prev[pos % WINDOW_SIZE] = self.head[h];
        self.head[h] = pos as u32;
    }

    /// Finds the best match at the current position.  Returns `(distance,
    /// length)` or `(0, 0)` if no match ≥ `MIN_MATCH` exists.
    fn find_match(&self) -> (usize, usize) {
        let pos = self.pos;
        let available = self.base + self.buf.len();
        let remaining = available - pos;
        let max_len = MAX_MATCH.min(remaining);

        if max_len < MIN_MATCH {
            return (0, 0);
        }

        let buf_pos = pos - self.base;
        let min_pos = pos.saturating_sub(WINDOW_SIZE);
        let h = hash3(&self.buf, buf_pos);
        let mut chain_pos = self.head[h];
        let mut best_len = MIN_MATCH - 1;
        let mut best_dist = 0;

        let mut steps = 0;
        while chain_pos != NIL && steps < MAX_CHAIN {
            let candidate = chain_pos as usize;

            // Stale entry — stop.
            if candidate < min_pos {
                break;
            }

            let ci = candidate - self.base;
            let len = common_prefix_len(&self.buf[ci..], &self.buf[buf_pos..], max_len);
            if len > best_len {
                best_len = len;
                best_dist = pos - candidate;
                if best_len == max_len {
                    break;
                }
            }

            chain_pos = self.prev[candidate % WINDOW_SIZE];
            steps += 1;
        }

        if best_len >= MIN_MATCH {
            (best_dist, best_len)
        } else {
            (0, 0)
        }
    }
}

/// LZ77 decoder that reconstructs raw bytes from a stream of [`Sequence`]
/// tokens.
///
/// Maintains a 64 KB circular window of previously decoded output so that
/// back-references can be resolved.
pub(crate) struct Decoder {
    /// Circular buffer of recently decoded bytes.
    window: Vec<u8>,
    /// Write cursor (absolute byte count; index via `pos % WINDOW_SIZE`).
    pos: usize,
}

impl Decoder {
    /// Creates a new decoder with an empty window.
    pub(crate) fn new() -> Self {
        Self {
            window: vec![0; WINDOW_SIZE],
            pos: 0,
        }
    }

    /// Resets the decoder for a new block.
    pub(crate) fn reset(&mut self) {
        self.window.fill(0);
        self.pos = 0;
    }

    /// Decodes one sequence, appending the result to `output`.
    ///
    /// `literals` must have exactly `seq.literal_len` bytes.
    pub(crate) fn decode(&mut self, seq: Sequence, literals: &[u8], output: &mut Vec<u8>) {
        debug_assert_eq!(literals.len(), seq.literal_len as usize);

        // 1. Copy literal bytes.
        for &b in literals {
            output.push(b);
            self.window[self.pos % WINDOW_SIZE] = b;
            self.pos += 1;
        }

        // 2. Copy the back-reference, byte-by-byte (handles overlapping runs).
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::bench::lipsum_bytes;

    /// Helper: valid literal slice from a token.
    fn token_literals(token: &Token) -> &[u8] {
        &token.literals[..token.seq.literal_len as usize]
    }

    /// Encode-then-decode helper using feed/finish/next. Returns decoded bytes.
    fn roundtrip_codec(data: &[u8]) -> Vec<u8> {
        let mut enc = Encoder::new();
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
        roundtrip_codec(b"");
    }

    #[test]
    fn lipsum_roundtrip_compresses() {
        let data = lipsum_bytes(65_536);

        let mut enc = Encoder::new();
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

        // Compressed size proxy: 4-byte sequence header + literal bytes.
        let encoded_size = seq_count * size_of::<Sequence>() + total_literal_bytes;

        #[allow(clippy::cast_precision_loss)]
        let ratio = 100.0 * (1.0 - encoded_size as f64 / data.len() as f64);
        eprintln!(
            "lz77 compression ratio: {ratio:.1}% ({} → {} bytes, {} sequences)",
            data.len(),
            encoded_size,
            seq_count,
        );

        assert_eq!(data, &decoded[..]);
        assert!(encoded_size < data.len());
    }

    #[test]
    fn larger_than_window() {
        // Exercises the stale-chain-entry path in find_match (candidate < min_pos)
        // and buffer compaction.
        let data = lipsum_bytes(128 * 1024);
        roundtrip_codec(&data);
    }

    #[test]
    fn multi_feed_roundtrip() {
        let data = lipsum_bytes(4096);

        let mut enc = Encoder::new();
        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        // Feed in small chunks, draining tokens between feeds.
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

        let mut enc = Encoder::new();
        enc.feed(&data);

        // Drain some tokens via next().
        let mut tokens_before_drain = Vec::new();
        while let Some(token) = enc.next() {
            tokens_before_drain.push(token);
        }

        // Drain pending literals.
        if let Some(token) = enc.drain() {
            tokens_before_drain.push(token);
        }

        // Feed more data and finish.
        let more = lipsum_bytes(512);
        enc.feed(&more);
        enc.finish();

        let mut all_tokens = tokens_before_drain;
        while let Some(token) = enc.next() {
            all_tokens.push(token);
        }

        // Decode all and verify.
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
    fn reset_between_blocks() {
        let block1 = lipsum_bytes(2048);
        let block2 = vec![0x42; 2048]; // different data

        // Encode block 1.
        let mut enc = Encoder::new();
        enc.feed(&block1);
        enc.finish();
        let mut tokens1 = Vec::new();
        while let Some(token) = enc.next() {
            tokens1.push(token);
        }

        // Reset and encode block 2.
        enc.reset();
        enc.feed(&block2);
        enc.finish();
        let mut tokens2 = Vec::new();
        while let Some(token) = enc.next() {
            tokens2.push(token);
        }

        // Decode both blocks with reset between.
        let mut dec = Decoder::new();
        let mut decoded1 = Vec::new();
        for token in &tokens1 {
            dec.decode(token.seq, token_literals(token), &mut decoded1);
        }
        assert_eq!(block1, decoded1);

        dec.reset();
        let mut decoded2 = Vec::new();
        for token in &tokens2 {
            dec.decode(token.seq, token_literals(token), &mut decoded2);
        }
        assert_eq!(block2, decoded2);
    }

    proptest! {
        #[test]
        fn roundtrip(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            roundtrip_codec(&data);
        }

        #[test]
        fn sequence_invariants(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            let mut enc = Encoder::new();
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
            let mut enc = Encoder::new();
            let mut dec = Decoder::new();
            let mut decoded = Vec::new();

            // Feed in random-sized chunks.
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
    }
}
