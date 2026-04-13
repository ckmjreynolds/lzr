//! Greedy LZ77 encoder and decoder, one sequence at a time.
//!
//! A [`Sequence`] describes zero or more literal bytes followed by an optional
//! back-reference (match). The encoder uses a hash-chain match finder with a
//! 64 KB sliding window. The decoder maintains the same window to resolve
//! back-references.
//!
//! # Pipeline (future integration)
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

/// A [`Sequence`] together with the literal bytes it references.
///
/// The literal slice borrows from the encoder's input — no allocation per
/// token.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Token<'a> {
    pub(crate) seq: Sequence,
    pub(crate) literals: &'a [u8],
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

/// Greedy LZ77 encoder with hash-chain match finding.
///
/// Created with the full input slice; call [`next`](Self::next) repeatedly to
/// emit one [`Token`] at a time until `None` is returned.
pub(crate) struct Encoder<'a> {
    /// Full input being compressed.
    input: &'a [u8],
    /// Current read position.
    pos: usize,
    /// Hash-chain heads: `head[hash] = most recent absolute position`.
    head: Vec<u32>,
    /// Hash-chain links: `prev[pos % WINDOW_SIZE] = previous position with
    /// the same hash`.
    prev: Vec<u32>,
}

impl<'a> Encoder<'a> {
    /// Creates a new encoder over `input`.
    pub(crate) fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            head: vec![NIL; HASH_SIZE],
            prev: vec![NIL; WINDOW_SIZE],
        }
    }

    /// Returns the next token, or `None` if all input has been consumed.
    pub(crate) fn next(&mut self) -> Option<Token<'a>> {
        if self.pos >= self.input.len() {
            return None;
        }

        let lit_start = self.pos;
        let mut lit_len: usize = 0;

        loop {
            // Max literals reached — flush them without a match.
            if lit_len == MAX_LITERALS {
                #[allow(clippy::cast_possible_truncation)]
                return Some(Token {
                    seq: Sequence {
                        literal_len: lit_len as u8,
                        match_len: 0,
                        match_distance: 0,
                    },
                    literals: &self.input[lit_start..lit_start + lit_len],
                });
            }

            // No more input — flush remaining literals.
            if self.pos >= self.input.len() {
                break;
            }

            // Try to find a match at the current position.
            let (dist, mlen) = self.find_match();

            if mlen >= MIN_MATCH {
                // Insert hashes for every position inside the match so future
                // searches can find them.
                self.insert_hash(self.pos);
                for i in 1..mlen {
                    self.insert_hash(self.pos + i);
                }
                self.pos += mlen;

                #[allow(clippy::cast_possible_truncation)]
                return Some(Token {
                    seq: Sequence {
                        literal_len: lit_len as u8,
                        match_len: mlen as u8,
                        match_distance: dist as u16,
                    },
                    literals: &self.input[lit_start..lit_start + lit_len],
                });
            }

            // No match — accumulate as a literal.
            self.insert_hash(self.pos);
            self.pos += 1;
            lit_len += 1;
        }

        // Trailing literals (end of input).
        #[allow(clippy::cast_possible_truncation)]
        Some(Token {
            seq: Sequence {
                literal_len: lit_len as u8,
                match_len: 0,
                match_distance: 0,
            },
            literals: &self.input[lit_start..lit_start + lit_len],
        })
    }

    /// Inserts `self.pos` (which must have ≥ 3 bytes remaining) into the hash
    /// chain.
    fn insert_hash(&mut self, pos: usize) {
        if pos + 2 >= self.input.len() {
            return;
        }
        let h = hash3(self.input, pos);
        #[allow(clippy::cast_possible_truncation)]
        {
            self.prev[pos % WINDOW_SIZE] = self.head[h];
            self.head[h] = pos as u32;
        }
    }

    /// Finds the best match at the current position.  Returns `(distance,
    /// length)` or `(0, 0)` if no match ≥ `MIN_MATCH` exists.
    fn find_match(&self) -> (usize, usize) {
        let pos = self.pos;
        let remaining = self.input.len() - pos;
        let max_len = MAX_MATCH.min(remaining);

        if max_len < MIN_MATCH {
            return (0, 0);
        }

        let min_pos = pos.saturating_sub(WINDOW_SIZE);
        let h = hash3(self.input, pos);
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

            let len = common_prefix_len(&self.input[candidate..], &self.input[pos..], max_len);
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

    /// Encode-then-decode helper. Returns the decoded bytes.
    fn roundtrip_codec(data: &[u8]) -> Vec<u8> {
        let mut enc = Encoder::new(data);
        let mut dec = Decoder::new();
        let mut decoded = Vec::new();

        while let Some(token) = enc.next() {
            dec.decode(token.seq, token.literals, &mut decoded);
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

        let mut enc = Encoder::new(&data);
        let mut dec = Decoder::new();
        let mut decoded = Vec::with_capacity(data.len());
        let mut seq_count = 0_usize;
        let mut total_literal_bytes = 0_usize;

        while let Some(token) = enc.next() {
            seq_count += 1;
            total_literal_bytes += token.seq.literal_len as usize;
            dec.decode(token.seq, token.literals, &mut decoded);
        }

        // Compressed size proxy: 4-byte sequence header + literal bytes.
        // Matched bytes are "free" (replaced by the back-reference).
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
        // Exercises the stale-chain-entry path in find_match (candidate < min_pos).
        let data = lipsum_bytes(128 * 1024);
        roundtrip_codec(&data);
    }

    proptest! {
        #[test]
        fn roundtrip(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            roundtrip_codec(&data);
        }

        #[test]
        fn sequence_invariants(data in prop::collection::vec(any::<u8>(), 0..4_096)) {
            let mut enc = Encoder::new(&data);
            let mut total_bytes = 0_usize;

            while let Some(token) = enc.next() {
                prop_assert_eq!(token.literals.len(), token.seq.literal_len as usize);
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
    }
}
