//! Escape-byte LZ77 byte preprocessor.
//!
//! A classic sliding-window LZ77 that runs on the repair-tokenized byte stream (just before the
//! entropy coder) and replaces long back-references with compact match tokens. It emits literals
//! verbatim and introduces each match with one **marker** byte followed by a `(length, distance)`
//! pair, all uleb128-u22:
//!
//! ```text
//! [MODE]                       0x00 = pass-through (rest = input verbatim)
//!                              0x01 = folded, followed by:
//! [MARKER]                     the chosen marker byte
//! [payload …]                  literals and match tokens:
//!     b (b != MARKER)          a literal byte
//!     MARKER, uleb22(0)        a literal MARKER byte (escape; length 0 never names a match)
//!     MARKER, uleb22(len), uleb22(dist)   a match: copy `len` bytes from `dist` back
//! ```
//!
//! The marker is the **least-used** byte value in the input (lowest count, lowest byte value on a
//! tie — deterministic, so the decoder needs no side channel; it reads the marker from the header).
//! The upstream Re-Pair stage reserves symbol id 0, so byte `0x00` is normally absent and is chosen
//! as the marker with zero escapes; when the marker still occurs (e.g. Re-Pair off, or a literal
//! NUL) each such byte is escaped, so the stage stays reversible for *any* input.
//!
//! A match is emitted only when it pays for itself: by default when `len > 1 + u22_len(len) +
//! u22_len(dist)` (a strict shrink, so matches can never expand the stream); a fixed
//! [`Lz77::min_match`] instead emits any match of at least that length (which may expand). If the
//! whole folded stream is not smaller than the input, the stage falls back to an exact pass-through.

use anyhow::{Result, anyhow, ensure};
use arbitrary_int::u22;

use super::{MAX_EXPANSION_BYTES, MODE_FOLDED, Transform, inverse_framed, passthrough};
use crate::uleb128::{decode_u22, encode_u22, u22_len};

/// Largest match length/distance a u22 varint can hold (`u22::MAX`). Also the upper bound on a
/// caller-supplied fixed `min_match`.
pub(crate) const MAX_MATCH_LEN: u32 = 0x3F_FFFF;
/// Largest back-reference distance — the same u22 ceiling as [`MAX_MATCH_LEN`], named apart because
/// it bounds a distinct field.
const MAX_DISTANCE: u32 = MAX_MATCH_LEN;
/// Shortest emittable match. Length 0 is the literal-marker escape and length 1 never wins, so a
/// real match covers at least two bytes; a fixed `min_match` is validated to be `>= MIN_MATCH_LEN`.
pub(crate) const MIN_MATCH_LEN: u32 = 2;
/// Bytes hashed to seed the match search; also the shortest reference a chain lookup can find.
const HASH_BYTES: usize = 4;
/// Hash-table size (power of two): `head` has this many buckets.
const HASH_BITS: u32 = 17;
/// Sliding-window size (power of two, one past [`MAX_DISTANCE`]) for the ring-buffered `prev`
/// chain, so its memory is bounded regardless of input size.
const WINDOW: usize = 1 << 22;
/// Longest hash chain the matcher walks per position — a speed/ratio knob.
const MAX_CHAIN: usize = 128;
/// Sentinel for an empty hash slot (no real position equals it: positions are `< n <= u32::MAX`).
const NONE: u32 = u32::MAX;

/// An escape-byte LZ77 byte preprocessor. See the module docs for the on-stream format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lz77 {
    /// Fixed minimum match length. `None` uses the dynamic "only emit a winning match" rule;
    /// `Some(n)` (validated `>= MIN_MATCH_LEN`) emits any match of length `>= n`, which may expand.
    pub(crate) min_match: Option<u32>,
}

/// Reads the four bytes at `input[pos..]` as a little-endian `u32` for hashing. The caller
/// guarantees `pos + HASH_BYTES <= input.len()`.
fn load4(input: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes([input[pos], input[pos + 1], input[pos + 2], input[pos + 3]])
}

/// Fibonacci-hashes a 4-byte key into a `HASH_BITS`-wide bucket index.
const fn hash(key: u32) -> usize {
    (key.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
}

/// The length of the common prefix of `input[a..]` and `input[b..]`, capped at `max`. `a < b`, so
/// an overlapping run (`b - a < len`, i.e. RLE) is compared against the input itself and reproduced
/// byte-for-byte on decode.
fn common_prefix(input: &[u8], a: usize, b: usize, max: usize) -> usize {
    let mut k = 0;
    while k < max && input[a + k] == input[b + k] {
        k += 1;
    }
    k
}

impl Transform for Lz77 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "positions are < n <= u32::MAX and match lengths are <= MAX_MATCH_LEN, so every \
                  usize->u32 cast here is exact"
    )]
    fn forward(&self, input: Vec<u8>) -> Vec<u8> {
        let n = input.len();
        // Positions are stored as u32 in the hash chains; refuse (pass through) anything that would
        // not fit, mirroring the Re-Pair stage's u32 position bound.
        if u32::try_from(n).is_err() {
            return passthrough(&input);
        }

        // Marker = least-used byte value (lowest count, lowest value on a tie).
        let mut counts = [0u32; 256];
        for &byte in &input {
            counts[usize::from(byte)] += 1;
        }
        let marker = (0u8..=255).min_by_key(|&b| counts[usize::from(b)]).unwrap_or(0);

        let mask = WINDOW - 1;
        let mut head = vec![NONE; 1 << HASH_BITS];
        let mut prev = vec![NONE; n.min(WINDOW)];
        // Inserts position `p` at the head of its hash chain (skipping the final `HASH_BYTES - 1`
        // positions, which cannot seed a 4-byte hash).
        let insert = |head: &mut [u32], prev: &mut [u32], p: usize| {
            if p + HASH_BYTES <= n {
                let h = hash(load4(&input, p));
                prev[p & mask] = head[h];
                head[h] = p as u32;
            }
        };

        let mut out = Vec::with_capacity(n + 2);
        out.push(MODE_FOLDED);
        out.push(marker);

        let mut i = 0;
        while i < n {
            let (len, dist) = find_match(&input, i, &head, &prev, mask);
            if self.should_emit(len, dist) {
                out.push(marker);
                encode_u22(u22::new(len), &mut out);
                encode_u22(u22::new(dist), &mut out);
                let end = i + len as usize;
                while i < end {
                    insert(&mut head, &mut prev, i);
                    i += 1;
                }
            } else {
                if input[i] == marker {
                    // Escape a literal marker byte as `MARKER, uleb22(0)`.
                    out.push(marker);
                    encode_u22(u22::new(0), &mut out);
                } else {
                    out.push(input[i]);
                }
                insert(&mut head, &mut prev, i);
                i += 1;
            }
        }

        // Self-describing: keep the folded stream only if it actually beat a pass-through.
        if out.len() < n + 1 {
            out
        } else {
            // Folding lost: free the discarded buffer before allocating the pass-through copy so
            // peak memory stays ~2n rather than ~3n on incompressible input.
            drop(out);
            passthrough(&input)
        }
    }

    fn inverse(&self, input: Vec<u8>) -> Result<Vec<u8>> {
        inverse_framed(&input, "lz77", |rest| {
            let (&marker, payload) =
                rest.split_first().ok_or_else(|| anyhow!("lz77 header truncated: missing marker byte"))?;
            // The output is at least the payload length (literals alone), so seed the buffer with
            // that floor to avoid reallocating the whole stream as it grows.
            let mut out: Vec<u8> = Vec::with_capacity(payload.len());
            let mut pos = 0;
            while pos < payload.len() {
                let byte = payload[pos];
                pos += 1;
                if byte != marker {
                    out.push(byte);
                    continue;
                }
                let len = decode_u22(payload, &mut pos)?.value();
                if len == 0 {
                    // Escape: a literal marker byte.
                    out.push(marker);
                    continue;
                }
                let dist = decode_u22(payload, &mut pos)?.value() as usize;
                ensure!(dist >= 1 && dist <= out.len(), "lz77 match distance {dist} out of range");
                ensure!(
                    out.len() as u64 + u64::from(len) <= MAX_EXPANSION_BYTES,
                    "lz77 expansion exceeds {MAX_EXPANSION_BYTES} bytes"
                );
                let start = out.len() - dist;
                let len = len as usize;
                if len <= dist {
                    // Non-overlapping: a single bulk copy (which reserves internally).
                    out.extend_from_within(start..start + len);
                } else {
                    // Overlapping run (RLE): copy byte-by-byte as the output grows.
                    out.reserve(len);
                    for k in 0..len {
                        out.push(out[start + k]);
                    }
                }
            }
            Ok(out)
        })
    }
}

impl Lz77 {
    /// Whether a match of `(len, dist)` at the current position should be emitted: never below
    /// [`MIN_MATCH_LEN`]; then either the fixed `min_match` floor or the dynamic "it must shrink the
    /// stream" rule.
    fn should_emit(self, len: u32, dist: u32) -> bool {
        // Never below the two-byte floor (length 0 is the escape, length 1 never wins); then either
        // the fixed `min_match` floor or the dynamic "the match must shrink the stream" rule.
        len >= MIN_MATCH_LEN
            && self.min_match.map_or_else(|| len as usize > 1 + u22_len(len) + u22_len(dist), |floor| len >= floor)
    }
}

/// Finds the longest back-reference for the position `i` by walking its hash chain (most-recent
/// first, so distance only grows — the walk stops as soon as it passes [`MAX_DISTANCE`]). Returns
/// `(len, dist)`, with `len == 0` meaning "no usable match".
#[expect(
    clippy::cast_possible_truncation,
    reason = "candidate positions are < i <= n <= u32::MAX and the match length is capped at \
              MAX_MATCH_LEN, so both u32 casts are exact"
)]
fn find_match(input: &[u8], i: usize, head: &[u32], prev: &[u32], mask: usize) -> (u32, u32) {
    let n = input.len();
    if i + HASH_BYTES > n {
        return (0, 0);
    }
    let max_len = (n - i).min(MAX_MATCH_LEN as usize);
    let mut cand = head[hash(load4(input, i))];
    let (mut best_len, mut best_dist) = (0u32, 0u32);
    let mut chain = MAX_CHAIN;
    while cand != NONE && chain > 0 {
        let c = cand as usize;
        let dist = i - c;
        if dist > MAX_DISTANCE as usize {
            break;
        }
        let len = common_prefix(input, c, i, max_len);
        if len as u32 > best_len {
            best_len = len as u32;
            best_dist = dist as u32;
            if len >= max_len {
                break;
            }
        }
        cand = prev[c & mask];
        chain -= 1;
    }
    (best_len, best_dist)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::super::MODE_PASSTHROUGH;
    use super::*;

    /// The default (dynamic) LZ77 stage.
    fn lz() -> Lz77 {
        Lz77 {
            min_match: None,
        }
    }

    proptest! {
        /// LZ77 must round-trip *any* byte input exactly, via both the folded and pass-through paths.
        #[test]
        fn roundtrip_any(data in prop::collection::vec(any::<u8>(), 0..4096)) {
            let s = lz();
            prop_assert_eq!(s.inverse(s.forward(data.clone())).unwrap(), data);
        }

        /// A small alphabet produces many long, overlapping matches (RLE) — the copy path most
        /// likely to be mishandled.
        #[test]
        fn roundtrip_small_alphabet(data in prop::collection::vec(0u8..3, 0..8192)) {
            let s = lz();
            prop_assert_eq!(s.inverse(s.forward(data.clone())).unwrap(), data);
        }

        /// Text drawn from repeated phrases exercises the winning-match path heavily.
        #[test]
        fn roundtrip_repetitive(reps in 0usize..400) {
            let data = b"the quick brown fox ".repeat(reps);
            let s = lz();
            prop_assert_eq!(s.inverse(s.forward(data.clone())).unwrap(), data);
        }

        /// A fixed `min_match` (which may expand) must still round-trip exactly.
        #[test]
        fn roundtrip_fixed_min_match(
            data in prop::collection::vec(0u8..6, 0..4096),
            floor in MIN_MATCH_LEN..16,
        ) {
            let s = Lz77 { min_match: Some(floor) };
            prop_assert_eq!(s.inverse(s.forward(data.clone())).unwrap(), data);
        }

        /// Inverse runs on decoded, possibly-corrupt data — it may error but must never panic.
        #[test]
        fn inverse_never_panics(data in prop::collection::vec(any::<u8>(), 0..2048)) {
            drop(lz().inverse(data));
        }

        /// Inputs that saturate every byte value force the marker to be a byte that still occurs,
        /// exercising the escape path.
        #[test]
        fn dense_roundtrip(extra in prop::collection::vec(any::<u8>(), 0..512)) {
            let mut data: Vec<u8> = (0..=255).collect();
            data.extend_from_slice(&b"AAAAAAAAAAAAAAAA".repeat(8));
            data.extend(extra);
            let s = lz();
            prop_assert_eq!(s.inverse(s.forward(data.clone())).unwrap(), data);
        }
    }

    #[test]
    fn known_values_roundtrip() {
        let inputs: [&[u8]; 6] =
            [b"", b"a", b"abcabcabc", b"aaaaaaaaaa", b"the the the the", b"mississippi mississippi"];
        for input in inputs {
            assert_eq!(lz().inverse(lz().forward(input.to_vec())).unwrap(), input, "roundtrip failed for {input:?}");
        }
    }

    #[test]
    fn compresses_repetitive_input() {
        // A long run is replaced by one match token, so the folded stream is far shorter.
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(64);
        let folded = lz().forward(data.clone());
        assert_eq!(folded[0], MODE_FOLDED);
        assert!(folded.len() < data.len() / 2, "weak match coverage: {} bytes", folded.len());
        assert_eq!(lz().inverse(folded).unwrap(), data);
    }

    #[test]
    fn incompressible_input_passes_through() {
        // No exploitable repetition (and every byte value used) → the folded stream cannot beat a
        // pass-through, so the stage emits one.
        let mut data = vec![0u8; 256];
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        for byte in &mut data {
            state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *byte = state.to_be_bytes()[0];
        }
        assert_eq!(lz().forward(data.clone())[0], MODE_PASSTHROUGH);
        assert_eq!(lz().inverse(lz().forward(data.clone())).unwrap(), data);
    }
}
