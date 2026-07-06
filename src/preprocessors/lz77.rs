//! Token-aware escape-byte LZ77 preprocessor.
//!
//! Runs on the Re-Pair-tokenized byte stream (just before the entropy coder). Its input is a flat
//! sequence of canonical ULEB128-**u22** varints — the Re-Pair grammar-and-sequence stream — so this
//! stage parses those varints into a token array and matches **whole tokens**, never byte spans that
//! could straddle a varint boundary. Literals are complete tokens; a match covers whole tokens with
//! its `(length, distance)` measured in tokens. The emitted payload is therefore itself a clean
//! sequence of complete u22 integers (literals, plus `0x00`-introduced control records).
//!
//! ```text
//! [MODE]                          0x00 = pass-through (rest = input verbatim)
//!                                 0x01 = folded, followed by the payload:
//! [payload …]                     literals and control records:
//!     <uleb22 v> (v != 0)         a literal token (its verbatim, 0x00-free varint bytes)
//!     0x00, uleb22(0)             an escaped literal value-0 token (the only token whose varint
//!                                 starts with 0x00; see below)
//!     0x00, uleb22(len>=1), uleb22(dist)   a match: copy `len` tokens from `dist` tokens back
//! ```
//!
//! **The marker is the fixed byte `0x00`.** In a canonical u22 stream the only value whose encoding
//! begins with `0x00` is the value 0 (every multi-byte form sets the continuation bit on its
//! non-terminal bytes, and canonical decoding rejects a `0x00` terminating byte). Re-Pair reserves
//! symbol id 0, so its reduced sequence never contains a value-0 token; a value-0 token can still
//! appear as a terminal rule's raw byte child (a literal NUL in the pre-repair stream) or for an
//! empty grammar, and each such literal is escaped as `0x00, uleb22(0)`. So `0x00` is an unambiguous
//! match-introducer and no per-stream marker byte is needed. The first u22 after the marker is a
//! two-way discriminator: `0` escapes a literal value-0 token, any `len >= 1` is a match.
//!
//! A match is emitted only when it pays for itself: by default when the **summed varint width** of
//! the matched token span exceeds the record cost `1 + u22_len(len) + u22_len(dist)`. A fixed
//! [`Lz77::min_match`] instead emits any match of at least that many tokens (which may expand). If
//! the whole folded stream is not smaller than the input, the stage falls back to an exact
//! pass-through — the backstop against net expansion. When the input is not a clean canonical u22
//! stream (e.g. Re-Pair disabled, or arbitrary bytes), parsing bails and the stage passes through,
//! so it stays reversible for *any* input.

use anyhow::{Result, ensure};
use arbitrary_int::u22;

use super::{MAX_EXPANSION_BYTES, MODE_FOLDED, Transform, inverse_framed, passthrough};
use crate::uleb128::{decode_u22, encode_u22, u22_len};

/// Largest match length/distance a u22 varint can hold (`u22::MAX`), now measured in **tokens**. Also
/// the upper bound on a caller-supplied fixed `min_match`.
pub(crate) const MAX_MATCH_LEN: u32 = 0x3F_FFFF;
/// Largest back-reference distance in tokens — the same u22 ceiling as [`MAX_MATCH_LEN`], named apart
/// because it bounds a distinct field.
const MAX_DISTANCE: u32 = MAX_MATCH_LEN;
/// Shortest emittable match, in tokens. Length 0 is the value-0 escape and a single-token match can
/// never pay for itself, so a real match covers at least two tokens; a fixed `min_match` is validated
/// to be `>= MIN_MATCH_LEN`.
pub(crate) const MIN_MATCH_LEN: u32 = 2;
/// Consecutive token values hashed to seed the match search. Must equal [`MIN_MATCH_LEN`]: hashing
/// `k` tokens can only seed matches of length `>= k`, and the shortest legal match is two tokens.
const HASH_TOKENS: usize = 2;
/// Hash-table size (power of two): `head` has this many buckets.
const HASH_BITS: u32 = 17;
/// Sliding-window size (power of two, one past [`MAX_DISTANCE`]) for the ring-buffered chains, so
/// their memory is bounded regardless of input size.
const WINDOW: usize = 1 << 22;
/// Longest hash chain the matcher walks per position — a speed/ratio knob.
const MAX_CHAIN: usize = 128;
/// Sentinel for an empty hash slot (no real token position equals it: positions are `< T <= u32::MAX`).
const NONE: u32 = u32::MAX;

/// A token-aware escape-byte LZ77 preprocessor. See the module docs for the on-stream format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lz77 {
    /// Fixed minimum match length, in tokens. `None` uses the dynamic "only emit a winning match"
    /// rule; `Some(n)` (validated `>= MIN_MATCH_LEN`) emits any match of `>= n` tokens, which may expand.
    pub(crate) min_match: Option<u32>,
}

/// Packs two 22-bit token values into a 44-bit key and Fibonacci-hashes it into a `HASH_BITS`-wide
/// bucket index.
fn hash2(a: u32, b: u32) -> usize {
    let key = (u64::from(a) << 22) | u64::from(b);
    ((key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - HASH_BITS)) as u32) as usize
}

/// The length (in tokens) of the common prefix of `tokens[a..]` and `tokens[b..]`, capped at `max`.
/// `a < b`, so an overlapping run (`b - a < len`, i.e. RLE) is compared against the growing sequence
/// and reproduced token-for-token on decode.
fn common_prefix(tokens: &[u32], a: usize, b: usize, max: usize) -> usize {
    let mut k = 0;
    while k < max && tokens[a + k] == tokens[b + k] {
        k += 1;
    }
    k
}

impl Transform for Lz77 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "token positions and byte offsets are all < n <= u32::MAX, so every usize->u32 cast \
                  here is exact"
    )]
    fn forward(&self, input: Vec<u8>) -> Vec<u8> {
        let n = input.len();
        // Positions are stored as u32 in the hash chains; refuse (pass through) anything that would
        // not fit, mirroring the Re-Pair stage's u32 position bound.
        if u32::try_from(n).is_err() {
            return passthrough(&input);
        }

        // Parse the whole input into a flat token array plus each token's starting byte offset. Any
        // non-canonical or truncated varint means this is not a clean u22 stream, so pass through
        // unchanged (also the graceful path when the input did not come from Re-Pair). Because
        // `decode_u22` consumes whole canonical varints or errors, the loop can only land exactly on
        // `n` or bail — so `serialize(parse(input)) == input` holds on every folded path.
        // Every token is at least one byte, so `n / 2` is a safe under-estimate of the token count that
        // avoids the early doubling reallocations of a large buffer without over-committing memory.
        let mut tokens: Vec<u32> = Vec::with_capacity(n / 2);
        let mut offsets: Vec<u32> = Vec::with_capacity(n / 2 + 1);
        let mut pos = 0;
        while pos < n {
            offsets.push(pos as u32);
            match decode_u22(&input, &mut pos) {
                Ok(v) => tokens.push(v.value()),
                Err(_) => return passthrough(&input),
            }
        }
        offsets.push(n as u32); // sentinel: byte offset just past the last token
        let t = tokens.len();

        let mask = WINDOW - 1;
        let mut head = vec![NONE; 1 << HASH_BITS];
        let mut prev = vec![NONE; t.min(WINDOW)];
        // Inserts token position `p` at the head of its hash chain (skipping the final
        // `HASH_TOKENS - 1` positions, which cannot seed a two-token hash).
        let insert = |head: &mut [u32], prev: &mut [u32], p: usize| {
            if p + HASH_TOKENS <= t {
                let h = hash2(tokens[p], tokens[p + 1]);
                prev[p & mask] = head[h];
                head[h] = p as u32;
            }
        };

        let mut out = Vec::with_capacity(n + 1);
        out.push(MODE_FOLDED);

        let mut i = 0;
        while i < t {
            let (len, dist) = find_match(&tokens, i, &head, &prev, mask);
            // Byte width of the token span a match would replace: the summed varint widths of its
            // tokens, read off the offset prefix sum in O(1).
            let span = (offsets[i + len as usize] - offsets[i]) as usize;
            if self.should_emit(len, dist, span) {
                out.push(0x00);
                encode_u22(u22::new(len), &mut out);
                encode_u22(u22::new(dist), &mut out);
                let end = i + len as usize;
                while i < end {
                    insert(&mut head, &mut prev, i);
                    i += 1;
                }
            } else if tokens[i] == 0 {
                // Escape the only token whose varint starts with the 0x00 marker.
                out.push(0x00);
                encode_u22(u22::new(0), &mut out);
                insert(&mut head, &mut prev, i);
                i += 1;
            } else {
                // Literal: emit its verbatim (0x00-free) canonical varint bytes.
                out.extend_from_slice(&input[offsets[i] as usize..offsets[i + 1] as usize]);
                insert(&mut head, &mut prev, i);
                i += 1;
            }
        }

        // Self-describing: keep the folded stream only if it actually beat a pass-through (both carry
        // a single lead byte, so this is still `1 + payload < 1 + n`).
        if out.len() < n + 1 {
            out
        } else {
            // Folding lost: free the discarded buffer before allocating the pass-through copy so peak
            // memory stays lower on incompressible input.
            drop(out);
            passthrough(&input)
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "out.len() is guarded to <= MAX_EXPANSION_BYTES < 2^32, so the u32 offset cast is exact"
    )]
    fn inverse(&self, input: Vec<u8>) -> Result<Vec<u8>> {
        inverse_framed(&input, "lz77", |payload| {
            let mask = WINDOW - 1;
            // Reconstruct directly into the output byte buffer, copying literal and matched bytes
            // verbatim. `tok_off` is a WINDOW-sized ring of each recent token's start offset in `out`
            // (grown lazily, capped at WINDOW entries), enough to resolve any in-range back-reference.
            let mut out: Vec<u8> = Vec::with_capacity(payload.len());
            let mut tok_off: Vec<u32> = Vec::new();
            let mut t: usize = 0; // tokens emitted so far
            let mut pos = 0;
            while pos < payload.len() {
                if payload[pos] != 0x00 {
                    // Literal token: copy its raw canonical varint bytes verbatim.
                    let start = pos;
                    let _ = decode_u22(payload, &mut pos)?; // advance over the varint; validates canonical
                    ensure!(
                        out.len() as u64 + (pos - start) as u64 <= MAX_EXPANSION_BYTES,
                        "lz77 expansion exceeds {MAX_EXPANSION_BYTES} bytes"
                    );
                    push_off(&mut tok_off, t, out.len() as u32);
                    out.extend_from_slice(&payload[start..pos]);
                    t += 1;
                    continue;
                }
                pos += 1; // consume the 0x00 marker
                let first = decode_u22(payload, &mut pos)?.value();
                if first == 0 {
                    // Escaped literal value-0 token.
                    ensure!(
                        (out.len() as u64) < MAX_EXPANSION_BYTES,
                        "lz77 expansion exceeds {MAX_EXPANSION_BYTES} bytes"
                    );
                    push_off(&mut tok_off, t, out.len() as u32);
                    out.push(0x00);
                    t += 1;
                    continue;
                }
                // Match of `first` tokens at `dist` tokens back.
                let len = first as usize;
                let dist = decode_u22(payload, &mut pos)?.value() as usize;
                ensure!(dist >= 1 && dist <= t, "lz77 match distance {dist} out of range");
                let src = t - dist;
                for k in 0..len {
                    // The source token is fully materialized (`src + k < t`, and within WINDOW), so
                    // its bytes lie wholly before the write point — an in-bounds, non-overlapping copy.
                    let s = tok_off[(src + k) & mask] as usize;
                    let mut sp = s;
                    let _ = decode_u22(&out, &mut sp)?; // advance over the source token: width = sp - s
                    ensure!(
                        out.len() as u64 + (sp - s) as u64 <= MAX_EXPANSION_BYTES,
                        "lz77 expansion exceeds {MAX_EXPANSION_BYTES} bytes"
                    );
                    push_off(&mut tok_off, t, out.len() as u32);
                    out.extend_from_within(s..sp);
                    t += 1;
                }
            }
            Ok(out)
        })
    }

    /// The number of match records in the folded output, for the CLI's per-stage trace (how many
    /// token back-references were emitted). `None` for a pass-through stream (nothing folded) or a
    /// malformed one.
    fn trace_detail(&self, output: &[u8]) -> Option<(u64, &'static str)> {
        let (&mode, payload) = output.split_first()?;
        if mode != MODE_FOLDED {
            return None;
        }
        let mut pos = 0;
        let mut matches = 0u64;
        while pos < payload.len() {
            if payload[pos] != 0x00 {
                let _ = decode_u22(payload, &mut pos).ok()?; // literal: skip its varint
                continue;
            }
            pos += 1;
            if decode_u22(payload, &mut pos).ok()?.value() != 0 {
                matches += 1;
                let _ = decode_u22(payload, &mut pos).ok()?; // skip the distance
            }
        }
        Some((matches, "matches"))
    }
}

impl Lz77 {
    /// Whether a match of `(len, dist)` tokens whose replaced span is `span_bytes` wide should be
    /// emitted: never below [`MIN_MATCH_LEN`] tokens; then either the fixed `min_match` floor or the
    /// dynamic "the match record must be smaller than the bytes it replaces" rule. `1 + u22_len(len) +
    /// u22_len(dist)` is exactly the record cost (marker byte plus the two varints).
    fn should_emit(self, len: u32, dist: u32, span_bytes: usize) -> bool {
        len >= MIN_MATCH_LEN
            && self.min_match.map_or_else(|| span_bytes > 1 + u22_len(len) + u22_len(dist), |floor| len >= floor)
    }
}

/// Records token `t`'s start offset `off`, keeping only the most recent `WINDOW` entries: absolute
/// indices while the ring is filling (`t < WINDOW`), then ring reuse. `t & (WINDOW - 1)` equals the
/// absolute index while `t < WINDOW`, so reads via the mask are consistent across the switch.
fn push_off(tok_off: &mut Vec<u32>, t: usize, off: u32) {
    if t < WINDOW {
        tok_off.push(off);
    } else {
        tok_off[t & (WINDOW - 1)] = off;
    }
}

/// Finds the longest back-reference for token position `i` by walking its hash chain (most-recent
/// first, so distance only grows — the walk stops as soon as it passes [`MAX_DISTANCE`]). Returns
/// `(len, dist)` in tokens, with `len == 0` meaning "no usable match".
#[expect(
    clippy::cast_possible_truncation,
    reason = "candidate positions are < i <= T <= u32::MAX, the match length is capped at MAX_MATCH_LEN, \
              and the distance is capped at MAX_DISTANCE, so both u32 casts are exact"
)]
fn find_match(tokens: &[u32], i: usize, head: &[u32], prev: &[u32], mask: usize) -> (u32, u32) {
    let t = tokens.len();
    if i + HASH_TOKENS > t {
        return (0, 0);
    }
    let max_len = (t - i).min(MAX_MATCH_LEN as usize);
    let mut cand = head[hash2(tokens[i], tokens[i + 1])];
    let (mut best_len, mut best_dist) = (0u32, 0u32);
    let mut chain = MAX_CHAIN;
    while cand != NONE && chain > 0 {
        let c = cand as usize;
        let dist = i - c;
        if dist > MAX_DISTANCE as usize {
            break;
        }
        let len = common_prefix(tokens, c, i, max_len);
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

    /// The default (dynamic) token-aware LZ77 stage.
    fn lz() -> Lz77 {
        Lz77 {
            min_match: None,
        }
    }

    /// Serializes token values as a canonical u22 stream — the shape Re-Pair produces and this stage
    /// consumes. Every value must be `<= u22::MAX`.
    fn encode_tokens(values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &v in values {
            encode_u22(u22::new(v), &mut out);
        }
        out
    }

    proptest! {
        /// LZ77 must round-trip *any* byte input exactly, via both the folded and pass-through paths.
        #[test]
        fn roundtrip_any(data in prop::collection::vec(any::<u8>(), 0..4096)) {
            let s = lz();
            prop_assert_eq!(s.inverse(s.forward(data.clone())).unwrap(), data);
        }

        /// Any canonical u22 token stream must round-trip — this directly exercises the fold path and
        /// the `serialize(parse(x)) == x` invariant on real token input.
        #[test]
        fn fold_path_roundtrips_token_streams(values in prop::collection::vec(0u32..=MAX_MATCH_LEN, 0..2048)) {
            let s = encode_tokens(&values);
            let lz = lz();
            prop_assert_eq!(lz.inverse(lz.forward(s.clone())).unwrap(), s);
        }

        /// A small alphabet (single-byte tokens including value 0) produces many long, overlapping
        /// matches (RLE) and exercises the value-0 escape — the paths most likely to be mishandled.
        #[test]
        fn roundtrip_small_alphabet(data in prop::collection::vec(0u8..3, 0..8192)) {
            let s = lz();
            prop_assert_eq!(s.inverse(s.forward(data.clone())).unwrap(), data);
        }

        /// Text drawn from repeated phrases (single-byte tokens) exercises the winning-match path.
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
    fn folds_repeated_token_run() {
        // A repeated multi-token phrase (mixed 1/2/3-byte tokens) must fold and round-trip.
        let phrase = [1u32, 1000, 40000, 7, 0x3F_FFFF];
        let mut values = Vec::new();
        for _ in 0..64 {
            values.extend_from_slice(&phrase);
        }
        let s = encode_tokens(&values);
        let folded = lz().forward(s.clone());
        assert_eq!(folded[0], MODE_FOLDED);
        assert!(folded.len() < s.len() / 2, "weak folding: {} vs {} bytes", folded.len(), s.len());
        assert_eq!(lz().inverse(folded).unwrap(), s);
    }

    #[test]
    fn value_zero_tokens_survive_folding() {
        // Value-0 tokens (a literal-NUL terminal in a Re-Pair grammar) must survive via the escape,
        // both as bounded literals and inside a folded repeat.
        let phrase = [0u32, 1234, 0, 99];
        let mut values = Vec::new();
        for _ in 0..64 {
            values.extend_from_slice(&phrase);
        }
        let s = encode_tokens(&values);
        let folded = lz().forward(s.clone());
        assert_eq!(folded[0], MODE_FOLDED);
        assert_eq!(lz().inverse(folded).unwrap(), s);
    }

    #[test]
    fn empty_and_single_zero_pass_through() {
        let s = lz();
        assert_eq!(s.forward(Vec::new())[0], MODE_PASSTHROUGH);
        assert_eq!(s.inverse(s.forward(Vec::new())).unwrap(), Vec::<u8>::new());
        // A single value-0 token `[0x00]`: can't match (needs two tokens) → escaped → fold loses →
        // pass-through, still reversible.
        let one_zero = vec![0x00u8];
        assert_eq!(s.forward(one_zero.clone())[0], MODE_PASSTHROUGH);
        assert_eq!(s.inverse(s.forward(one_zero.clone())).unwrap(), one_zero);
    }

    #[test]
    fn non_canonical_input_passes_through() {
        // A trailing dangling continuation byte is not a clean u22 stream → pass-through, reversible.
        let mut dangling = encode_tokens(&[5, 300, 5, 300, 5, 300]);
        dangling.push(0x80);
        let s = lz();
        assert_eq!(s.forward(dangling.clone())[0], MODE_PASSTHROUGH);
        assert_eq!(s.inverse(s.forward(dangling.clone())).unwrap(), dangling);
        // A non-canonical two-byte form `[0x80, 0x00]` is likewise rejected → pass-through.
        let non_canonical = vec![0x80u8, 0x00];
        assert_eq!(s.forward(non_canonical.clone())[0], MODE_PASSTHROUGH);
        assert_eq!(s.inverse(s.forward(non_canonical.clone())).unwrap(), non_canonical);
    }

    #[test]
    fn overlapping_rle_roundtrips() {
        // Many copies of one multi-byte token → a dist-1 RLE match; inverse takes the `len > dist` path.
        let s = encode_tokens(&vec![1000u32; 500]);
        let folded = lz().forward(s.clone());
        assert_eq!(folded[0], MODE_FOLDED);
        assert!(folded.len() < s.len() / 2, "weak RLE folding: {} bytes", folded.len());
        assert_eq!(lz().inverse(folded).unwrap(), s);
    }

    #[test]
    fn inverse_rejects_out_of_range_distance() {
        // Marker, len=2, dist=5 with nothing emitted yet → distance out of range (must error, not panic).
        let mut stream = vec![MODE_FOLDED, 0x00];
        encode_u22(u22::new(2), &mut stream);
        encode_u22(u22::new(5), &mut stream);
        assert!(lz().inverse(stream).is_err());
    }

    #[test]
    fn inverse_rejects_truncated_match() {
        // A marker at end-of-payload with no following length → truncated (must error, not panic).
        let stream = vec![MODE_FOLDED, 0x00];
        assert!(lz().inverse(stream).is_err());
    }
}
