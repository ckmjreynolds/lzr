//! LZ77 back-referencing over the `u15` token stream.
//!
//! This is a token preprocessor (`Transform<u15>`): it runs *after* the tokenizer and folds
//! repeated runs of tokens into back-references before the entropy coder sees them. A match is
//! emitted as three tokens — a reserved flag, then the match length and distance:
//!
//! ```text
//! [0x7FFE] [length] [distance]
//! ```
//!
//! Every other token is a literal, copied through verbatim. The flag `0x7FFE` is reserved: the
//! Re-Pair tokenizer caps its vocabulary one below it (see [`crate::tokenizers`]), so a literal
//! token can never equal it and the escape is unambiguous with no literal-escaping needed. The
//! `length`/`distance` *operands* may themselves take any `u15` value (including `0x7FFE`/`0x7FFF`):
//! the decoder reads them positionally, never re-scanning them as flags.
//!
//! Matches are at least [`MIN_MATCH`] tokens long, at most [`MAX_MATCH`], and reference a source at
//! most [`MAX_DISTANCE`] tokens back. Matching uses a bounded hash-chain finder with greedy parsing.

use anyhow::{Result, ensure};
use arbitrary_int::u15;
use static_assertions::const_assert;

use super::Transform;

/// Reserved flag token marking a match. Never a valid literal (the tokenizer caps its vocabulary
/// below it), so it can only introduce a `[flag, length, distance]` triple.
const LZ_FLAG: u16 = 0x7FFE;
/// Shortest match worth encoding, in tokens; also the width of the hash window. A match costs three
/// tokens, so length 2 expands the raw token count (3 for 2) yet still wins under the entropy coder
/// at large scale (a frequent flag plus a small length/distance code beats two literal tokens once
/// the corpus is repetitive enough).
///
/// NOTE: 2 is tuned for the large enwik8/enwik9 target, where it is best (−2.9% on full enwik8). It
/// is NOT the right value for a general-purpose compressor: a sweep showed the optimum drifts down
/// with input size (small text <1 MB wants 4, medium 2–10 MB wants 3, ≥40 MB wants 2), and 2 REGRESSES
/// small text badly (+7–9% on alice29/news/book1). The genuinely-robust single value is 3 (Pareto over
/// 4). Picking `MIN_MATCH` properly for a general compressor — ideally adaptive to input size, or a
/// size-gated choice — needs further attention.
const MIN_MATCH: usize = 2;
/// Longest single match, in tokens (must fit a `u15`). Longer runs are emitted as consecutive
/// matches.
const MAX_MATCH: usize = 0x7FFF;
/// Farthest a match may reference back, in tokens (`u15::MAX`, so the distance fits a `u15`).
const MAX_DISTANCE: usize = 0x7FFF;
/// Cap on hash-chain positions examined per starting position, bounding encode work to `O(n)`.
const MAX_CHAIN: usize = 128;
/// Log2 of the hash-table size (buckets). 2^17 buckets ≈ 512 KiB of `u32` heads.
const HASH_BITS: u32 = 17;
/// "No position" sentinel for the hash head/chain arrays.
const NIL_POS: u32 = u32::MAX;
/// Decompression-bomb ceiling for `inverse`: back-references can amplify a short stream, so the
/// reconstructed token count is bounded regardless of the (untrusted) input. A legitimate stream
/// (even enwik9's) stays far below this; the cap only bounds what a corrupt stream can demand.
const MAX_EXPANSION_TOKENS: usize = 1 << 31;

const_assert!(MIN_MATCH >= 2); // Below 2, a length-1 "match" is three tokens for one — always an expansion.
const_assert!(MAX_MATCH <= 0x7FFF); // A length must fit a `u15`.
const_assert!(MAX_DISTANCE <= 0x7FFF); // A distance must fit a `u15`.
const_assert!(LZ_FLAG < 0x8000); // The flag must be a valid `u15`.

/// Hash the `MIN_MATCH`-token window into a table bucket: fold each 15-bit lane through a
/// multiply-add, then Fibonacci-mix. Works for any window width.
fn hash(window: &[u15]) -> usize {
    let mut acc: u64 = 0;
    for token in window {
        acc = acc.wrapping_mul(0x0100_0000_01B3).wrapping_add(u64::from(token.value()));
    }
    let mixed = acc.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (mixed >> (64 - HASH_BITS)) as usize
}

/// LZ77 token preprocessor. See the module docs for the wire format.
pub(crate) struct Lz77;

impl Transform<u15> for Lz77 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "best_len <= MAX_MATCH and best_dist <= MAX_DISTANCE both fit u15; positions are < u32::MAX by the size guard"
    )]
    fn forward(&self, input: &[u15]) -> Vec<u15> {
        debug_assert!(
            input.iter().all(|token| token.value() != LZ_FLAG),
            "LZ77 input must not contain the reserved flag 0x7FFE"
        );
        let n = input.len();
        // Nothing to match, or too large to index with u32 positions: pass through unchanged.
        if n < MIN_MATCH || u32::try_from(n).is_err() {
            return input.to_vec();
        }
        let flag = u15::new(LZ_FLAG);
        let mut out = Vec::with_capacity(n);
        let mut head = vec![NIL_POS; 1usize << HASH_BITS];
        let mut prev = vec![NIL_POS; n];
        let mut i = 0usize;
        while i < n {
            // Only a full MIN_MATCH window can start (or be indexed as) a match.
            if i + MIN_MATCH > n {
                out.push(input[i]);
                i += 1;
                continue;
            }
            let bucket = hash(&input[i..i + MIN_MATCH]);
            let max_len = (n - i).min(MAX_MATCH);
            let mut best_len = 0usize;
            let mut best_dist = 0usize;
            let mut cand = head[bucket];
            let mut chain = MAX_CHAIN;
            while cand != NIL_POS && chain > 0 {
                let cpos = cand as usize;
                let dist = i - cpos;
                // The chain runs newest-to-oldest, so once one link is out of range all older ones
                // are too.
                if dist > MAX_DISTANCE {
                    break;
                }
                let mut len = 0usize;
                while len < max_len && input[cpos + len] == input[i + len] {
                    len += 1;
                }
                if len > best_len {
                    best_len = len;
                    best_dist = dist;
                    if len == max_len {
                        break;
                    }
                }
                cand = prev[cpos];
                chain -= 1;
            }
            if best_len >= MIN_MATCH {
                out.push(flag);
                out.push(u15::new(best_len as u16));
                out.push(u15::new(best_dist as u16));
                // Index every position the match covers so later matches can reference inside it.
                let end = i + best_len;
                while i < end {
                    if i + MIN_MATCH <= n {
                        let b = hash(&input[i..i + MIN_MATCH]);
                        prev[i] = head[b];
                        head[b] = i as u32;
                    }
                    i += 1;
                }
            } else {
                prev[i] = head[bucket];
                head[bucket] = i as u32;
                out.push(input[i]);
                i += 1;
            }
        }
        out
    }

    fn inverse(&self, input: &[u15]) -> Result<Vec<u15>> {
        let mut out: Vec<u15> = Vec::with_capacity(input.len());
        let mut p = 0usize;
        while p < input.len() {
            let token = input[p];
            if token.value() != LZ_FLAG {
                out.push(token);
                p += 1;
                continue;
            }
            // A match flag consumes two following operands.
            ensure!(p + 2 < input.len(), "LZ77 match flag is missing its length/distance operands");
            let len = usize::from(input[p + 1].value());
            let dist = usize::from(input[p + 2].value());
            p += 3;
            ensure!(len >= MIN_MATCH, "LZ77 match length {len} is below the minimum {MIN_MATCH}");
            ensure!(
                dist >= 1 && dist <= out.len(),
                "LZ77 match distance {dist} is out of range (output length {})",
                out.len()
            );
            ensure!(
                out.len().saturating_add(len) <= MAX_EXPANSION_TOKENS,
                "LZ77 expansion exceeds {MAX_EXPANSION_TOKENS} tokens"
            );
            // Copy one token at a time so overlapping matches (dist < len) reproduce the run.
            let start = out.len() - dist;
            let mut k = 0usize;
            while k < len {
                let copied = out[start + k];
                out.push(copied);
                k += 1;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use arbitrary_int::u15;
    use proptest::prelude::*;

    use super::{LZ_FLAG, Lz77};
    use crate::preprocessors::Transform;

    /// Build a token vector from raw `u16` values.
    fn tokens(values: &[u16]) -> Vec<u15> {
        values.iter().copied().map(u15::new).collect()
    }

    proptest! {
        /// Round-trips over any flag-free stream (the tokenizer guarantees no literal `0x7FFE`).
        #[test]
        fn roundtrip_any(raw in prop::collection::vec(0u16..=0x7FFF, 0..4096)) {
            let data: Vec<u15> = raw
                .into_iter()
                .map(|v| if v == LZ_FLAG { u15::new(0x7FFD) } else { u15::new(v) })
                .collect();
            let lz = Lz77;
            let restored = lz.inverse(&lz.forward(&data)).unwrap();
            prop_assert_eq!(restored, data);
        }

        /// A tiny alphabet maximizes matches (long overlapping runs, back-to-back triples).
        #[test]
        fn roundtrip_small_alphabet(raw in prop::collection::vec(0u16..4, 0..4096)) {
            let data = tokens(&raw);
            let lz = Lz77;
            let restored = lz.inverse(&lz.forward(&data)).unwrap();
            prop_assert_eq!(restored, data);
        }

        /// `inverse` must never panic on arbitrary token input — only `Ok` or `Err`. Values stay
        /// small so any decoded match length keeps the expansion cheap for the proptest budget.
        #[test]
        fn inverse_never_panics(
            raw in prop::collection::vec(prop_oneof![Just(LZ_FLAG), 0u16..=64], 0..256)
        ) {
            drop(Lz77.inverse(&tokens(&raw)));
        }
    }

    #[test]
    fn repeated_run_produces_a_match() {
        // A repeating 4-token pattern must be back-referenced: the output is shorter and carries
        // at least one flag triple, and it round-trips.
        let data = tokens(&[1, 2, 3, 4].repeat(64));
        let lz = Lz77;
        let encoded = lz.forward(&data);
        assert!(encoded.iter().any(|t| t.value() == LZ_FLAG), "expected at least one match flag");
        assert!(encoded.len() < data.len(), "matches must shrink the stream");
        assert_eq!(lz.inverse(&encoded).unwrap(), data);
    }

    #[test]
    fn overlapping_copy_reproduces_the_run() {
        // [literal 7] [flag] [len 6] [dist 1] expands to seven 7s (dist < len self-reference).
        let encoded = tokens(&[7, LZ_FLAG, 6, 1]);
        assert_eq!(Lz77.inverse(&encoded).unwrap(), tokens(&[7; 7]));
    }

    #[test]
    fn inverse_rejects_malformed_matches() {
        let lz = Lz77;
        // Truncated: a flag with fewer than two operands.
        assert!(lz.inverse(&tokens(&[LZ_FLAG])).is_err());
        assert!(lz.inverse(&tokens(&[LZ_FLAG, 5])).is_err());
        // Length below the minimum (MIN_MATCH is 2, so 1 is the largest illegal length).
        assert!(lz.inverse(&tokens(&[9, LZ_FLAG, 1, 1])).is_err());
        // Distance of zero, and a distance reaching past the output so far.
        assert!(lz.inverse(&tokens(&[9, LZ_FLAG, 4, 0])).is_err());
        assert!(lz.inverse(&tokens(&[9, LZ_FLAG, 4, 5])).is_err());
    }

    #[test]
    fn short_and_empty_inputs_pass_through() {
        let lz = Lz77;
        for raw in [vec![], vec![1u16], vec![1, 2, 3]] {
            let data = tokens(&raw);
            assert_eq!(lz.forward(&data), data);
            assert_eq!(lz.inverse(&lz.forward(&data)).unwrap(), data);
        }
    }
}
