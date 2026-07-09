//! Byte-BPE Re-Pair grammar tokenizer — a byte→byte [`Transform`].
//!
//! Re-Pair repeatedly replaces the most frequent adjacent digram with a new non-terminal, building a
//! grammar over the input. This byte-centric variant claims the block's **unused byte values** as its
//! non-terminals and runs until they are exhausted — so after it runs (given enough redundancy) all
//! 256 byte values are in use, and the whole thing is one plain byte stream:
//!
//! ```text
//! [R] [(sym_1, left_1, right_1)] … [(sym_R, left_R, right_R)] [sequence …]
//! ```
//!
//! `R` (a ULEB128 `u64`) is the number of grammar rules; each rule is three raw bytes — the
//! non-terminal byte `sym` and its two children `left`/`right` (each itself either a literal byte or
//! another non-terminal). A byte not named as a rule's `sym` is a literal that expands to itself, so
//! the grammar is fully described by the rules alone. The reduced sequence follows as plain bytes.
//!
//! The grammar is built by the space-efficient frequency-based Re-Pair of Bille–Gørtz–Prezza (2017)
//! (see [`builder`]), capped at a 256-symbol vocabulary so every symbol id maps onto a distinct byte
//! value. Rule order is not guaranteed topological, so `inverse` treats the grammar as a DAG and
//! rejects cycles.

mod builder;

use anyhow::{Result, bail, ensure};

use super::MAX_EXPANSION_BYTES;
use crate::transform::Transform;
use crate::uleb128::{decode_u64, encode_u64};

/// Total symbol-id ceiling (exclusive) for the grammar build: ids run `0..256`, so every symbol maps
/// to a distinct byte value and Re-Pair stops once the block's free byte slots are all consumed. Ids
/// are `u16` in the builder (256 fits, leaving `u16::MAX` free as the builder's `BLANK` sentinel).
const VOCAB_CAP: u16 = 256;

/// A resolved grammar symbol used by `inverse`, one per byte value.
#[derive(Debug, Clone, Copy)]
enum Def {
    /// Terminal: expands to this byte (the default for every byte not named by a rule).
    Byte(u8),
    /// Non-terminal: the concatenation of two other symbols (each a byte value).
    Pair(u8, u8),
}

/// The byte-BPE Re-Pair grammar tokenizer.
///
/// `forward` claims the input's unused byte values as non-terminals, derives a per-input grammar with
/// the space-efficient Re-Pair builder ([`builder`]) capped at 256 symbols, and serializes it as plain
/// bytes inline ahead of the reduced sequence; `inverse` parses that self-describing grammar (as a
/// cycle-checked DAG) and expands the sequence back to bytes. There are no per-run parameters — the
/// stage runs until the free byte slots are exhausted.
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RepairTokenizer;

impl Transform for RepairTokenizer {
    #[expect(clippy::cast_possible_truncation, reason = "terminal count and t are <= 256, which fits u16")]
    fn forward(&self, input: Vec<u8>) -> Vec<u8> {
        if input.is_empty() {
            return Vec::new();
        }
        // Distinct present bytes become terminals (ascending byte order); absent bytes are the free
        // slots the grammar's non-terminals will claim.
        let mut present = [false; 256];
        for &byte in &input {
            present[usize::from(byte)] = true;
        }
        let mut terminals: Vec<u8> = Vec::new();
        let mut byte_to_id = [0u16; 256];
        for byte in 0u8..=255 {
            if present[usize::from(byte)] {
                // `terminals.len() < 256` here, so the cast to `u16` is exact.
                byte_to_id[usize::from(byte)] = terminals.len() as u16;
                terminals.push(byte);
            }
        }
        let free: Vec<u8> = (0u8..=255).filter(|&b| !present[usize::from(b)]).collect();
        let t = terminals.len();
        // Map the input to compact terminal symbol ids, then free the byte buffer before the grammar
        // build — the builder works on `symbols`, and holding the input too would cost ~1 GB at 1 GiB blocks.
        let symbols: Vec<u16> = input.iter().map(|&byte| byte_to_id[usize::from(byte)]).collect();
        let fits = u32::try_from(input.len()).is_ok();
        drop(input);
        // Build the grammar to its natural floor, bounded by the 256-symbol cap so each non-terminal id
        // `t + k` maps onto the `k`-th free byte. `t <= 256` so the `u16` cast is exact. Identity
        // fallback if the input exceeds u32 positions.
        let (rules, sequence) = if fits {
            builder::build_grammar(symbols, t as u16, VOCAB_CAP)
        } else {
            (Vec::new(), symbols)
        };
        // Map every symbol id to its byte value: terminal id `i` -> the i-th present byte, non-terminal
        // id `t + k` -> the k-th free byte. `rules.len() <= free.len()` by the vocab cap.
        let mut id_to_byte: Vec<u8> = terminals;
        id_to_byte.extend_from_slice(&free[..rules.len()]);
        // Serialize `[R] [(sym, left, right)] * R [sequence]`, all as plain bytes.
        let mut out = Vec::with_capacity(1 + rules.len() * 3 + sequence.len());
        encode_u64(rules.len() as u64, &mut out);
        for (k, &(left, right)) in rules.iter().enumerate() {
            out.push(id_to_byte[t + k]);
            out.push(id_to_byte[usize::from(left)]);
            out.push(id_to_byte[usize::from(right)]);
        }
        for &sym in &sequence {
            out.push(id_to_byte[usize::from(sym)]);
        }
        out
    }

    fn inverse(&self, input: Vec<u8>) -> Result<Vec<u8>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let mut pos = 0;
        let rule_count = decode_u64(&input, &mut pos)?;
        // Every byte defaults to a literal expanding to itself; the rules override the non-terminals.
        let mut dict = [Def::Byte(0); 256];
        for b in 0u8..=255 {
            dict[usize::from(b)] = Def::Byte(b);
        }
        for _ in 0..rule_count {
            ensure!(pos + 3 <= input.len(), "Re-Pair rule table is truncated");
            let sym = input[pos];
            let left = input[pos + 1];
            let right = input[pos + 2];
            pos += 3;
            dict[usize::from(sym)] = Def::Pair(left, right);
        }
        // Rule order is not guaranteed topological, so resolve expansion lengths over the grammar DAG
        // (this also rejects cycles). Then stream the sequence, bounding each byte's expansion against
        // the decompression-bomb ceiling *before* expanding it.
        let lengths = expansion_lengths(&dict)?;
        let mut total: u64 = 0;
        let mut output = Vec::new();
        let mut stack: Vec<u8> = Vec::new();
        for &byte in &input[pos..] {
            total = total.saturating_add(lengths[usize::from(byte)]);
            ensure!(total <= MAX_EXPANSION_BYTES, "Re-Pair expansion exceeds {MAX_EXPANSION_BYTES} bytes");
            stack.push(byte);
            while let Some(symbol) = stack.pop() {
                match dict[usize::from(symbol)] {
                    Def::Byte(b) => output.push(b),
                    Def::Pair(left, right) => {
                        stack.push(right);
                        stack.push(left);
                    }
                }
            }
        }
        Ok(output)
    }

    /// The Re-Pair rule count for the CLI trace: the leading `[R]` header of a non-empty stream. Empty
    /// output (empty input) reports nothing.
    fn trace_detail(&self, output: &[u8]) -> Option<(u64, &'static str)> {
        let mut pos = 0;
        decode_u64(output, &mut pos).ok().map(|r| (r, "rules"))
    }
}

/// Resolves each grammar symbol's expansion length in bytes over the (possibly non-topological) DAG
/// via an iterative post-order walk, rejecting cycles. Lengths saturate so a decompression bomb stays
/// finite for the caller's cap.
fn expansion_lengths(dict: &[Def; 256]) -> Result<Vec<u64>> {
    // Per-symbol state: 0 = unvisited, 1 = on the current path, 2 = resolved.
    let mut lengths = vec![0u64; 256];
    let mut state = [0u8; 256];
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..256 {
        if state[start] != 0 {
            continue;
        }
        stack.push(start);
        while let Some(&node) = stack.last() {
            if state[node] == 2 {
                let _ = stack.pop();
                continue;
            }
            match dict[node] {
                Def::Byte(_) => {
                    lengths[node] = 1;
                    state[node] = 2;
                    let _ = stack.pop();
                }
                Def::Pair(left, right) => {
                    let (left, right) = (usize::from(left), usize::from(right));
                    if state[node] == 0 {
                        state[node] = 1;
                        for child in [left, right] {
                            match state[child] {
                                1 => bail!("Re-Pair grammar contains a cycle"),
                                0 => stack.push(child),
                                _ => {}
                            }
                        }
                    } else {
                        lengths[node] = lengths[left].saturating_add(lengths[right]);
                        state[node] = 2;
                        let _ = stack.pop();
                    }
                }
            }
        }
    }
    Ok(lengths)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::path::Path;

    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::RepairTokenizer;
    use crate::transform::Transform;
    use crate::uleb128::encode_u64;

    fn tok() -> RepairTokenizer {
        RepairTokenizer
    }

    fn corpus(rel: &str) -> Option<Vec<u8>> {
        std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).ok()
    }

    proptest! {
        #[test]
        fn roundtrip_any(data in prop::collection::vec(any::<u8>(), 0..4096)) {
            let restored = tok().inverse(tok().forward(data.clone())).unwrap();
            prop_assert_eq!(restored, data);
        }

        #[test]
        fn roundtrip_small_alphabet(data in prop::collection::vec(0u8..4, 0..4096)) {
            let restored = tok().inverse(tok().forward(data.clone())).unwrap();
            prop_assert_eq!(restored, data);
        }

        /// `inverse` must never panic on arbitrary byte input — only `Ok` or `Err`.
        #[test]
        fn inverse_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
            drop(tok().inverse(bytes));
        }
    }

    #[test]
    fn known_values_roundtrip() {
        let inputs: [&[u8]; 5] = [b"", b"a", b"aaaa", b"abracadabra", b"mississippi"];
        for input in inputs {
            assert_eq!(tok().inverse(tok().forward(input.to_vec())).unwrap(), input);
        }
    }

    #[test]
    fn compresses_repetitive_input() {
        let data = vec![b'a'; 8192];
        assert!(tok().forward(data.clone()).len() <= data.len());
    }

    #[test]
    fn consumes_free_bytes_on_redundant_text() {
        // A redundant text input over a small alphabet has many free byte values; Re-Pair should claim
        // them, so the output uses substantially more distinct byte values than the input did.
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(256);
        let mut present_in = [false; 256];
        for &b in &data {
            present_in[usize::from(b)] = true;
        }
        let out = tok().forward(data.clone());
        let mut present_out = [false; 256];
        for &b in &out {
            present_out[usize::from(b)] = true;
        }
        let d_in = present_in.iter().filter(|&&p| p).count();
        let d_out = present_out.iter().filter(|&&p| p).count();
        assert!(d_out > d_in, "expected Re-Pair to claim free byte values ({d_in} -> {d_out})");
        assert_eq!(tok().inverse(out).unwrap(), data);
    }

    #[test]
    fn rule_count_header_is_reported() {
        let data = b"abracadabra abracadabra abracadabra".to_vec();
        let out = tok().forward(data.clone());
        let (rules, unit) = tok().trace_detail(&out).unwrap();
        assert_eq!(unit, "rules");
        assert!(rules >= 1, "a repetitive input should yield at least one rule");
        assert_eq!(tok().inverse(out).unwrap(), data);
    }

    #[test]
    fn full_alphabet_input_is_passthrough_grammar() {
        // Input using all 256 byte values has no free slots, so Re-Pair emits zero rules.
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let out = tok().forward(data.clone());
        assert_eq!(out[0], 0, "no free bytes -> zero rules");
        assert_eq!(tok().inverse(out).unwrap(), data);
    }

    #[test]
    fn inverse_rejects_truncated_rule_table() {
        // Claim one rule but supply no rule bytes.
        let mut bytes = Vec::new();
        encode_u64(1, &mut bytes);
        assert!(tok().inverse(bytes).is_err());
    }

    #[test]
    fn inverse_rejects_cycle() {
        // A single rule whose non-terminal references itself is a cycle.
        let mut bytes = Vec::new();
        encode_u64(1, &mut bytes);
        bytes.extend_from_slice(&[0x00, 0x00, 0x01]); // sym 0 = (0, 1) -> self-reference
        bytes.push(0x00); // a sequence byte referencing the cyclic symbol
        assert!(tok().inverse(bytes).is_err());
    }

    #[test]
    fn bible_roundtrip() {
        let Some(data) = corpus("corpora/large/bible.txt") else {
            return;
        };
        let slice = &data[..data.len().min(256 * 1024)];
        assert_eq!(tok().inverse(tok().forward(slice.to_vec())).unwrap(), slice);
    }
}
