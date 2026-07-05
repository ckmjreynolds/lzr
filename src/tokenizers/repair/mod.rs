//! Capped Re-Pair grammar tokenizer.
//!
//! Re-Pair repeatedly replaces the most frequent adjacent digram with a new non-terminal, building
//! a grammar over the input. This tokenizer runs Re-Pair to a hard `u15` vocabulary cap (32766
//! symbols = up to 256 terminals + non-terminals) and emits the grammar *inline* ahead of the
//! reduced symbol sequence, so the whole thing is one `u15` token stream the codec can carry
//! without a separate dictionary channel:
//!
//! ```text
//! [V-1] [(left_0, right_0)] … [(left_{V-1}, right_{V-1})] [sequence …]
//! ```
//!
//! `V` is the symbol count (at most 32766). Two high `u15` values are reserved and never used as
//! symbol ids: `0x7FFF` = `u15::MAX` is the NIL sentinel (marks a terminal), and `0x7FFE` is left
//! free for the downstream LZ77 token stage's flag (see [`crate::preprocessors`]).
//! Every symbol is a uniform `(left, right)` pair — a terminal is `(byte, NIL)`, a non-terminal is
//! `(left, right)` referencing two other symbols. Symbol ids are assigned by **descending reference
//! frequency** — id 0 is the most-used symbol — so low-valued (short) ids code cheaply; this
//! matters most for a non-entropy back end. Frequency order is not topological, so `inverse` treats
//! the grammar as a DAG and rejects cycles. The grammar is built by the space-efficient
//! frequency-based Re-Pair of Bille–Gørtz–Prezza (2017); see [`builder`].

mod builder;

use anyhow::{Result, anyhow, bail, ensure};
use arbitrary_int::u15;
use static_assertions::const_assert;

use crate::tokenizers::Tokenize;

/// Reserved sentinel symbol (`u15::MAX`); never a valid symbol id, so it marks a terminal as the
/// right child of a uniform `(left, right)` grammar definition.
const NIL_SYM: u16 = 0x7FFF;
/// Vocabulary ceiling: symbol ids run `0..VOCAB_CAP`, so the grammar holds at most 32766 symbols.
/// This stays strictly below both reserved sentinels — [`NIL_SYM`] (`0x7FFF`) and the LZ77 flag
/// `0x7FFE` the downstream token stage claims — so neither ever appears as a symbol id.
const VOCAB_CAP: u32 = 0x7FFE;
/// Decompression-bomb ceiling for `inverse`: a small grammar can expand exponentially, so the
/// total expansion is bounded regardless of the (untrusted) grammar.
const MAX_EXPANSION_BYTES: u64 = 1 << 31;

const_assert!(VOCAB_CAP == 0x7FFE);

/// A resolved grammar symbol used by `inverse`.
#[derive(Debug, Clone, Copy)]
enum Def {
    /// Terminal: expands to this byte.
    Byte(u8),
    /// Non-terminal: the concatenation of two earlier symbols.
    Pair(u16, u16),
}

/// A capped Re-Pair grammar tokenizer.
///
/// `forward` derives a per-input grammar with the space-efficient frequency-based Re-Pair builder
/// ([`builder`]), renumbers its symbols by descending reference frequency (id 0 = most used), and
/// emits it inline ahead of the reduced sequence; `inverse` parses that self-describing grammar (as
/// a cycle-checked DAG) and expands the sequence back to bytes. The tokenizer is stateless — all
/// per-input grammar state travels in the token stream.
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct RepairTokenizer;

impl Tokenize for RepairTokenizer {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "vocab <=32768 so vocab-1 and ranks fit u16; terminal count <=256"
    )]
    fn forward(&self, input: Vec<u8>) -> Vec<u15> {
        if input.is_empty() {
            return Vec::new();
        }
        // Compact terminals to the distinct present bytes, in ascending byte order.
        let mut present = [false; 256];
        for &byte in &input {
            present[usize::from(byte)] = true;
        }
        let mut terminals: Vec<u8> = Vec::new();
        let mut byte_to_id = [0u16; 256];
        for byte in 0u8..=255 {
            if present[usize::from(byte)] {
                byte_to_id[usize::from(byte)] = terminals.len() as u16;
                terminals.push(byte);
            }
        }
        let t = terminals.len();
        // Map the input to compact terminal symbol ids, then free the byte buffer before the grammar
        // build — the builder works on `symbols`, and holding the input too would cost ~1 GB at enwik9.
        let symbols: Vec<u16> = input.iter().map(|&byte| byte_to_id[usize::from(byte)]).collect();
        let fits = u32::try_from(input.len()).is_ok();
        drop(input);
        // Build the capped grammar (identity fallback if the input exceeds u32 position indices).
        let (rules, sequence) = if fits {
            builder::build_grammar(symbols, t as u16)
        } else {
            (Vec::new(), symbols)
        };
        // Assemble the topological grammar as a flat definition list (terminals then rules).
        let vocab = t + rules.len();
        let mut dict: Vec<Def> = Vec::with_capacity(vocab);
        for &byte in &terminals {
            dict.push(Def::Byte(byte));
        }
        for &(left, right) in &rules {
            dict.push(Def::Pair(left, right));
        }
        // Count how often each symbol is *referenced* (in the sequence and as a rule child), then
        // renumber by descending frequency so id 0 is the most-used symbol (short ids code cheaply).
        let mut freq = vec![0u64; vocab];
        for &sym in &sequence {
            freq[usize::from(sym)] += 1;
        }
        for &(left, right) in &rules {
            freq[usize::from(left)] += 1;
            freq[usize::from(right)] += 1;
        }
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_unstable_by(|&a, &b| freq[b].cmp(&freq[a]).then(a.cmp(&b)));
        let mut new_id = vec![0u16; vocab];
        for (rank, &old) in order.iter().enumerate() {
            new_id[old] = rank as u16;
        }
        // Emit `[V-1]`, then each definition — uniformly `(left, right)`, a terminal being
        // `(byte, NIL)` — in new-id order, then the remapped sequence.
        let mut out = Vec::with_capacity(1 + 2 * vocab + sequence.len());
        out.push(u15::new((vocab - 1) as u16));
        for &old in &order {
            match dict[old] {
                Def::Byte(byte) => {
                    out.push(u15::new(u16::from(byte)));
                    out.push(u15::new(NIL_SYM));
                }
                Def::Pair(left, right) => {
                    out.push(u15::new(new_id[usize::from(left)]));
                    out.push(u15::new(new_id[usize::from(right)]));
                }
            }
        }
        for &sym in &sequence {
            out.push(u15::new(new_id[usize::from(sym)]));
        }
        out
    }

    fn inverse(&self, input: &[u15]) -> Result<Vec<u8>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let vocab = usize::from(input[0].value()) + 1;
        // Parse the uniform definition list: each symbol is `(left, right)`, a terminal being
        // `(byte, NIL)` and a non-terminal a pair of symbol ids.
        let mut dict: Vec<Def> = Vec::with_capacity(vocab);
        let mut pos = 1;
        for _ in 0..vocab {
            ensure!(pos + 1 < input.len(), "Re-Pair definition list is truncated");
            let left = input[pos].value();
            let right = input[pos + 1].value();
            pos += 2;
            if right == NIL_SYM {
                let byte = u8::try_from(left).map_err(|_| anyhow!("Re-Pair terminal token {left} is not a byte"))?;
                dict.push(Def::Byte(byte));
            } else {
                ensure!(
                    usize::from(left) < vocab && usize::from(right) < vocab,
                    "Re-Pair rule references an undefined symbol"
                );
                dict.push(Def::Pair(left, right));
            }
        }
        let sequence = &input[pos..];
        // Frequency order is not topological, so resolve expansion lengths over the grammar DAG
        // (this also rejects cycles), then bound the total against a decompression bomb.
        let lengths = expansion_lengths(&dict)?;
        let mut total: u64 = 0;
        for &token in sequence {
            let index = usize::from(token.value());
            ensure!(index < vocab, "Re-Pair sequence symbol {} is out of range", token.value());
            total = total.saturating_add(lengths[index]);
            ensure!(total <= MAX_EXPANSION_BYTES, "Re-Pair expansion exceeds {MAX_EXPANSION_BYTES} bytes");
        }
        // Expand via an explicit stack (the grammar is an acyclic DAG, so this terminates).
        let capacity = usize::try_from(total.min(1 << 24)).unwrap_or(0);
        let mut output = Vec::with_capacity(capacity);
        let mut stack: Vec<u16> = Vec::new();
        for &token in sequence {
            stack.push(token.value());
            while let Some(symbol) = stack.pop() {
                match dict[usize::from(symbol)] {
                    Def::Byte(byte) => output.push(byte),
                    Def::Pair(left, right) => {
                        stack.push(right);
                        stack.push(left);
                    }
                }
            }
        }
        Ok(output)
    }
}

/// Resolves each grammar symbol's expansion length in bytes over the (possibly non-topological)
/// DAG via an iterative post-order walk, rejecting cycles. Lengths saturate so a decompression
/// bomb stays finite for the caller's cap.
fn expansion_lengths(dict: &[Def]) -> Result<Vec<u64>> {
    // Per-symbol state: 0 = unvisited, 1 = on the current path, 2 = resolved.
    let mut lengths = vec![0u64; dict.len()];
    let mut state = vec![0u8; dict.len()];
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..dict.len() {
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

    use arbitrary_int::u15;
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::RepairTokenizer;
    use crate::tokenizers::Tokenize;

    fn corpus(rel: &str) -> Option<Vec<u8>> {
        std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).ok()
    }

    proptest! {
        #[test]
        fn roundtrip_any(data in prop::collection::vec(any::<u8>(), 0..4096)) {
            let restored = RepairTokenizer.inverse(&RepairTokenizer.forward(data.clone())).unwrap();
            prop_assert_eq!(restored, data);
        }

        #[test]
        fn roundtrip_small_alphabet(data in prop::collection::vec(0u8..4, 0..4096)) {
            let restored = RepairTokenizer.inverse(&RepairTokenizer.forward(data.clone())).unwrap();
            prop_assert_eq!(restored, data);
        }

        /// `inverse` must never panic on arbitrary token input — only `Ok` or `Err`.
        #[test]
        fn inverse_never_panics(raw in prop::collection::vec(0u16..=0x7FFF, 0..512)) {
            let tokens: Vec<u15> = raw.into_iter().map(u15::new).collect();
            drop(RepairTokenizer.inverse(&tokens));
        }
    }

    #[test]
    fn known_values_roundtrip() {
        let inputs: [&[u8]; 5] = [b"", b"a", b"aaaa", b"abracadabra", b"mississippi"];
        let tok = RepairTokenizer;
        for input in inputs {
            assert_eq!(tok.inverse(&tok.forward(input.to_vec())).unwrap(), input);
        }
    }

    #[test]
    fn compresses_repetitive_input() {
        let data = vec![b'a'; 8192];
        assert!(RepairTokenizer.forward(data.clone()).len() <= data.len());
    }

    #[test]
    fn large_grammar_stays_within_vocabulary() {
        // A pseudo-random 256 KiB block makes Re-Pair create thousands of rules; the emitted
        // vocabulary must stay within the u15 cap and still round-trip.
        let mut data = vec![0u8; 256 * 1024];
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        for byte in &mut data {
            state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *byte = state.to_be_bytes()[0];
        }
        let tok = RepairTokenizer;
        let tokens = tok.forward(data.clone());
        let vocab = usize::from(tokens[0].value()) + 1;
        assert!(vocab <= 32766, "vocabulary {vocab} exceeds the reserved-sentinel cap");
        // Every token is a valid u15 and the reserved LZ77 flag `0x7FFE` never appears.
        assert!(tokens.iter().all(|token| token.value() <= 0x7FFF && token.value() != 0x7FFE));
        assert_eq!(tok.inverse(&tokens).unwrap(), data);
    }

    #[test]
    fn ids_are_frequency_ordered() {
        // No pair repeats in "zzab", so the vocabulary is just the terminals, renumbered by
        // descending reference frequency: the most-frequent byte 'z' becomes id 0 even though it is
        // the largest byte value present.
        let tok = RepairTokenizer;
        let tokens = tok.forward(b"zzab".to_vec());
        assert_eq!(tokens[0].value(), 2); // V - 1 (three terminals, no rules)
        assert_eq!(tokens[1].value(), u16::from(b'z')); // def_0 left = 'z', the most frequent byte
        assert_eq!(tokens[2].value(), 0x7FFF); // ...right = NIL, marking it a terminal
        assert_eq!(tok.inverse(&tokens).unwrap(), b"zzab".to_vec());
    }

    #[test]
    fn inverse_rejects_malformed_tokens() {
        let tok = RepairTokenizer;
        let mut tokens = tok.forward(b"abracadabra".to_vec());
        tokens.push(u15::new(0x7FFF)); // dangling reference to a nonexistent symbol
        assert!(tok.inverse(&tokens).is_err());
        assert!(tok.inverse(&[u15::new(0x7FFF)]).is_err());
    }

    #[test]
    fn bible_roundtrip() {
        let Some(data) = corpus("corpora/large/bible.txt") else {
            return;
        };
        let slice = &data[..data.len().min(256 * 1024)];
        let tok = RepairTokenizer;
        assert_eq!(tok.inverse(&tok.forward(slice.to_vec())).unwrap(), slice);
    }
}
