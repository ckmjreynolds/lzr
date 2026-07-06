//! Capped Re-Pair grammar tokenizer — a byte→byte [`Transform`].
//!
//! Re-Pair repeatedly replaces the most frequent adjacent digram with a new non-terminal, building
//! a grammar over the input. This tokenizer runs Re-Pair to a caller-supplied vocabulary cap
//! ([`RepairTokenizer::num_tokens`], default `u22::MAX - 1`), renumbers the symbols, and serializes
//! the grammar *inline* ahead of the reduced symbol sequence with the [`crate::uleb128`] u22 varint
//! codec, so the whole thing is one byte stream the codec can carry without a separate dictionary
//! channel:
//!
//! ```text
//! [V] [(left_1, right_1)] … [(left_V, right_V)] [sequence …]      (each value uleb128-u22)
//! ```
//!
//! `V` is the symbol count (at most `u22::MAX - 1`), and the header carries the *max id* `V`. Real
//! symbol ids run `1..=V`: **id 0 is reserved and never emitted**, so the byte value `0x00` stays
//! out of the serialized stream and is a free marker for the downstream LZ77 stage. Two `u22`
//! values are thus special — id 0 (reserved) and [`NIL_SYM`] = `u22::MAX`, the NIL sentinel that
//! marks a terminal. Every symbol is a uniform `(left, right)` pair — a terminal is `(byte, NIL)`,
//! a non-terminal is `(left, right)` referencing two other symbols. Ids are assigned by
//! **descending reference frequency** — id 1 is the most-used symbol — so low-valued (short) ids
//! serialize to fewer bytes; this matters most for a non-entropy back end. Frequency order is not
//! topological, so `inverse` treats the grammar as a DAG and rejects cycles. The grammar is built
//! by the space-efficient frequency-based Re-Pair of Bille–Gørtz–Prezza (2017); see [`builder`].

mod builder;
mod prune;

use anyhow::{Result, anyhow, bail, ensure};
use arbitrary_int::u22;

use super::MAX_EXPANSION_BYTES;
use crate::transform::Transform;
use crate::uleb128::{decode_u22, encode_u22};

/// Reserved sentinel symbol (`u22::MAX`); never a valid symbol id, so it marks a terminal as the
/// right child of a uniform `(left, right)` grammar definition. This is the only reserved value.
const NIL_SYM: u32 = 0x3F_FFFF; // == u22::MAX
/// Default vocabulary cap: symbol ids run `0..DEFAULT_NUM_TOKENS`, one below [`NIL_SYM`], so the
/// sentinel can never collide with a real id.
pub(crate) const DEFAULT_NUM_TOKENS: u32 = NIL_SYM - 1; // 0x3F_FFFE
/// Smallest sensible cap: the input can use all 256 byte values, so the cap must admit the terminals.
pub(crate) const MIN_NUM_TOKENS: u32 = 256;

/// A resolved grammar symbol used by `inverse`.
#[derive(Debug, Clone, Copy)]
enum Def {
    /// Terminal: expands to this byte.
    Byte(u8),
    /// Non-terminal: the concatenation of two earlier symbols.
    Pair(u32, u32),
}

/// A capped Re-Pair grammar tokenizer.
///
/// `forward` derives a per-input grammar with the space-efficient frequency-based Re-Pair builder
/// ([`builder`]), renumbers its symbols by descending reference frequency (id 0 = most used), and
/// serializes it (uleb128-u22) inline ahead of the reduced sequence; `inverse` parses that
/// self-describing grammar (as a cycle-checked DAG) and expands the sequence back to bytes. The only
/// per-run parameter is the vocabulary cap `num_tokens` (encode-side only — `inverse` reads the
/// actual vocabulary from the stream's `[V-1]` prefix).
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct RepairTokenizer {
    /// Vocabulary ceiling for this run. Under `cost_stop` the grammar is built to its natural floor
    /// (bounded only by the safety ceiling [`DEFAULT_NUM_TOKENS`]) and this is ignored; otherwise it
    /// is the post-build count cap — the surviving vocabulary is trimmed to at most `num_tokens`.
    pub(crate) num_tokens: u32,
    /// Prune the built grammar by the serialized-byte cost model rather than to a fixed `num_tokens`
    /// count (see [`prune`]). The default; the CLI turns it off when an explicit `--num-tokens` asks
    /// for a hard count cap instead.
    pub(crate) cost_stop: bool,
}

impl Default for RepairTokenizer {
    fn default() -> Self {
        Self {
            num_tokens: DEFAULT_NUM_TOKENS,
            cost_stop: true,
        }
    }
}

impl Transform for RepairTokenizer {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "vocab <= num_tokens <= 0x3F_FFFE, so vocab, ids (rank+1), and ranks all fit u32/u22"
    )]
    fn forward(&self, input: Vec<u8>) -> Vec<u8> {
        if input.is_empty() {
            return Vec::new();
        }
        // Compact terminals to the distinct present bytes, in ascending byte order.
        let mut present = [false; 256];
        for &byte in &input {
            present[usize::from(byte)] = true;
        }
        let mut terminals: Vec<u8> = Vec::new();
        let mut byte_to_id = [0u32; 256];
        for byte in 0u8..=255 {
            if present[usize::from(byte)] {
                byte_to_id[usize::from(byte)] = terminals.len() as u32;
                terminals.push(byte);
            }
        }
        let t = terminals.len();
        // Map the input to compact terminal symbol ids, then free the byte buffer before the grammar
        // build — the builder works on `symbols`, and holding the input too would cost ~1 GB at enwik9.
        let symbols: Vec<u32> = input.iter().map(|&byte| byte_to_id[usize::from(byte)]).collect();
        let fits = u32::try_from(input.len()).is_ok();
        drop(input);
        // Build the grammar to its natural floor (bounded by the safety ceiling), then prune it —
        // either by the serialized-byte cost model or down to the `num_tokens` count cap. Both run
        // *after* the build so the pruning sees true reference frequencies. Identity fallback if the
        // input exceeds u32 position indices.
        let (rules, sequence) = if fits {
            let (rules, sequence) = builder::build_grammar(symbols, t as u32, DEFAULT_NUM_TOKENS);
            prune::prune_grammar(t as u32, rules, sequence, self.num_tokens, self.cost_stop)
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
            freq[sym as usize] += 1;
        }
        for &(left, right) in &rules {
            freq[left as usize] += 1;
            freq[right as usize] += 1;
        }
        let mut order: Vec<usize> = (0..vocab).collect();
        order.sort_unstable_by(|&a, &b| freq[b].cmp(&freq[a]).then(a.cmp(&b)));
        // Real ids run `1..=V`: id 0 is reserved and never emitted, so the byte value `0x00`
        // stays out of the serialized stream (a free marker for the downstream LZ77 stage). The
        // header therefore carries the *max id* `V` rather than `V-1`.
        let mut new_id = vec![0u32; vocab];
        for (rank, &old) in order.iter().enumerate() {
            new_id[old] = rank as u32 + 1;
        }
        // Emit `[V]` (the max id), then each definition — uniformly `(left, right)`, a terminal
        // being `(byte, NIL)` — in new-id order, then the remapped sequence, all as uleb128-u22
        // bytes. Pre-size to skip reallocation churn on the (at enwik9 scale) multi-hundred-MB
        // stream: ids are frequency-reordered, so most sequence symbols are low ids of 1–2 bytes.
        let mut out = Vec::with_capacity(4 + vocab * 4 + sequence.len() * 2);
        encode_u22(u22::new(vocab as u32), &mut out);
        for &old in &order {
            match dict[old] {
                Def::Byte(byte) => {
                    encode_u22(u22::new(u32::from(byte)), &mut out);
                    encode_u22(u22::new(NIL_SYM), &mut out);
                }
                Def::Pair(left, right) => {
                    encode_u22(u22::new(new_id[left as usize]), &mut out);
                    encode_u22(u22::new(new_id[right as usize]), &mut out);
                }
            }
        }
        for &sym in &sequence {
            encode_u22(u22::new(new_id[sym as usize]), &mut out);
        }
        out
    }

    fn inverse(&self, input: Vec<u8>) -> Result<Vec<u8>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let mut pos = 0;
        // The header is the max id `V`; real ids run `1..=V`, so there are `V + 1` dict slots
        // (slot 0 is a reserved placeholder that valid data never references — see `forward`).
        let vocab = decode_u22(&input, &mut pos)?.value() as usize + 1;
        // Parse the uniform definition list: each symbol is `(left, right)`, a terminal being
        // `(byte, NIL)` and a non-terminal a pair of symbol ids. Cap the reservation so a corrupt
        // `[V]` claiming a huge vocabulary cannot demand a large allocation before the decode fails.
        let mut dict: Vec<Def> = Vec::with_capacity(vocab.min(1 << 16));
        // Reserved id 0: a placeholder terminal so positional indexing stays valid; it expands to a
        // single byte if a corrupt stream ever references it, which the container checksum rejects.
        dict.push(Def::Byte(0));
        for _ in 1..vocab {
            let left = decode_u22(&input, &mut pos)?.value();
            let right = decode_u22(&input, &mut pos)?.value();
            if right == NIL_SYM {
                let byte = u8::try_from(left).map_err(|_| anyhow!("Re-Pair terminal token {left} is not a byte"))?;
                dict.push(Def::Byte(byte));
            } else {
                ensure!(
                    (left as usize) < vocab && (right as usize) < vocab,
                    "Re-Pair rule references an undefined symbol"
                );
                dict.push(Def::Pair(left, right));
            }
        }
        // Frequency order is not topological, so resolve expansion lengths over the grammar DAG (this
        // also rejects cycles). Then stream the sequence: bound each token's expansion against the
        // decompression-bomb ceiling *before* expanding it, so `output` never exceeds the cap.
        let lengths = expansion_lengths(&dict)?;
        let mut total: u64 = 0;
        let mut output = Vec::new();
        let mut stack: Vec<u32> = Vec::new();
        while pos < input.len() {
            let token = decode_u22(&input, &mut pos)?.value();
            let index = token as usize;
            ensure!(index < vocab, "Re-Pair sequence symbol {token} is out of range");
            total = total.saturating_add(lengths[index]);
            ensure!(total <= MAX_EXPANSION_BYTES, "Re-Pair expansion exceeds {MAX_EXPANSION_BYTES} bytes");
            stack.push(token);
            while let Some(symbol) = stack.pop() {
                match dict[symbol as usize] {
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

    /// The Re-Pair vocabulary size for the CLI trace: the leading `[V]` header (the max symbol id,
    /// i.e. the token count) of a non-empty stream. Empty output (empty input) reports nothing.
    fn trace_detail(&self, output: &[u8]) -> Option<(u64, &'static str)> {
        let mut pos = 0;
        decode_u22(output, &mut pos).ok().map(|v| (u64::from(v.value()), "tokens"))
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
                    let (left, right) = (left as usize, right as usize);
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

    use arbitrary_int::u22;
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::{NIL_SYM, RepairTokenizer};
    use crate::transform::Transform;
    use crate::uleb128::{decode_u22, encode_u22};

    /// A default tokenizer (`num_tokens = u22::MAX - 1`).
    fn tok() -> RepairTokenizer {
        RepairTokenizer::default()
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
    fn reserves_zero_so_output_has_no_nul_byte() {
        // Reserving id 0 keeps the byte value `0x00` out of the serialized stream for any input
        // free of literal NUL bytes — this is the free marker the downstream LZ77 stage relies on.
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(64);
        assert!(!tok().forward(data).contains(&0), "reserved id 0 leaked a 0x00 byte");
    }

    #[test]
    fn large_grammar_stays_within_vocabulary() {
        // A pseudo-random 256 KiB block makes Re-Pair create thousands of rules; the emitted
        // vocabulary must stay within the cap and still round-trip.
        let mut data = vec![0u8; 256 * 1024];
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        for byte in &mut data {
            state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *byte = state.to_be_bytes()[0];
        }
        let bytes = tok().forward(data.clone());
        let mut pos = 0;
        // The header is the max id `V`; real ids `1..=V` must stay below `NIL_SYM`.
        let max_id = decode_u22(&bytes, &mut pos).unwrap().value() as usize;
        assert!(max_id <= 0x3F_FFFE, "max id {max_id} exceeds the cap");
        assert_eq!(tok().inverse(bytes).unwrap(), data);
    }

    #[test]
    fn ids_are_frequency_ordered() {
        // No pair repeats in "zzab", so the vocabulary is just the terminals, renumbered by
        // descending reference frequency: the most-frequent byte 'z' becomes id 0 even though it is
        // the largest byte value present.
        let bytes = tok().forward(b"zzab".to_vec());
        let mut pos = 0;
        assert_eq!(decode_u22(&bytes, &mut pos).unwrap().value(), 3); // V = max id (three terminals)
        assert_eq!(decode_u22(&bytes, &mut pos).unwrap().value(), u32::from(b'z')); // id 1 left = 'z'
        assert_eq!(decode_u22(&bytes, &mut pos).unwrap().value(), NIL_SYM); // ...right = NIL (terminal)
        assert_eq!(tok().inverse(bytes).unwrap(), b"zzab".to_vec());
    }

    #[test]
    fn inverse_rejects_malformed_tokens() {
        // Append an out-of-range sequence symbol (id == V+1, one past the largest valid id V).
        let mut bytes = tok().forward(b"abracadabra".to_vec());
        let mut pos = 0;
        let v = decode_u22(&bytes, &mut pos).unwrap().value() + 1; // max_id + 1 = V + 1
        encode_u22(u22::new(v), &mut bytes);
        assert!(tok().inverse(bytes).is_err());
        // A stream claiming a huge vocabulary with no definitions following is truncated.
        let mut header_only = Vec::new();
        encode_u22(u22::new(NIL_SYM - 1), &mut header_only);
        assert!(tok().inverse(header_only).is_err());
    }

    #[test]
    fn num_tokens_cap_is_respected() {
        // A tiny cap must bound the emitted vocabulary and still round-trip.
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(64);
        let capped = RepairTokenizer {
            num_tokens: 300,
            cost_stop: false,
        };
        let bytes = capped.forward(data.clone());
        let mut pos = 0;
        // The header is the max id `V`, which equals the real symbol count (ids `1..=V`).
        let symbols = decode_u22(&bytes, &mut pos).unwrap().value() as usize;
        assert!(symbols <= 300, "vocabulary {symbols} exceeds the num_tokens cap");
        assert_eq!(capped.inverse(bytes).unwrap(), data);
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
