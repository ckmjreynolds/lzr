//! Capped Re-Pair grammar tokenizer.
//!
//! Re-Pair repeatedly replaces the best-scoring adjacent digram with a new non-terminal, building
//! a grammar over the input. This tokenizer runs Re-Pair to a hard `u15` vocabulary cap (32768
//! symbols = up to 256 terminals + non-terminals) and emits the grammar *inline* ahead of the
//! reduced symbol sequence, so the whole thing is one `u15` token stream the codec can carry
//! without a separate dictionary channel:
//!
//! ```text
//! [V-1] [(left_0, right_0)] … [(left_{V-1}, right_{V-1})] [sequence …]
//! ```
//!
//! `V` is the symbol count (at most 32767; `0x7FFF` = `u15::MAX` is the reserved NIL sentinel).
//! Every symbol is a uniform `(left, right)` pair — a terminal is `(byte, NIL)`, a non-terminal is
//! `(left, right)` referencing two other symbols. Symbol ids are assigned by **descending reference
//! frequency** — id 0 is the most-used symbol — so low-valued (short) ids code cheaply; this
//! matters most for a non-entropy back end. Frequency order is not topological, so `inverse` treats
//! the grammar as a DAG and rejects cycles. Which digram is "best" is chosen by a pluggable
//! [`cost::MergeCost`]; the default is [`cost::Entropy`], the bit-saving criterion that suits this
//! codec's entropy-coded output.

pub(crate) mod cost;

use std::collections::{BinaryHeap, HashMap};

use anyhow::{Result, anyhow, bail, ensure};
use arbitrary_int::u15;
use static_assertions::const_assert;

use self::cost::{Entropy, MergeCost};
use crate::tokenizers::Tokenize;

/// Sentinel "no node" index into the working sequence.
const NIL: u32 = u32::MAX;
/// Sentinel symbol marking a tombstoned (merged-away) sequence slot.
const NONE_SYM: u16 = u16::MAX;
/// Reserved sentinel symbol (`u15::MAX`); never a valid symbol id, so it marks a terminal as the
/// right child of a uniform `(left, right)` grammar definition.
const NIL_SYM: u16 = 0x7FFF;
/// Vocabulary ceiling: symbol ids run `0..VOCAB_CAP`, staying strictly below [`NIL_SYM`], so the
/// grammar holds at most 32767 symbols.
const VOCAB_CAP: u32 = 0x7FFF;
/// Decompression-bomb ceiling for `inverse`: a small grammar can expand exponentially, so the
/// total expansion is bounded regardless of the (untrusted) grammar.
const MAX_EXPANSION_BYTES: u64 = 1 << 31;

const_assert!(VOCAB_CAP == 0x7FFF);

/// One slot in the working sequence, doubly linked both in sequence order and within its digram's
/// occurrence list.
#[derive(Debug, Clone, Copy)]
struct Node {
    /// Symbol id, or [`NONE_SYM`] once the slot has been merged away.
    sym: u16,
    /// Previous live slot in sequence order, or [`NIL`].
    prev: u32,
    /// Next live slot in sequence order, or [`NIL`].
    next: u32,
    /// Previous slot in this slot's digram occurrence list, or [`NIL`].
    occ_prev: u32,
    /// Next slot in this slot's digram occurrence list, or [`NIL`].
    occ_next: u32,
}

/// Occurrence bookkeeping for one active digram.
#[derive(Debug, Clone, Copy)]
struct PairInfo {
    /// Number of live occurrences.
    count: u32,
    /// First slot in the occurrence list, or [`NIL`].
    head: u32,
}

/// An `f64` merge score with a total order (scores are always finite here).
#[derive(Debug, Clone, Copy)]
struct Score(f64);

impl PartialEq for Score {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == core::cmp::Ordering::Equal
    }
}

impl Eq for Score {}

impl PartialOrd for Score {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Score {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// A lazily-revalidated max-heap entry: the digram and the score it had when pushed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    /// Score at push time; re-checked against the live score on pop.
    key: Score,
    /// The digram `(a, b)`.
    pair: (u16, u16),
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Max-heap on score; on ties the lexicographically smaller digram wins (is treated as
        // greater), for a fully deterministic merge order.
        self.key.cmp(&other.key).then_with(|| other.pair.cmp(&self.pair))
    }
}

/// A resolved grammar symbol used by `inverse`.
#[derive(Debug, Clone, Copy)]
enum Def {
    /// Terminal: expands to this byte.
    Byte(u8),
    /// Non-terminal: the concatenation of two earlier symbols.
    Pair(u16, u16),
}

/// Working state for one capped Re-Pair run over a single input.
struct Repair<'a> {
    /// The working sequence as a linked list with tombstones.
    nodes: Vec<Node>,
    /// Live occurrence bookkeeping per digram.
    pairs: HashMap<(u16, u16), PairInfo>,
    /// Lazy max-heap of merge candidates.
    heap: BinaryHeap<Candidate>,
    /// Rules in creation order; rule `k` has id `first_rule_id + k`.
    rules: Vec<(u16, u16)>,
    /// Live occurrence count per symbol id (indexed by id; length == `next_rule_id`).
    counts: Vec<u32>,
    /// Total live token count.
    total: u32,
    /// Id that the next rule will receive (`== T + rules.len()`).
    next_rule_id: u16,
    /// Reused occurrence-position buffer for [`Repair::replace_pair`], so the per-merge
    /// snapshot allocates at most once across the whole run.
    scratch: Vec<u32>,
    /// The pluggable merge-selection criterion.
    cost: &'a dyn MergeCost,
}

impl<'a> Repair<'a> {
    /// Builds the working state for `symbols` (already mapped to `0..T` terminal ids, with
    /// `counts` their occurrence counts) using `cost` to score merges.
    #[expect(clippy::cast_possible_truncation, reason = "caller guarantees symbols.len() <= u32::MAX")]
    fn new(symbols: &[u16], n_terminals: u16, counts: Vec<u32>, cost: &'a dyn MergeCost) -> Self {
        let n = symbols.len();
        let mut nodes = Vec::with_capacity(n);
        for (i, &sym) in symbols.iter().enumerate() {
            nodes.push(Node {
                sym,
                prev: if i == 0 {
                    NIL
                } else {
                    (i - 1) as u32
                },
                next: if i + 1 < n {
                    (i + 1) as u32
                } else {
                    NIL
                },
                occ_prev: NIL,
                occ_next: NIL,
            });
        }
        Self {
            nodes,
            pairs: HashMap::new(),
            heap: BinaryHeap::new(),
            rules: Vec::new(),
            counts,
            total: n as u32,
            next_rule_id: n_terminals,
            scratch: Vec::new(),
            cost,
        }
    }

    /// Scores the digram `pair` occurring `freq` times against the current symbol counts.
    fn score_pair(&self, pair: (u16, u16), freq: u32) -> f64 {
        let (a, b) = pair;
        let count_a = self.counts[usize::from(a)];
        let count_b = self.counts[usize::from(b)];
        self.cost.score(freq, count_a, count_b, self.total, a == b)
    }

    /// Inserts the digram starting at `pos` (which must have a live `next`) into its occurrence
    /// list, bumps its count, and pushes a fresh heap candidate.
    fn add_occurrence(&mut self, pos: u32) {
        let a = self.nodes[pos as usize].sym;
        let b = self.nodes[self.nodes[pos as usize].next as usize].sym;
        let key = (a, b);
        let (head, count) = {
            let info = self.pairs.entry(key).or_insert(PairInfo {
                count: 0,
                head: NIL,
            });
            let head = info.head;
            info.head = pos;
            info.count += 1;
            (head, info.count)
        };
        self.nodes[pos as usize].occ_prev = NIL;
        self.nodes[pos as usize].occ_next = head;
        if head != NIL {
            self.nodes[head as usize].occ_prev = pos;
        }
        let score = self.score_pair(key, count);
        self.heap.push(Candidate {
            key: Score(score),
            pair: key,
        });
    }

    /// Removes the digram occurrence starting at `pos` (which must have a live `next`) from its
    /// occurrence list and decrements its count, dropping or re-scoring the digram accordingly.
    fn remove_occurrence(&mut self, pos: u32) {
        let a = self.nodes[pos as usize].sym;
        let b = self.nodes[self.nodes[pos as usize].next as usize].sym;
        let key = (a, b);
        let (occ_prev, occ_next) = (self.nodes[pos as usize].occ_prev, self.nodes[pos as usize].occ_next);
        if occ_prev != NIL {
            self.nodes[occ_prev as usize].occ_next = occ_next;
        }
        if occ_next != NIL {
            self.nodes[occ_next as usize].occ_prev = occ_prev;
        }
        let remaining = self.pairs.get_mut(&key).map(|info| {
            if occ_prev == NIL {
                info.head = occ_next;
            }
            info.count -= 1;
            info.count
        });
        match remaining {
            Some(0) => drop(self.pairs.remove(&key)),
            Some(count) => {
                let score = self.score_pair(key, count);
                self.heap.push(Candidate {
                    key: Score(score),
                    pair: key,
                });
            }
            None => {}
        }
    }

    /// Pops the highest-scoring digram whose live score is still current and profitable, or `None`
    /// if the best remaining pair is not worth merging.
    fn pop_best(&mut self) -> Option<(u16, u16)> {
        while let Some(candidate) = self.heap.pop() {
            let live = self.pairs.get(&candidate.pair).map_or(0, |info| info.count);
            if live == 0 {
                continue;
            }
            let score = self.score_pair(candidate.pair, live);
            if Score(score) == candidate.key {
                return self.cost.is_profitable(score, live).then_some(candidate.pair);
            }
            self.heap.push(Candidate {
                key: Score(score),
                pair: candidate.pair,
            });
        }
        None
    }

    /// Replaces every (non-overlapping) occurrence of `pair` with the new rule symbol `new_id`.
    fn replace_pair(&mut self, pair: (u16, u16), new_id: u16) {
        let (a, b) = pair;
        // Snapshot the occurrence positions in ascending (= sequence) order; node indices are never
        // reordered, so sorting yields left-to-right processing and correct overlap handling. The
        // buffer is taken from `self` and returned below so its capacity is reused across merges.
        let mut positions = std::mem::take(&mut self.scratch);
        positions.clear();
        let mut p = self.pairs.get(&pair).map_or(NIL, |info| info.head);
        while p != NIL {
            positions.push(p);
            p = self.nodes[p as usize].occ_next;
        }
        positions.sort_unstable();
        for &i in &positions {
            if self.nodes[i as usize].sym != a {
                continue; // consumed by an earlier overlapping merge
            }
            let j = self.nodes[i as usize].next;
            if j == NIL || self.nodes[j as usize].sym != b {
                continue;
            }
            let prev = self.nodes[i as usize].prev;
            let next = self.nodes[j as usize].next;
            // Retire the three affected digrams (keys read before any structural change).
            self.remove_occurrence(i);
            if prev != NIL {
                self.remove_occurrence(prev);
            }
            if next != NIL {
                self.remove_occurrence(j);
            }
            // Splice: i becomes the rule symbol, j leaves the sequence.
            self.nodes[i as usize].sym = new_id;
            self.nodes[i as usize].next = next;
            if next != NIL {
                self.nodes[next as usize].prev = i;
            }
            self.nodes[j as usize].sym = NONE_SYM;
            // Maintain live symbol counts (a == b decrements a twice).
            self.counts[usize::from(a)] -= 1;
            self.counts[usize::from(b)] -= 1;
            self.counts[usize::from(new_id)] += 1;
            self.total -= 1;
            // Register the new digrams around the rule symbol.
            if prev != NIL {
                self.add_occurrence(prev);
            }
            if next != NIL {
                self.add_occurrence(i);
            }
        }
        self.scratch = positions;
    }

    /// Runs Re-Pair: seed the initial digrams, then merge the best pair until none is profitable or
    /// the vocabulary cap is reached.
    fn run(&mut self) {
        let mut i = if self.nodes.is_empty() {
            NIL
        } else {
            0
        };
        while i != NIL {
            let next = self.nodes[i as usize].next;
            if next != NIL {
                self.add_occurrence(i);
            }
            i = next;
        }
        while u32::from(self.next_rule_id) < VOCAB_CAP {
            let Some(pair) = self.pop_best() else {
                break;
            };
            let new_id = self.next_rule_id;
            self.rules.push(pair);
            self.counts.push(0);
            self.next_rule_id += 1;
            self.replace_pair(pair, new_id);
        }
    }

    /// Walks the surviving sequence in order.
    fn collect_sequence(&self) -> Vec<u16> {
        let mut sequence = Vec::new();
        let mut cur = if self.nodes.is_empty() {
            NIL
        } else {
            0
        };
        while cur != NIL {
            sequence.push(self.nodes[cur as usize].sym);
            cur = self.nodes[cur as usize].next;
        }
        sequence
    }
}

/// A capped Re-Pair grammar tokenizer.
///
/// `forward` derives a per-input grammar, renumbers its symbols by descending reference frequency
/// (id 0 = most used), and emits it inline ahead of the reduced sequence; `inverse` parses that
/// self-describing grammar (as a cycle-checked DAG) and expands the sequence back to bytes. The
/// tokenizer is stateless apart from its merge criterion — all per-input grammar state travels in
/// the token stream.
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[derive(Debug)]
pub(crate) struct RepairTokenizer {
    cost: Box<dyn MergeCost>,
}

impl RepairTokenizer {
    /// Creates a Re-Pair tokenizer that selects merges with `cost`.
    #[cfg_attr(feature = "bench-internals", visibility::make(pub))]
    #[must_use]
    pub(crate) fn new(cost: impl MergeCost + 'static) -> Self {
        Self {
            cost: Box::new(cost),
        }
    }
}

impl Default for RepairTokenizer {
    fn default() -> Self {
        Self::new(Entropy::default())
    }
}

impl Tokenize for RepairTokenizer {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "vocab <=32768 so vocab-1 and ranks fit u16; terminal count <=256"
    )]
    fn forward(&self, input: &[u8]) -> Vec<u15> {
        if input.is_empty() {
            return Vec::new();
        }
        // Compact terminals to the distinct present bytes, in ascending byte order.
        let mut present = [false; 256];
        for &byte in input {
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
        // Map the input to symbol ids and seed terminal counts.
        let mut counts = vec![0u32; t];
        let symbols: Vec<u16> = input
            .iter()
            .map(|&byte| {
                let id = byte_to_id[usize::from(byte)];
                counts[usize::from(id)] += 1;
                id
            })
            .collect();
        // Build the capped grammar (identity fallback if the input exceeds u32 node indices).
        let (rules, sequence) = if u32::try_from(input.len()).is_ok() {
            let mut builder = Repair::new(&symbols, t as u16, counts, self.cost.as_ref());
            builder.run();
            let sequence = builder.collect_sequence();
            (builder.rules, sequence)
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
    use super::cost::{Entropy, Frequency};
    use crate::tokenizers::Tokenize;

    /// Both shipped cost models — round-trip correctness is independent of the criterion.
    fn tokenizers() -> Vec<RepairTokenizer> {
        vec![RepairTokenizer::new(Frequency), RepairTokenizer::new(Entropy::default())]
    }

    fn corpus(rel: &str) -> Option<Vec<u8>> {
        std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).ok()
    }

    proptest! {
        #[test]
        fn roundtrip_any(data in prop::collection::vec(any::<u8>(), 0..4096)) {
            for tok in tokenizers() {
                prop_assert_eq!(tok.inverse(&tok.forward(&data)).unwrap(), data.clone());
            }
        }

        #[test]
        fn roundtrip_small_alphabet(data in prop::collection::vec(0u8..4, 0..4096)) {
            for tok in tokenizers() {
                prop_assert_eq!(tok.inverse(&tok.forward(&data)).unwrap(), data.clone());
            }
        }

        /// `inverse` must never panic on arbitrary token input — only `Ok` or `Err`.
        #[test]
        fn inverse_never_panics(raw in prop::collection::vec(0u16..=0x7FFF, 0..512)) {
            let tokens: Vec<u15> = raw.into_iter().map(u15::new).collect();
            drop(RepairTokenizer::default().inverse(&tokens));
        }
    }

    #[test]
    fn known_values_roundtrip() {
        let inputs: [&[u8]; 5] = [b"", b"a", b"aaaa", b"abracadabra", b"mississippi"];
        for tok in tokenizers() {
            for input in inputs {
                assert_eq!(tok.inverse(&tok.forward(input)).unwrap(), input);
            }
        }
    }

    #[test]
    fn compresses_repetitive_input() {
        let data = vec![b'a'; 8192];
        for tok in tokenizers() {
            assert!(tok.forward(&data).len() <= data.len());
        }
    }

    #[test]
    fn large_grammar_stays_within_vocabulary() {
        // A pseudo-random 256 KiB block makes classic Re-Pair create thousands of rules; the
        // emitted vocabulary must stay within the u15 cap and still round-trip. (The hard 32768
        // cap in `Repair::run` is additionally covered by the no-panic proptests.)
        let mut data = vec![0u8; 256 * 1024];
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        for byte in &mut data {
            state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *byte = state.to_be_bytes()[0];
        }
        let tok = RepairTokenizer::new(Frequency);
        let tokens = tok.forward(&data);
        let vocab = usize::from(tokens[0].value()) + 1;
        assert!(vocab <= 32767, "vocabulary {vocab} exceeds the reserved-NIL cap");
        assert!(tokens.iter().all(|token| token.value() <= 0x7FFF));
        assert_eq!(tok.inverse(&tokens).unwrap(), data);
    }

    #[test]
    fn ids_are_frequency_ordered() {
        // With no profitable merges (Entropy on a tiny input) the vocabulary is just the terminals,
        // renumbered by descending frequency: the most-frequent byte 'z' becomes id 0 even though it
        // is the largest byte value present.
        let tok = RepairTokenizer::new(Entropy::default());
        let tokens = tok.forward(b"zzzab");
        assert_eq!(tokens[0].value(), 2); // V - 1 (three terminals, no rules)
        assert_eq!(tokens[1].value(), u16::from(b'z')); // def_0 left = 'z', the most frequent byte
        assert_eq!(tokens[2].value(), 0x7FFF); // ...right = NIL, marking it a terminal
        assert_eq!(tok.inverse(&tokens).unwrap(), b"zzzab".to_vec());
    }

    #[test]
    fn inverse_rejects_malformed_tokens() {
        let tok = RepairTokenizer::default();
        let mut tokens = tok.forward(b"abracadabra");
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
        for tok in tokenizers() {
            assert_eq!(tok.inverse(&tok.forward(slice)).unwrap(), slice);
        }
    }
}
