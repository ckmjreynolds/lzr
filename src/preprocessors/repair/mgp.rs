//! Minimal-grammar-parse (MGP): an optimal re-parse of the Re-Pair top-level *sequence*.
//!
//! Re-Pair's sequence is a by-product of the greedy digram-replacement order, not a least-cost cover
//! of the text by the available symbols. MGP leaves the grammar's rules untouched (so their binary
//! right-hand sides — and the u22 serialization — are unaffected) and rewrites only the sequence: it
//! finds the minimum-cost cover of the original terminal text using the surviving symbols (terminals
//! plus non-terminals), which can use fewer / cheaper symbols than the greedy parse.
//!
//! The cover is a shortest-path DP over text positions: node = a position `0..=n`, an edge `i → j`
//! covers `text[i..j]` with a symbol whose byte expansion equals that span, its weight the symbol's
//! (proxy) coding cost. Symbol occurrences are found by walking a trie built over the symbols'
//! expansions. A valid cover always exists — every terminal is a length-1 symbol — so the re-parse is
//! lossless by construction: expanding the new sequence reproduces the same text, hence the same
//! grammar output the decoder already knows how to invert. Nothing here is serialized; MGP is purely
//! an encode-side choice of *which* symbols spell the sequence.
//!
//! The cost function is a deliberately swappable proxy (per-symbol self-information over the current
//! reference frequencies). Because frequency-rank ids are assigned *after* the parse, the re-parse is
//! iterated EM-style ([`MGP_ROUNDS`]): parse → recount → re-cost → re-parse. The proxy only steers the
//! search; the arbiter is measured end-to-end bpb.

/// Symbols whose byte expansion is longer than this are not offered as cover candidates. Their spans
/// are still coverable by shorter symbols (ultimately terminals), so validity is unaffected; the cap
/// just bounds the trie's depth and size when MGP is (unusually) run on a large-vocabulary grammar.
const MAX_MGP_PATTERN: u64 = 4096;

/// Safety ceiling on the trie's node count. Terminals are inserted first (they guarantee a valid
/// cover), then rules until this budget is reached; any rules beyond it are simply not offered as
/// candidates. Bounds memory when MGP is run on a large grammar rather than the intended small vocab.
const MAX_MGP_NODES: usize = 16_000_000;

/// EM-style refinement rounds: each re-costs symbols from the previous round's sequence frequencies
/// and re-parses. Convergence is quick at the small operating point; the loop also stops early once a
/// round leaves the sequence unchanged.
const MGP_ROUNDS: usize = 3;

/// A trie over symbol byte-expansions (in terminal-id space, alphabet `< t <= 256`). Children are held
/// as a small `(terminal id, node index)` list — the branching factor is low — so a walk is a short
/// linear scan per level.
#[derive(Default)]
struct TrieNode {
    /// `(terminal id, child node index)` edges out of this node.
    children: Vec<(u32, usize)>,
    /// Symbols whose expansion ends exactly here (usually one; more only when distinct symbols share
    /// an identical expansion). The DP picks the cheapest at relax time, so cost need not be baked in.
    syms: Vec<u32>,
}

impl TrieNode {
    /// The child node reached by terminal `term`, if any.
    fn child(&self, term: u32) -> Option<usize> {
        self.children.iter().find(|&&(k, _)| k == term).map(|&(_, n)| n)
    }
}

/// Expand symbol `s` to its terminal-id string into `out` (reused buffer). Terminals (`id < t`) expand
/// to themselves; a rule to the concatenation of its two children. Ids are topologically ordered
/// (a rule only names smaller ids), but a plain stack walk resolves any DAG regardless.
fn expand_into(s: u32, t: usize, rules: &[(u32, u32)], out: &mut Vec<u32>, stack: &mut Vec<u32>) {
    out.clear();
    stack.clear();
    stack.push(s);
    while let Some(x) = stack.pop() {
        let xi = x as usize;
        if xi < t {
            out.push(x);
        } else {
            let (a, b) = rules[xi - t];
            stack.push(b);
            stack.push(a);
        }
    }
}

/// Per-symbol expansion length in terminals: `1` for a terminal, `len(left) + len(right)` for a rule.
/// Ids are topologically ordered so a single forward pass resolves them; lengths saturate so a huge
/// expansion cannot overflow (it is simply excluded by [`MAX_MGP_PATTERN`]).
fn expansion_lengths(t: usize, rules: &[(u32, u32)]) -> Vec<u64> {
    let mut len = vec![1u64; t + rules.len()];
    for (k, &(a, b)) in rules.iter().enumerate() {
        len[t + k] = len[a as usize].saturating_add(len[b as usize]);
    }
    len
}

/// The child of `nodes[cur]` for terminal `term`, adding a fresh node when absent. Kept a free
/// function so both the lookup and the append borrow the arena in sequence rather than nesting.
fn get_or_add_child(nodes: &mut Vec<TrieNode>, cur: usize, term: u32) -> usize {
    if let Some(n) = nodes[cur].child(term) {
        return n;
    }
    let n = nodes.len();
    nodes.push(TrieNode::default());
    nodes[cur].children.push((term, n));
    n
}

/// Build the candidate trie over every symbol's expansion (terminals first, then rules within the
/// [`MAX_MGP_NODES`] budget and the [`MAX_MGP_PATTERN`] length cap). Returns the node arena; node 0 is
/// the root.
#[expect(
    clippy::needless_range_loop,
    reason = "`s` is a symbol id used both to index `len` and as the stored value/expansion root"
)]
#[expect(clippy::cast_possible_truncation, reason = "symbol id s < vocab <= 0x3F_FFFE fits u32")]
fn build_trie(t: usize, rules: &[(u32, u32)], len: &[u64]) -> Vec<TrieNode> {
    let vocab = t + rules.len();
    let mut nodes: Vec<TrieNode> = vec![TrieNode::default()];
    let mut exp: Vec<u32> = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    // Terminals first: single-terminal expansions that guarantee every position is coverable.
    for s in 0..vocab {
        if len[s] > MAX_MGP_PATTERN {
            continue;
        }
        if s >= t && nodes.len() >= MAX_MGP_NODES {
            continue; // rule budget exhausted; terminals (s < t) are always admitted
        }
        expand_into(s as u32, t, rules, &mut exp, &mut stack);
        let mut cur = 0usize;
        for &term in &exp {
            cur = get_or_add_child(&mut nodes, cur, term);
        }
        nodes[cur].syms.push(s as u32);
    }
    nodes
}

/// Per-symbol proxy coding cost from the current reference frequencies: self-information
/// `-log2(freq / total)`, where `freq = sequence occurrences + rule-RHS references` (mirroring the
/// frequency-rank id assignment in [`super::RepairTokenizer::forward`]). Frequencies are smoothed to
/// at least 1 so an as-yet-unused symbol has a finite (large) cost rather than `+inf`.
#[expect(clippy::cast_precision_loss, reason = "reference counts are proxy weights; f64 precision is ample")]
fn symbol_costs(seq_freq: &[u64], rule_ref: &[u64]) -> Vec<f64> {
    let total: u64 = seq_freq.iter().zip(rule_ref).map(|(&a, &b)| a + b).sum();
    let total = total.max(1) as f64;
    seq_freq
        .iter()
        .zip(rule_ref)
        .map(|(&a, &b)| {
            let freq = (a + b).max(1) as f64;
            -(freq / total).log2()
        })
        .collect()
}

/// Count each symbol's occurrences in `sequence`.
fn sequence_frequencies(sequence: &[u32], vocab: usize) -> Vec<u64> {
    let mut freq = vec![0u64; vocab];
    for &s in sequence {
        freq[s as usize] += 1;
    }
    freq
}

/// Minimum-cost cover of `text` using the trie's symbols under `cost`. Returns the covering symbol
/// sequence, or `None` if some position is unreachable (impossible while terminals are present —
/// defensive, so a degenerate trie falls back to the caller's existing sequence rather than looping).
fn dp_cover(text: &[u32], nodes: &[TrieNode], cost: &[f64]) -> Option<Vec<u32>> {
    let n = text.len();
    let mut dp = vec![f64::INFINITY; n + 1];
    let mut back_sym = vec![u32::MAX; n + 1];
    let mut back_prev = vec![usize::MAX; n + 1];
    dp[0] = 0.0;
    for i in 0..n {
        let di = dp[i];
        if !di.is_finite() {
            continue;
        }
        // Walk the trie along the text from `i`, relaxing every symbol whose expansion ends at `j`.
        let mut node = 0usize;
        let mut j = i;
        while j < n {
            let Some(next) = nodes[node].child(text[j]) else {
                break;
            };
            node = next;
            j += 1;
            for &s in &nodes[node].syms {
                let c = di + cost[s as usize];
                if c < dp[j] {
                    dp[j] = c;
                    back_sym[j] = s;
                    back_prev[j] = i;
                }
            }
        }
    }
    if !dp[n].is_finite() {
        return None;
    }
    // Walk the back-pointers from `n` to `0` and reverse into forward order.
    let mut out: Vec<u32> = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let sym = back_sym[pos];
        let prev = back_prev[pos];
        if sym == u32::MAX || prev >= pos {
            return None; // broken chain — defensive
        }
        out.push(sym);
        pos = prev;
    }
    out.reverse();
    Some(out)
}

/// Re-parse the top-level `sequence` into a minimum-cost cover of the same text using the fixed symbol
/// set (terminals `0..t` plus `rules`, rule `k` having id `t + k`). Iterates the cost/parse EM loop a
/// few rounds and returns the new sequence; falls back to the input sequence if no cover is found.
///
/// The grammar (terminals, rules) is unchanged — only the sequence is rewritten — so the caller's
/// subsequent frequency renumber and serialization stay valid and the result is losslessly invertible.
pub(super) fn reparse_sequence(t: u32, rules: &[(u32, u32)], sequence: Vec<u32>) -> Vec<u32> {
    let t = t as usize;
    let vocab = t + rules.len();
    if rules.is_empty() || sequence.is_empty() {
        return sequence; // no non-terminals to re-cover with, or nothing to cover
    }
    // Reconstruct the original terminal-id text by expanding the current sequence.
    let mut text: Vec<u32> = Vec::new();
    {
        let mut stack: Vec<u32> = Vec::new();
        for &s in &sequence {
            stack.push(s);
            while let Some(x) = stack.pop() {
                let xi = x as usize;
                if xi < t {
                    text.push(x);
                } else {
                    let (a, b) = rules[xi - t];
                    stack.push(b);
                    stack.push(a);
                }
            }
        }
    }
    let len = expansion_lengths(t, rules);
    let nodes = build_trie(t, rules, &len);
    // Rule-RHS reference counts are fixed across rounds (the rules never change); only the sequence
    // frequencies move, so they alone are recounted each round.
    let mut rule_ref = vec![0u64; vocab];
    for &(a, b) in rules {
        rule_ref[a as usize] += 1;
        rule_ref[b as usize] += 1;
    }
    let mut current = sequence;
    for _ in 0..MGP_ROUNDS {
        let seq_freq = sequence_frequencies(&current, vocab);
        let cost = symbol_costs(&seq_freq, &rule_ref);
        match dp_cover(&text, &nodes, &cost) {
            Some(next) if next != current => current = next,
            _ => break, // no improvement / converged / degenerate — keep the last good sequence
        }
    }
    current
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::super::RepairTokenizer;
    use super::*;
    use crate::transform::Transform;

    /// A tokenizer with MGP enabled.
    fn mgp_tok() -> RepairTokenizer {
        RepairTokenizer {
            mgp: true,
            ..RepairTokenizer::default()
        }
    }

    /// Map bytes to compact terminal ids `0..t` (as `forward` does).
    fn terminals(data: &[u8]) -> (Vec<u32>, u32) {
        let mut present = [false; 256];
        for &d in data {
            present[usize::from(d)] = true;
        }
        let mut id = [0u32; 256];
        let mut t = 0u32;
        for b in 0..=255usize {
            if present[b] {
                id[b] = t;
                t += 1;
            }
        }
        (data.iter().map(|&d| id[usize::from(d)]).collect(), t)
    }

    /// Expand a grammar (`rules[k]` has id `t + k`) back to its terminal-id sequence.
    fn expand(rules: &[(u32, u32)], sequence: &[u32], t: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut stack = Vec::new();
        for &s in sequence {
            stack.push(s);
            while let Some(x) = stack.pop() {
                if x < t {
                    out.push(x);
                } else {
                    let (l, r) = rules[(x - t) as usize];
                    stack.push(r);
                    stack.push(l);
                }
            }
        }
        out
    }

    proptest! {
        /// A tokenizer with MGP on must still round-trip arbitrary input.
        #[test]
        fn mgp_roundtrips_any(data in prop::collection::vec(any::<u8>(), 0..4096)) {
            let restored = mgp_tok().inverse(mgp_tok().forward(data.clone())).unwrap();
            prop_assert_eq!(restored, data);
        }

        /// Small alphabets build deep hierarchies with heavy expansion overlap — the case most likely
        /// to expose a re-parse cover bug — and must still round-trip.
        #[test]
        fn mgp_roundtrips_small_alphabet(data in prop::collection::vec(0u8..4, 0..4096)) {
            let restored = mgp_tok().inverse(mgp_tok().forward(data.clone())).unwrap();
            prop_assert_eq!(restored, data);
        }

        /// `inverse` must never panic on arbitrary bytes for an MGP tokenizer either.
        #[test]
        fn mgp_inverse_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
            drop(mgp_tok().inverse(bytes));
        }

        /// Re-parsing must be an exact cover: the re-parsed sequence expands to the same text.
        #[test]
        fn reparse_preserves_text(data in prop::collection::vec(0u8..6, 0..4096)) {
            use super::super::builder::build_grammar;
            let (symbols, t) = terminals(&data);
            let (rules, sequence) = build_grammar(symbols.clone(), t, super::super::DEFAULT_NUM_TOKENS);
            let reparsed = reparse_sequence(t, &rules, sequence);
            prop_assert_eq!(expand(&rules, &reparsed, t), symbols);
        }
    }

    #[test]
    fn mgp_known_values_roundtrip() {
        let inputs: [&[u8]; 5] = [b"", b"a", b"aaaa", b"abracadabra", b"mississippi"];
        for input in inputs {
            assert_eq!(mgp_tok().inverse(mgp_tok().forward(input.to_vec())).unwrap(), input);
        }
    }

    #[test]
    fn empty_and_ruleless_pass_through() {
        // No rules: nothing to re-cover with, so the sequence is returned unchanged.
        let seq = vec![0u32, 1, 2];
        assert_eq!(reparse_sequence(3, &[], seq.clone()), seq);
        assert!(reparse_sequence(3, &[(0, 1)], Vec::new()).is_empty());
    }
}
