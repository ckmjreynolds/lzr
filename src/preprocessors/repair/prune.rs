//! Post-build grammar pruning.
//!
//! Re-Pair, run to its natural frequency floor, mints a long tail of rules that do not earn their
//! keep once the grammar is serialized: a rule replaced by a *wider* symbol id (after the
//! descending-frequency renumbering that gives rare symbols long ids) can cost more than the pair it
//! folds. Rather than decide this *during* construction — where the eventual serialized widths are
//! unknown and greedily refusing a rule severs the hierarchy that later rules build on — we build the
//! full grammar first, then prune it here with the widths in hand.
//!
//! Two modes, both peeling rules off the grammar and *inlining* them (replacing a rule's id by its
//! two-symbol definition at every use site):
//! - **cost prune** (the default): drop every rule whose serialized-byte cost model says inlining it
//!   shrinks the stream — `keep ⟺ f·(C − N) > C`, with `C = width(a) + width(b)` the rule's RHS cost,
//!   `N = width(x)` the width of each of its `f` references, and `width` the byte length of a
//!   symbol's frequency-rank id.
//! - **count cap** (an explicit `num_tokens`): keep the `num_tokens` most-referenced symbols, peeling
//!   the least-referenced rules until the vocabulary fits.
//!
//! Inlining is exact, so pruning is always lossless — only the serialized size changes. A rule is
//! only ever dropped when **no surviving rule references it** (it is used solely from the reduced
//! sequence), which keeps the surviving grammar downward-closed: every kept rule's children survive,
//! so rule right-hand sides stay binary and only the sequence is rewritten.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::uleb128::u22_len;

/// Bound on the cost-prune width-refinement passes. Dropping rules narrows the surviving ids (a
/// smaller vocabulary ranks everything shorter), which can expose further non-paying rules; each pass
/// re-ranks and re-tests. Convergence is fast (2–3 passes in practice); the cap only guards runtime.
const MAX_COST_PASSES: usize = 10;

/// Prune the built grammar after ranking its symbols by reference frequency. `rules` are in creation
/// order (rule `k` has id `n_terminals + k`); `sequence` is the reduced symbol stream. Returns a
/// compacted `(rules, sequence)` in a fresh id space — terminals `0..n_terminals` unchanged, surviving
/// rules renumbered after them — ready for [`super::RepairTokenizer::forward`]'s frequency renumber.
pub(super) fn prune_grammar(
    n_terminals: u32,
    rules: Vec<(u32, u32)>,
    sequence: Vec<u32>,
    num_tokens: u32,
    cost_stop: bool,
) -> (Vec<(u32, u32)>, Vec<u32>) {
    if rules.is_empty() {
        return (rules, sequence);
    }
    let mut pruner = Pruner::new(n_terminals, &rules, &sequence);
    if cost_stop {
        pruner.cost_prune(&rules);
    } else {
        pruner.count_cap(num_tokens, &rules);
    }
    pruner.finish(&rules, &sequence)
}

/// Mutable pruning state over one built grammar. Symbol ids `0..t` are terminals, `t..t + r` are the
/// rules; `keep[i]` tracks whether rule `i` survives. `seq_freq` and `rule_ref` are kept current as
/// rules are dropped, so `seq_freq[s] + rule_ref[s]` is `s`'s live reference count at any moment.
struct Pruner {
    /// Terminal count (`0..t` are terminal ids).
    t: usize,
    /// Rule count (`t..t + r` are rule ids; rule `i` has id `t + i`).
    r: usize,
    /// Survival flag per rule index.
    keep: Vec<bool>,
    /// Occurrences of each symbol in the (progressively inlined) reduced sequence.
    seq_freq: Vec<u64>,
    /// Number of *surviving* rules that name each symbol in their right-hand side.
    rule_ref: Vec<u32>,
}

impl Pruner {
    fn new(n_terminals: u32, rules: &[(u32, u32)], sequence: &[u32]) -> Self {
        let t = n_terminals as usize;
        let r = rules.len();
        let vocab = t + r;
        let mut seq_freq = vec![0u64; vocab];
        for &s in sequence {
            seq_freq[s as usize] += 1;
        }
        let mut rule_ref = vec![0u32; vocab];
        for &(a, b) in rules {
            rule_ref[a as usize] += 1;
            rule_ref[b as usize] += 1;
        }
        Self {
            t,
            r,
            keep: vec![true; r],
            seq_freq,
            rule_ref,
        }
    }

    /// A symbol's live reference count: sequence occurrences plus surviving-rule references. This is
    /// exactly how many times its id appears in the serialized stream, so ranking by it reproduces
    /// `forward`'s renumbering.
    fn total_freq(&self, s: usize) -> u64 {
        self.seq_freq[s] + u64::from(self.rule_ref[s])
    }

    /// Serialized-id byte width of every symbol, from the current descending-frequency ranking (id 1
    /// is the most-referenced symbol, so the shortest). Recomputed between cost passes as drops
    /// reshuffle the ranks.
    #[expect(clippy::cast_possible_truncation, reason = "vocab <= num_tokens <= 0x3F_FFFE fits u32")]
    fn widths(&self) -> Vec<usize> {
        let vocab = self.t + self.r;
        let mut order: Vec<u32> = (0..vocab as u32).collect();
        order.sort_unstable_by(|&x, &y| self.total_freq(y as usize).cmp(&self.total_freq(x as usize)).then(x.cmp(&y)));
        let mut width = vec![0usize; vocab];
        for (rank, &s) in order.iter().enumerate() {
            width[s as usize] = u22_len(rank as u32 + 1);
        }
        width
    }

    /// Whether rule index `i` is a peel candidate: alive and named by no surviving rule (so it is
    /// referenced only from the sequence, and inlining it rewrites the sequence alone).
    fn droppable(&self, i: usize) -> bool {
        self.keep[i] && self.rule_ref[self.t + i] == 0
    }

    /// Drop rule `i = (a, b)`: mark it dead, move its `f` sequence occurrences onto `a` and `b`
    /// (inlining), and release its two RHS references. Any child that thereby loses its last
    /// referencing rule becomes a fresh peel candidate and is pushed to `newly`.
    #[expect(
        clippy::many_single_char_names,
        reason = "a,b the pair; s the rule id; f its frequency — Re-Pair notation"
    )]
    fn drop_rule(&mut self, i: usize, a: u32, b: u32, newly: &mut Vec<usize>) {
        let s = self.t + i;
        let f = self.seq_freq[s];
        self.keep[i] = false;
        self.seq_freq[a as usize] += f;
        self.seq_freq[b as usize] += f;
        self.seq_freq[s] = 0;
        for child in [a as usize, b as usize] {
            self.rule_ref[child] -= 1;
            if child >= self.t && self.keep[child - self.t] && self.rule_ref[child] == 0 {
                newly.push(child - self.t);
            }
        }
    }

    /// Cost-model prune: repeatedly rank the survivors, then peel every droppable rule whose inlining
    /// would not grow the stream (`f·(C − N) ≤ C`). Re-ranks and repeats until a full pass drops
    /// nothing — narrower post-drop ids can price out rules a previous pass kept.
    #[expect(clippy::many_single_char_names, reason = "a,b the pair; f frequency; c,n the RHS/reference widths")]
    fn cost_prune(&mut self, rules: &[(u32, u32)]) {
        for _ in 0..MAX_COST_PASSES {
            let width = self.widths();
            let mut queue: Vec<usize> = (0..self.r).filter(|&i| self.droppable(i)).collect();
            let mut dropped = false;
            while let Some(i) = queue.pop() {
                if !self.droppable(i) {
                    continue; // already dropped, or gained a reference — re-judged elsewhere
                }
                let (a, b) = rules[i];
                let f = self.seq_freq[self.t + i];
                let c = width[a as usize] + width[b as usize];
                let n = width[self.t + i];
                // Keep only if the rule pays: each of its `f` references saves `(c - n)` bytes over the
                // inlined pair, and that must beat the one-time `c`-byte RHS definition.
                if c > n && f * (c - n) as u64 > c as u64 {
                    continue;
                }
                self.drop_rule(i, a, b, &mut queue);
                dropped = true;
            }
            if !dropped {
                break;
            }
        }
    }

    /// Count-cap prune: peel the least-referenced rules (respecting downward closure) until the
    /// vocabulary — terminals plus survivors — is at most `num_tokens`. Widths are irrelevant here;
    /// only the frequency order matters. A lazy min-heap keyed on live reference count drives it;
    /// once a rule is droppable its count no longer changes (nothing references it), so heap entries
    /// never go stale — a `keep` check on pop is enough.
    #[expect(clippy::cast_possible_truncation, reason = "rule index < r <= vocab <= 0x3F_FFFE fits u32")]
    fn count_cap(&mut self, num_tokens: u32, rules: &[(u32, u32)]) {
        let target_rules = (num_tokens as usize).saturating_sub(self.t);
        let mut kept = self.r;
        if kept <= target_rules {
            return;
        }
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = (0..self.r)
            .filter(|&i| self.droppable(i))
            .map(|i| Reverse((self.total_freq(self.t + i), i as u32)))
            .collect();
        while kept > target_rules {
            let Some(Reverse((_, i))) = heap.pop() else {
                break; // nothing left to peel while staying binary
            };
            let i = i as usize;
            if !self.droppable(i) {
                continue;
            }
            let (a, b) = rules[i];
            let mut newly = Vec::new();
            self.drop_rule(i, a, b, &mut newly);
            kept -= 1;
            for j in newly {
                heap.push(Reverse((self.total_freq(self.t + j), j as u32)));
            }
        }
    }

    /// Materialize the pruned grammar: compact surviving symbol ids (terminals unchanged, kept rules
    /// renumbered after), remap the kept rules' right-hand sides, and rewrite the sequence — expanding
    /// every dropped rule to its surviving frontier via a stack. Downward closure guarantees a kept
    /// rule never names a dropped symbol, so RHS remapping needs no expansion.
    #[expect(clippy::cast_possible_truncation, reason = "compacted id < vocab <= 0x3F_FFFE fits u32")]
    fn finish(self, rules: &[(u32, u32)], sequence: &[u32]) -> (Vec<(u32, u32)>, Vec<u32>) {
        let vocab = self.t + self.r;
        let mut map = vec![u32::MAX; vocab];
        for (id, slot) in map.iter_mut().enumerate().take(self.t) {
            *slot = id as u32;
        }
        let mut next = self.t as u32;
        let mut new_rules: Vec<(u32, u32)> = Vec::new();
        for i in 0..self.r {
            if self.keep[i] {
                map[self.t + i] = next;
                next += 1;
                new_rules.push(rules[i]);
            }
        }
        for rule in &mut new_rules {
            rule.0 = map[rule.0 as usize];
            rule.1 = map[rule.1 as usize];
        }
        let mut new_seq: Vec<u32> = Vec::with_capacity(sequence.len());
        let mut stack: Vec<u32> = Vec::new();
        for &tok in sequence {
            stack.push(tok);
            while let Some(x) = stack.pop() {
                let xi = x as usize;
                if xi < self.t || self.keep[xi - self.t] {
                    new_seq.push(map[xi]);
                } else {
                    let (a, b) = rules[xi - self.t];
                    stack.push(b);
                    stack.push(a);
                }
            }
        }
        (new_rules, new_seq)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::super::builder::build_grammar;
    use super::*;

    /// The default vocabulary ceiling (`u22::MAX - 1`).
    const CAP: u32 = 0x3F_FFFE;

    /// Map bytes to compact terminal ids `0..t` (as `forward` does), returning the symbol vector and
    /// terminal count.
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
        /// Cost pruning only inlines rules, so the pruned grammar must expand to the original stream.
        #[test]
        fn cost_prune_is_lossless(data in prop::collection::vec(any::<u8>(), 0..2048)) {
            let (symbols, t) = terminals(&data);
            let (rules, sequence) = build_grammar(symbols.clone(), t, CAP);
            let (pr, ps) = prune_grammar(t, rules, sequence, CAP, true);
            prop_assert_eq!(expand(&pr, &ps, t), symbols);
        }

        /// Small alphabets build deep hierarchies (heavy overlap) — the DAG-expansion path most likely
        /// to be mishandled — and must still round-trip after pruning.
        #[test]
        fn cost_prune_is_lossless_small_alphabet(data in prop::collection::vec(0u8..4, 0..4096)) {
            let (symbols, t) = terminals(&data);
            let (rules, sequence) = build_grammar(symbols.clone(), t, CAP);
            let (pr, ps) = prune_grammar(t, rules, sequence, CAP, true);
            prop_assert_eq!(expand(&pr, &ps, t), symbols);
        }

        /// The count cap must be lossless and hold the surviving vocabulary (terminals + rules) at or
        /// below `num_tokens`.
        #[test]
        fn count_cap_is_lossless_and_bounded(
            data in prop::collection::vec(any::<u8>(), 0..2048),
            cap in 256u32..1200,
        ) {
            let (symbols, t) = terminals(&data);
            let (rules, sequence) = build_grammar(symbols.clone(), t, CAP);
            let (pr, ps) = prune_grammar(t, rules, sequence, cap, false);
            prop_assert!(t as usize + pr.len() <= cap as usize);
            prop_assert_eq!(expand(&pr, &ps, t), symbols);
        }
    }

    #[test]
    fn cost_prune_shrinks_a_low_value_tail() {
        // Text with a long common phrase plus a one-off repeated pair: Re-Pair mints a rule for the
        // rare pair that the cost model then prunes, so the survivor count is below the full grammar's.
        let data = b"the quick brown fox jumps over the lazy dog. ".repeat(50);
        let (symbols, t) = terminals(&data);
        let (rules, sequence) = build_grammar(symbols.clone(), t, CAP);
        let full = rules.len();
        let (pruned, ps) = prune_grammar(t, rules, sequence, CAP, true);
        assert!(pruned.len() <= full, "pruning must not add rules ({} -> {})", full, pruned.len());
        assert_eq!(expand(&pruned, &ps, t), symbols);
    }

    #[test]
    fn empty_grammar_passes_through() {
        // No rules (e.g. a non-repeating input): pruning is a no-op in either mode.
        let seq = vec![0u32, 1, 2];
        let (r, s) = prune_grammar(3, Vec::new(), seq.clone(), CAP, true);
        assert!(r.is_empty());
        assert_eq!(s, seq);
    }
}
