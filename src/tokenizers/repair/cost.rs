//! Merge-selection cost criteria for the Re-Pair tokenizer.
//!
//! The builder always merges the highest-[`MergeCost::score`]d adjacent digram. Which merge is
//! "best" depends on the downstream coder: for a raw token count, [`Frequency`] (classic Re-Pair)
//! is right; for an *entropy-coded* token stream — what this codec uses — [`Entropy`] scores by
//! the bits actually saved and picks better merges. The criterion is a trait so it can be swapped
//! and A/B-tested without touching the builder.

/// Scores candidate digram merges for the Re-Pair builder. Higher score = merge sooner.
///
/// Implementations are immutable and thread-safe (`Send + Sync`), so a [`super::RepairTokenizer`]
/// carrying one can be shared across threads.
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) trait MergeCost: core::fmt::Debug + Send + Sync {
    /// Score merging the digram `(a, b)` that occurs `freq` times, given the current live
    /// occurrence counts of its constituents (`count_a`, `count_b`), the current total token
    /// count (`total`), and whether `a == b`. Higher is more worth merging.
    fn score(&self, freq: u32, count_a: u32, count_b: u32, total: u32, same: bool) -> f64;

    /// Whether the best remaining pair — with recomputed `score` and current `freq` — is worth
    /// turning into a rule. Returning `false` stops the build.
    fn is_profitable(&self, score: f64, freq: u32) -> bool;
}

/// Classic Re-Pair: score is the raw occurrence count; a pair is worth merging once it repeats.
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[cfg_attr(
    not(any(test, feature = "bench-internals")),
    expect(
        dead_code,
        reason = "Alternative merge criterion for benchmarks/experiments; the default pipeline uses Entropy."
    )
)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Frequency;

impl MergeCost for Frequency {
    fn score(&self, freq: u32, _count_a: u32, _count_b: u32, _total: u32, _same: bool) -> f64 {
        f64::from(freq)
    }

    fn is_profitable(&self, _score: f64, freq: u32) -> bool {
        freq >= 2
    }
}

/// Order-0 entropy criterion: score is the bits the replacement saves under an order-0 model.
///
/// The saving is `≈ freq · PMI(a, b)`, computed via `g(x) = x·log2 x` deltas over the live symbol
/// counts minus the amortized per-rule definition cost. It prefers *predictable* (high
/// mutual-information) pairs over merely frequent ones — the right objective when the tokens are
/// entropy-coded.
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Entropy {
    /// Profitability threshold: the minimum order-0 code-length saving (in bits) a merge must yield
    /// to justify emitting its rule. It models the real cost of *materializing* a rule — its
    /// `(left, right)` definition is two extra `u15` tokens entropy-coded into the grammar list
    /// (~25+ bits here), far above the `g(f)` term for the new symbol itself. Subtracted uniformly,
    /// it shifts every score equally, so it never changes merge *ranking* — only where the build
    /// stops. The bpb-optimal value clusters near 24 across small/medium inputs; at large scale the
    /// `u15` vocabulary cap ([`super::VOCAB_CAP`]) binds first, so the exact value stops mattering
    /// (e.g. enwik8 saturates the cap at any `rule_cost` in 6..64).
    rule_cost: f64,
}

impl Entropy {
    /// Creates an [`Entropy`] cost that charges `rule_cost` bits per new rule.
    #[cfg_attr(feature = "bench-internals", visibility::make(pub))]
    #[must_use]
    pub(crate) const fn new(rule_cost: f64) -> Self {
        Self {
            rule_cost,
        }
    }
}

impl Default for Entropy {
    fn default() -> Self {
        // Empirically bpb-optimal across the corpus (paper1..bible); ~4× the naive self-information
        // charge, reflecting the true coded cost of emitting a rule's definition. See `rule_cost`.
        Self::new(24.0)
    }
}

/// `g(x) = x · log2(x)`, with `g(x) = 0` for `x <= 0`. Callers pass count *differences* as `f64`,
/// so a fully consumed symbol (`count − freq == 0`) collapses to `0` with no branch.
#[inline]
fn g(x: f64) -> f64 {
    if x > 0.0 {
        x * x.log2()
    } else {
        0.0
    }
}

impl MergeCost for Entropy {
    fn score(&self, freq: u32, count_a: u32, count_b: u32, total: u32, same: bool) -> f64 {
        let n = f64::from(total);
        let f = f64::from(freq);
        let global = g(n) - g(n - f);
        let ca = f64::from(count_a);
        let local = if same {
            g(2.0f64.mul_add(-f, ca)) - g(ca)
        } else {
            let cb = f64::from(count_b);
            (g(ca - f) - g(ca)) + (g(cb - f) - g(cb))
        };
        global + local + g(f) - self.rule_cost
    }

    fn is_profitable(&self, score: f64, _freq: u32) -> bool {
        score > 0.0
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn frequency_scores_by_count() {
        let cost = Frequency;
        assert!(cost.score(10, 0, 0, 0, false) > cost.score(3, 0, 0, 0, false));
        assert!(cost.is_profitable(0.0, 2));
        assert!(!cost.is_profitable(0.0, 1));
    }

    #[test]
    fn entropy_default_rule_cost_reflects_grammar_emission() {
        // The default charges the empirically bpb-optimal per-rule cost, well above the naive
        // self-information charge, because emitting a rule adds two entropy-coded grammar tokens.
        assert!((Entropy::default().rule_cost - 24.0).abs() < f64::EPSILON);
    }

    #[test]
    fn entropy_prefers_predictable_over_merely_frequent() {
        // In a corpus of N tokens:
        //   P1 = "qu": rare but perfectly predictable — q always followed by u, so
        //        freq == count_q == count_u == 100 (high pointwise mutual information).
        //   P2 = "e"+space: both very common, co-occur often but ~independently —
        //        freq 500 with count_e 5000, count_space 8000 (low PMI despite high freq).
        let cost = Entropy::default();
        let n = 1_000_000;
        let predictable = cost.score(100, 100, 100, n, false);
        let frequent = cost.score(500, 5000, 8000, n, false);
        assert!(predictable > frequent, "predictable {predictable} should beat frequent {frequent}");
    }

    #[test]
    fn entropy_profitability_is_positive_saving() {
        let cost = Entropy::new(6.0);
        assert!(cost.is_profitable(0.5, 2));
        assert!(!cost.is_profitable(0.0, 2));
        assert!(!cost.is_profitable(-1.0, 1000));
    }
}
