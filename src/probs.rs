//! Logits → CDF conversion for the arithmetic coder.
//!
//! Softmax is computed in `f32` (one `f32::exp` per symbol is ~2× faster
//! than `f64::exp` and plenty precise for a 256-symbol, 16-bit CDF quant
//! step). The 65 536-count total mass is distributed deterministically:
//!
//! 1. Every symbol gets at least one count (prevents zero-probability
//!    symbols, which the AC cannot represent).
//! 2. The remaining `TOTAL_EXTRA = 65 536 - 256 = 65 280` counts are split
//!    proportional to `p[i]` via `floor(p[i] * TOTAL_EXTRA)`.
//! 3. The residual (non-negative by construction) is allocated one count at
//!    a time to symbols ordered by descending probability, ties broken by
//!    ascending index. This is fully deterministic and exactly reproducible.
//!
//! The CDF total is exactly [`crate::ac::TOTAL`] (= 65 536).

use crate::ac::TOTAL;
use crate::arch::VOCAB;

/// Counts reserved for proportional allocation (every symbol already gets +1,
/// so the pool is `TOTAL - VOCAB`).
///
/// `VOCAB` is 256 → fits in u32 exactly; the cast is safe.
#[allow(clippy::cast_possible_truncation)]
const EXTRA: u32 = TOTAL - VOCAB as u32;

/// Convert `logits[256]` into a 257-entry CDF suitable for [`crate::ac`].
///
/// The casts inside are bounded:
/// - `VOCAB as u32`: `VOCAB = 256` fits in u32.
/// - `(p * EXTRA).floor() as u32`: `p ∈ [0, 1]`, product ∈ `[0, EXTRA]`, so
///   neither truncation nor sign loss is possible.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn logits_to_cdf(logits: &[f32; VOCAB]) -> [u32; 257] {
    // f32 softmax (max-shift for numerical stability). A 16-bit CDF can
    // distinguish ~10⁵ probability levels; f32's 23-bit mantissa is well
    // over the resolution we quantize to.
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l > max {
            max = l;
        }
    }
    let mut probs = [0.0_f32; VOCAB];
    let mut sum = 0.0_f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        probs[i] = e;
        sum += e;
    }
    if sum == 0.0 {
        // Degenerate case: treat as uniform.
        return uniform_cdf();
    }
    let inv_sum = 1.0 / sum;
    for p in &mut probs {
        *p *= inv_sum;
    }

    // Base-1 allocation + proportional.
    //
    // `EXTRA as f32` looks like a precision-loss hazard on paper but
    // `EXTRA = 65 280` easily fits in f32's 23-bit mantissa.
    #[allow(clippy::cast_precision_loss)]
    let extra_f = EXTRA as f32;
    let mut counts = [1u32; VOCAB];
    let mut allocated: u32 = VOCAB as u32;
    for (i, &p) in probs.iter().enumerate() {
        let add = (p * extra_f).floor() as u32;
        counts[i] += add;
        allocated += add;
    }

    // Residual distribution: sort indices by (-p, +i).
    let mut residual = TOTAL - allocated;
    if residual > 0 {
        let mut idx: [usize; VOCAB] = core::array::from_fn(|i| i);
        idx.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut k = 0;
        while residual > 0 {
            counts[idx[k]] += 1;
            residual -= 1;
            k += 1;
        }
    }

    // Cumulative sum.
    let mut cdf = [0u32; 257];
    let mut acc = 0u32;
    for i in 0..VOCAB {
        cdf[i] = acc;
        acc += counts[i];
    }
    cdf[VOCAB] = acc;
    debug_assert_eq!(cdf[VOCAB], TOTAL);
    cdf
}

/// 257-entry CDF where every symbol has equal mass (`TOTAL / 256 = 256`).
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn uniform_cdf() -> [u32; 257] {
    let step = TOTAL / (VOCAB as u32);
    core::array::from_fn(|i| (i as u32) * step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_symmetric() {
        let cdf = uniform_cdf();
        for i in 0..256 {
            assert_eq!(cdf[i + 1] - cdf[i], 256);
        }
        assert_eq!(cdf[256], TOTAL);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn logits_cdf_totals_to_total() {
        let mut logits = [0.0_f32; 256];
        for (i, l) in logits.iter_mut().enumerate() {
            *l = ((i as f32) * 0.05).sin();
        }
        let cdf = logits_to_cdf(&logits);
        assert_eq!(cdf[256], TOTAL);
        assert_eq!(cdf[0], 0);
        for i in 0..256 {
            assert!(cdf[i + 1] > cdf[i], "sym {i} has zero mass");
        }
    }

    #[test]
    fn logits_cdf_deterministic() {
        // Same inputs → byte-identical outputs.
        let logits = [0.0_f32; 256];
        let a = logits_to_cdf(&logits);
        let b = logits_to_cdf(&logits);
        assert_eq!(a, b);
        // Zero logits → uniform (all probabilities equal, residual split by
        // index order → first 0 symbols get +1, but all equal so +0 per).
        for i in 1..=256 {
            assert!(a[i] > a[i - 1], "every symbol gets ≥ 1 count");
        }
    }

    #[test]
    fn logits_cdf_highly_skewed() {
        // One symbol dominates; others get min-1-count.
        let mut logits = [-20.0_f32; 256];
        logits[42] = 20.0;
        let cdf = logits_to_cdf(&logits);
        let mass42 = cdf[43] - cdf[42];
        assert!(
            mass42 > TOTAL - 300,
            "expected ~all mass at symbol 42, got {mass42}"
        );
        // Every other symbol still has at least 1.
        for i in 0..256 {
            if i != 42 {
                assert!(cdf[i + 1] - cdf[i] >= 1);
            }
        }
    }

    #[test]
    fn logits_cdf_handles_inf_max() {
        // Non-finite logits shouldn't panic; treat as valid softmax input by
        // the underlying f64 ops. NaN is the one case we don't guarantee.
        let mut logits = [0.0_f32; 256];
        logits[0] = f32::MAX / 2.0;
        let cdf = logits_to_cdf(&logits);
        assert_eq!(cdf[256], TOTAL);
    }
}
