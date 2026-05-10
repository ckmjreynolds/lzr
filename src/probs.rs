//! Logits → CDF conversion for the arithmetic coder.
//!
//! Softmax is computed in `f32` (one `f32::exp` per symbol is ~2× faster
//! than `f64::exp` and plenty precise for the 16-bit CDF quant step). The
//! 65 536-count total mass is distributed deterministically:
//!
//! 1. Every symbol gets at least one count (prevents zero-probability
//!    symbols, which the AC cannot represent).
//! 2. The remaining `TOTAL_EXTRA = 65 536 - VOCAB` counts are split
//!    proportional to `p[i]` via `floor(p[i] * TOTAL_EXTRA)`.
//! 3. The residual (non-negative by construction) is allocated one count at
//!    a time to symbols ordered by descending probability, ties broken by
//!    ascending index. This is fully deterministic and exactly reproducible.
//!
//! The CDF total is exactly [`crate::ac::TOTAL`] (= 65 536).

// `[u32; CDF_LEN]` (= 16 388 bytes at VOCAB=4096) is a few bytes over
// clippy's `large_stack_arrays` threshold but fine in practice — we
// build at most one CDF per token, never inside a tight inner loop, and
// the alternative (boxing) costs an allocation per token.
#![allow(clippy::large_stack_arrays)]

use crate::ac::TOTAL;
use crate::arch::{CDF_LEN, VOCAB};

/// Counts reserved for proportional allocation (every symbol already gets +1,
/// so the pool is `TOTAL - VOCAB`).
///
/// Cast is safe for `VOCAB ≤ TOTAL`.
#[allow(clippy::cast_possible_truncation)]
const EXTRA: u32 = TOTAL - VOCAB as u32;

const _: () = {
    assert!(
        VOCAB <= TOTAL as usize,
        "VOCAB must be ≤ TOTAL so every symbol can get ≥1 count"
    );
};

/// Convert `logits[VOCAB]` into a `CDF_LEN`-entry CDF for [`crate::ac`].
///
/// The casts inside are bounded:
/// - `VOCAB as u32` fits when `VOCAB ≤ TOTAL`.
/// - `(p * EXTRA).floor() as u32`: `p ∈ [0, 1]`, product ∈ `[0, EXTRA]`,
///   so neither truncation nor sign loss is possible.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn logits_to_cdf(logits: &[f32; VOCAB]) -> [u32; CDF_LEN] {
    // f32 softmax (max-shift for numerical stability).
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
        return uniform_cdf();
    }
    let inv_sum = 1.0 / sum;
    for p in &mut probs {
        *p *= inv_sum;
    }

    // Base-1 allocation + proportional.
    //
    // `EXTRA as f32` looks like a precision-loss hazard on paper but
    // `EXTRA ≤ 65 536` easily fits in f32's 23-bit mantissa.
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
        let mut idx: Vec<usize> = (0..VOCAB).collect();
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
    let mut cdf = [0u32; CDF_LEN];
    let mut acc = 0u32;
    for i in 0..VOCAB {
        cdf[i] = acc;
        acc += counts[i];
    }
    cdf[VOCAB] = acc;
    debug_assert_eq!(cdf[VOCAB], TOTAL);
    cdf
}

/// `CDF_LEN`-entry CDF where every symbol has equal mass (mostly).
/// Distributes `TOTAL` counts with rounding so totals match exactly.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn uniform_cdf() -> [u32; CDF_LEN] {
    let v = VOCAB as u32;
    let step_q = TOTAL / v;
    let step_r = TOTAL % v;
    let mut cdf = [0u32; CDF_LEN];
    let mut acc: u32 = 0;
    for (i, slot) in cdf.iter_mut().enumerate().take(VOCAB) {
        *slot = acc;
        let extra = u32::from(u32::try_from(i).unwrap() < step_r);
        acc += step_q + extra;
    }
    cdf[VOCAB] = TOTAL;
    debug_assert_eq!(cdf[VOCAB], TOTAL);
    cdf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_totals_to_total() {
        let cdf = uniform_cdf();
        assert_eq!(cdf[VOCAB], TOTAL);
        // Every symbol has at least 1 count.
        for i in 0..VOCAB {
            assert!(cdf[i + 1] > cdf[i], "symbol {i} has zero mass");
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn logits_cdf_totals_to_total() {
        let mut logits = [0.0_f32; VOCAB];
        for (i, l) in logits.iter_mut().enumerate() {
            *l = ((i as f32) * 0.05).sin();
        }
        let cdf = logits_to_cdf(&logits);
        assert_eq!(cdf[VOCAB], TOTAL);
        assert_eq!(cdf[0], 0);
        for i in 0..VOCAB {
            assert!(cdf[i + 1] > cdf[i], "sym {i} has zero mass");
        }
    }

    #[test]
    fn logits_cdf_deterministic() {
        let logits = [0.0_f32; VOCAB];
        let a = logits_to_cdf(&logits);
        let b = logits_to_cdf(&logits);
        assert_eq!(a, b);
        for i in 1..=VOCAB {
            assert!(a[i] > a[i - 1], "every symbol gets ≥ 1 count");
        }
    }

    #[test]
    fn logits_cdf_highly_skewed() {
        let mut logits = [-20.0_f32; VOCAB];
        logits[42] = 20.0;
        let cdf = logits_to_cdf(&logits);
        let mass42 = cdf[43] - cdf[42];
        // Allow generous slack — with a wider VOCAB the residual gets
        // larger but mode mass should still dominate.
        let vocab_u32 = u32::try_from(VOCAB).unwrap();
        assert!(
            mass42 > TOTAL - 2 * vocab_u32,
            "expected ~all mass at symbol 42, got {mass42}"
        );
        for i in 0..VOCAB {
            if i != 42 {
                assert!(cdf[i + 1] - cdf[i] >= 1);
            }
        }
    }

    #[test]
    fn logits_cdf_handles_inf_max() {
        let mut logits = [0.0_f32; VOCAB];
        logits[0] = f32::MAX / 2.0;
        let cdf = logits_to_cdf(&logits);
        assert_eq!(cdf[VOCAB], TOTAL);
    }
}
