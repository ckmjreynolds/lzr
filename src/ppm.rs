//! PPM-D byte predictor.
//!
//! Order-N Prediction by Partial Matching with PPM-D escape weighting
//! and full exclusion. Built as a single-event predictor (one AC
//! emission per byte) rather than the multi-event "encode escape, try
//! lower order, encode escape, ..." formulation: the single 256-symbol
//! pmf is mathematically equivalent and lets PPM drop straight into
//! the existing `ProbSource` slot in the codec.
//!
//! ## Algorithm
//!
//! For each byte, walk from the longest available context (order
//! `min(history, max_order)`) down to order 0, with an excluded set
//! growing as we go:
//!
//!  - At order k, look up the context. If it has been seen and has
//!    at least one un-excluded symbol, those un-excluded symbols
//!    receive `remaining × (2c / (2T + n))` of the pmf each, escape
//!    receives `remaining × (n / (2T + n))`. The escape mass becomes
//!    the new `remaining` for the lower-order pass; symbols just
//!    allocated mass join the excluded set.
//!  - If the context is unseen or all-excluded, descend silently
//!    (no pmf change, no AC event).
//!
//! After visiting order 0, any leftover `remaining` is split uniformly
//! over symbols still not in the excluded set (the order-(-1)
//! fallback). The pmf then sums to 1.0 over all 256 symbols (every
//! symbol has positive mass — the actual symbol being encoded is
//! always in the union of excluded ∪ usable).
//!
//! ## Storage
//!
//! Sparse: `HashMap<Vec<u8>, ContextNode>` per order. Visited contexts
//! cost ~80–120 B each. For order-4 PPM on the non-letter sub-stream
//! of enwik9 (~330 MB), expect a few-MB working set.

use std::collections::HashMap;

use crate::ac::TOTAL;
use crate::arch::{CDF_LEN, VOCAB};

const _: () = assert!(VOCAB == 256, "PPM module assumes byte-level VOCAB=256");

#[derive(Default)]
struct ContextNode {
    counts: HashMap<u8, u32>,
}

/// PPM-D predictor. `max_order` is the longest context length tracked;
/// classical text bounds are 4–6.
///
/// Caller-managed context: each call to `observe`/`build_cdf` takes a
/// `ctx` slice carrying the recent byte history. This lets callers
/// share a single mixed-stream history across multiple predictors —
/// e.g. the type-routed codec uses ONE byte history and queries either
/// the letter or non-letter PPM at each position depending on the byte
/// class. Cross-stream context is the cmix-style refinement that pulls
/// surrounding-markup priors into the letter arm and vice versa.
pub(crate) struct Ppm {
    max_order: usize,
    contexts: Vec<HashMap<Vec<u8>, ContextNode>>,
}

impl Ppm {
    pub(crate) fn new(max_order: usize) -> Self {
        let mut contexts = Vec::with_capacity(max_order + 1);
        for _ in 0..=max_order {
            contexts.push(HashMap::new());
        }
        Self {
            max_order,
            contexts,
        }
    }

    /// Length of the longest context this PPM uses. Callers should
    /// keep at least this many trailing bytes in their history buffer.
    pub(crate) const fn max_order(&self) -> usize {
        self.max_order
    }

    /// Suffix of `ctx` at order `k` (`k <= ctx.len() <= max_order`
    /// expected; caller is responsible for length).
    fn suffix(ctx: &[u8], k: usize) -> &[u8] {
        let n = k.min(ctx.len());
        &ctx[ctx.len() - n..]
    }

    /// Increment counts at every visited context for `sym`. `ctx`
    /// carries the byte history immediately preceding the symbol; only
    /// the trailing `max_order` bytes are used.
    pub(crate) fn observe(&mut self, ctx: &[u8], sym: u8) {
        let n = ctx.len().min(self.max_order);
        let trimmed_ctx = &ctx[ctx.len() - n..];
        for k in 0..=n {
            let key = Self::suffix(trimmed_ctx, k).to_vec();
            let node = self.contexts[k].entry(key).or_default();
            *node.counts.entry(sym).or_insert(0) += 1;
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub(crate) fn build_cdf(&self, ctx: &[u8]) -> [u32; CDF_LEN] {
        let mut mass = [0.0_f64; VOCAB];
        let mut excluded = [false; VOCAB];
        let mut remaining = 1.0_f64;

        let max_k = ctx.len().min(self.max_order);
        let trimmed_ctx = &ctx[ctx.len() - max_k..];
        for k in (0..=max_k).rev() {
            let suf = Self::suffix(trimmed_ctx, k);
            let Some(node) = self.contexts[k].get(suf) else {
                continue;
            };

            let mut t: u32 = 0;
            let mut n: u32 = 0;
            for (&s, &c) in &node.counts {
                if !excluded[s as usize] {
                    t += c;
                    n += 1;
                }
            }
            if n == 0 {
                continue;
            }
            let denom = f64::from(2 * t + n);
            for (&s, &c) in &node.counts {
                if !excluded[s as usize] {
                    mass[s as usize] = remaining * (2.0 * f64::from(c)) / denom;
                    excluded[s as usize] = true;
                }
            }
            let e = f64::from(n) / denom;
            remaining *= e;
        }

        // Order -1 fallback: uniform over symbols not yet excluded.
        // Edge case: if every symbol has already been given mass at
        // some higher order (typical once order-0 has seen all 256
        // bytes), `remaining` represents "escape further" probability
        // that has nowhere to go. Fold it back proportionally so the
        // pmf still sums to 1.0 — equivalent to recognizing that the
        // last order's escape coefficient should have been 0 since
        // there are no further symbols to escape to.
        if remaining > 0.0 {
            let n_usable = (0..VOCAB).filter(|&i| !excluded[i]).count();
            if n_usable > 0 {
                let m = remaining / n_usable as f64;
                for i in 0..VOCAB {
                    if !excluded[i] {
                        mass[i] = m;
                    }
                }
            } else {
                let existing: f64 = mass.iter().sum();
                if existing > 0.0 {
                    let scale = (existing + remaining) / existing;
                    for m in &mut mass {
                        *m *= scale;
                    }
                }
            }
        }

        // Convert pmf to integer counts with each gap ≥ 1 and total =
        // TOTAL. Same allocation strategy as `crate::probs::logits_to_cdf`.
        let mut counts = [1u32; VOCAB];
        let mut allocated: u32 = VOCAB as u32;
        let extra = f64::from(TOTAL - VOCAB as u32);
        let mut frac_idx: Vec<(f64, usize)> = Vec::with_capacity(VOCAB);
        for (i, &p) in mass.iter().enumerate() {
            let prop = (p * extra).floor();
            let prop_u = prop as u32;
            counts[i] += prop_u;
            allocated += prop_u;
            let frac = p.mul_add(extra, -prop);
            frac_idx.push((frac, i));
        }
        frac_idx.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let mut residual = TOTAL - allocated;
        let mut idx = 0;
        while residual > 0 {
            counts[frac_idx[idx].1] += 1;
            residual -= 1;
            idx += 1;
        }

        let mut cdf = [0u32; CDF_LEN];
        let mut acc = 0u32;
        for i in 0..VOCAB {
            cdf[i] = acc;
            acc += counts[i];
        }
        cdf[VOCAB] = TOTAL;
        debug_assert_eq!(cdf[VOCAB], TOTAL);
        cdf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_pmf_is_uniform() {
        let p = Ppm::new(4);
        let cdf = p.build_cdf(&[]);
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[VOCAB], TOTAL);
        let nominal = TOTAL / u32::try_from(VOCAB).unwrap();
        for i in 0..VOCAB {
            let m = cdf[i + 1] - cdf[i];
            assert!(
                m == nominal || m == nominal + 1,
                "sym {i} has mass {m}, expected {nominal} ± 1"
            );
        }
    }

    #[test]
    fn cdf_stays_valid_through_long_run() {
        let mut p = Ppm::new(4);
        let mut history: Vec<u8> = Vec::with_capacity(4);
        for step in 0u32..20_000 {
            let sym = u8::try_from(step % 256).unwrap();
            p.observe(&history, sym);
            if history.len() == 4 {
                history.remove(0);
            }
            history.push(sym);
            let cdf = p.build_cdf(&history);
            assert_eq!(cdf[VOCAB], TOTAL, "step {step}: total drift");
            for i in 0..VOCAB {
                assert!(cdf[i + 1] > cdf[i], "step {step}: sym {i} zero mass");
            }
        }
    }

    #[test]
    fn learns_periodic_pattern() {
        let mut p = Ppm::new(4);
        let cycle = *b"abcd";
        let mut history: Vec<u8> = Vec::with_capacity(4);
        for i in 0..2_000 {
            let sym = cycle[i % 4];
            p.observe(&history, sym);
            if history.len() == 4 {
                history.remove(0);
            }
            history.push(sym);
        }
        // History ends with "...abcd". Querying with that history should
        // predict 'a' overwhelmingly as the next cycle position.
        let cdf = p.build_cdf(&history);
        let a_mass = cdf[b'a' as usize + 1] - cdf[b'a' as usize];
        let z_mass = cdf[b'z' as usize + 1] - cdf[b'z' as usize];
        assert!(
            a_mass > 1000 * z_mass,
            "after periodic training, expected 'a' to dominate; got a={a_mass} z={z_mass}"
        );
    }
}
