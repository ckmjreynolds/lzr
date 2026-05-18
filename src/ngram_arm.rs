//! Runtime byte-level n-gram arm — Order-2 fixed-size count table.
//!
//! Both encoder and decoder build the same table by feeding bytes in
//! the same order, so no model state is shipped (zero `L(D)` cost).
//! Memory at runtime: 65 536 contexts × 257 u32 counts ≈ 67 MB,
//! comfortably inside the 10 GB judge budget.
//!
//! Counts are Laplace-smoothed with `+1` floor so every byte gets at
//! least one unit of AC mass, even contexts seen zero times. At cold
//! start (first ~2 bytes) the table falls back to a uniform CDF —
//! same convention as [`crate::moe_arm`].
//!
//! Counts are capped at `COUNT_CAP` (16-bit equivalent) and divided
//! by 2 when any context reaches the cap. This keeps the CDF
//! responsive to non-stationary text (e.g., when the article topic
//! shifts) without needing a learning-rate hyperparameter.
//!
//! The CDF is produced in 16-bit fixed-point against `crate::ac::TOTAL`
//! with strict-increasing enforcement, matching the AC contract.

#![allow(dead_code)]

use crate::ac::TOTAL;

const ORDER: usize = 2;
const N_CONTEXTS: usize = 1 << (8 * ORDER); // 65 536 for Order-2
const COUNT_CAP: u32 = 1 << 15;

pub(crate) struct NgramArm {
    /// `counts[context_idx * 257 + byte]` holds the count of `byte`
    /// following the 2-byte `context_idx`, with `[256]` reserved for
    /// the running total per context.
    counts: Vec<u32>,
    /// Rolling 2-byte context of the most recent bytes fed.
    context: u16,
    /// Bytes fed since last reset; once `>= ORDER` the context is
    /// reliable, before that we predict uniform.
    fed_count: usize,
}

impl NgramArm {
    pub(crate) fn new() -> Self {
        Self {
            counts: vec![0_u32; N_CONTEXTS * 257],
            context: 0,
            fed_count: 0,
        }
    }

    pub(crate) const fn fed_count(&self) -> usize {
        self.fed_count
    }

    /// Reset to fresh state. Frees no memory — the counts vector
    /// stays allocated and is zeroed in place.
    pub(crate) fn reset(&mut self) {
        for c in &mut self.counts {
            *c = 0;
        }
        self.context = 0;
        self.fed_count = 0;
    }

    /// Feed one source byte: update the count for `(context, byte)`,
    /// roll the context forward, and divide if any context reached
    /// the count cap.
    pub(crate) fn feed(&mut self, byte: u8) {
        if self.fed_count >= ORDER {
            let ctx = self.context as usize;
            let slot = ctx * 257 + byte as usize;
            self.counts[slot] = self.counts[slot].saturating_add(1);
            self.counts[ctx * 257 + 256] = self.counts[ctx * 257 + 256].saturating_add(1);
            if self.counts[ctx * 257 + 256] >= COUNT_CAP {
                // Halve every count in this context, total included.
                // The +1 floor preserved on AC emit ensures any byte
                // that drops to 0 still gets representable mass.
                for slot in &mut self.counts[ctx * 257..(ctx + 1) * 257] {
                    *slot /= 2;
                }
            }
        }
        self.context = (self.context << 8) | u16::from(byte);
        self.fed_count += 1;
    }

    /// Produce a 257-entry strict-monotonic AC CDF for the next byte
    /// given the current context. At cold start (<= ORDER bytes fed)
    /// returns the uniform CDF.
    #[allow(clippy::needless_range_loop, clippy::cast_possible_truncation)]
    pub(crate) fn predict_byte_cdf(&self, out: &mut [u32; 257]) {
        if self.fed_count < ORDER {
            uniform_cdf(out);
            return;
        }
        let ctx = self.context as usize;
        let total_raw = self.counts[ctx * 257 + 256];
        if total_raw == 0 {
            uniform_cdf(out);
            return;
        }

        // Apply +1 Laplace floor to every byte (256 added) and recompute
        // the total. This keeps zero-observed bytes representable
        // without dragging predicted-byte mass too far down.
        let smoothed_total = u64::from(total_raw) + 256;
        out[0] = 0;
        let mut acc = 0_u64;
        let total_u64 = u64::from(TOTAL);
        for i in 0..256 {
            let c = u64::from(self.counts[ctx * 257 + i]) + 1;
            acc += c;
            let scaled = (acc * total_u64) / smoothed_total;
            out[i + 1] = scaled as u32;
        }
        out[256] = TOTAL;

        enforce_monotonic(out);
    }
}

impl Default for NgramArm {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for NgramArm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `counts` is a 67 MB Vec — print its length, not contents.
        f.debug_struct("NgramArm")
            .field("order", &ORDER)
            .field("n_contexts", &N_CONTEXTS)
            .field("counts_len", &self.counts.len())
            .field("fed_count", &self.fed_count)
            .field("context", &self.context)
            .finish()
    }
}

fn uniform_cdf(out: &mut [u32; 257]) {
    out[0] = 0;
    let per = TOTAL / 256;
    let leftover = TOTAL - per * 256;
    for i in 0..256 {
        let extra = u32::from(i < leftover as usize);
        out[i + 1] = out[i] + per + extra;
    }
    debug_assert_eq!(out[256], TOTAL);
}

fn enforce_monotonic(out: &mut [u32; 257]) {
    let mut prev = 0_u32;
    for slot in out.iter_mut().take(257).skip(1) {
        if *slot <= prev {
            *slot = prev + 1;
        }
        prev = *slot;
    }
    if out[256] != TOTAL {
        let last = out[256];
        for slot in out.iter_mut().take(257).skip(1) {
            let scaled = u64::from(*slot) * u64::from(TOTAL) / u64::from(last);
            *slot = u32::try_from(scaled).unwrap_or(u32::MAX).max(1);
        }
        out[256] = TOTAL;
        let mut prev = 0_u32;
        for slot in out.iter_mut().take(257).skip(1) {
            if *slot <= prev {
                *slot = prev + 1;
            }
            prev = *slot;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_cold_start_is_uniform() {
        let arm = NgramArm::new();
        let mut cdf = [0_u32; 257];
        arm.predict_byte_cdf(&mut cdf);
        // Uniform: per-symbol mass ≈ TOTAL/256.
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[256], TOTAL);
        let per = TOTAL / 256;
        for i in 0..256 {
            let mass = cdf[i + 1] - cdf[i];
            // Allow ±1 for the leftover distribution.
            assert!(
                (per - 1..=per + 1).contains(&mass),
                "uniform mass at {i}: {mass} (expected ~{per})"
            );
        }
    }

    #[test]
    fn ngram_learns_obvious_pattern() {
        // Feed "ababababab..." — context (a,b) should strongly predict 'a'.
        let mut arm = NgramArm::new();
        for _ in 0..100 {
            arm.feed(b'a');
            arm.feed(b'b');
        }
        // After feeding many cycles, context is (a,b), next byte should
        // be 'a' with very high probability.
        let mut cdf = [0_u32; 257];
        arm.predict_byte_cdf(&mut cdf);
        let mass_a = cdf[b'a' as usize + 1] - cdf[b'a' as usize];
        let mass_b = cdf[b'b' as usize + 1] - cdf[b'b' as usize];
        assert!(
            mass_a > 10 * mass_b,
            "ngram should predict 'a' much more than 'b' after ababab: mass_a={mass_a}, mass_b={mass_b}"
        );
    }

    #[test]
    fn ngram_cdf_strictly_monotonic() {
        // Feed varied bytes, ensure CDF stays well-formed.
        let mut arm = NgramArm::new();
        for i in 0..1000_u16 {
            arm.feed((i.wrapping_mul(31) & 0xff) as u8);
        }
        let mut cdf = [0_u32; 257];
        arm.predict_byte_cdf(&mut cdf);
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[256], TOTAL);
        for i in 1..=256 {
            assert!(
                cdf[i] > cdf[i - 1],
                "non-monotonic at {i}: {} vs {}",
                cdf[i - 1],
                cdf[i]
            );
        }
    }

    #[test]
    fn ngram_count_cap_halves_safely() {
        let mut arm = NgramArm::new();
        // Get into a stable 2-byte context first.
        arm.feed(b'a');
        arm.feed(b'b');
        // Feed enough 'c's after context (a,b)→c to exceed the cap.
        // Each new (b,c) pair extends the context, so we need to
        // repeat the (a,b)→c pattern. Easier: feed (a,b,c) repeatedly.
        for _ in 0..COUNT_CAP {
            arm.feed(b'c');
            arm.feed(b'a');
            arm.feed(b'b');
        }
        let mut cdf = [0_u32; 257];
        arm.predict_byte_cdf(&mut cdf);
        // Still well-formed after multiple cap-halvings.
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[256], TOTAL);
    }
}
