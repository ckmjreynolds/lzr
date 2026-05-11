//! Adaptive probability models for the arithmetic coder.
//!
//! - `Order0<N>` — N-symbol Order-0 with Laplace `+1` smoothing and
//!   online halving when the running total approaches `TOTAL/2`.
//! - `Order1Bytes` — 256-symbol Order-1 over the previous byte. Same
//!   smoothing/halving discipline applied per context row.
//!
//! Halving keeps `total ≤ TOTAL/2` so the rescale to `TOTAL` mass in
//! `cdf_to` never collapses a slot to zero — the AC requires every
//! symbol to have ≥ 1 unit of mass.

use crate::ac::TOTAL;

/// Rescale threshold. Halving fires once `total` exceeds this.
const RESCALE_THRESHOLD: u64 = (TOTAL / 2) as u64;

/// N-symbol Order-0 adaptive model.
#[derive(Debug)]
pub(crate) struct Order0<const N: usize> {
    counts: [u32; N],
    total: u64,
}

impl<const N: usize> Order0<N> {
    pub(crate) const fn new() -> Self {
        Self {
            counts: [1; N],
            total: N as u64,
        }
    }

    /// Write the current CDF into `out`. `out.len()` must equal `N + 1`.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn cdf_to(&self, out: &mut [u32]) {
        debug_assert_eq!(out.len(), N + 1);
        let total = self.total;
        let mut acc: u64 = 0;
        for (slot, &c) in out.iter_mut().zip(self.counts.iter()).take(N) {
            *slot = ((acc * u64::from(TOTAL)) / total) as u32;
            acc += u64::from(c);
        }
        out[N] = TOTAL;
    }

    pub(crate) fn observe(&mut self, sym: usize) {
        debug_assert!(sym < N);
        self.counts[sym] += 1;
        self.total += 1;
        if self.total > RESCALE_THRESHOLD {
            self.rescale();
        }
    }

    fn rescale(&mut self) {
        let mut new_total: u64 = 0;
        for c in &mut self.counts {
            *c = (*c >> 1).max(1);
            new_total += u64::from(*c);
        }
        self.total = new_total;
    }
}

/// 256-symbol Order-1 over the previous byte: 256 context rows of 256
/// counts each. Memory is `256 * 256 * 4 = 256 KiB` of counts plus
/// `256 * 8 = 2 KiB` of u64 totals — well within budget. The count
/// table is a `Vec<[u32; 256]>` rather than a `Box<[[u32; 256]; 256]>`
/// to avoid materializing the 256 KiB initializer on the stack.
#[derive(Debug)]
pub(crate) struct Order1Bytes {
    counts: Vec<[u32; 256]>,
    totals: [u64; 256],
    /// Previous byte in the modeled stream. `None` only at cold
    /// start — once the model has observed a single byte we always
    /// have context.
    last: Option<u8>,
}

impl Order1Bytes {
    pub(crate) fn new() -> Self {
        Self {
            counts: vec![[1u32; 256]; 256],
            totals: [256u64; 256],
            last: None,
        }
    }

    /// CDF (257 entries) conditioned on the current `last` byte.
    /// Falls back to a flat uniform CDF when `last` is `None`.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn cdf_to(&self, out: &mut [u32]) {
        debug_assert_eq!(out.len(), 257);
        if let Some(prev) = self.last {
            let ctx = prev as usize;
            let total = self.totals[ctx];
            let row = &self.counts[ctx];
            let mut acc: u64 = 0;
            for (i, slot) in out.iter_mut().enumerate().take(256) {
                *slot = ((acc * u64::from(TOTAL)) / total) as u32;
                acc += u64::from(row[i]);
            }
        } else {
            let n = 256u64;
            for (i, slot) in out.iter_mut().enumerate().take(256) {
                *slot = ((i as u64 * u64::from(TOTAL)) / n) as u32;
            }
        }
        out[256] = TOTAL;
    }

    pub(crate) fn observe(&mut self, byte: u8) {
        if let Some(prev) = self.last {
            let ctx = prev as usize;
            self.counts[ctx][byte as usize] += 1;
            self.totals[ctx] += 1;
            if self.totals[ctx] > RESCALE_THRESHOLD {
                self.rescale_row(ctx);
            }
        }
        self.last = Some(byte);
    }

    fn rescale_row(&mut self, ctx: usize) {
        let mut new_total: u64 = 0;
        for c in &mut self.counts[ctx] {
            *c = (*c >> 1).max(1);
            new_total += u64::from(*c);
        }
        self.totals[ctx] = new_total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cdf_is_valid(cdf: &[u32]) -> bool {
        if cdf[0] != 0 {
            return false;
        }
        if *cdf.last().unwrap() != TOTAL {
            return false;
        }
        cdf.windows(2).all(|w| w[1] > w[0])
    }

    #[test]
    fn order0_cold_start_uniform_cdf() {
        let m: Order0<256> = Order0::new();
        let mut cdf = vec![0u32; 257];
        m.cdf_to(&mut cdf);
        assert!(cdf_is_valid(&cdf));
        // Symmetry: every gap should equal TOTAL/N at cold start.
        let expected_gap = TOTAL / 256;
        for w in cdf.windows(2) {
            assert_eq!(w[1] - w[0], expected_gap);
        }
    }

    #[test]
    fn order0_skewed_concentrates_mass() {
        let mut m: Order0<256> = Order0::new();
        for _ in 0..10_000 {
            m.observe(b'a' as usize);
        }
        let mut cdf = vec![0u32; 257];
        m.cdf_to(&mut cdf);
        assert!(cdf_is_valid(&cdf));
        let a_mass = cdf[b'a' as usize + 1] - cdf[b'a' as usize];
        let other_mass = cdf[b'b' as usize + 1] - cdf[b'b' as usize];
        assert!(a_mass > 50 * other_mass);
    }

    #[test]
    fn order1_learns_alternating_pattern() {
        let mut m = Order1Bytes::new();
        for _ in 0..1000 {
            m.observe(b'a');
            m.observe(b'b');
        }
        // Last observed was 'b'. Probe context = 'a' by feeding one
        // more 'a' and checking CDF before observing.
        m.observe(b'a');
        let mut cdf = vec![0u32; 257];
        m.cdf_to(&mut cdf);
        assert!(cdf_is_valid(&cdf));
        let b_mass = cdf[b'b' as usize + 1] - cdf[b'b' as usize];
        let c_mass = cdf[b'c' as usize + 1] - cdf[b'c' as usize];
        assert!(b_mass > 100 * c_mass);
    }

    #[test]
    fn order0_cdf_valid_through_long_run() {
        let mut m: Order0<256> = Order0::new();
        let mut cdf = vec![0u32; 257];
        for step in 0..50_000u32 {
            #[allow(clippy::cast_possible_truncation)]
            m.observe((step % 256) as usize);
            m.cdf_to(&mut cdf);
            assert!(cdf_is_valid(&cdf), "step {step}");
        }
    }
}
