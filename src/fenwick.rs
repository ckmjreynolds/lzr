//! Fenwick (binary-indexed) tree over `u32` counts.
//!
//! Point-update and prefix/range sums in O(log n). The reusable structure behind
//! frequency-table models; the order-0 model is its first consumer.

/// A Fenwick tree of `u32` counts over a fixed number of slots.
#[derive(Clone, Debug)]
pub(crate) struct Fenwick {
    // 1-based internally; `tree[0]` is unused.
    tree: Vec<u32>,
}

impl Fenwick {
    /// New tree of `n` zeroed slots (logical indices `0..n`).
    pub(crate) fn new(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1],
        }
    }

    /// Add `delta` to slot `i`.
    pub(crate) fn add(&mut self, i: usize, delta: u32) {
        let mut j = i + 1;
        while j < self.tree.len() {
            self.tree[j] += delta;
            j += j & j.wrapping_neg();
        }
    }

    /// Sum of slots `[0, i)`.
    pub(crate) fn prefix_sum(&self, i: usize) -> u32 {
        let mut s = 0;
        let mut j = i;
        while j > 0 {
            s += self.tree[j];
            j -= j & j.wrapping_neg();
        }
        s
    }

    /// Sum of slots `[lo, hi)`.
    pub(crate) fn range_sum(&self, lo: usize, hi: usize) -> u32 {
        self.prefix_sum(hi) - self.prefix_sum(lo)
    }

    /// Sum of all slots.
    pub(crate) fn total(&self) -> u32 {
        self.prefix_sum(self.tree.len() - 1)
    }

    /// Reset every slot to zero, keeping the allocation.
    pub(crate) fn clear(&mut self) {
        self.tree.fill(0);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]
    use super::*;

    #[test]
    fn matches_brute_force() {
        let n = 256;
        let mut fw = Fenwick::new(n);
        let mut reference = vec![0u32; n];
        let mut state = 12345u64;
        for _ in 0..10_000 {
            // simple LCG to drive a deterministic update pattern
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let i = (state >> 33) as usize % n;
            let d = ((state >> 17) as u32) % 7 + 1;
            fw.add(i, d);
            reference[i] += d;

            let hi = (state as usize) % (n + 1);
            let expect: u32 = reference[..hi].iter().sum();
            assert_eq!(fw.prefix_sum(hi), expect);
        }
        let total: u32 = reference.iter().sum();
        assert_eq!(fw.total(), total);
        assert_eq!(fw.range_sum(10, 20), reference[10..20].iter().sum());
    }
}
