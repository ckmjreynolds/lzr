//! Adaptive frequency model for arithmetic coding.
//!
//! Maintains per-symbol frequency counts (256 symbols, one per byte value) with a
//! fixed total count of [`TOTAL_COUNTS`]. A Fenwick tree provides O(log N) prefix-sum
//! queries and updates, enabling efficient CDF lookups for encoding and decoding.
//!
//! Adaptation uses round-robin victim selection: each [`Model::update`] call
//! decrements one victim symbol and increments the observed symbol, keeping the
//! total constant. Symbols with a frequency of 1 are skipped as victims so that
//! every symbol always has a non-zero probability.

#[cfg(test)]
use std::fmt;

use static_assertions::const_assert;

const_assert!(TOTAL_COUNTS.is_power_of_two());
const_assert!(TOTAL_COUNTS == NUM_SYMBOLS * INITIAL_COUNT);

/// Total frequency count across all symbols. Must be a power of two.
const TOTAL_COUNTS: usize = 1 << 12;

/// Initial frequency assigned to every symbol in a fresh model.
const INITIAL_COUNT: usize = TOTAL_COUNTS / NUM_SYMBOLS;

/// Number of distinct symbols (one per possible byte value).
const NUM_SYMBOLS: usize = 1 << 8;

// Guard that the pre-computed tables below match the constants above.
const_assert!(TOTAL_COUNTS == 4_096);
const_assert!(INITIAL_COUNT == 16);
const_assert!(NUM_SYMBOLS == 256);

/// Pre-computed Fenwick tree for a uniform distribution where every symbol has
/// a frequency of [`INITIAL_COUNT`].
///
/// Index 0 is unused (Fenwick trees are 1-based). Each internal node stores the
/// partial sum of a range determined by the lowest set bit of its index.
#[rustfmt::skip]
pub(crate) const UNIFORM_TREE: [i16; NUM_SYMBOLS + 1] = [
    0,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 512,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 1024,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 512,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 2048,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 512,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 1024,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 512,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 256,
    16, 32, 16, 64, 16, 32, 16, 128,
    16, 32, 16, 64, 16, 32, 16, 4096,
];

/// Flat frequency table for a uniform distribution.
const UNIFORM_FREQ: [u16; NUM_SYMBOLS] = [16u16; NUM_SYMBOLS];

/// Adaptive frequency model backed by a Fenwick tree.
///
/// Tracks the frequency of each byte value and maintains a Fenwick tree for
/// efficient cumulative-frequency queries. The total count is always
/// [`TOTAL_COUNTS`], so updates redistribute probability mass rather than
/// growing the table.
///
/// # Examples
///
/// ```text
/// let mut m = Model::new();
/// assert_eq!(m.freq(0x41), 16);   // uniform at start
/// m.update(0x41);                  // boost 'A'
/// assert_eq!(m.freq(0x41), 17);
/// ```
pub(crate) struct Model {
    /// Fenwick (binary indexed) tree over symbol frequencies.
    tree: [i16; NUM_SYMBOLS + 1],
    /// Raw frequency count for each symbol.
    freq: [u16; NUM_SYMBOLS],
    /// Round-robin cursor for victim selection during updates.
    cursor: u8,
}

impl Model {
    /// Creates a new model with a uniform frequency distribution.
    pub(crate) const fn new() -> Self {
        Self {
            tree: UNIFORM_TREE,
            freq: UNIFORM_FREQ,
            cursor: 0xFF,
        }
    }

    /// Resets the model to a uniform frequency distribution.
    pub(crate) const fn reset(&mut self) {
        *self = Self::new();
    }

    /// Returns the fixed total count across all symbols.
    pub(crate) const fn total() -> u64 {
        TOTAL_COUNTS as u64
    }

    /// Returns the current frequency of `byte`.
    pub(crate) const fn freq(&self, byte: u8) -> u64 {
        self.freq[byte as usize] as u64
    }

    /// Adapts the model by boosting `byte`'s frequency by 1.
    ///
    /// A round-robin victim symbol is decremented to keep the total constant.
    /// Symbols with a frequency of 1 are never chosen as victims, ensuring
    /// every symbol retains a non-zero probability.
    pub(crate) const fn update(&mut self, byte: u8) {
        // Cursor is the _LAST_ victim, bump it first.
        self.cursor = self.cursor.wrapping_add(1);

        // Find a victim that _CAN_ be decremented.
        while self.freq[self.cursor as usize] <= 1 {
            self.cursor = self.cursor.wrapping_add(1);
        }

        // Just return if this is going to be a no-op.
        if self.cursor == byte {
            return;
        }

        // Update raw frequencies.
        self.freq[self.cursor as usize] -= 1;
        self.freq[byte as usize] += 1;

        // Update the tree.
        self.update_inner(self.cursor as usize, -1);
        self.update_inner(byte as usize, 1);
    }

    /// Applies `delta` to the Fenwick tree at position `i`.
    const fn update_inner(&mut self, mut i: usize, delta: i16) {
        // Fenwick trees are 1-based for more efficient index math.
        i += 1;

        while i <= NUM_SYMBOLS {
            self.tree[i] = self.tree[i].wrapping_add(delta);
            i += i & i.wrapping_neg();
        }
    }

    /// Returns the cumulative frequency for all symbols `< byte`.
    ///
    /// This is the lower bound of `byte`'s interval in the CDF, used by the
    /// arithmetic encoder.
    pub(crate) const fn prefix_sum(&self, byte: u8) -> u64 {
        self.prefix_sum_inner(byte as usize).cast_unsigned() as u64
    }

    /// Fenwick prefix-sum query over the first `i` elements.
    const fn prefix_sum_inner(&self, mut i: usize) -> i16 {
        let mut sum = 0i16;

        while i > 0 {
            sum += self.tree[i];
            i -= i & i.wrapping_neg();
        }

        sum
    }

    /// Finds the symbol whose CDF interval contains `target`.
    ///
    /// Returns the largest `byte` such that `prefix_sum(byte) <= target`.
    /// Used by the arithmetic decoder to map a code point back to a symbol.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const fn find(&self, target: u64) -> u8 {
        let mut target = target.cast_signed() as i16;
        let mut bit = NUM_SYMBOLS >> 1;
        let mut i = 0;

        while bit > 0 {
            let next = i + bit;

            if self.tree[next] <= target {
                target -= self.tree[next];
                i = next;
            }

            bit >>= 1;
        }

        i as u8
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
impl fmt::Debug for Model {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model {{")?;
        for i in 0..NUM_SYMBOLS {
            let count = self.freq[i];
            // Bar length = floor(log2(count)) + 1, i.e. the bit-width of the count.
            let bar_len = u16::BITS - count.leading_zeros();
            let bar: String = "#".repeat(bar_len as usize);
            writeln!(f, "    freq[{i:03}]({count:04}) {bar}")?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    //use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    /// Verifies that the Fenwick tree and raw frequency table are in sync.
    fn in_sync(model: &Model) -> bool {
        let mut remaining = Model::total();

        for b in (0..=255u8).rev() {
            remaining -= model.freq(b);

            if remaining != model.prefix_sum(b) {
                return false;
            }
        }

        true
    }

    #[test]
    fn uniform_model() {
        let mut model = Model::new();

        for b in 0..=255u8 {
            model.update(b);
        }

        model.reset();
        model.freq.iter().for_each(|freq| assert_eq!(*freq as usize, INITIAL_COUNT));

        assert!(in_sync(&model));
    }

    proptest! {
        #[test]
        fn roundtrip(updates in prop::collection::vec(any::<u8>(), 0..8_192), byte: u8) {
            let mut model = Model::new();

            for &b in &updates {
                model.update(b);
            }

            prop_assert_eq!(model.find(model.prefix_sum(byte)), byte);
            prop_assert!(in_sync(&model));
        }
    }
}
