//! Adaptive frequency models for arithmetic coding.
//!
//! Two model types are provided:
//!
//! - [`Model`] — 256-symbol model (one per byte value), backed by a Fenwick tree
//!   for O(log N) prefix-sum queries.
//! - [`TagModel`] — 4-symbol model for the token tag alphabet (LINE, EOH, EOS, EOO),
//!   using a flat array with linear scan.
//!
//! Both implement the [`FreqModel`] trait consumed by the arithmetic coder.
//!
//! Adaptation increments the observed symbol's frequency on each update.
//! When the total reaches [`RESCALE_AT`], all frequencies are halved (with a
//! floor of 1) to provide exponential decay of old observations while keeping
//! the total bounded for arithmetic precision.

#[cfg(test)]
use std::fmt;

use static_assertions::const_assert;

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Rescale threshold. When total reaches this value, all frequencies are halved.
const RESCALE_AT: u16 = 16_384;

// ---------------------------------------------------------------------------
// FreqModel trait
// ---------------------------------------------------------------------------

/// Trait for adaptive frequency models used by the arithmetic coder.
///
/// Provides CDF queries (`cum_freq`, `freq`, `total`, `find`) and adaptation
/// (`update`, `reset`). Implementations must maintain synchronized state so
/// that encoder and decoder produce identical model evolution.
pub(crate) trait FreqModel {
    /// Returns the cumulative frequency for all symbols `< symbol`.
    fn cum_freq(&self, symbol: u8) -> u32;

    /// Returns the frequency of `symbol`.
    fn freq(&self, symbol: u8) -> u32;

    /// Returns the total of all frequencies.
    fn total(&self) -> u32;

    /// Finds the symbol whose CDF interval contains `value`.
    ///
    /// Returns the largest `s` such that `cum_freq(s) <= value < cum_freq(s) + freq(s)`.
    fn find(&self, value: u32) -> u8;

    /// Adapts the model after encoding/decoding `symbol`.
    fn update(&mut self, symbol: u8);

    /// Resets the model to a uniform distribution.
    fn reset(&mut self);
}

// ===========================================================================
// Model — 256-symbol byte model (Fenwick tree)
// ===========================================================================

const_assert!(TOTAL_COUNTS == NUM_SYMBOLS * INITIAL_COUNT);

/// Initial total frequency count across all symbols.
const TOTAL_COUNTS: usize = 1 << 8;

/// Initial frequency assigned to every symbol in a fresh model.
const INITIAL_COUNT: usize = TOTAL_COUNTS / NUM_SYMBOLS;

/// Number of distinct symbols (one per possible byte value).
const NUM_SYMBOLS: usize = 1 << 8;

// Guard that the pre-computed tables below match the constants above.
const_assert!(TOTAL_COUNTS == 256);
const_assert!(INITIAL_COUNT == 1);
const_assert!(NUM_SYMBOLS == 256);
const_assert!(RESCALE_AT as usize >= TOTAL_COUNTS);

/// Pre-computed Fenwick tree for a uniform distribution where every symbol has
/// a frequency of [`INITIAL_COUNT`].
///
/// Index 0 is unused (Fenwick trees are 1-based). Each internal node stores the
/// partial sum of a range determined by the lowest set bit of its index.
#[rustfmt::skip]
pub(crate) const UNIFORM_TREE: [i16; NUM_SYMBOLS + 1] = [
    0,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 16,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 32,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 16,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 64,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 16,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 32,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 16,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 128,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 16,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 32,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 16,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 64,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 16,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 32,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 16,
    1, 2, 1, 4, 1, 2, 1, 8,
    1, 2, 1, 4, 1, 2, 1, 256,
];

/// Flat frequency table for a uniform distribution.
const UNIFORM_FREQ: [u16; NUM_SYMBOLS] = [1u16; NUM_SYMBOLS];

/// Adaptive frequency model for 256 byte symbols, backed by a Fenwick tree.
///
/// Tracks the frequency of each byte value and maintains a Fenwick tree for
/// efficient cumulative-frequency queries. The total grows by 1 on each update
/// and is halved (with floor 1) when it reaches [`RESCALE_AT`].
pub(crate) struct Model {
    /// Fenwick (binary indexed) tree over symbol frequencies.
    tree: [i16; NUM_SYMBOLS + 1],
    /// Raw frequency count for each symbol.
    freq: [u16; NUM_SYMBOLS],
    /// Current total of all frequencies.
    total: u16,
}

impl Model {
    /// Creates a new model with a uniform frequency distribution.
    pub(crate) const fn new() -> Self {
        Self {
            tree: UNIFORM_TREE,
            freq: UNIFORM_FREQ,
            total: 256,
        }
    }

    /// Halves all frequencies (floor 1) and rebuilds the Fenwick tree.
    fn rescale(&mut self) {
        let mut total = 0u16;

        for f in &mut self.freq {
            *f = (*f >> 1).max(1);
            total += *f;
        }

        self.total = total;
        self.rebuild_tree();
    }

    /// Reconstructs the Fenwick tree from the raw frequency table.
    #[allow(clippy::cast_possible_wrap)]
    fn rebuild_tree(&mut self) {
        self.tree[0] = 0;

        for i in 1..=NUM_SYMBOLS {
            self.tree[i] = self.freq[i - 1] as i16;
        }

        for i in 1..=NUM_SYMBOLS {
            let parent = i + (i & i.wrapping_neg());
            if parent <= NUM_SYMBOLS {
                self.tree[parent] += self.tree[i];
            }
        }
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

    /// Fenwick prefix-sum query over the first `i` elements.
    const fn prefix_sum_inner(&self, mut i: usize) -> i16 {
        let mut sum = 0i16;

        while i > 0 {
            sum += self.tree[i];
            i -= i & i.wrapping_neg();
        }

        sum
    }
}

impl FreqModel for Model {
    fn cum_freq(&self, symbol: u8) -> u32 {
        u32::from(self.prefix_sum_inner(symbol as usize).cast_unsigned())
    }

    fn freq(&self, symbol: u8) -> u32 {
        u32::from(self.freq[symbol as usize])
    }

    fn total(&self) -> u32 {
        u32::from(self.total)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn find(&self, value: u32) -> u8 {
        let mut target = value as i16;
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

    fn update(&mut self, symbol: u8) {
        self.freq[symbol as usize] += 1;
        self.update_inner(symbol as usize, 1);
        self.total += 1;

        if self.total >= RESCALE_AT {
            self.rescale();
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

// ===========================================================================
// TagModel — 4-symbol tag model (flat array, linear scan)
// ===========================================================================

/// Number of tag symbols.
const TAG_SYMBOLS: usize = 4;

/// Adaptive frequency model for the 4-symbol tag alphabet.
///
/// Uses a flat frequency array with linear-scan CDF queries. For 4 symbols,
/// a linear scan is faster than any tree structure due to branch prediction
/// and cache locality.
pub(crate) struct TagModel {
    freq: [u16; TAG_SYMBOLS],
    total: u16,
}

impl TagModel {
    /// Creates a new tag model with a uniform frequency distribution.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const fn new() -> Self {
        Self {
            freq: [1u16; TAG_SYMBOLS],
            total: TAG_SYMBOLS as u16,
        }
    }
}

impl FreqModel for TagModel {
    fn cum_freq(&self, symbol: u8) -> u32 {
        let mut sum = 0u32;
        for i in 0..symbol as usize {
            sum += u32::from(self.freq[i]);
        }
        sum
    }

    fn freq(&self, symbol: u8) -> u32 {
        u32::from(self.freq[symbol as usize])
    }

    fn total(&self) -> u32 {
        u32::from(self.total)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn find(&self, value: u32) -> u8 {
        let mut cum = 0u32;
        // Only check first N-1 symbols; the last symbol is the implicit remainder.
        for (i, &f) in self.freq[..TAG_SYMBOLS - 1].iter().enumerate() {
            cum += u32::from(f);
            if cum > value {
                return i as u8;
            }
        }
        (TAG_SYMBOLS - 1) as u8
    }

    fn update(&mut self, symbol: u8) {
        self.freq[symbol as usize] += 1;
        self.total += 1;

        if self.total >= RESCALE_AT {
            self.rescale();
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

impl TagModel {
    /// Halves all frequencies (floor 1) and recomputes total.
    fn rescale(&mut self) {
        let mut total = 0u16;
        for f in &mut self.freq {
            *f = (*f >> 1).max(1);
            total += *f;
        }
        self.total = total;
    }
}

// ===========================================================================
// Debug impl (test only)
// ===========================================================================

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
            writeln!(f, "    freq[{i:03}]({count:05}) {bar}")?;
        }
        write!(f, "}}")
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    // -----------------------------------------------------------------------
    // Model (256-symbol) tests
    // -----------------------------------------------------------------------

    /// Verifies that the Fenwick tree and raw frequency table are in sync.
    fn model_in_sync(model: &Model) -> bool {
        let mut remaining = model.total();

        for b in (0..=255u8).rev() {
            remaining -= model.freq(b);

            if remaining != model.cum_freq(b) {
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
        model.freq.iter().for_each(|freq| assert_eq!(usize::from(*freq), INITIAL_COUNT));

        assert!(model_in_sync(&model));
    }

    proptest! {
        #[test]
        fn model_roundtrip(updates in prop::collection::vec(any::<u8>(), 0..8_192), byte: u8) {
            let mut model = Model::new();

            for &b in &updates {
                model.update(b);
            }

            prop_assert_eq!(model.find(model.cum_freq(byte)), byte);
            prop_assert!(model_in_sync(&model));
        }
    }

    // -----------------------------------------------------------------------
    // TagModel (4-symbol) tests
    // -----------------------------------------------------------------------

    #[allow(clippy::cast_possible_truncation)]
    fn tag_in_sync(model: &TagModel) -> bool {
        let mut remaining = model.total();

        for s in (0..TAG_SYMBOLS as u8).rev() {
            remaining -= model.freq(s);
            if remaining != model.cum_freq(s) {
                return false;
            }
        }

        true
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn tag_uniform() {
        let model = TagModel::new();
        assert_eq!(model.total(), TAG_SYMBOLS as u32);
        for s in 0..TAG_SYMBOLS as u8 {
            assert_eq!(model.freq(s), 1);
            assert_eq!(model.cum_freq(s), u32::from(s));
        }
        assert!(tag_in_sync(&model));
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn tag_reset() {
        let mut model = TagModel::new();
        for _ in 0..100 {
            model.update(0);
        }
        model.reset();
        assert_eq!(model.total(), TAG_SYMBOLS as u32);
        for s in 0..TAG_SYMBOLS as u8 {
            assert_eq!(model.freq(s), 1);
        }
    }

    proptest! {
        #[test]
        fn tag_roundtrip(updates in prop::collection::vec(0u8..4, 0..8_192), symbol in 0u8..4) {
            let mut model = TagModel::new();

            for &s in &updates {
                model.update(s);
            }

            prop_assert_eq!(model.find(model.cum_freq(symbol)), symbol);
            prop_assert!(tag_in_sync(&model));
        }

        #[test]
        fn tag_find_covers_range(updates in prop::collection::vec(0u8..4, 0..1_000)) {
            let mut model = TagModel::new();
            for &s in &updates {
                model.update(s);
            }

            // Every value in 0..total should map to a valid symbol.
            for v in 0..model.total() {
                let s = model.find(v);
                #[allow(clippy::cast_possible_truncation)]
                { prop_assert!(s < TAG_SYMBOLS as u8, "find({v}) returned {s}"); }
                let lo = model.cum_freq(s);
                let hi = lo + model.freq(s);
                prop_assert!(v >= lo && v < hi, "find({v})={s}, but cum_freq={lo}, hi={hi}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Generic FreqModel tests (run on both types)
    // -----------------------------------------------------------------------

    #[allow(clippy::cast_possible_truncation)]
    fn verify_freq_model<M: FreqModel>(model: &mut M, num_symbols: u16) {
        assert_eq!(model.total(), u32::from(num_symbols));

        // Update each symbol once.
        for s in 0..num_symbols {
            model.update(s as u8);
        }
        assert_eq!(model.total(), 2 * u32::from(num_symbols));

        // find(cum_freq(s)) == s for each symbol.
        for s in 0..num_symbols {
            let s8 = s as u8;
            assert_eq!(model.find(model.cum_freq(s8)), s8);
        }

        // Reset brings it back.
        model.reset();
        assert_eq!(model.total(), u32::from(num_symbols));
    }

    #[test]
    fn generic_model_256() {
        verify_freq_model(&mut Model::new(), 256);
    }

    #[test]
    fn generic_tag_model() {
        verify_freq_model(&mut TagModel::new(), 4);
    }

    proptest! {
        #[test]
        fn tag_model_rescale(
            symbols in prop::collection::vec(0u8..4, 16_384..20_000),
        ) {
            let mut model = TagModel::new();
            for &s in &symbols {
                model.update(s);
            }
            // After 16384+ updates, rescale must have fired. Verify consistency.
            prop_assert!(model.total() < u32::from(RESCALE_AT), "total {} should be < {RESCALE_AT}", model.total());
            #[allow(clippy::cast_possible_truncation)]
            for s in 0..TAG_SYMBOLS as u8 {
                prop_assert!(model.freq(s) >= 1, "freq({s}) must be >= 1 after rescale");
                prop_assert_eq!(model.find(model.cum_freq(s)), s);
            }
        }

        #[test]
        fn model_rescale(
            symbols in prop::collection::vec(any::<u8>(), 16_384..20_000),
        ) {
            let mut model = Model::new();
            for &s in &symbols {
                model.update(s);
            }
            // After 16384+ updates, rescale must have fired. Verify consistency.
            for s in 0..=255u8 {
                prop_assert_eq!(model.find(model.cum_freq(s)), s);
            }
        }
    }
}
