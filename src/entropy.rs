//! Adaptive arithmetic coding using u64 range coding with byte-oriented I/O.
//!
//! This module provides an adaptive frequency [`Model`] and a matched
//! [`Encoder`]/[`Decoder`] pair. The model tracks per-symbol frequencies with
//! a fixed total budget, and the coder narrows a u64 interval to produce or
//! consume compressed bytes.
//!
//! Models are **not** owned by the coder — they are passed by reference on each
//! call, allowing the caller to switch between multiple models (e.g. one per
//! context) while sharing a single coding stream.

use static_assertions::const_assert;

/// Total frequency budget across all symbols. Must be a power of two so that
/// `range / total` is a simple shift (though the compiler handles this).
const TOTAL: usize = 512;

/// Number of distinct symbols (one per byte value).
const SYMBOLS: usize = 256;

/// Starting count for every symbol: `TOTAL / SYMBOLS`.
const INITIAL_COUNT: usize = TOTAL / SYMBOLS;

const_assert!(TOTAL.is_power_of_two());
const_assert!(TOTAL == SYMBOLS * INITIAL_COUNT);

pub(crate) struct Encoder {}

impl Encoder {
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(crate) fn encode(&mut self, _model: &mut Model, _symbol: u8, _output: &mut Vec<u8>) {
        todo!()
    }

    #[allow(unused_mut)]
    pub(crate) fn finish(mut self, _output: &mut Vec<u8>) {
        todo!()
    }
}

pub(crate) struct Decoder {}

impl Decoder {
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(crate) fn decode(&mut self, _model: &mut Model, _input: &mut &[u8]) -> u8 {
        todo!()
    }
}

/// Adaptive frequency model for arithmetic coding.
///
/// Maintains per-symbol frequency counts that always sum to [`TOTAL`].
/// After each symbol is coded, [`update`](Model::update) transfers one count
/// from a victim symbol to the observed symbol, keeping the distribution
/// adaptive without ever rescaling the entire table.
pub(crate) struct Model {
    /// Frequency count for each of the 256 byte values.
    counts: [u8; SYMBOLS],
    /// Rotating victim pointer for the steal-one-count update strategy.
    cursor: u8,
}

impl Model {
    /// Creates a model with a uniform distribution (`INITIAL_COUNT` per symbol).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const fn new() -> Self {
        Self {
            counts: [INITIAL_COUNT as u8; SYMBOLS],
            cursor: 0,
        }
    }

    /// Adapts the model after observing `symbol`.
    ///
    /// Finds a victim symbol whose count is > 1, decrements it, and increments
    /// the observed symbol's count. This keeps the total fixed at [`TOTAL`].
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(crate) const fn update(&mut self, _symbol: u8) {
        todo!()
    }

    /// Returns the frequency count for `symbol`.
    pub(crate) const fn symbol_to_count(&self, _symbol: u8) -> u16 {
        todo!()
    }

    /// Returns the cumulative frequency for all symbols before `symbol`
    /// (i.e. the sum of counts for symbols `0..symbol`).
    pub(crate) fn symbol_to_cumulative(&self, _symbol: u8) -> u16 {
        todo!()
    }

    /// Returns the symbol whose cumulative frequency range contains `value`.
    ///
    /// This is the inverse of [`symbol_to_cumulative`](Model::symbol_to_cumulative):
    /// it finds the smallest symbol `s` such that the cumulative frequency up
    /// through `s` exceeds `value`.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn cumulative_to_symbol(&self, _value: u16) -> u8 {
        todo!()
    }
}
