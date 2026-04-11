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
const TOTAL: usize = 4_096;

/// Number of distinct symbols (one per byte value).
const SYMBOLS: usize = 256;

/// Starting count for every symbol: `TOTAL / SYMBOLS`.
const INITIAL_COUNT: usize = TOTAL / SYMBOLS;

const_assert!(TOTAL.is_power_of_two());
const_assert!(TOTAL == SYMBOLS * INITIAL_COUNT);

/// Renormalization threshold. When `range` drops below this value the encoder
/// shifts out (or the decoder shifts in) one byte and restores precision.
/// Chosen so that `BOTTOM / TOTAL = 2^36`, giving ample precision per symbol.
const BOTTOM: u64 = 1 << 48;

/// Adaptive arithmetic encoder using u64 range coding with byte-oriented output.
///
/// Uses a cache/pending mechanism to handle carry propagation without
/// backtracking through the output buffer. Models are passed per-call,
/// allowing the caller to switch between multiple models.
#[derive(Debug)]
#[allow(missing_copy_implementations)]
pub(crate) struct Encoder {
    low: u64,
    range: u64,
    /// Last extracted byte that has not yet been emitted (may be incremented
    /// by a future carry).
    cache: u8,
    /// Number of consecutive `0xFF` bytes waiting behind `cache`. On carry
    /// these become `0x00`; otherwise they are emitted as-is.
    pending: u32,
    /// `true` until the first byte has been extracted into `cache`.
    first_byte: bool,
}

impl Encoder {
    /// Creates a new encoder with full-range initial state.
    pub(crate) const fn new() -> Self {
        Self {
            low: 0,
            range: u64::MAX,
            cache: 0,
            pending: 0,
            first_byte: true,
        }
    }

    /// Encodes a single symbol using the given model, appending compressed
    /// bytes to `output`.
    ///
    /// The model is updated after encoding so that encoder and decoder stay
    /// in sync.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn encode(&mut self, symbol: u8, model: &mut Model, output: &mut Vec<u8>) {
        let total = u64::from(Model::total());
        let cum = u64::from(model.symbol_to_cumulative(symbol));
        let freq = u64::from(model.count(symbol));

        let r = self.range / total;
        let old_low = self.low;
        self.low = self.low.wrapping_add(cum * r);
        self.range = freq * r;

        if self.low < old_low {
            self.propagate_carry(output);
        }

        model.update(symbol);
        self.shift_low(output);
    }

    /// Flushes all remaining encoder state to `output`. Consumes the encoder.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn finish(mut self, output: &mut Vec<u8>) {
        // Shift out all 8 bytes of `low` through the carry-safe mechanism.
        for _ in 0..8 {
            let byte = (self.low >> 56) as u8;
            if self.first_byte {
                self.cache = byte;
                self.first_byte = false;
            } else if byte == 0xFF {
                self.pending += 1;
            } else {
                output.push(self.cache);
                Self::emit_pending(output, 0xFF, self.pending);
                self.cache = byte;
                self.pending = 0;
            }
            self.low <<= 8;
        }
        // Flush final cache and pending.
        if !self.first_byte {
            output.push(self.cache);
            Self::emit_pending(output, 0xFF, self.pending);
        }
    }

    /// A carry occurred — increment the cached byte and flush pending as `0x00`.
    fn propagate_carry(&mut self, output: &mut Vec<u8>) {
        if !self.first_byte {
            output.push(self.cache.wrapping_add(1));
            Self::emit_pending(output, 0x00, self.pending);
        }
        self.pending = 0;
        self.first_byte = true;
    }

    /// Renormalize: while `range < BOTTOM`, extract the top byte of `low`
    /// into the cache/pending pipeline and shift both `low` and `range` left
    /// by 8 bits.
    #[allow(clippy::cast_possible_truncation)]
    fn shift_low(&mut self, output: &mut Vec<u8>) {
        while self.range < BOTTOM {
            let byte = (self.low >> 56) as u8;
            if self.first_byte {
                self.cache = byte;
                self.first_byte = false;
            } else if byte == 0xFF {
                self.pending += 1;
            } else {
                output.push(self.cache);
                Self::emit_pending(output, 0xFF, self.pending);
                self.cache = byte;
                self.pending = 0;
            }
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    /// Pushes `count` copies of `byte` to `output`.
    fn emit_pending(output: &mut Vec<u8>, byte: u8, count: u32) {
        for _ in 0..count {
            output.push(byte);
        }
    }
}

/// Adaptive arithmetic decoder using u64 range coding with byte-oriented input.
///
/// The decoder reads 8 initial bytes to seed the code register, then consumes
/// additional bytes during renormalization. Models are passed per-call,
/// allowing the caller to switch between multiple models.
#[derive(Debug)]
#[allow(missing_copy_implementations)]
pub(crate) struct Decoder {
    low: u64,
    range: u64,
    /// Current code value read from the compressed stream. Always lies within
    /// the interval `[low, low + range)` (modulo wrapping).
    code: u64,
}

impl Decoder {
    /// Creates a new decoder, reading 8 bytes from `input` to seed the code
    /// register.
    ///
    /// The initial read is required because `range` starts at `u64::MAX`
    /// (above [`BOTTOM`]), so the first [`decode`](Decoder::decode) call would
    /// not trigger renormalization to pull bytes in.
    pub(crate) fn new(input: &mut &[u8]) -> Self {
        let mut code: u64 = 0;
        for _ in 0..8 {
            code = (code << 8) | u64::from(Self::read_byte(input));
        }
        Self {
            low: 0,
            range: u64::MAX,
            code,
        }
    }

    /// Decodes a single symbol using the given model, consuming bytes from
    /// `input` as needed during renormalization.
    ///
    /// The model is updated after decoding so that encoder and decoder stay
    /// in sync.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn decode(&mut self, model: &mut Model, input: &mut &[u8]) -> u8 {
        let total = u64::from(Model::total());
        let r = self.range / total;

        // Determine which symbol the code falls in. The `.min(total - 1)`
        // clamp handles the rounding remainder from `range / total`.
        let offset = (self.code.wrapping_sub(self.low) / r).min(total - 1) as u16;
        let symbol = model.cumulative_to_symbol(offset);

        let cum = u64::from(model.symbol_to_cumulative(symbol));
        let freq = u64::from(model.count(symbol));

        self.low = self.low.wrapping_add(cum * r);
        self.range = freq * r;

        model.update(symbol);

        // Renormalize: shift in bytes while range is below threshold.
        while self.range < BOTTOM {
            self.code = (self.code << 8) | u64::from(Self::read_byte(input));
            self.low <<= 8;
            self.range <<= 8;
        }

        symbol
    }

    /// Reads one byte from `input`, advancing the slice. Returns `0` if the
    /// input is exhausted (trailing padding).
    fn read_byte(input: &mut &[u8]) -> u8 {
        if input.is_empty() {
            0
        } else {
            let byte = input[0];
            *input = &input[1..];
            byte
        }
    }
}

/// Adaptive frequency model for arithmetic coding.
///
/// Maintains per-symbol frequency counts that always sum to [`TOTAL`] (4 096).
/// After each symbol is coded, [`update`](Model::update) transfers one count
/// from a victim symbol to the observed symbol, keeping the distribution
/// adaptive without ever rescaling the entire table.
pub(crate) struct Model {
    /// Frequency count for each of the 256 byte values.
    counts: [u16; SYMBOLS],
    /// Rotating victim pointer for the steal-one-count update strategy.
    cursor: u8,
}

impl Model {
    /// Creates a model with a uniform distribution (`INITIAL_COUNT` per symbol).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const fn new() -> Self {
        Self {
            counts: [INITIAL_COUNT as u16; SYMBOLS],
            cursor: 0,
        }
    }

    /// Adapts the model after observing `symbol`.
    ///
    /// Finds a victim symbol whose count is > 1, decrements it, and increments
    /// the observed symbol's count. This keeps the total fixed at [`TOTAL`].
    pub(crate) const fn update(&mut self, symbol: u8) {
        // Find a victim and decrement their count.
        if self.counts[self.cursor as usize] <= 1 || self.cursor == symbol {
            loop {
                self.cursor = self.cursor.wrapping_add(1);

                if self.counts[self.cursor as usize] > 1 {
                    break;
                }
            }
        }

        self.counts[self.cursor as usize] -= 1;

        // Now, we know that the increment here cannot overflow.
        self.counts[symbol as usize] += 1;
    }

    /// Returns the frequency count for `symbol`.
    pub(crate) const fn count(&self, symbol: u8) -> u16 {
        self.counts[symbol as usize]
    }

    /// Returns the cumulative frequency for all symbols before `symbol`
    /// (i.e. the sum of counts for symbols `0..symbol`).
    pub(crate) fn symbol_to_cumulative(&self, symbol: u8) -> u16 {
        let mut cumulative = 0u16;

        for i in 0..symbol {
            cumulative += self.counts[i as usize];
        }

        cumulative
    }

    /// Returns the symbol whose cumulative frequency range contains `value`.
    ///
    /// This is the inverse of [`symbol_to_cumulative`](Model::symbol_to_cumulative):
    /// it finds the smallest symbol `s` such that the cumulative frequency up
    /// through `s` exceeds `value`.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn cumulative_to_symbol(&self, value: u16) -> u8 {
        let mut cumulative = 0u16;

        for i in 0..SYMBOLS {
            cumulative += self.counts[i];
            if cumulative > value {
                return i as u8;
            }
        }

        unreachable!()
    }

    /// Returns the fixed total frequency budget ([`TOTAL`]).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) const fn total() -> u16 {
        TOTAL as u16
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    /// Encodes `data`, then decodes it. Returns `(encoded_len, decoded_data)`.
    fn roundtrip(data: &[u8]) -> (usize, Vec<u8>) {
        let mut buf = Vec::new();
        let mut model = Model::new();
        let mut enc = Encoder::new();
        for &b in data {
            enc.encode(b, &mut model, &mut buf);
        }
        enc.finish(&mut buf);
        let encoded_len = buf.len();

        let mut input = buf.as_slice();
        let mut dec = Decoder::new(&mut input);
        let mut model = Model::new();
        let mut out = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            out.push(dec.decode(&mut model, &mut input));
        }
        (encoded_len, out)
    }

    proptest! {
        #[test]
        fn roundtrip_lipsum(len in 1..10_000usize) {
            let data = crate::_bench::lipsum_bytes(len);
            let (encoded_len, decoded) = roundtrip(&data);
            prop_assert_eq!(&decoded, &data);
            // The model starts uniform and adapts slowly (1 count per symbol),
            // so short inputs don't compress. 500+ bytes of text reliably do.
            if len >= 500 {
                prop_assert!(encoded_len < data.len(), "expected compression, got {encoded_len} >= {}", data.len());
            }
        }
    }

    proptest! {
        #[test]
        fn roundtrip_arbitrary(data in prop::collection::vec(any::<u8>(), 0..10_000)) {
            let (_, decoded) = roundtrip(&data);
            prop_assert_eq!(decoded, data);
        }
    }

    #[test]
    fn empty() {
        let (_, decoded) = roundtrip(&[]);
        assert_eq!(decoded, Vec::<u8>::new());
    }

    #[test]
    fn single_byte() {
        let (_, decoded) = roundtrip(&[42]);
        assert_eq!(decoded, vec![42]);
    }

    #[test]
    fn all_zeros() {
        let data = vec![0u8; 4096];
        let (encoded_len, decoded) = roundtrip(&data);
        assert_eq!(decoded, data);
        assert!(encoded_len < data.len(), "expected compression, got {encoded_len} >= {}", data.len());
    }

    #[test]
    fn all_0xff() {
        let data = vec![0xFFu8; 4096];
        let (encoded_len, decoded) = roundtrip(&data);
        assert_eq!(decoded, data);
        assert!(encoded_len < data.len(), "expected compression, got {encoded_len} >= {}", data.len());
    }
}
