//! Bit-prediction models and the shared [`Context`] they read.
//!
//! Each byte is coded as an 8-bit MSB-first bit-tree. A [`TokenModel`] is queried
//! once per bit ([`TokenModel::predict`]) and then told the actual bit
//! ([`TokenModel::update`]). Predictions are returned in the stretched (logit)
//! domain so the [`crate::mixer::Mixer`] can combine them directly.

pub(crate) mod null;
pub(crate) mod order0;
pub(crate) mod statemap;

/// Bits per coded symbol. The entropy coder operates on bytes, so a symbol is a
/// depth-8 bit-tree.
pub(crate) const SYMBOL_BITS: u32 = u8::BITS;

/// Minimum / maximum index width for a hashed model table, bounding skeleton
/// memory regardless of input size.
const MIN_HASH_BITS: u32 = 16;
const MAX_HASH_BITS: u32 = 22;

/// Index width (in bits) for a hashed table sized to roughly `capacity` distinct
/// contexts, clamped to `[MIN_HASH_BITS, MAX_HASH_BITS]`.
///
/// Both the encoder and decoder derive this from the same framed token count, so
/// their hashed tables are identically sized and their predictions stay in lock
/// step — the determinism the codec depends on. Computed via `leading_zeros`
/// rather than `next_power_of_two`, which would overflow (panic) near `usize::MAX`
/// — and the token count reaching here comes from untrusted input.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "Table-sizing helper for the hashed models to come (used by the removed order-1).")
)]
pub(crate) fn hashed_bits(capacity: usize) -> u32 {
    (usize::BITS - capacity.max(1).leading_zeros()).clamp(MIN_HASH_BITS, MAX_HASH_BITS)
}

/// Mutable per-stream prediction state shared by every model.
///
/// `c0` walks the bit-tree of the current byte; `s1` is the previous finalized
/// byte (the order-1 key).
#[derive(Debug)]
pub(crate) struct Context {
    /// Partial current byte: a leading-1 sentinel followed by the bits coded so
    /// far. Kept as `u32` (headroom to spare) — it transiently reaches `2^9 - 1`
    /// after the 8th bit (sentinel at bit 8) before [`Context::push_symbol`] strips it.
    pub(crate) c0: u32,
    /// Bits of the current byte already coded (`0..=7`).
    pub(crate) bpos: u8,
    /// The previous finalized byte's value (`0` before any) — order-1 key.
    pub(crate) s1: u8,
}

impl Context {
    /// A fresh context at the start of the first symbol.
    pub(crate) const fn new() -> Self {
        Self {
            c0: 1,
            bpos: 0,
            s1: 0,
        }
    }

    /// Append one freshly-coded bit to the partial current symbol.
    pub(crate) fn push_bit(&mut self, bit: u8) {
        self.c0 = (self.c0 << 1) | u32::from(bit);
        self.bpos += 1;
    }

    /// Finalize the current byte once all [`SYMBOL_BITS`] bits are in, recording it
    /// as `s1` and resetting for the next. The `& 0xff` strips the leading-1
    /// sentinel (at bit 8), leaving the 8-bit byte value.
    pub(crate) const fn push_symbol(&mut self) {
        self.s1 = (self.c0 & 0xff) as u8;
        self.c0 = 1;
        self.bpos = 0;
    }
}

/// A model that predicts the next bit from the [`Context`].
pub(crate) trait TokenModel {
    /// Predict P(next bit == 1) in the stretched (logit) domain, roughly
    /// `[-2047, 2047]`. Called before the bit is known.
    fn predict(&mut self, ctx: &Context) -> i32;

    /// Observe the actual `bit`. `ctx` still reflects the pre-bit state.
    fn update(&mut self, ctx: &Context, bit: u8);
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// Feeding a byte's 8 bits MSB-first, then finalizing, must recover the byte
        /// as `s1` — the round-trip the whole codec relies on. The full `0..=255`
        /// range catches a wrong sentinel mask in [`Context::push_symbol`].
        #[test]
        fn push_bits_then_symbol_recovers_value(raw in 0u8..=0xFF) {
            let mut ctx = Context::new();
            for k in (0..SYMBOL_BITS).rev() {
                ctx.push_bit((raw >> k) & 1);
            }
            prop_assert_eq!(u32::from(ctx.bpos), SYMBOL_BITS);
            ctx.push_symbol();
            prop_assert_eq!(ctx.s1, raw);
            prop_assert_eq!(ctx.c0, 1);
            prop_assert_eq!(ctx.bpos, 0);
        }
    }

    #[test]
    fn hashed_bits_is_clamped_and_deterministic() {
        assert_eq!(hashed_bits(0), MIN_HASH_BITS);
        assert_eq!(hashed_bits(1), MIN_HASH_BITS);
        assert_eq!(hashed_bits(usize::MAX), MAX_HASH_BITS);
        // A pure function of capacity: encode and decode must agree.
        assert_eq!(hashed_bits(1_000_000), hashed_bits(1_000_000));
    }
}
