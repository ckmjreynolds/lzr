//! Bit-prediction models and the shared [`Context`] they read.
//!
//! Each byte is coded as an 8-bit MSB-first bit-tree. A [`Model`] is queried
//! once per bit ([`Model::predict`]) and then told the actual bit
//! ([`Model::update`]). Predictions are returned in the stretched (logit)
//! domain so the [`crate::mixer::Mixer`] can combine them directly.
//!
//! The models are **byte-level**: the only per-stream state on [`Context`] is the partial current
//! byte (`c0`/`bpos`). Higher-order context — the previous bytes — is read from the `hist` window the
//! entropy driver borrows from its output buffer, so no byte-history ring is kept here.

pub(crate) mod iddelta;
pub(crate) mod match_model;
pub(crate) mod nnlm;
pub(crate) mod null;
pub(crate) mod ordern;
pub(crate) mod run;
pub(crate) mod sparse;
pub(crate) mod statemap;
pub(crate) mod xmltag;

/// Bits per coded symbol. The entropy coder operates on bytes, so a symbol is a
/// depth-8 bit-tree.
pub(crate) const SYMBOL_BITS: u32 = u8::BITS;

/// Minimum / maximum index width for a hashed model table, bounding skeleton
/// memory regardless of input size.
const MIN_HASH_BITS: u32 = 16;
const MAX_HASH_BITS: u32 = 28;

/// Index width (in bits) for a hashed table sized to roughly `capacity` distinct
/// contexts, clamped to `[MIN_HASH_BITS, MAX_HASH_BITS]`.
///
/// Both the encoder and decoder derive this from the same framed token count, so
/// their hashed tables are identically sized and their predictions stay in lock
/// step — the determinism the codec depends on. Computed via `leading_zeros`
/// rather than `next_power_of_two`, which would overflow (panic) near `usize::MAX`
/// — and the token count reaching here comes from untrusted input.
pub(crate) fn hashed_bits(capacity: usize) -> u32 {
    (usize::BITS - capacity.max(1).leading_zeros()).clamp(MIN_HASH_BITS, MAX_HASH_BITS)
}

/// Golden-ratio multiplier for the byte-context multiplicative hash, shared by the order-N and
/// sparse models via [`byte_hash`] so a change to the mixing function cannot silently desync them.
pub(crate) const HASH_MULT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Fold a set of previous byte values into a multiplicative context hash seeded from `node` (the
/// in-progress bit-tree node `c0`). A wrapping multiply-accumulate, so it never overflows regardless
/// of how many bytes are folded in. Shared by [`ordern::OrderN`] and [`sparse::SparseModel`], whose
/// only difference is which byte positions they pass here.
pub(crate) fn byte_hash(node: u64, bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h = node.wrapping_mul(HASH_MULT);
    for byte in bytes {
        h = (h ^ u64::from(byte)).wrapping_mul(HASH_MULT);
    }
    h
}

/// Mutable per-stream prediction state shared by every model: the partial current byte.
///
/// `c0` walks the bit-tree of the current byte (a leading-1 sentinel followed by the bits coded so
/// far), and `bpos` counts how many of its bits are in. That is the whole of the shared state — the
/// models are byte-level and read their higher-order context (previous bytes) from the `hist` window
/// the entropy driver passes, so nothing else needs to live here.
#[derive(Debug)]
pub(crate) struct Context {
    /// Partial current byte: a leading-1 sentinel followed by the bits coded so
    /// far. Kept as `u32` (headroom to spare) — it transiently reaches `2^9 - 1`
    /// after the 8th bit (sentinel at bit 8) before [`Context::push_symbol`] strips it.
    pub(crate) c0: u32,
    /// Bits of the current byte already coded (`0..=7`).
    pub(crate) bpos: u8,
}

impl Context {
    /// A fresh context at the start of the first symbol.
    pub(crate) const fn new() -> Self {
        Self {
            c0: 1,
            bpos: 0,
        }
    }

    /// The bit-tree node for the current bit: the leading-1 sentinel plus the bits coded so far. It is
    /// `< 256` during any prediction (`bpos <= 7`), so it indexes a dense 256-slot order-0 table
    /// directly and seeds the hashed models' [`byte_hash`].
    pub(crate) const fn node(&self) -> u64 {
        self.c0 as u64
    }

    /// Append one freshly-coded bit to the partial current symbol.
    pub(crate) fn push_bit(&mut self, bit: u8) {
        self.c0 = (self.c0 << 1) | u32::from(bit);
        self.bpos += 1;
    }

    /// Close the current byte once all [`SYMBOL_BITS`] bits are in: reset for the next symbol. The
    /// finalized byte is already in the driver's output buffer (the models' `hist` window), so nothing
    /// else needs recording here.
    pub(crate) const fn push_symbol(&mut self) {
        self.c0 = 1;
        self.bpos = 0;
    }
}

/// A model that predicts the next bit from the [`Context`] and a window of recent bytes.
pub(crate) trait Model {
    /// Predict P(next bit == 1) in the stretched (logit) domain, roughly
    /// `[-2047, 2047]`. Called before the bit is known. `hist` is the finalized
    /// bytes preceding the current byte, newest last (`hist[hist.len() - 1]` is the
    /// previous byte); it is a borrowed window (bounded length), not owned.
    fn predict(&mut self, ctx: &Context, hist: &[u8]) -> i32;

    /// Observe the actual `bit`. `ctx` and `hist` still reflect the pre-bit state.
    fn update(&mut self, ctx: &Context, hist: &[u8], bit: u8);
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// Feeding a byte's 8 bits MSB-first assembles it in `c0` (a leading-1 sentinel
        /// above the byte value); finalizing then resets the context for the next symbol.
        /// The full `0..=255` range catches a wrong sentinel mask in [`Context::push_symbol`].
        #[test]
        fn push_bits_then_symbol_recovers_value(raw in 0u8..=0xFF) {
            let mut ctx = Context::new();
            for k in (0..SYMBOL_BITS).rev() {
                ctx.push_bit((raw >> k) & 1);
            }
            prop_assert_eq!(u32::from(ctx.bpos), SYMBOL_BITS);
            // c0 holds the sentinel (bit 8) plus the assembled byte value.
            prop_assert_eq!(ctx.c0 & 0xff, u32::from(raw));
            prop_assert_eq!(ctx.c0 >> SYMBOL_BITS, 1);
            ctx.push_symbol();
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
