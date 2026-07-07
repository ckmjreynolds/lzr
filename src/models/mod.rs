//! Bit-prediction models and the shared [`Context`] they read.
//!
//! Each byte is coded as an 8-bit MSB-first bit-tree. A [`TokenModel`] is queried
//! once per bit ([`TokenModel::predict`]) and then told the actual bit
//! ([`TokenModel::update`]). Predictions are returned in the stretched (logit)
//! domain so the [`crate::mixer::Mixer`] can combine them directly.

pub(crate) mod match_model;
pub(crate) mod null;
pub(crate) mod ordern;
pub(crate) mod sparse;
pub(crate) mod statemap;
pub(crate) mod varint;

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

/// Number of previous complete u22 token values retained for token-context models. Indexed `1..`
/// (newest at `[1]`); `[0]` is unused. Bounds how far back order-N / sparse token contexts can see.
pub(crate) const TOKEN_HISTORY: usize = 8;

/// Golden-ratio multiplier for the token-context multiplicative hash, shared by the order-N and
/// sparse models via [`token_hash`] so a change to the mixing function cannot silently desync them.
pub(crate) const HASH_MULT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Fold a set of previous token values into a multiplicative context hash seeded from `base` (the
/// in-progress token, see [`Context::hash_base`]). A wrapping multiply-accumulate, so it never
/// overflows regardless of how many tokens are folded in. Shared by [`ordern::OrderN`] and
/// [`sparse::SparseModel`], whose only difference is which token positions they pass here.
pub(crate) fn token_hash(base: u64, toks: impl Iterator<Item = u32>) -> u64 {
    let mut h = base.wrapping_mul(HASH_MULT);
    for tok in toks {
        h = (h ^ u64::from(tok)).wrapping_mul(HASH_MULT);
    }
    h
}

/// Mutable per-stream prediction state shared by every model: the partial current byte plus the
/// shared u22-token parser.
///
/// `c0` walks the bit-tree of the current byte. The context also carries the running u22-varint parse
/// of the stream — the in-progress token's accumulated payload and the last [`TOKEN_HISTORY`] complete
/// token values — advanced once per finalized byte in [`Context::push_symbol`]. Holding one parser here
/// (rather than one per model) keeps every token-context model in lock-step, and is deterministic on
/// both encode and decode since both advance it from the same coded byte sequence.
#[derive(Debug)]
pub(crate) struct Context {
    /// Partial current byte: a leading-1 sentinel followed by the bits coded so
    /// far. Kept as `u32` (headroom to spare) — it transiently reaches `2^9 - 1`
    /// after the 8th bit (sentinel at bit 8) before [`Context::push_symbol`] strips it.
    pub(crate) c0: u32,
    /// Bits of the current byte already coded (`0..=7`).
    pub(crate) bpos: u8,
    /// Payload bits accumulated so far in the in-progress u22 token (`0`, 7, or 14 bits).
    pub(crate) cur_payload: u32,
    /// Finalized bytes of the in-progress token so far, `∈ {0, 1, 2}`.
    pub(crate) cur_bytes: u8,
    /// The last complete u22 token values, newest at `[1]`; `[0]` unused. Zero-initialized, so
    /// contexts before enough tokens exist read as `0` identically on both sides.
    pub(crate) prev_tokens: [u32; TOKEN_HISTORY],
}

impl Context {
    /// A fresh context at the start of the first symbol.
    pub(crate) const fn new() -> Self {
        Self {
            c0: 1,
            bpos: 0,
            cur_payload: 0,
            cur_bytes: 0,
            prev_tokens: [0; TOKEN_HISTORY],
        }
    }

    /// The multiplicative-hash seed shared by the token-context models: the in-progress token's
    /// accumulated payload shifted above the bit-tree node `c0`. `cur_payload <= 0x3FFF` and
    /// `c0 & 0xff <= 0xFF`, so this is always `<= 2^22 - 1` — order 0 can index its dense `2^22` table
    /// with it directly, and the hashed models fold their previous tokens into it via [`token_hash`].
    pub(crate) fn hash_base(&self) -> u64 {
        (u64::from(self.cur_payload) << SYMBOL_BITS) | u64::from(self.c0 & 0xff)
    }

    /// Append one freshly-coded bit to the partial current symbol.
    pub(crate) fn push_bit(&mut self, bit: u8) {
        self.c0 = (self.c0 << 1) | u32::from(bit);
        self.bpos += 1;
    }

    /// Close the current byte once all [`SYMBOL_BITS`] bits are in: advance the u22-token parser by the
    /// finalized byte (`c0 & 0xff`, since `c0 == (1 << 8) | byte` here), then reset for the next symbol.
    /// The advance happens *after* the byte's predictions/updates, so models saw the pre-byte framing.
    pub(crate) fn push_symbol(&mut self) {
        self.advance_token((self.c0 & 0xff) as u8);
        self.c0 = 1;
        self.bpos = 0;
    }

    /// Advance the incremental u22 parser by one finalized byte, mirroring [`crate::uleb128::decode_u22`]
    /// (7 + 7 + 8 layout; the third byte always terminates, carrying all 8 bits).
    fn advance_token(&mut self, b: u8) {
        match self.cur_bytes {
            0 => {
                if b & 0x80 == 0 {
                    self.complete_token(u32::from(b));
                } else {
                    self.cur_payload = u32::from(b & 0x7f);
                    self.cur_bytes = 1;
                }
            }
            1 => {
                if b & 0x80 == 0 {
                    self.complete_token(self.cur_payload | (u32::from(b) << 7));
                } else {
                    self.cur_payload |= u32::from(b & 0x7f) << 7;
                    self.cur_bytes = 2;
                }
            }
            // `cur_bytes == 2`: the third byte terminates, carrying all 8 bits.
            _ => self.complete_token(self.cur_payload | (u32::from(b) << 14)),
        }
    }

    /// Record a completed token: shift it into `prev_tokens[1]` (newest first) and reset the
    /// in-progress token. The shift runs regardless of any model's order.
    fn complete_token(&mut self, value: u32) {
        self.prev_tokens.copy_within(1..TOKEN_HISTORY - 1, 2);
        self.prev_tokens[1] = value;
        self.cur_payload = 0;
        self.cur_bytes = 0;
    }
}

/// A model that predicts the next bit from the [`Context`] and a window of recent bytes.
pub(crate) trait TokenModel {
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

    /// Feed a u22 token's encoded bytes through the context (one full 8-bit symbol per byte) and
    /// assert its value lands in `prev_tokens[1]`, for the 1-, 2-, and 3-byte forms; the in-progress
    /// token resets after each. Exercises the shared parser advanced in `push_symbol`.
    #[test]
    fn token_parser_recovers_values() {
        use arbitrary_int::u22;

        use crate::uleb128::encode_u22;
        for raw in [0u32, 1, 0x7f, 0x80, 0x3fff, 0x4000, 0x3f_ffff] {
            let mut ctx = Context::new();
            let mut buf = Vec::new();
            encode_u22(u22::new(raw), &mut buf);
            for b in buf {
                for k in (0..SYMBOL_BITS).rev() {
                    ctx.push_bit((b >> k) & 1);
                }
                ctx.push_symbol();
            }
            assert_eq!(ctx.prev_tokens[1], raw, "token {raw:#x}");
            assert_eq!(ctx.cur_bytes, 0, "in-progress reset for {raw:#x}");
            assert_eq!(ctx.cur_payload, 0, "payload reset for {raw:#x}");
        }
    }

    /// Successive complete tokens shift through the ring, newest at `[1]`.
    #[test]
    fn token_parser_ring_shifts_newest_first() {
        use arbitrary_int::u22;

        use crate::uleb128::encode_u22;
        let mut ctx = Context::new();
        for raw in [0x11u32, 0x22, 0x33] {
            let mut buf = Vec::new();
            encode_u22(u22::new(raw), &mut buf);
            for b in buf {
                for k in (0..SYMBOL_BITS).rev() {
                    ctx.push_bit((b >> k) & 1);
                }
                ctx.push_symbol();
            }
        }
        assert_eq!(ctx.prev_tokens[1], 0x33); // newest
        assert_eq!(ctx.prev_tokens[2], 0x22);
        assert_eq!(ctx.prev_tokens[3], 0x11);
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
