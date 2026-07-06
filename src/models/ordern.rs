//! Generic order-`N` **u22-token** model: a `StateMap` over the in-progress u22 token and the last
//! `N` complete tokens, plus the bit-tree node.
//!
//! The stream feeding the entropy coder is (by construction of the Re-Pair → LZ77 pipeline) a dense
//! sequence of ULEB128 **u22** varints. This model works at that token granularity rather than on raw
//! bytes. It reads the shared u22 parse state carried on the [`Context`] — the payload bits accumulated
//! so far in the *in-progress* token, and the values of the last few *complete* tokens — so it needs no
//! parser of its own (the driver advances one parser per byte in [`Context::push_symbol`]).
//!
//! Order 0 conditions on the in-progress token only — its accumulated payload bits (`<= 14`) plus the
//! bit-tree node `c0` (`8`) — the token-level analog of a byte order-0 model keying on the partial
//! current byte. That is `<= 22` bits, so at scale it indexes a dense `2^22` map directly. Each higher
//! order additionally folds in the previous `order` complete token values (22 bits each), overflowing
//! any exact table, so orders `>= 1` hash into a table sized to the input by [`hashed_bits`]. Small
//! inputs never need the full `2^22` table, so order 0 hashes too when the input is too small for
//! [`hashed_bits`] to reach 22 (keeping tiny streams and the test suite cheap).

use super::statemap::StateMap;
use super::{Context, TOKEN_HISTORY, TokenModel, hashed_bits, token_hash};

/// Widest context width (in bits) eligible for the direct (exact, unhashed) dense table: order 0's
/// in-progress token (`14` payload bits) plus the bit-tree node (`8`) = `22`. Order 1 upward adds a
/// full 22-bit token per order, exceeding this, so they hash. The direct path is taken only when the
/// input is also large enough for [`hashed_bits`] to reach 22 (else even order 0 hashes).
const MAX_DIRECT_BITS: usize = 22;

/// Predicts each bit from the in-progress u22 token plus the last `order` complete tokens.
#[derive(Debug)]
pub(crate) struct OrderN {
    /// How many previous complete tokens this model folds into its context (`0` = order-0).
    order: usize,
    /// Right-shift folding the multiplicative hash down to the table's index width; `0` marks the
    /// direct (exact-index) dense path used by order 0 at scale.
    shift: u32,
    /// The adaptive probability map, one slot per (hashed) context.
    sm: StateMap,
    /// Slot chosen by the last [`OrderN::predict`], reused by the paired [`OrderN::update`].
    idx: usize,
}

impl OrderN {
    /// A fresh order-`order` model. `capacity` (the framed byte count) sizes the table so encode and
    /// decode agree: order 0 gets the exact dense `2^22` map when the input is large enough for
    /// [`hashed_bits`] to reach 22, otherwise a hashed table; orders `>= 1` always hash.
    pub(crate) fn new(order: usize, capacity: usize) -> Self {
        debug_assert!(order < TOKEN_HISTORY, "only orders below {TOKEN_HISTORY} are supported");
        let context_bits = 22 * order + 22;
        let bits = hashed_bits(capacity);
        if context_bits <= MAX_DIRECT_BITS && bits as usize >= context_bits {
            // Direct-indexed dense table: `shift == 0` selects the exact-index arm in `slot`.
            return Self {
                order,
                shift: 0,
                sm: StateMap::new(1 << context_bits),
                idx: 0,
            };
        }
        Self {
            order,
            shift: u64::BITS - bits,
            sm: StateMap::new(1 << bits),
            idx: 0,
        }
    }

    /// The map slot for the current bit: the in-progress token payload shifted above the bit-tree node
    /// `c0` (order 0), with the previous `order` complete tokens folded in by a multiplicative hash
    /// (orders `>= 1`, and order 0 on small inputs). A wrapping multiply-accumulate, so it never
    /// overflows even at high order (`22 * (order + 1)` would exceed 64 bits under naive shifting).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The direct value fits the 2^22 table; the hashed value is folded down to `shift` bits."
    )]
    fn slot(&self, ctx: &Context) -> usize {
        // `base <= 2^22 - 1` for ANY input (see `Context::hash_base`), so it can never index the dense
        // `2^22` table out of bounds.
        let base = ctx.hash_base();
        if self.shift == 0 {
            return base as usize;
        }
        // Fold in the previous `order` contiguous complete tokens (newest at `[1]`).
        let toks = ctx.prev_tokens.iter().skip(1).take(self.order).copied();
        (token_hash(base, toks) >> self.shift) as usize
    }
}

impl TokenModel for OrderN {
    fn predict(&mut self, ctx: &Context, _hist: &[u8]) -> i32 {
        self.idx = self.slot(ctx);
        self.sm.predict(self.idx)
    }

    fn update(&mut self, _ctx: &Context, _hist: &[u8], bit: u8) {
        self.sm.update(self.idx, bit);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Order 0 indexes the dense `2^22` table directly when the input is large enough for `hashed_bits`
    /// to reach 22 bits; a small input falls back to a hashed table.
    #[test]
    fn order0_dense_at_scale_hashed_when_small() {
        assert_eq!(OrderN::new(0, 1 << 22).shift, 0); // large input -> direct/dense
        assert_ne!(OrderN::new(0, 100).shift, 0); // small input -> hashed
    }

    /// Order 1 folds in a full 22-bit previous token, exceeding the direct width, so it always hashes.
    #[test]
    fn order1_is_hashed() {
        assert_ne!(OrderN::new(1, 1 << 22).shift, 0);
    }

    /// Order 0 keys only on the in-progress token (`cur_payload`, `c0`), ignoring completed tokens.
    #[test]
    fn order0_ignores_token_history() {
        let m = OrderN::new(0, 1 << 22);
        let mut a = Context::new();
        let mut b = Context::new();
        a.prev_tokens[1] = 5;
        b.prev_tokens[1] = 6;
        assert_eq!(m.slot(&a), m.slot(&b));
    }

    /// Order 1 folds in the previous complete token, so differing token history moves the slot.
    #[test]
    fn order1_keys_on_previous_token() {
        let m = OrderN::new(1, 1 << 22);
        let mut a = Context::new();
        let mut b = Context::new();
        a.prev_tokens[1] = 5;
        b.prev_tokens[1] = 6;
        assert_ne!(m.slot(&a), m.slot(&b));
    }
}
