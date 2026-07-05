//! Generic order-`N` byte model: a direct `StateMap` over the last `N` bytes and the bit-tree.
//!
//! Generalizes the order-0 model. Each bit's context is the last `order` finalized bytes (read from
//! the driver's history window) with the partial-symbol node `c0` appended, mapped to one slot of a
//! `StateMap`. Order 0 uses no history, so its context is just `c0` over a `2^8`-slot map — identical
//! to the original order-0 model. Low orders index the map directly (exact, collision-free); higher
//! orders, whose `256^order` context space is infeasible to hold, hash into a table sized to the input
//! by [`hashed_bits`].

use super::statemap::StateMap;
use super::{Context, SYMBOL_BITS, TokenModel, hashed_bits};

/// Widest context width (in bits) still indexed directly rather than hashed. Order 0 needs 8 bits
/// (`c0`) and order 1 needs 16 (`prev << 8 | c0`); both fit, giving exact `2^8` / `2^16` maps. Order
/// 2 upward (`>= 24` bits) is hashed instead of allocating a multi-million-slot map.
const MAX_DIRECT_BITS: usize = 16;

/// Golden-ratio multiplier for the multiplicative context hash (orders `>= 2`).
const HASH_MULT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Predicts each bit from the last `order` finalized bytes plus the current bit-tree node `c0`.
#[derive(Debug)]
pub(crate) struct OrderN {
    /// How many finalized bytes of context this model reads (`0` = order-0).
    order: usize,
    /// Right-shift folding the multiplicative hash down to the table's index width; `0` marks the
    /// direct (exact-index) path used by the low orders.
    shift: u32,
    /// The adaptive probability map, one slot per (hashed) context.
    sm: StateMap,
    /// Slot chosen by the last [`OrderN::predict`], reused by the paired [`OrderN::update`].
    idx: usize,
}

impl OrderN {
    /// A fresh order-`order` model. `capacity` (the framed byte count) sizes the hashed table for
    /// orders `>= 2` so encode and decode agree; direct orders ignore it.
    pub(crate) fn new(order: usize, capacity: usize) -> Self {
        let context_bits = 8 * order + SYMBOL_BITS as usize;
        if context_bits <= MAX_DIRECT_BITS {
            // Direct-indexed: `shift == 0` selects the exact-table arm in `slot`.
            return Self {
                order,
                shift: 0,
                sm: StateMap::new(1 << context_bits),
                idx: 0,
            };
        }
        let bits = hashed_bits(capacity);
        Self {
            order,
            shift: u64::BITS - bits,
            sm: StateMap::new(1 << bits),
            idx: 0,
        }
    }

    /// The map slot for the current bit: the last `order` bytes (newest first) shifted above `c0`,
    /// used directly for low orders or multiplicatively hashed for high ones.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The hashed value is folded down to `shift`; the direct value fits the table."
    )]
    fn slot(&self, c0: u32, hist: &[u8]) -> usize {
        let mut cv = 0u64;
        for k in 1..=self.order {
            // hist is newest-last; the k-th most recent byte, or 0 before enough history exists.
            let byte = hist.len().checked_sub(k).map_or(0, |i| hist[i]);
            cv = (cv << 8) | u64::from(byte);
        }
        let raw = (cv << SYMBOL_BITS) | u64::from(c0 & 0xff);
        if self.shift == 0 {
            raw as usize
        } else {
            (raw.wrapping_mul(HASH_MULT) >> self.shift) as usize
        }
    }
}

impl TokenModel for OrderN {
    fn predict(&mut self, ctx: &Context, hist: &[u8]) -> i32 {
        self.idx = self.slot(ctx.c0, hist);
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

    /// Order 0 reads no history, so every prediction ignores `hist` and indexes only by `c0` — the
    /// original order-0 behavior. Its table is the exact `2^8` slots.
    #[test]
    fn order0_ignores_history() {
        let m = OrderN::new(0, 0);
        assert_eq!(m.shift, 0); // direct-indexed
        // The same c0 yields the same slot regardless of preceding bytes.
        let ctx = Context::new();
        assert_eq!(m.slot(ctx.c0, b""), m.slot(ctx.c0, b"abc"));
    }

    /// Order 1 keys on the previous byte: different history bytes select different slots (a distinct
    /// `2^16` direct table), while order 0 would collapse them.
    #[test]
    fn order1_keys_on_previous_byte() {
        let m = OrderN::new(1, 0);
        assert_eq!(m.shift, 0); // direct-indexed
        let ctx = Context::new();
        assert_ne!(m.slot(ctx.c0, b"a"), m.slot(ctx.c0, b"b"));
        // With no history yet, order 1 falls back to a zero context byte.
        assert_eq!(m.slot(ctx.c0, b""), m.slot(ctx.c0, &[0]));
    }

    /// Order 2 exceeds the direct-index width and switches to the hashed table.
    #[test]
    fn order2_is_hashed() {
        let m = OrderN::new(2, 1 << 20);
        assert_ne!(m.shift, 0); // hashed
    }
}
