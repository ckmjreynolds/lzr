//! Generic order-`N` **byte** model: a `StateMap` over the last `N` finalized bytes plus the
//! bit-tree node.
//!
//! Order 0 conditions on the bit-tree node `c0` only (the partial current byte) — the classic
//! order-0 model — and indexes a dense 256-slot table exactly, no hashing. Each higher order folds in
//! the previous `order` finalized bytes (read from the `hist` window), overflowing any exact table,
//! so orders `>= 1` hash into a table sized to the input by [`hashed_bits`].

use super::statemap::StateMap;
use super::{Context, Model, byte_hash, hashed_bits};

/// Predicts each bit from the bit-tree node plus the last `order` finalized bytes.
#[derive(Debug)]
pub(crate) struct OrderN {
    /// How many previous finalized bytes this model folds into its context (`0` = order-0).
    order: usize,
    /// Right-shift folding the multiplicative hash down to the table's index width; `0` marks the
    /// direct (exact-index) dense path used by order 0.
    shift: u32,
    /// The adaptive probability map, one slot per (hashed) context.
    sm: StateMap,
    /// Slot chosen by the last [`OrderN::predict`], reused by the paired [`OrderN::update`].
    idx: usize,
}

impl OrderN {
    /// A fresh order-`order` model. `capacity` (the framed byte count) sizes the hashed table so
    /// encode and decode agree; order 0 is always the exact dense 256-slot table.
    pub(crate) fn new(order: usize, capacity: usize) -> Self {
        if order == 0 {
            // Direct-indexed dense table over the bit-tree node (`< 256`): `shift == 0` selects the
            // exact-index arm in `slot`.
            return Self {
                order,
                shift: 0,
                sm: StateMap::new(1 << u8::BITS),
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

    /// The map slot for the current bit: the bit-tree node `c0` (order 0), with the previous `order`
    /// finalized bytes folded in by a multiplicative hash (orders `>= 1`).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The direct node fits the 256 table; the hashed value is folded down to `shift` bits."
    )]
    fn slot(&self, ctx: &Context, hist: &[u8]) -> usize {
        let node = ctx.node();
        if self.shift == 0 {
            // `node < 256` during any prediction, so this cannot index the dense 256 table out of bounds.
            return node as usize;
        }
        // Fold in the previous `order` finalized bytes (newest first).
        let bytes = hist.iter().rev().take(self.order).copied();
        (byte_hash(node, bytes) >> self.shift) as usize
    }
}

impl Model for OrderN {
    fn predict(&mut self, ctx: &Context, hist: &[u8]) -> i32 {
        self.idx = self.slot(ctx, hist);
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

    /// Order 0 is always the exact dense table (`shift == 0`), regardless of input size.
    #[test]
    fn order0_is_dense() {
        assert_eq!(OrderN::new(0, 1 << 22).shift, 0);
        assert_eq!(OrderN::new(0, 100).shift, 0);
    }

    /// Orders `>= 1` always hash.
    #[test]
    fn order1_is_hashed() {
        assert_ne!(OrderN::new(1, 1 << 22).shift, 0);
    }

    /// Order 0 keys only on the bit-tree node, ignoring the byte history.
    #[test]
    fn order0_ignores_byte_history() {
        let m = OrderN::new(0, 1 << 22);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx, &[5]), m.slot(&ctx, &[6]));
    }

    /// Order 1 folds in the previous finalized byte, so differing history moves the slot.
    #[test]
    fn order1_keys_on_previous_byte() {
        let m = OrderN::new(1, 1 << 22);
        let ctx = Context::new();
        assert_ne!(m.slot(&ctx, &[5]), m.slot(&ctx, &[6]));
    }
}
