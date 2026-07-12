//! Generic order-`N` **byte** model: keyed on the last `N` finalized bytes plus the bit-tree node.
//!
//! Order 0 conditions on the bit-tree node `c0` only (the partial current byte) — the classic
//! order-0 model — and indexes a dense 256-slot [`StateMap`] exactly, no hashing. Each higher order
//! folds in the previous `order` finalized bytes (read from the `hist` window), overflowing any exact
//! table, so orders `>= 1` hash into a table sized to the input by [`hashed_bits`].
//!
//! Order 0 keeps a full `(probability, count)` `StateMap` slot per context: its 256 contexts are dense
//! and heavily visited, so a per-context probability converges well. Every higher order instead stores
//! a 1-byte **bit-history state** per context (see [`super::state`]) and shares one small `StateMap`
//! mapping state → probability. Those contexts are sparse — most are seen only a few times — so pooling
//! statistics by bit history calibrates far better than a lonely per-context estimate, and the state
//! byte is a third the memory of a full slot (which matters at enwik9 scale). Measured on a 20 MB
//! enwik8 prefix, the switch improves the entropy stage ~0.015 bpb while running ~20% faster (a byte is
//! more cache-friendly than the 3-byte slot) and cutting peak RSS ~40%.

use super::state::BitHistory;
use super::statemap::StateMap;
use super::{Context, Model, byte_hash, hashed_bits};

/// Orders at or above this use the shared bit-history state map; lower orders keep a per-context
/// `StateMap` slot. Only order 0 (256 dense, heavily-visited contexts) is left on the direct path;
/// from order 1 up the contexts are sparse enough that the pooled state map wins (swept on a 20 MB
/// enwik8 prefix: threshold 1 beat 2 and 3).
const BITHIST_MIN_ORDER: usize = 1;

/// Predicts each bit from the bit-tree node plus the last `order` finalized bytes.
#[derive(Debug)]
pub(crate) struct OrderN {
    /// How many previous finalized bytes this model folds into its context (`0` = order-0).
    order: usize,
    /// Right-shift folding the multiplicative hash down to the table's index width; `0` marks the
    /// direct (exact-index) dense path used by order 0.
    shift: u32,
    /// The direct per-context probability map, used only on the direct path (order 0); `None` on the
    /// bit-history path.
    sm: Option<StateMap>,
    /// The bit-history predictor, used only for high orders (`>= BITHIST_MIN_ORDER`); `None` on the
    /// direct path.
    bits: Option<BitHistory>,
    /// Direct-path `StateMap` index chosen by the last [`OrderN::predict`], reused by the paired
    /// `update` (the context slot, i.e. the bit-tree node for order 0).
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
                sm: Some(StateMap::new(1 << u8::BITS)),
                bits: None,
                idx: 0,
            };
        }
        let bits = hashed_bits(capacity);
        let bithist = order >= BITHIST_MIN_ORDER;
        Self {
            order,
            shift: u64::BITS - bits,
            // Direct path: a full per-context slot for every hashed context. Bit-history path: a state
            // byte per context plus one shared 256-state map.
            sm: (!bithist).then(|| StateMap::new(1 << bits)),
            bits: bithist.then(|| BitHistory::new(1 << bits)),
            idx: 0,
        }
    }

    /// The context slot for the current bit: the bit-tree node `c0` (order 0), with the previous
    /// `order` finalized bytes folded in by a multiplicative hash (orders `>= 1`).
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
        let slot = self.slot(ctx, hist);
        // Bit-history path delegates to the shared predictor; direct path indexes its own StateMap.
        if let Some(bits) = &mut self.bits {
            return bits.predict(slot);
        }
        self.idx = slot;
        self.sm.as_ref().map_or(0, |sm| sm.predict(slot))
    }

    fn update(&mut self, _ctx: &Context, _hist: &[u8], bit: u8) {
        if let Some(bits) = &mut self.bits {
            bits.update(bit);
        } else if let Some(sm) = &mut self.sm {
            sm.update(self.idx, bit);
        }
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

    /// Order 0 keeps a per-context slot (direct `StateMap`, no bit history); every higher order
    /// switches to the shared bit-history predictor.
    #[test]
    fn bithist_engages_for_high_orders() {
        let o0 = OrderN::new(0, 1 << 22);
        assert!(o0.bits.is_none() && o0.sm.is_some());
        assert!(OrderN::new(BITHIST_MIN_ORDER, 1 << 22).bits.is_some());
        assert!(OrderN::new(6, 1 << 22).bits.is_some());
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

    /// A bit-history model advances a context's state on update and predicts a moving probability
    /// after a run of one bit — end-to-end exercise of the state path.
    #[test]
    fn bithist_state_advances_and_predicts() {
        let mut m = OrderN::new(6, 1 << 16);
        let ctx = Context::new();
        let hist = [1u8, 2, 3, 4, 5, 6];
        let before = m.predict(&ctx, &hist);
        for _ in 0..64 {
            let _ = m.predict(&ctx, &hist);
            m.update(&ctx, &hist, 1);
        }
        let after = m.predict(&ctx, &hist);
        assert!(after > before, "a run of ones should raise the prediction: before={before} after={after}");
    }
}
