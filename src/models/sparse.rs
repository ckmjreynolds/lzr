//! Generic sparse (skip-gram) **byte** model: a hashed `StateMap` keyed on a *non-contiguous* set of
//! previous finalized byte values plus the bit-tree node.
//!
//! Where [`super::ordern::OrderN`] folds in the last `order` **contiguous** bytes, a `SparseModel`
//! reads bytes at arbitrary back-offsets `gaps` (`1` = the most recent finalized byte). `[2]` keys on
//! the byte two back while skipping the immediately preceding one; `[1, 3]` skips the byte two back.
//! Such gapped contexts capture correlations a contiguous order-`n` context misses (structure at a
//! fixed byte stride). It hashes the context down to the table width, so encode and decode agree
//! bit-for-bit.

use super::statemap::StateMap;
use super::{Context, Model, byte_hash, hashed_bits};

/// Predicts each bit from a fixed set of gapped previous byte values plus the bit-tree node.
#[derive(Debug)]
pub(crate) struct SparseModel {
    /// Back-offsets into the `hist` window (`1` = most recent finalized byte), newest-relative.
    gaps: &'static [usize],
    /// Right-shift folding the multiplicative hash down to the table's index width.
    shift: u32,
    /// The adaptive probability map, one slot per (hashed) context.
    sm: StateMap,
    /// Slot chosen by the last [`SparseModel::predict`], reused by the paired [`SparseModel::update`].
    idx: usize,
}

impl SparseModel {
    /// A fresh sparse model over `gaps`. `capacity` (the framed byte count) sizes the hashed table so
    /// encode and decode agree.
    pub(crate) fn new(gaps: &'static [usize], capacity: usize) -> Self {
        let bits = hashed_bits(capacity);
        Self {
            gaps,
            shift: u64::BITS - bits,
            sm: StateMap::new(1 << bits),
            idx: 0,
        }
    }

    /// The map slot for the current bit: the bit-tree node `c0` with the gapped previous byte values
    /// folded in by a multiplicative hash. A gap past the available history reads as `0`
    /// (deterministically on both sides).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The hashed value is folded down to `shift` bits, so it indexes the table exactly."
    )]
    fn slot(&self, ctx: &Context, hist: &[u8]) -> usize {
        // Fold in the gapped previous bytes (`k` back from the newest); a gap past history reads as 0.
        let bytes = self.gaps.iter().map(|&k| hist.len().checked_sub(k).map_or(0, |i| hist[i]));
        (byte_hash(ctx.node(), bytes) >> self.shift) as usize
    }
}

impl Model for SparseModel {
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

    /// A single-gap `[2]` model keys on the byte two back and ignores the immediately preceding one:
    /// changing the gap-2 byte moves the slot, changing only the gap-1 byte does not.
    #[test]
    fn keys_on_gapped_byte_only() {
        let model = SparseModel::new(&[2], 1 << 16);
        let ctx = Context::new();
        // hist is newest-last, so gap-2 is hist[len-2] and gap-1 is hist[len-1].
        assert_ne!(model.slot(&ctx, &[0xAA, 0x00]), model.slot(&ctx, &[0xBB, 0x00]));
        // The gap-1 byte is skipped, so varying it leaves the slot unchanged.
        assert_eq!(model.slot(&ctx, &[0x00, 0xAA]), model.slot(&ctx, &[0x00, 0xBB]));
    }

    /// Before enough bytes exist the missing positions read as zero, deterministically.
    #[test]
    fn short_history_is_deterministic() {
        let m = SparseModel::new(&[1, 3], 1 << 16);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx, &[]), m.slot(&ctx, &[]));
    }
}
