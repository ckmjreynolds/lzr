//! Generic sparse (skip-gram) **u22-token** model: a hashed `StateMap` keyed on a *non-contiguous* set
//! of previous complete token values plus the in-progress token and the bit-tree node.
//!
//! Where [`super::ordern::OrderN`] folds in the last `order` **contiguous** tokens, a `SparseModel`
//! reads tokens at arbitrary back-offsets `gaps` (`1` = the most recent complete token). `[2]` keys on
//! the token two back while skipping the immediately preceding one; `[1, 3]` skips the token two back.
//! Such gapped-*token* contexts capture correlations a contiguous order-`n` context misses (structure
//! at a fixed token stride) — and, unlike a byte-offset skip-gram, they actually align to token
//! boundaries in the varint stream. Like the high-order models it reads the shared parse state on the
//! [`Context`] and hashes the context down to the table width, so encode and decode agree bit-for-bit.

use super::statemap::StateMap;
use super::{Context, TokenModel, hashed_bits, token_hash};

/// Predicts each bit from a fixed set of gapped previous token values plus the in-progress token.
#[derive(Debug)]
pub(crate) struct SparseModel {
    /// Back-offsets into [`Context::prev_tokens`] (`1` = most recent complete token), newest-relative.
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

    /// The map slot for the current bit: the in-progress token payload above the bit-tree node `c0`,
    /// with the gapped previous token values folded in by a multiplicative hash. A gap past the
    /// retained history reads as `0` (deterministically on both sides).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The hashed value is folded down to `shift` bits, so it indexes the table exactly."
    )]
    fn slot(&self, ctx: &Context) -> usize {
        // Fold in the gapped previous tokens; a gap past the retained history reads as 0.
        let toks = self.gaps.iter().map(|&k| ctx.prev_tokens.get(k).copied().unwrap_or(0));
        (token_hash(ctx.hash_base(), toks) >> self.shift) as usize
    }
}

impl TokenModel for SparseModel {
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

    /// A single-gap `[2]` model keys on the token two back and ignores the immediately preceding one:
    /// changing the gap-2 token moves the slot, changing only the gap-1 token does not.
    #[test]
    fn keys_on_gapped_token_only() {
        let model = SparseModel::new(&[2], 1 << 16);
        let (mut gap2_lo, mut gap2_hi) = (Context::new(), Context::new());
        gap2_lo.prev_tokens[2] = 0xAA;
        gap2_hi.prev_tokens[2] = 0xBB;
        assert_ne!(model.slot(&gap2_lo), model.slot(&gap2_hi));
        // The gap-1 token is skipped, so varying it leaves the slot unchanged.
        let (mut gap1_lo, mut gap1_hi) = (Context::new(), Context::new());
        gap1_lo.prev_tokens[1] = 0xAA;
        gap1_hi.prev_tokens[1] = 0xBB;
        assert_eq!(model.slot(&gap1_lo), model.slot(&gap1_hi));
    }

    /// Before enough tokens exist the missing positions read as zero, deterministically.
    #[test]
    fn short_history_is_deterministic() {
        let m = SparseModel::new(&[1, 3], 1 << 16);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx), m.slot(&ctx));
    }
}
