//! Varint-phase (continuation) model: predicts each bit from the continuation-bit pattern of the
//! recent bytes plus the bit-tree node.
//!
//! The stream feeding the entropy coder is dense with ULEB128 varints — Re-Pair symbol ids and LZ77
//! lengths/distances — whose high bit is a **continuation flag**: set on every non-final byte, clear
//! on the last. That flag makes some bits near-deterministic (mid-varint the next high bit is almost
//! surely 1; just after a terminator the phase resets). The order-`n` and sparse token models parse
//! this framing to recover token *values*, but none keys directly on the continuation-bit *phase*
//! pattern; this model does, keying purely on the high (continuation) bits of the last [`SPAN`] bytes
//! plus the current node — an explicit "byte-k-of-a-varint" context at near-zero cost that is
//! empirically complementary to the token-value models (a clear win on enwik9, independent of them).
//! The context is derived only from the borrowed history window, so encode and decode agree bit-for-bit.

use super::statemap::StateMap;
use super::{Context, SYMBOL_BITS, TokenModel};

/// How many recent bytes' continuation bits form the phase context.
const SPAN: usize = 4;

/// Predicts each bit from the recent continuation-bit pattern and the current bit-tree node.
#[derive(Debug)]
pub(crate) struct VarintModel {
    /// The adaptive probability map, one slot per (continuation pattern, node) context.
    sm: StateMap,
    /// Slot chosen by the last [`VarintModel::predict`], reused by the paired [`VarintModel::update`].
    idx: usize,
    /// The continuation-bit prefix of the last `SPAN` bytes, already shifted above the node bits. It
    /// depends only on `hist`, which is constant across a byte's 8 bits, so it is refreshed once per
    /// byte (at `bpos == 0`) rather than rescanned every bit.
    cont: usize,
}

impl VarintModel {
    /// A fresh varint-phase model. Its context is `SPAN` continuation bits plus the [`SYMBOL_BITS`]
    /// node bits — a small, exact, cache-resident direct table (no hashing, no `capacity`).
    pub(crate) fn new() -> Self {
        Self {
            sm: StateMap::new(1 << (SPAN + SYMBOL_BITS as usize)),
            idx: 0,
            cont: 0,
        }
    }

    /// The continuation bits of the last `SPAN` bytes (newest first), shifted above the [`SYMBOL_BITS`]
    /// node — the part of the slot that depends only on `hist`, so it is stable across a byte's bits.
    fn cont_prefix(hist: &[u8]) -> usize {
        let mut cont = 0usize;
        for k in 1..=SPAN {
            // hist is newest-last; the k-th most recent byte's high bit, or 0 before enough history.
            let msb = hist.len().checked_sub(k).map_or(0, |i| usize::from(hist[i] >> 7));
            cont = (cont << 1) | msb;
        }
        cont << SYMBOL_BITS
    }
}

impl TokenModel for VarintModel {
    fn predict(&mut self, ctx: &Context, hist: &[u8]) -> i32 {
        // `hist` is the same for all 8 bits of the current byte, so recompute the continuation prefix
        // only when a new byte starts; per bit only the bit-tree node `c0` varies.
        if ctx.bpos == 0 {
            self.cont = Self::cont_prefix(hist);
        }
        self.idx = self.cont | (ctx.c0 & 0xff) as usize;
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

    /// The context is the *high* bit of recent bytes only: two histories that differ solely in a
    /// low bit share a continuation prefix; flipping a recent high bit moves it.
    #[test]
    fn keys_on_continuation_bits_only() {
        // 0x01 and 0x7F share high bit 0 -> same prefix; 0x80 has high bit 1 -> different prefix.
        assert_eq!(VarintModel::cont_prefix(&[0x01]), VarintModel::cont_prefix(&[0x7F]));
        assert_ne!(VarintModel::cont_prefix(&[0x01]), VarintModel::cont_prefix(&[0x80]));
    }
}
