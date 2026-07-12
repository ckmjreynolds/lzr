//! Character-class **run** model: a hashed `StateMap` keyed on the current run of same-class bytes
//! immediately preceding the current byte, plus the bit-tree node.
//!
//! Parameterised by a byte-class predicate. The registered use is the **word** model
//! (`is_ascii_alphabetic`): the partial word being typed. Ignoring the punctuation before a word
//! generalises its prefix across contexts, and long words reach further back than the order-N chain;
//! the entropy stream is post-casefold, so letters are already lowercased (case-insensitive). Any byte
//! class works — a digit-class run was also swept but measured neutral (number structure is
//! cross-token, not in the trailing digits) and is not registered.
//!
//! It abstains when the previous byte is not of the class (no run to condition on), so it never
//! competes with the order-N models on the bytes it has nothing to say about. Like the other byte
//! models it is a pure function of the finalized bytes (`hist`) and the framed byte count, so encode
//! and decode stay in lock-step; it only feeds a probability to the mixer, so a bug degrades ratio,
//! never correctness.

use super::statemap::SlotMap;
use super::{Context, Model, hashed_bits, hashed_slot};

/// Cap on the number of trailing same-class bytes folded into the context. Longer runs share their
/// tail context — past this the prefix is already distinctive — and the cap bounds the backward scan.
const MAX_RUN: usize = 32;

/// Predicts each bit from the current run of `class` bytes plus the bit-tree node.
#[derive(Debug)]
pub(crate) struct RunModel {
    /// Byte-class predicate defining the run (e.g. `u8::is_ascii_alphabetic` for words).
    class: fn(&u8) -> bool,
    /// Right-shift folding the multiplicative hash down to the table's index width.
    shift: u32,
    /// The adaptive probability map, keyed on the (hashed) run context; abstains off a run.
    map: SlotMap,
}

impl RunModel {
    /// A fresh run model over the byte class `class`. `capacity` (the framed byte count) sizes the
    /// hashed table via [`hashed_bits`] so encode and decode agree.
    pub(crate) fn new(class: fn(&u8) -> bool, capacity: usize) -> Self {
        let bits = hashed_bits(capacity);
        Self {
            class,
            shift: u64::BITS - bits,
            map: SlotMap::new(1 << bits),
        }
    }

    /// The map slot for the current bit, or `None` to abstain: the bit-tree node `c0` with the trailing
    /// run of `class` bytes folded in (newest-first, capped at [`MAX_RUN`]). Abstains when the previous
    /// byte is not of the class (no run to condition on).
    fn slot(&self, ctx: &Context, hist: &[u8]) -> Option<usize> {
        let run = hist.iter().rev().copied().take_while(|b| (self.class)(b)).take(MAX_RUN);
        hashed_slot(ctx.node(), run, self.shift)
    }
}

impl Model for RunModel {
    fn predict(&mut self, ctx: &Context, hist: &[u8]) -> i32 {
        let slot = self.slot(ctx, hist);
        self.map.predict(slot)
    }

    fn update(&mut self, _ctx: &Context, _hist: &[u8], bit: u8) {
        self.map.update(bit);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn word(capacity: usize) -> RunModel {
        RunModel::new(u8::is_ascii_alphabetic, capacity)
    }

    /// Abstains at run start (previous byte not of the class) and on empty history.
    #[test]
    fn abstains_off_run() {
        let m = word(1 << 16);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx, &[]), None);
        assert_eq!(m.slot(&ctx, b"hello "), None); // trailing space => no partial word
        assert_eq!(m.slot(&ctx, b"123"), None); // digits are not letters
        // The digit model is the mirror image: it runs on digits, abstains on letters.
        let d = RunModel::new(u8::is_ascii_digit, 1 << 16);
        assert!(d.slot(&ctx, b"123").is_some());
        assert_eq!(d.slot(&ctx, b"abc"), None);
    }

    /// Keys on the trailing run only, ignoring the bytes before it — the same partial word after
    /// different separators lands on the same slot.
    #[test]
    fn keys_on_trailing_run_only() {
        let m = word(1 << 16);
        let ctx = Context::new();
        assert!(m.slot(&ctx, b"the").is_some());
        assert_eq!(m.slot(&ctx, b"foo bar the"), m.slot(&ctx, b"...!the"));
        assert_ne!(m.slot(&ctx, b"the"), m.slot(&ctx, b"teh"));
    }

    /// Deterministic: the same inputs always produce the same slot (encode/decode must agree).
    #[test]
    fn is_deterministic() {
        let m = word(1 << 16);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx, b"compression"), m.slot(&ctx, b"compression"));
    }
}
