//! Previous-word (word-bigram) context model: keyed on the **previous completed word** folded with
//! the **current partial word** and the bit-tree node.
//!
//! The registered `word` model ([`super::run`]) keys on the *current* partial word only — the letters
//! being typed — so it captures within-word structure but nothing about which word came *before*. This
//! model adds that missing axis: it hashes the previous completed word (the maximal letter run that
//! ended before the current one) together with the current partial word, so a common word transition
//! ("united"→"states", "new"→"york") is predicted from the byte the previous word implies. Folding the
//! partial word in as well makes the context position-aware *within* the current word rather than
//! sharing one slot across all of its bytes.
//!
//! The stream is post-casefold, so letters are already lowercased (case-insensitive). It uses the
//! lpaq-class bit-history state machine ([`super::state`]): word-pair contexts are sparse — most are
//! seen a handful of times — so pooling statistics by short bit history calibrates far better than a
//! lonely per-context probability, and a state byte is a third the memory of a full slot (which matters
//! at enwik9 scale). It abstains when there is no previous word within a short gap (start of stream, or
//! a long non-letter span before the cursor), so it never competes where it has nothing to say. Like
//! the other byte models it is a pure function of the finalized bytes (`hist`) and the framed byte
//! count, so encode and decode stay in lock-step; it only feeds the mixer, so a bug degrades ratio,
//! never correctness.

use super::state::BitHistory;
use super::{Context, Model, byte_hash, hashed_bits};

/// Cap on letters folded from each word (previous and partial). Past this a word's prefix is already
/// distinctive, and the cap bounds the backward scan.
const MAX_WORD: usize = 32;

/// Cap on the non-letter gap scanned between the current and previous word. A short separator ("` `",
/// "`. `", markup punctuation) still reaches the previous word; a longer span is treated as no adjacent
/// previous word (abstain), which also bounds the backward scan.
const MAX_GAP: usize = 8;

/// Delimiter folded between the previous word and the partial word so that "ab"+"c" and "a"+"bc" hash
/// to distinct contexts. Any value not produced by a letter run works; `0` never appears in a word.
const DELIM: u8 = 0;

/// Predicts each bit from the previous completed word plus the current partial word and the bit-tree node.
#[derive(Debug)]
pub(crate) struct PrevWordModel {
    /// Right-shift folding the multiplicative hash down to the table's index width.
    shift: u32,
    /// The bit-history predictor keyed on the (hashed) word-bigram context.
    bits: BitHistory,
    /// Whether the last [`PrevWordModel::predict`] found a context (else it abstained): the paired
    /// `update` advances a state only when a context was active.
    active: bool,
}

impl PrevWordModel {
    /// A fresh model. `capacity` (the framed byte count) sizes the hashed table via [`hashed_bits`] so
    /// encode and decode agree.
    pub(crate) fn new(capacity: usize) -> Self {
        let bits = hashed_bits(capacity);
        Self {
            shift: u64::BITS - bits,
            bits: BitHistory::new(1 << bits),
            active: false,
        }
    }

    /// The `[start, end)` bounds of the previous completed word and the current partial word within
    /// `hist`, or `None` to abstain when there is no adjacent previous word.
    ///
    /// Walks back from the cursor: the trailing letter run is the current *partial* word (possibly
    /// empty at a word boundary); a bounded run of non-letters is the *gap*; the letter run before that
    /// is the *previous* word. Abstains when the gap exceeds [`MAX_GAP`] (no adjacent word) or no
    /// previous word exists (stream start).
    fn spans(hist: &[u8]) -> Option<(&[u8], &[u8])> {
        let is_letter = u8::is_ascii_alphabetic;
        let n = hist.len();
        // Current partial word: trailing letters, capped.
        let mut p = n;
        while p > 0 && is_letter(&hist[p - 1]) && n - p < MAX_WORD {
            p -= 1;
        }
        let partial = &hist[p..n];
        // Gap: non-letters between the two words, capped. If the cap is hit while still on non-letters,
        // the previous-word scan below finds none and we abstain.
        let mut g = p;
        while g > 0 && !is_letter(&hist[g - 1]) && p - g < MAX_GAP {
            g -= 1;
        }
        // Previous word: letters before the gap, capped.
        let mut w = g;
        while w > 0 && is_letter(&hist[w - 1]) && g - w < MAX_WORD {
            w -= 1;
        }
        let prev = &hist[w..g];
        (!prev.is_empty()).then_some((prev, partial))
    }

    /// The context slot for the current bit, or `None` to abstain: the previous word and the current
    /// partial word (separated by [`DELIM`]) folded with the bit-tree node.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The hashed value is folded down to `shift` bits, so it indexes the table exactly."
    )]
    fn slot(&self, ctx: &Context, hist: &[u8]) -> Option<usize> {
        let (prev, partial) = Self::spans(hist)?;
        let ctx_bytes = prev.iter().copied().chain(std::iter::once(DELIM)).chain(partial.iter().copied());
        Some((byte_hash(ctx.node(), ctx_bytes) >> self.shift) as usize)
    }
}

impl Model for PrevWordModel {
    fn predict(&mut self, ctx: &Context, hist: &[u8]) -> i32 {
        if let Some(slot) = self.slot(ctx, hist) {
            self.active = true;
            self.bits.predict(slot)
        } else {
            self.active = false;
            0
        }
    }

    fn update(&mut self, _ctx: &Context, _hist: &[u8], bit: u8) {
        if self.active {
            self.bits.update(bit);
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Abstains with no previous word (stream start, all one word) and keys on the previous word once
    /// one exists.
    #[test]
    fn abstains_without_previous_word() {
        let m = PrevWordModel::new(1 << 16);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx, b""), None); // empty history
        assert_eq!(m.slot(&ctx, b"hello"), None); // one partial word, nothing before it
        assert!(m.slot(&ctx, b"new y").is_some()); // "new" is the previous word, "y" the partial
        assert!(m.slot(&ctx, b"new ").is_some()); // at a word boundary the partial is empty
    }

    /// Abstains when the previous word is separated by more than [`MAX_GAP`] non-letters.
    #[test]
    fn abstains_across_a_long_gap() {
        let m = PrevWordModel::new(1 << 16);
        let ctx = Context::new();
        assert!(m.slot(&ctx, b"new    y").is_some()); // 4-byte gap is within MAX_GAP
        let long = [b"new", &[b' '; MAX_GAP + 1][..], b"y"].concat();
        assert_eq!(m.slot(&ctx, &long), None); // gap exceeds MAX_GAP => no adjacent previous word
    }

    /// Different previous words land on different contexts even for the same partial word — the signal
    /// the current-word `run` model cannot see.
    #[test]
    fn keys_on_the_previous_word() {
        let m = PrevWordModel::new(1 << 16);
        let ctx = Context::new();
        assert_ne!(m.slot(&ctx, b"new y"), m.slot(&ctx, b"old y"));
        // The delimiter keeps the word split unambiguous: "ab c" != "a bc".
        assert_ne!(m.slot(&ctx, b"ab c"), m.slot(&ctx, b"a bc"));
    }

    /// Deterministic (encode/decode must agree).
    #[test]
    fn is_deterministic() {
        let m = PrevWordModel::new(1 << 16);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx, b"united states"), m.slot(&ctx, b"united states"));
    }

    /// A run of ones at an active context raises its prediction while an abstaining position stays
    /// neutral — end-to-end exercise of the bit-history path and the abstain guard.
    #[test]
    fn active_context_learns_and_abstain_is_neutral() {
        let mut m = PrevWordModel::new(1 << 16);
        let ctx = Context::new();
        let hist = b"new y";
        let before = m.predict(&ctx, hist);
        for _ in 0..64 {
            let _ = m.predict(&ctx, hist);
            m.update(&ctx, hist, 1);
        }
        let after = m.predict(&ctx, hist);
        assert!(after > before, "a run of ones should raise the prediction: before={before} after={after}");
        // Abstaining position returns the neutral stretched 0 and does not panic on update.
        assert_eq!(m.predict(&ctx, b"hello"), 0);
        m.update(&ctx, b"hello", 1);
    }
}
