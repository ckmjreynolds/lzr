//! Two-word context model: keyed on the previous **two** completed words folded with the current
//! partial word and the bit-tree node.
//!
//! Where [`super::prevword`] conditions on the single previous word (a word bigram), this conditions on
//! the two words before the cursor (a word trigram): "united states"→"of", "new york"→"city". The two
//! preceding words pin down a continuation the single previous word leaves ambiguous. Folding the
//! current partial word in as well keeps the context position-aware within the word being typed rather
//! than sharing one slot across all of its bytes.
//!
//! The stream is post-casefold, so letters are already lowercased. It uses the lpaq-class bit-history
//! state machine ([`super::state`]): two-word contexts are even sparser than one-word contexts, so
//! pooling by short bit history is what makes them usable, and a state byte is a third the memory of a
//! full slot. It abstains unless *two* previous words sit within short gaps of the cursor, so deep in a
//! long word (or after a long non-letter span) it stays silent and the order-N, `word`, and `prevword`
//! models carry the context. Like the other byte models it is a pure function of the finalized bytes
//! (`hist`) and the framed byte count, so encode and decode stay in lock-step; it only feeds the mixer,
//! so a bug degrades ratio, never correctness.

use super::state::BitHistory;
use super::{Context, Model, byte_hash, hashed_bits};

/// Cap on letters folded from each word. Past this a word's prefix is already distinctive, and the cap
/// bounds the backward scan.
const MAX_WORD: usize = 32;

/// Cap on each non-letter gap scanned between adjacent words. A short separator still bridges two words;
/// a longer span means no adjacent word (abstain), which also bounds the backward scan.
const MAX_GAP: usize = 8;

/// Delimiter folded between the words and the partial so that different splits of the same letters hash
/// to distinct contexts. `0` never appears in a word.
const DELIM: u8 = 0;

/// Predicts each bit from the previous two completed words plus the current partial word and bit-tree node.
#[derive(Debug)]
pub(crate) struct PrevWord2Model {
    /// Right-shift folding the multiplicative hash down to the table's index width.
    shift: u32,
    /// The bit-history predictor keyed on the (hashed) two-word context.
    bits: BitHistory,
    /// Whether the last [`PrevWord2Model::predict`] found a context (else it abstained): the paired
    /// `update` advances a state only when a context was active.
    active: bool,
}

impl PrevWord2Model {
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

    /// The previous two completed words and the current partial word within `hist`, or `None` to abstain
    /// when two adjacent previous words do not exist.
    ///
    /// Walks back from the cursor: the trailing letter run is the *partial* word; then, twice, a bounded
    /// non-letter *gap* followed by a letter *word*. Abstains as soon as an expected word is missing (a
    /// gap over [`MAX_GAP`], or the stream start).
    fn spans(hist: &[u8]) -> Option<(&[u8], &[u8], &[u8])> {
        let is_letter = u8::is_ascii_alphabetic;
        // Scan one letter run ending at `end` (exclusive), capped at MAX_WORD; return its start index.
        let word_start = |end: usize| {
            let mut s = end;
            while s > 0 && is_letter(&hist[s - 1]) && end - s < MAX_WORD {
                s -= 1;
            }
            s
        };
        // Scan one non-letter gap ending at `end` (exclusive), capped at MAX_GAP; return its start index.
        let gap_start = |end: usize| {
            let mut s = end;
            while s > 0 && !is_letter(&hist[s - 1]) && end - s < MAX_GAP {
                s -= 1;
            }
            s
        };

        let n = hist.len();
        let p = word_start(n); // current partial word: hist[p..n]
        let g1 = gap_start(p);
        let w1 = word_start(g1); // previous word: hist[w1..g1]
        if w1 == g1 {
            return None; // no first previous word
        }
        let g2 = gap_start(w1);
        let w2 = word_start(g2); // word before that: hist[w2..g2]
        if w2 == g2 {
            return None; // no second previous word
        }
        Some((&hist[w2..g2], &hist[w1..g1], &hist[p..n]))
    }

    /// The context slot for the current bit, or `None` to abstain: the two previous words and the current
    /// partial word (each [`DELIM`]-separated) folded with the bit-tree node.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The hashed value is folded down to `shift` bits, so it indexes the table exactly."
    )]
    fn slot(&self, ctx: &Context, hist: &[u8]) -> Option<usize> {
        let (w2, w1, partial) = Self::spans(hist)?;
        let ctx_bytes = w2
            .iter()
            .copied()
            .chain(std::iter::once(DELIM))
            .chain(w1.iter().copied())
            .chain(std::iter::once(DELIM))
            .chain(partial.iter().copied());
        Some((byte_hash(ctx.node(), ctx_bytes) >> self.shift) as usize)
    }
}

impl Model for PrevWord2Model {
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

    /// Abstains until two previous words exist, then keys on the pair.
    #[test]
    fn needs_two_previous_words() {
        let m = PrevWord2Model::new(1 << 16);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx, b""), None);
        assert_eq!(m.slot(&ctx, b"york c"), None); // only one previous word ("york")
        assert!(m.slot(&ctx, b"new york c").is_some()); // "new" and "york" precede the partial "c"
        assert!(m.slot(&ctx, b"new york ").is_some()); // at a word boundary the partial is empty
    }

    /// Abstains when either gap between the words exceeds [`MAX_GAP`].
    #[test]
    fn abstains_across_a_long_gap() {
        let m = PrevWord2Model::new(1 << 16);
        let ctx = Context::new();
        let long = [b"new", &[b' '; MAX_GAP + 1][..], b"york c"].concat();
        assert_eq!(m.slot(&ctx, &long), None); // gap before "new" is too wide to reach the 2nd word
    }

    /// The second word back changes the context — the signal a one-word model cannot see.
    #[test]
    fn keys_on_the_second_word_back() {
        let m = PrevWord2Model::new(1 << 16);
        let ctx = Context::new();
        assert_ne!(m.slot(&ctx, b"new york c"), m.slot(&ctx, b"old york c"));
        // Distinct word splits of the same letters stay distinct.
        assert_ne!(m.slot(&ctx, b"a bc d"), m.slot(&ctx, b"ab c d"));
    }

    /// Deterministic (encode/decode must agree).
    #[test]
    fn is_deterministic() {
        let m = PrevWord2Model::new(1 << 16);
        let ctx = Context::new();
        assert_eq!(m.slot(&ctx, b"united states of"), m.slot(&ctx, b"united states of"));
    }

    /// A run of ones at an active context raises its prediction while an abstaining position stays
    /// neutral and does not panic on update.
    #[test]
    fn active_context_learns_and_abstain_is_neutral() {
        let mut m = PrevWord2Model::new(1 << 16);
        let ctx = Context::new();
        let hist = b"new york c";
        let before = m.predict(&ctx, hist);
        for _ in 0..64 {
            let _ = m.predict(&ctx, hist);
            m.update(&ctx, hist, 1);
        }
        let after = m.predict(&ctx, hist);
        assert!(after > before, "a run of ones should raise the prediction: before={before} after={after}");
        assert_eq!(m.predict(&ctx, b"york c"), 0); // one previous word => abstain, neutral
        m.update(&ctx, b"york c", 1);
    }
}
