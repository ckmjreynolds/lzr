//! Bit-prediction models and the shared [`Context`] they read.
//!
//! A model is queried once per bit ([`Model::predict`]) and then told the actual
//! bit ([`Model::update`]). Predictions are in the stretched (logit) domain so
//! the mixer can combine them directly.

pub(crate) mod context;
pub(crate) mod finder;
#[cfg(test)]
mod lstm_arm;
#[cfg(test)]
mod lstm_spike;
pub(crate) mod match_model;
pub(crate) mod statemap;

/// Mutable per-stream prediction context shared by every model.
///
/// Holds the full finalized-byte history (so long-range models like the match
/// model can read arbitrarily far back), plus the partial current byte and a
/// 4-byte rolling window the low-order context models index directly.
#[derive(Debug)]
pub(crate) struct Context {
    history: Vec<u8>,
    /// Partial current byte: a leading-1 sentinel followed by the bits coded so far.
    pub(crate) c0: u32,
    /// Number of bits of the current byte already coded (`0..=7`).
    pub(crate) bpos: u8,
    /// The last four finalized bytes; the most recent is in the low 8 bits.
    pub(crate) c4: u32,
    /// Rolling hash of the current word's letters so far (`0` between words).
    /// A "letter" is `a..=z` only: the stream is post-fold, so `A..=Z` never
    /// means a letter — those bytes are dictionary codes and act as boundaries.
    pub(crate) word_hash: u64,
}

/// Mixing multiplier for folding a letter into the rolling word hash.
const WORD_PRIME: u64 = 0x0100_0000_01b3;

impl Context {
    /// New, empty context with room reserved for `capacity` finalized bytes.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            history: Vec::with_capacity(capacity),
            c0: 1,
            bpos: 0,
            c4: 0,
            word_hash: 0,
        }
    }

    /// The finalized byte `i` positions back (`i` ≥ 1); `0` before that much
    /// history exists.
    pub(crate) fn byte_back(&self, i: usize) -> u8 {
        let n = self.history.len();
        if i <= n { self.history[n - i] } else { 0 }
    }

    /// All finalized bytes so far, in order. The most recent is the last element.
    pub(crate) fn history(&self) -> &[u8] {
        &self.history
    }

    /// Append one freshly-coded bit to the partial current byte.
    pub(crate) fn push_bit(&mut self, bit: u8) {
        self.c0 = (self.c0 << 1) | u32::from(bit);
        self.bpos += 1;
    }

    /// Finalize the current byte once all 8 bits are in, and reset for the next.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn push_byte(&mut self) {
        let b = self.c0 as u8; // low 8 bits are the byte; the sentinel is bit 8
        self.history.push(b);
        self.c4 = (self.c4 << 8) | u32::from(b);
        if b.is_ascii_lowercase() {
            self.word_hash = self
                .word_hash
                .wrapping_mul(WORD_PRIME)
                .wrapping_add(u64::from(b) + 1);
        } else {
            self.word_hash = 0;
        }
        self.c0 = 1;
        self.bpos = 0;
    }
}

/// A model that predicts the next bit from the [`Context`].
pub(crate) trait Model {
    /// Predict P(next bit == 1) in the stretched (logit) domain, clamped to
    /// roughly `[-2047, 2047]`. Called before the bit is known.
    fn predict(&mut self, ctx: &Context) -> i32;
    /// Observe the actual `bit`. `ctx` still reflects the pre-bit state.
    fn update(&mut self, ctx: &Context, bit: u8);
}
