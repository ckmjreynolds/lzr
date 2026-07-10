//! ID-delta model: predicts the digits of the number currently being emitted from the **previously
//! completed number**, exploiting the near-monotonic id/timestamp sequences in `MediaWiki` dumps.
//!
//! Consecutive `<id>` / `<revision><id>` / `<timestamp>` values share a long high-order prefix
//! (`12345670` → `12345671`), so once a digit of the current number has been seen, the next digit is
//! very often the digit at the same position in the previous number. This is a *cross-token* structure
//! that neither the order-N context nor the trailing-digit run model can see — the previous number is
//! usually thousands of bytes back (a whole page), far outside any history window — so the model keeps
//! the previous number in its own state, updated as bytes finalize.
//!
//! Mechanics mirror the byte-level [`super::match_model::MatchModel`]: while a match (here, the
//! predicted digit) holds and still agrees with the bits coded so far this byte, it predicts each bit
//! from the predicted digit, with a confidence bucket that grows with the position in the number. It
//! abstains at the first digit of a number (no continuation context yet) and off digit runs entirely,
//! so it never competes on non-numeric bytes. It is a pure function of the finalized bytes, so encode
//! and decode stay in lock-step; it only feeds the mixer, so a bug degrades ratio, never correctness.

use super::statemap::StateMap;
use super::{Context, Model};

/// Cap on the digits tracked per number, bounding the state buffers on a pathological all-digit
/// stream. Well past any real id/timestamp length.
const MAX_DIGITS: usize = 32;

/// Largest position bucket fed to the `StateMap` (5 bits): digit index within the current number,
/// saturating here. Numbers rarely run past this, and past it the prediction is already well
/// characterised.
const POS_BUCKET_MAX: usize = 31;

/// Predicts each bit of the current number's digits from the previous number's digit at the same
/// position.
#[derive(Debug)]
pub(crate) struct IdDeltaModel {
    /// Probability per `(position bucket, predicted bit, bit-tree node)` context.
    sm: StateMap,
    /// Slot chosen by the last [`IdDeltaModel::predict`], reused by the paired update; `None` when the
    /// model abstained (no active digit prediction).
    idx: Option<usize>,
    /// Digits (ASCII `'0'..='9'`) of the last completed number — the prediction source.
    prev: Vec<u8>,
    /// Digits of the number currently being emitted; its length is the position of the next digit.
    cur: Vec<u8>,
}

impl IdDeltaModel {
    /// A fresh model. Its `StateMap` is a small fixed table — `(POS_BUCKET_MAX + 1) × 2 × 256` — so it
    /// needs no capacity sizing.
    pub(crate) fn new() -> Self {
        Self {
            sm: StateMap::new((POS_BUCKET_MAX + 1) * 2 * 256),
            idx: None,
            prev: Vec::new(),
            cur: Vec::new(),
        }
    }

    /// The predicted ASCII digit for the current position, or `None` to abstain: the previous number's
    /// digit at position `cur.len()`. Abstains at the first digit of a number (`cur` empty, no
    /// continuation context) and when the previous number is shorter than the current position.
    fn predicted_digit(&self) -> Option<u8> {
        if self.cur.is_empty() {
            return None;
        }
        self.prev.get(self.cur.len()).copied()
    }

    /// The `StateMap` slot for the current bit, or `None` to abstain — active only while the predicted
    /// digit still agrees with the bits coded so far this byte (mirrors the match model's mid-byte
    /// agreement check).
    fn slot(&self, ctx: &Context) -> Option<usize> {
        let predicted = u32::from(self.predicted_digit()?);
        let bpos = u32::from(ctx.bpos);
        let coded = ctx.c0 & ((1 << bpos) - 1);
        if predicted >> (8 - bpos) != coded {
            return None;
        }
        let predicted_bit = (predicted >> (7 - bpos)) & 1;
        let bucket = self.cur.len().min(POS_BUCKET_MAX);
        Some((bucket << 9) | ((predicted_bit as usize) << 8) | (ctx.c0 & 0xff) as usize)
    }

    /// Fold one finalized byte into the number state: extend the current number on a digit, or close it
    /// out (becoming the new `prev`) on any non-digit.
    fn append_byte(&mut self, b: u8) {
        if b.is_ascii_digit() {
            if self.cur.len() < MAX_DIGITS {
                self.cur.push(b);
            }
        } else if !self.cur.is_empty() {
            // The current number just ended: it becomes the prediction source for the next one.
            self.prev = std::mem::take(&mut self.cur);
        }
    }
}

impl Model for IdDeltaModel {
    fn predict(&mut self, ctx: &Context, _hist: &[u8]) -> i32 {
        self.idx = self.slot(ctx);
        self.idx.map_or(0, |idx| self.sm.predict(idx))
    }

    fn update(&mut self, ctx: &Context, _hist: &[u8], bit: u8) {
        if let Some(idx) = self.idx {
            self.sm.update(idx, bit);
        }
        // On the byte's last bit, reconstruct the finalized byte (as the match model does) and fold it
        // into the number state.
        if ctx.bpos == 7 {
            let byte = (((ctx.c0 << 1) | u32::from(bit)) & 0xff) as u8;
            self.append_byte(byte);
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::models::SYMBOL_BITS;

    /// Drive `coded` through the model one bit at a time exactly as the entropy driver does.
    fn run(model: &mut IdDeltaModel, coded: &[u8]) {
        let mut ctx = Context::new();
        for &byte in coded {
            for k in (0..SYMBOL_BITS).rev() {
                let bit = (byte >> k) & 1;
                let _ = model.predict(&ctx, &[]);
                model.update(&ctx, &[], bit);
                ctx.push_bit(bit);
            }
            ctx.push_symbol();
        }
    }

    /// After one number, the model predicts the shared prefix of the next number of the same length.
    #[test]
    fn predicts_continuation_from_previous_number() {
        let mut m = IdDeltaModel::new();
        run(&mut m, b"12345670 "); // first number completes; ' ' closes it
        assert_eq!(m.prev, b"12345670");
        // Start the next number: after the first digit, position 1 predicts prev[1] = '2'.
        run(&mut m, b"1");
        assert_eq!(m.cur, b"1");
        let ctx = Context::new();
        assert_eq!(m.predicted_digit(), Some(b'2'));
        assert!(m.slot(&ctx).is_some());
    }

    /// Abstains at the first digit of a number (no continuation context) and on non-digit bytes.
    #[test]
    fn abstains_at_number_start_and_off_digits() {
        let mut m = IdDeltaModel::new();
        run(&mut m, b"42 ");
        // Fresh number, no digit emitted yet: cur empty => abstain.
        assert_eq!(m.predicted_digit(), None);
        // A number longer than prev: once past prev's length, no source digit => abstain.
        run(&mut m, b"999");
        assert_eq!(m.predicted_digit(), None); // cur.len()=3 > prev.len()=2
    }

    /// Never panics on arbitrary/adversarial bytes.
    #[test]
    fn survives_adversarial_input() {
        for bytes in [vec![b'9'; 100], vec![0u8; 8], (0u8..=255).collect::<Vec<_>>()] {
            let mut m = IdDeltaModel::new();
            run(&mut m, &bytes);
        }
    }
}
