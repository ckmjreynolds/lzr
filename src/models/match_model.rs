//! Byte-level match model (lpaq-style): predicts long recurrences the order-N context models miss.
//!
//! It maintains its own growing buffer `r` of every finalized coded byte, hashes the last
//! [`HASH_LEN`] bytes into a table, and when the current context matches an earlier occurrence it
//! predicts the next byte from that position. While a match holds it predicts each of the next byte's
//! bits from the matched byte, with a confidence that grows with the match length; when the actual
//! byte diverges the match drops and the model abstains until it re-acquires one.
//!
//! It is a pure function of the finalized coded bytes and the framed byte count (which sizes the hash
//! table), so encode and decode stay in lock-step. It only contributes a probability to the mixer, so
//! a bug degrades ratio, never correctness — the range coder and the container's Adler-32 are the
//! reversibility backstop. It never panics on arbitrary input.

use super::statemap::SlotMap;
use super::{Context, Model, byte_hash, hashed_bits};

/// Number of most-recent bytes hashed to seed / re-acquire a match. Deliberately *long*: the order-N
/// models already predict short recurrences well, so the match model earns its keep only on long
/// repeats they miss. A longer seed also makes a raw hash hit far likelier to be a real match than a
/// collision.
const HASH_LEN: usize = 4;

/// Cap on the backward scan that verifies a hash hit and measures its true length. A true hit extends
/// back at least [`HASH_LEN`]; a hash collision falls short and is rejected. Capped because the
/// confidence bucket saturates at [`LEN_BUCKET_MAX`] anyway, so scanning further buys nothing.
const MAX_EXTEND: usize = 64;

/// Ceiling on the tracked match length, so the running counter cannot overflow on a pathological
/// self-similar stream. Far above the length at which the confidence bucket saturates.
const LEN_CAP: u32 = 0xFFFF;

/// Largest match-length bucket fed to the `StateMap` (6 bits). Longer matches all share this bucket —
/// past it the prediction is already near-certain, so finer buckets buy nothing.
const LEN_BUCKET_MAX: u32 = 63;

/// Sentinel for an empty hash-table slot (no real position equals it: positions are `< r.len()`, and
/// the model disables itself long before `r` could reach `u32::MAX`).
const NONE: u32 = u32::MAX;

/// Defensive cap on `r`'s length. Positions are stored as `u32`, so once `r` would exceed this the
/// model stops indexing rather than overflowing. The entropy decoder's decompression-bomb guard keeps
/// a real stream far below this.
const MAX_R_BYTES: u64 = 1 << 31;

/// Predicts each bit from a byte-level match over the coded stream seen so far.
#[derive(Debug)]
pub(crate) struct MatchModel {
    /// Probability per `(match-length bucket, predicted bit, bit-tree node)` context; abstains with no
    /// active match.
    map: SlotMap,
    /// Every finalized coded byte seen so far.
    r: Vec<u8>,
    /// `hash(last HASH_LEN bytes)` -> the position that followed that context, for match re-acquire.
    ht: Vec<u32>,
    /// Right-shift folding the multiplicative context hash down to `ht`'s index width.
    ht_shift: u32,
    /// Index in `r` of the byte the model predicts will come next (valid only while `mlen > 0`).
    ptr: usize,
    /// Current match length in bytes (`0` = no active match), the `StateMap`'s confidence bucket.
    mlen: u32,
    /// Set once `r` would exceed [`MAX_R_BYTES`]; the model then abstains and stops indexing.
    disabled: bool,
}

impl MatchModel {
    /// A fresh match model. `capacity` (the framed byte count) sizes the hash table via [`hashed_bits`]
    /// so encode and decode agree.
    pub(crate) fn new(capacity: usize) -> Self {
        let bits = hashed_bits(capacity);
        Self {
            // (LEN_BUCKET_MAX + 1) buckets * 2 (predicted bit) * 256 (node) = 1 << 15 states.
            map: SlotMap::new(1 << 15),
            // Pre-size the byte buffer to the framed count (capped at MAX_R_BYTES, the point the model
            // disables itself) so it fills without repeated doubling reallocations — which, at GB scale,
            // spike peak RSS via the allocate-and-copy transient. `with_capacity` reserves address space
            // only; pages fault in as bytes are appended, so an over-large (corrupt) capacity costs no
            // RSS beyond what is actually decoded.
            r: Vec::with_capacity(capacity.min(usize::try_from(MAX_R_BYTES).unwrap_or(usize::MAX))),
            ht: vec![NONE; 1 << bits],
            ht_shift: u64::BITS - bits,
            ptr: 0,
            mlen: 0,
            disabled: false,
        }
    }

    /// The `StateMap` slot for the current bit, or `None` to abstain: abstains unless a byte-level match
    /// is active *and* the matched byte still agrees with the bits coded so far this byte.
    fn slot(&self, ctx: &Context) -> Option<usize> {
        if self.disabled || self.mlen == 0 || self.ptr >= self.r.len() {
            return None;
        }
        // Predict the matched byte's next bit, abstaining if the match has broken mid-byte; the
        // confidence bucket is the (saturated) match length.
        let bucket = self.mlen.min(LEN_BUCKET_MAX) as usize;
        ctx.predicted_bit_slot(self.r[self.ptr], bucket)
    }

    /// Append one finalized coded byte to `r`, advancing the byte-level match: extend the active match
    /// if it predicted this byte, else re-acquire one from the hash table; then index the new context.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`r.len()` is bounded to < MAX_R_BYTES (2^31), and `ext` to MAX_EXTEND (64), so the \
                  usize->u32 casts are exact."
    )]
    fn append_byte(&mut self, b: u8) {
        if self.disabled {
            return;
        }
        let i = self.r.len();
        // Verify the standing prediction: if `b` is what the match foretold, extend it; else drop it.
        if self.mlen > 0 && self.ptr < i && self.r[self.ptr] == b {
            self.ptr += 1;
            self.mlen = (self.mlen + 1).min(LEN_CAP);
        } else {
            self.mlen = 0;
        }
        self.r.push(b);
        if self.r.len() as u64 > MAX_R_BYTES {
            self.disabled = true;
            return;
        }
        let len = self.r.len();
        if len >= HASH_LEN {
            let h = self.hash_ctx(len);
            // Only re-acquire when the standing match has broken; a live (longer) match is kept.
            if self.mlen == 0 {
                let cand = self.ht[h];
                if cand != NONE {
                    let cand = cand as usize;
                    // A stored slot points at the byte that followed an earlier occurrence of this
                    // context — the byte to predict next. Verify the hit and measure its true length by
                    // extending the match backwards; a hash collision falls short of HASH_LEN and is
                    // dropped. Seeding `mlen` with the real length gives the confidence bucket a true
                    // reading immediately instead of a pessimistic `HASH_LEN`.
                    let ext = self.back_extend(cand, len);
                    if ext >= HASH_LEN {
                        self.ptr = cand;
                        self.mlen = ext as u32;
                    }
                }
            }
            self.ht[h] = len as u32;
        }
    }

    /// The number of bytes for which the two suffixes ending just before `a` and `b` agree, capped at
    /// [`MAX_EXTEND`]. Used to both verify a hash hit (a real match reaches [`HASH_LEN`]) and read off
    /// its length. `a < b`, and both are `>= HASH_LEN`, so the backward reads stay in bounds.
    fn back_extend(&self, a: usize, b: usize) -> usize {
        let max = MAX_EXTEND.min(a);
        let mut k = 0;
        while k < max && self.r[a - 1 - k] == self.r[b - 1 - k] {
            k += 1;
        }
        k
    }

    /// The hash-table index for the [`HASH_LEN`]-byte context ending at `r[len - 1]` (caller guarantees
    /// `len >= HASH_LEN`).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`ht_shift` leaves at most MAX_HASH_BITS (28) bits, which fits usize on every target."
    )]
    fn hash_ctx(&self, len: usize) -> usize {
        let h = byte_hash(0, self.r[len - HASH_LEN..len].iter().copied());
        (h >> self.ht_shift) as usize
    }
}

impl Model for MatchModel {
    fn predict(&mut self, ctx: &Context, _hist: &[u8]) -> i32 {
        let slot = self.slot(ctx);
        self.map.predict(slot)
    }

    fn update(&mut self, ctx: &Context, _hist: &[u8], bit: u8) {
        self.map.update(bit);
        // On the byte's last bit, recover the finalized byte from the bit-tree node (which the driver
        // has not yet advanced) and fold it into the match buffer.
        if ctx.bpos == 7 {
            self.append_byte(ctx.completed_byte(bit));
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::models::SYMBOL_BITS;

    /// Drive `coded` through the model one bit at a time exactly as the entropy driver does.
    fn run(model: &mut MatchModel, coded: &[u8]) {
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

    /// After a long repeated passage the model holds an active match and predicts the recurring byte.
    #[test]
    fn acquires_match_on_long_repeat() {
        let pattern: Vec<u8> = b"the quick brown fox jumps over the lazy dog. ".repeat(8);
        let mut model = MatchModel::new(pattern.len());
        run(&mut model, &pattern);
        // Feed the pattern's prefix again; a match should be active partway through.
        run(&mut model, b"the quick brown fox");
        assert!(model.mlen > 0, "expected an active match after a long repeat");
        // At the start of the next byte the model must offer a prediction slot.
        let ctx = Context::new();
        assert!(model.slot(&ctx).is_some());
    }

    /// The model abstains on a stream with no recurrence.
    #[test]
    fn abstains_without_recurrence() {
        let mut model = MatchModel::new(4);
        run(&mut model, &[1, 2, 3]);
        assert_eq!(model.mlen, 0);
        assert!(model.slot(&Context::new()).is_none());
    }

    /// Never panics on arbitrary/adversarial bytes.
    #[test]
    fn survives_adversarial_input() {
        for bytes in [vec![0xffu8; 64], vec![0u8; 1], (0u8..=255).collect::<Vec<_>>()] {
            let mut model = MatchModel::new(bytes.len());
            run(&mut model, &bytes); // must not panic
        }
    }
}
