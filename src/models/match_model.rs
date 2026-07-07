//! LZ77-aware match model: predicts recurring content *through* the LZ77 encoding.
//!
//! The stream feeding the entropy coder is the LZ77 stage's output — a leading mode byte followed by
//! literal Re-Pair u22 tokens interleaved with `0x00`-introduced match records (see
//! [`crate::preprocessors::lz77`]). The token-value models already predict a recurring *literal* run
//! well, but a passage that LZ77 folded into a match record does **not** recur bit-for-bit: the
//! record's `dist` field differs between two occurrences, so the coded bytes diverge even though the
//! underlying content is identical.
//!
//! This model closes that gap. It maintains a private, incremental replay of [`crate::preprocessors::lz77`]'s
//! `inverse` — reconstructing the *pre-LZ77* byte stream `R` from the finalized coded bytes both the
//! encoder and decoder possess — and runs a classic (lpaq-style) byte-level match model over `R`.
//! Because `R` is the LZ77-*expanded* stream, the same passage is byte-identical at both occurrences
//! regardless of how LZ77 encoded it, so the match finder sees the repeat. When it holds an active
//! match it predicts the next coded byte's bits from the matched position; it abstains during a
//! record's control bytes (the `0x00` marker and the `len`/`dist` varints), which are not part of `R`.
//!
//! It is a pure function of the finalized coded bytes and the framed byte count (which sizes the hash
//! table), so encode and decode stay in lock-step. It only contributes a probability to the mixer, so
//! a bug degrades ratio, never correctness — the range coder and the container's Adler-32 are the
//! reversibility backstop. On a truncated or non-canonical stream the replay simply disables itself
//! (predicting the neutral logit thereafter) rather than panicking.

use super::statemap::StateMap;
use super::{Context, TokenModel, hashed_bits, token_hash};
use crate::uleb128::decode_u22;

/// Mode byte marking a folded LZ77 payload (`crate::preprocessors::MODE_FOLDED`, redeclared here since
/// that constant is private to the preprocessors module). Any other leading byte is treated as an
/// unfolded literal stream (a raw Re-Pair stream, or a non-LZ77 input), which has no match records.
const MODE_FOLDED: u8 = 1;

/// Number of most-recent `R` bytes hashed to seed / re-acquire a match. Deliberately *long*: the
/// order-N and varint models already predict short recurrences well, so the match model earns its keep
/// only on long repeats they miss. A longer seed also makes a raw hash hit far likelier to be a real
/// match than a collision.
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

/// Sentinel for an empty hash-table slot (no real `R` position equals it: positions are `< R.len()`,
/// and the replay disables itself long before `R` could reach `u32::MAX`).
const NONE: u32 = u32::MAX;

/// Sliding-window size (power of two, one past LZ77's `MAX_DISTANCE`) for the token-offset ring,
/// matching [`crate::preprocessors::lz77`] so every in-range back-reference resolves.
const WINDOW: usize = 1 << 22;

/// Defensive cap on the reconstructed stream's length, mirroring `crate::preprocessors::MAX_EXPANSION_BYTES`.
/// A malformed record can name a huge `len`; once `R` would exceed this the replay disables itself. The
/// real decode path enforces the same bound, so this never trips on a legitimate stream.
const MAX_R_BYTES: u64 = 1 << 31;

/// Where the incremental replay expects the next finalized coded byte to fit in the LZ77 grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The very first byte: the LZ77 mode byte.
    Mode,
    /// The start of a token slot: a literal token's first varint byte, or (when folded) a `0x00`
    /// match-record marker.
    TokenStart,
    /// A continuation byte of the literal varint in progress.
    LitCont,
    /// The first varint after a `0x00` marker: `0` escapes a literal value-0 token, `>= 1` is a match length.
    RecFirst,
    /// The distance varint of a match record.
    RecDist,
}

/// Predicts each bit from a byte-level match over the reconstructed pre-LZ77 stream.
#[derive(Debug)]
pub(crate) struct MatchModel {
    // --- adaptive prediction ---
    /// Probability per `(match-length bucket, predicted bit, bit-tree node)` context.
    sm: StateMap,
    /// Slot chosen by the last [`MatchModel::predict`], reused by the paired [`MatchModel::update`];
    /// `None` when the model abstained (no active match / a control byte) so `update` leaves `sm` alone.
    idx: Option<usize>,

    // --- reconstructed stream `R` and the match finder over it ---
    /// The reconstructed pre-LZ77 bytes.
    r: Vec<u8>,
    /// `hash(last HASH_LEN bytes of R)` -> the position that followed that context, for match re-acquire.
    ht: Vec<u32>,
    /// Right-shift folding the multiplicative context hash down to `ht`'s index width.
    ht_shift: u32,
    /// Index in `r` of the byte the model predicts will come next (valid only while `mlen > 0`).
    ptr: usize,
    /// Current match length in bytes (`0` = no active match), the `StateMap`'s confidence bucket.
    mlen: u32,

    // --- incremental LZ77 replay state (mirrors lz77::inverse) ---
    /// Ring of each recent token's start offset in `r`, sized [`WINDOW`]; resolves record back-references.
    tok_off: Vec<u32>,
    /// Tokens materialized into `r` so far.
    t: usize,
    /// Grammar position of the next finalized coded byte.
    phase: Phase,
    /// Whether the stream is folded (mode byte `0x01`); an unfolded stream never emits `0x00` records.
    folded: bool,
    /// Bytes of the literal varint in progress (`1..=3`), for its termination check.
    lit_len: usize,
    /// Accumulator for a record's `len`/`dist` varint (at most 3 bytes).
    vbuf: [u8; 3],
    /// Bytes accumulated in `vbuf`.
    vlen: usize,
    /// Pending record length, held between the `RecFirst` and `RecDist` phases.
    rec_len: u32,
    /// Bits of the coded byte in progress, MSB-first; the finalized byte is fed to the replay at `bpos == 7`.
    partial: u32,
    /// Set once the replay hits truncated/non-canonical/over-long input; the model then abstains forever.
    disabled: bool,
}

impl MatchModel {
    /// A fresh match model. `capacity` (the framed byte count) sizes the hash table via [`hashed_bits`]
    /// so encode and decode agree.
    pub(crate) fn new(capacity: usize) -> Self {
        let bits = hashed_bits(capacity);
        Self {
            // (LEN_BUCKET_MAX + 1) buckets * 2 (predicted bit) * 256 (node) = 1 << 15 states.
            sm: StateMap::new(1 << 15),
            idx: None,
            r: Vec::new(),
            ht: vec![NONE; 1 << bits],
            ht_shift: u64::BITS - bits,
            ptr: 0,
            mlen: 0,
            tok_off: Vec::new(),
            t: 0,
            phase: Phase::Mode,
            folded: false,
            lit_len: 0,
            vbuf: [0; 3],
            vlen: 0,
            rec_len: 0,
            partial: 0,
            disabled: false,
        }
    }

    /// The `StateMap` slot for the current bit, or `None` to abstain: abstains unless a byte-level match
    /// is active *and* the coded byte being predicted is literal content (a token byte, not a record's
    /// control byte) *and* the matched byte still agrees with the bits coded so far this byte.
    fn slot(&self, ctx: &Context) -> Option<usize> {
        if self.disabled || self.mlen == 0 || self.ptr >= self.r.len() {
            return None;
        }
        // Only token-content bytes live in `r`. A record's marker / len / dist bytes do not, so there is
        // nothing to predict there. `TokenStart` is optimistically treated as literal (most tokens are),
        // eating the occasional miss when it turns out to be a `0x00` marker.
        if !matches!(self.phase, Phase::TokenStart | Phase::LitCont) {
            return None;
        }
        let expected = u32::from(self.r[self.ptr]);
        let bpos = u32::from(ctx.bpos);
        // Bits of the current byte coded so far (the low `bpos` bits of `c0`, below the sentinel).
        let coded = ctx.c0 & ((1 << bpos) - 1);
        // If the matched byte's leading bits no longer agree with what is actually being coded, the
        // match has broken mid-byte — abstain for the rest of it.
        if expected >> (8 - bpos) != coded {
            return None;
        }
        let predicted_bit = (expected >> (7 - bpos)) & 1;
        let bucket = self.mlen.min(LEN_BUCKET_MAX);
        Some(((bucket << 9) | (predicted_bit << 8) | (ctx.c0 & 0xff)) as usize)
    }

    /// Append one reconstructed byte to `r`, advancing the byte-level match: extend the active match if
    /// it predicted this byte, else re-acquire one from the hash table; then index the new context.
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
        reason = "`ht_shift` leaves at most MAX_HASH_BITS (22) bits, which fits usize on every target."
    )]
    fn hash_ctx(&self, len: usize) -> usize {
        let h = token_hash(0, self.r[len - HASH_LEN..len].iter().map(|&b| u32::from(b)));
        (h >> self.ht_shift) as usize
    }

    /// Record token `self.t`'s start offset, keeping only the most recent [`WINDOW`] entries (absolute
    /// indices while filling, then ring reuse) — the [`crate::preprocessors::lz77`] `push_off` scheme.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "`r.len()` is bounded to < MAX_R_BYTES (2^31), so the usize->u32 cast is exact."
    )]
    fn push_off(&mut self) {
        let off = self.r.len() as u32;
        if self.t < WINDOW {
            self.tok_off.push(off);
        } else {
            self.tok_off[self.t & (WINDOW - 1)] = off;
        }
    }

    /// Feed one finalized coded byte to the incremental LZ77 replay, advancing `r`/`tok_off`/`t` exactly
    /// as `lz77::inverse` would.
    fn feed_byte(&mut self, b: u8) {
        if self.disabled {
            return;
        }
        match self.phase {
            Phase::Mode => {
                self.folded = b == MODE_FOLDED;
                self.phase = Phase::TokenStart;
            }
            Phase::TokenStart => {
                if self.folded && b == 0x00 {
                    self.vlen = 0;
                    self.phase = Phase::RecFirst;
                } else {
                    self.push_off();
                    self.lit_len = 1;
                    self.append_byte(b);
                    self.phase = if b & 0x80 == 0 {
                        self.t += 1;
                        Phase::TokenStart
                    } else {
                        Phase::LitCont
                    };
                }
            }
            Phase::LitCont => {
                self.lit_len += 1;
                self.append_byte(b);
                if self.lit_len >= 3 || b & 0x80 == 0 {
                    self.t += 1;
                    self.phase = Phase::TokenStart;
                }
            }
            Phase::RecFirst => {
                if let Some(first) = self.push_varint(b) {
                    if first == 0 {
                        // Escaped literal value-0 token.
                        self.push_off();
                        self.append_byte(0x00);
                        self.t += 1;
                        self.phase = Phase::TokenStart;
                    } else {
                        self.rec_len = first;
                        self.vlen = 0;
                        self.phase = Phase::RecDist;
                    }
                }
            }
            Phase::RecDist => {
                if let Some(dist) = self.push_varint(b) {
                    self.expand(self.rec_len, dist);
                    self.phase = Phase::TokenStart;
                }
            }
        }
    }

    /// Accumulate one byte of a record varint; once the u22 form terminates, decode and return its
    /// value (disabling the model on a truncated/non-canonical/over-long encoding).
    fn push_varint(&mut self, b: u8) -> Option<u32> {
        self.vbuf[self.vlen] = b;
        self.vlen += 1;
        // u22 is 1..=3 bytes: byte 0/1 terminate when their high bit is clear; byte 2 always terminates.
        let complete = self.vlen == 3 || b & 0x80 == 0;
        if !complete {
            return None;
        }
        let mut pos = 0;
        if let Ok(v) = decode_u22(&self.vbuf[..self.vlen], &mut pos) {
            Some(v.value())
        } else {
            self.disabled = true;
            None
        }
    }

    /// Expand a match of `len` tokens starting `dist` tokens back, copying token-by-token into `r`
    /// (honoring overlap/RLE, since each source token is materialized before it is read) — the
    /// [`crate::preprocessors::lz77`] `inverse` match arm.
    fn expand(&mut self, len: u32, dist: u32) {
        let dist = dist as usize;
        if dist < 1 || dist > self.t {
            self.disabled = true;
            return;
        }
        let src = self.t - dist;
        for k in 0..len as usize {
            let s = self.tok_off[(src + k) & (WINDOW - 1)] as usize;
            // Measure the source token's byte width in `r`.
            let mut sp = s;
            if decode_u22(&self.r, &mut sp).is_err() {
                self.disabled = true;
                return;
            }
            self.push_off();
            for j in s..sp {
                let byte = self.r[j];
                self.append_byte(byte);
                if self.disabled {
                    return;
                }
            }
            self.t += 1;
        }
    }
}

impl TokenModel for MatchModel {
    fn predict(&mut self, ctx: &Context, _hist: &[u8]) -> i32 {
        self.idx = self.slot(ctx);
        self.idx.map_or(0, |idx| self.sm.predict(idx))
    }

    fn update(&mut self, ctx: &Context, _hist: &[u8], bit: u8) {
        if let Some(idx) = self.idx {
            self.sm.update(idx, bit);
        }
        // Rebuild the finalized coded byte MSB-first; on its last bit, drive the replay one byte forward.
        self.partial = (self.partial << 1) | u32::from(bit);
        if ctx.bpos == 7 {
            let byte = (self.partial & 0xff) as u8;
            self.partial = 0;
            self.feed_byte(byte);
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use arbitrary_int::u22;

    use super::*;
    use crate::models::SYMBOL_BITS;
    use crate::preprocessors::Lz77;
    use crate::transform::Transform;
    use crate::uleb128::encode_u22;

    /// Drive `coded` through the model one bit at a time exactly as the entropy driver does, returning
    /// the reconstructed stream the replay built.
    fn reconstruct(coded: &[u8]) -> Vec<u8> {
        let mut model = MatchModel::new(coded.len());
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
        model.r
    }

    /// A folded LZ77 stream over `tokens` and the model's replay of it must both reproduce the exact
    /// pre-LZ77 byte stream — the replay mirrors `lz77::inverse` byte-for-byte.
    fn assert_replay_matches(tokens: &[u32]) {
        let mut pre = Vec::new();
        for &tok in tokens {
            encode_u22(u22::new(tok), &mut pre);
        }
        let coded = Lz77 {
            min_match: Some(2),
        }
        .forward(pre.clone());
        // The reconstructed pre-LZ77 stream equals the original token bytes...
        assert_eq!(reconstruct(&coded), pre, "replay != original for {tokens:?}");
        // ...and equals what the real lz77 inverse produces from the same folded stream.
        let inverse = Lz77 {
            min_match: None,
        }
        .inverse(coded.clone())
        .expect("lz77 inverse");
        assert_eq!(reconstruct(&coded), inverse, "replay != lz77::inverse for {tokens:?}");
    }

    /// A repeated token run (which LZ77 folds into a match record) round-trips through the replay.
    #[test]
    fn replay_reconstructs_folded_repeat() {
        assert_replay_matches(&[10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]);
    }

    /// Multi-byte tokens (2- and 3-byte u22 varints) reconstruct correctly.
    #[test]
    fn replay_reconstructs_multibyte_tokens() {
        assert_replay_matches(&[1, 0x80, 0x3fff, 0x4000, 0x3f_ffff, 1, 0x80, 0x3fff, 0x4000, 0x3f_ffff]);
    }

    /// An overlapping (RLE) run — `dist < len` — reconstructs token-for-token.
    #[test]
    fn replay_reconstructs_rle() {
        assert_replay_matches(&[7, 7, 7, 7, 7, 7, 7, 7]);
    }

    /// The replay never panics on arbitrary/adversarial bytes and, when it cannot make sense of them,
    /// disables itself rather than corrupting state.
    #[test]
    fn replay_survives_adversarial_input() {
        for bytes in [
            vec![1, 0x00],                   // folded marker with no following varint
            vec![1, 0x00, 0x02, 0x05],       // match distance past the (empty) history
            vec![1, 0x80, 0x80],             // truncated 3-byte varint
            vec![1, 0x00, 0x80, 0x80, 0x00], // non-canonical record length
            vec![0xff; 64],                  // unknown mode byte, junk payload
        ] {
            drop(reconstruct(&bytes)); // must not panic
        }
    }

    /// A pass-through (unfolded) stream is reconstructed as its raw literal bytes.
    #[test]
    fn replay_reconstructs_passthrough() {
        let coded = [0u8, 5, 6, 7, 8]; // MODE_PASSTHROUGH then literal varints
        assert_eq!(reconstruct(&coded), vec![5, 6, 7, 8]);
    }
}
