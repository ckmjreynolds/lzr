//! Generic context model: bit-history states + a `StateMap`.
//!
//! Keyed on either the last `n` bytes (order-`n`) or the current word's spelling
//! hash — the bit-history and `StateMap` machinery is shared; only the context
//! value differs.
//!
//! For each context (the last `order` finalized bytes) and each bit-tree node
//! (the partial byte `c0`), a one-`u16` cell holds a *bit history* — a bounded,
//! nonstationary `(n0, n1)` count pair. A per-model [`StateMap`] maps that state
//! to a calibrated probability. The split matters: the per-cell state adapts
//! locally and fast, while the `StateMap` pools statistics across every cell
//! that reaches the same state, so even a context seen once predicts well (its
//! state is "one 1 observed", whose probability the `StateMap` learned
//! globally). This is the `lpaq`/cmix design, and strictly richer than a single
//! per-cell probability (which discards the confidence the count carries).
//!
//! Indexing is direct for dense low orders (≤ 2: every context is used, no
//! collisions) and hashed into a fixed table for sparse high orders (≥ 3, where
//! a full `256^order` array is infeasible). Same code, chosen by `order`.

use super::statemap::StateMap;
use super::{Context, Model};

const MAX: u16 = 63; // count cap; with the 6-bit packing this bounds a state to 12 bits
const HASH_BITS: u32 = 28; // CAP on the hashed-table size (orders ≥ 3 / word / sparse).
// The actual size adapts to the input (`hashed_bits`) AND to the model's context
// space (`hashed`), so small inputs/tests stay tiny and a model is never given a
// table larger than the distinct contexts it can produce. At the cap: 256M cells
// × 2 B = 512 MB/model — but only the orders and word reach it; the sparse models
// (2–3 byte contexts) are bounded far below, keeping enwik9 RSS ~6.4 GB.
const SM_STATES: usize = 1 << 12; // (n0 << 6) | n1, each ≤ 63

/// One observed bit moves the cell to its next bit-history state: bump the seen
/// count (capped) and discount the opposite count so the pair tracks drift.
fn transition(s: u16, bit: u8) -> u16 {
    let n0 = s >> 6;
    let n1 = s & 63;
    let (n0, n1) = if bit == 1 {
        (discount(n0), (n1 + 1).min(MAX))
    } else {
        ((n0 + 1).min(MAX), discount(n1))
    };
    (n0 << 6) | n1
}

/// Soft discount of the opposite count when a bit flips the recent trend.
const fn discount(x: u16) -> u16 {
    if x > 3 { 3 + ((x - 3) >> 1) } else { x }
}

/// How a [`ContextModel`] derives its context value each bit.
#[derive(Debug, Clone, Copy)]
enum CtxKind {
    /// The last `n` finalized bytes.
    Order(usize),
    /// Hash of the current word's letters so far (spelling, variable length).
    Word,
    /// A non-contiguous set of finalized bytes: `mask` bit `i` selects
    /// `byte_back(i + 1)`. Captures skip-gram dependencies the contiguous orders
    /// miss (e.g. context one step removed from a noisy delimiter byte).
    Sparse(u32),
    /// Digit-run context: the field byte that preceded the run, the run-relative
    /// position, and the last digit — signal the fixed-offset orders lack (a
    /// digit's distribution depends on which field it's in and how far into the
    /// number it is, e.g. timestamp/id positions).
    Number,
}

/// A context model over bit-history states. The context is either the last `n`
/// bytes (orders ≤ 2 direct-indexed, ≥ 3 hashed) or a word-derived hash.
///
/// Hashed cells carry a **4-bit confirm tag** packed into the spare top bits of
/// the `u16` (the bit-history state needs only 12 bits, `(n0<<6)|n1 ≤ 4095`). On
/// a tag mismatch the slot is treated as fresh rather than inheriting a colliding
/// context's history — converting silent corruption into a clean eviction at
/// zero extra memory. Direct (order ≤ 2) cells never collide and ignore the tag.
#[derive(Debug)]
pub(crate) struct ContextModel {
    kind: CtxKind,
    direct: bool,
    shift: u32,
    cells: Vec<u16>,
    sm: StateMap,
    idx: usize,
    check: u16, // expected 4-bit tag for the current hashed slot
    cv: u64,    // context value cached at bpos==0 (byte-constant; only c0 varies per bit)
}

/// Hashed-table size (in bits) for a corpus of `capacity` preprocessed bytes:
/// roughly 2× the distinct-context estimate (one context per position), capped
/// at [`HASH_BITS`] and floored for tiny inputs. Both directions derive it from
/// the same `capacity` (carried in the stream's length prefix), so it
/// round-trips; small inputs and tests get small tables instead of allocating
/// hundreds of MB per model.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn hashed_bits(capacity: usize) -> u32 {
    let want = (capacity.max(1) as u64)
        .next_power_of_two()
        .trailing_zeros()
        + 1;
    want.clamp(12, HASH_BITS)
}

impl ContextModel {
    /// Keyed on the last `order` finalized bytes. Orders ≤ 2 are directly
    /// indexed (exact); higher orders hash into an input-sized table.
    pub(crate) fn new(order: usize, capacity: usize) -> Self {
        if order <= 2 {
            return Self {
                kind: CtxKind::Order(order),
                direct: true,
                shift: 0,
                cells: vec![0; 1usize << (8 * order + 8)],
                sm: StateMap::new(SM_STATES),
                idx: 0,
                check: 0,
                cv: 0,
            };
        }
        Self::hashed(CtxKind::Order(order), capacity)
    }

    /// A hashed-table model sized to the input (word/sparse/high-order contexts
    /// are sparse). A sparse model's distinct contexts are bounded by its byte
    /// count (`8 * bytes + 8` for the appended `c0`), so it is never allocated a
    /// table larger than that — a 2-byte sparse context fits exactly in 2^24
    /// cells, freeing RAM for the high orders to reach the full [`HASH_BITS`] cap.
    fn hashed(kind: CtxKind, capacity: usize) -> Self {
        let bits = match kind {
            CtxKind::Sparse(mask) => hashed_bits(capacity).min(8 * mask.count_ones() + 8),
            CtxKind::Number => hashed_bits(capacity).min(22), // small context space
            _ => hashed_bits(capacity),
        };
        Self {
            kind,
            direct: false,
            shift: 64 - bits,
            cells: vec![0; 1usize << bits],
            sm: StateMap::new(SM_STATES),
            idx: 0,
            check: 0,
            cv: 0,
        }
    }

    /// Keyed on the current word's letters so far.
    pub(crate) fn word(capacity: usize) -> Self {
        Self::hashed(CtxKind::Word, capacity)
    }

    /// Keyed on a non-contiguous set of recent bytes (`mask` bit `i` →
    /// `byte_back(i + 1)`), hashed like the high orders.
    pub(crate) fn sparse(mask: u32, capacity: usize) -> Self {
        Self::hashed(CtxKind::Sparse(mask), capacity)
    }

    /// Keyed on the digit-run context (field byte, run position, last digit).
    pub(crate) fn number(capacity: usize) -> Self {
        Self::hashed(CtxKind::Number, capacity)
    }

    fn context_value(&self, ctx: &Context) -> u64 {
        match self.kind {
            CtxKind::Order(order) => {
                let mut cv = 0u64;
                for i in 1..=order {
                    cv = (cv << 8) | u64::from(ctx.byte_back(i));
                }
                cv
            }
            CtxKind::Word => ctx.word_hash,
            CtxKind::Sparse(mask) => {
                // Seed with the mask so different sparse models don't collide in
                // the shared hash space when they read the same bytes.
                let mut cv = u64::from(mask);
                let mut m = mask;
                while m != 0 {
                    let i = m.trailing_zeros() as usize + 1;
                    cv = (cv << 8) | u64::from(ctx.byte_back(i));
                    m &= m - 1;
                }
                cv
            }
            CtxKind::Number => {
                (u64::from(ctx.num_field) << 16)
                    | (u64::from(ctx.num_pos.min(31)) << 8)
                    | u64::from(ctx.byte_back(1))
            }
        }
    }
}

const STATE_MASK: u16 = 0x0FFF; // bit-history state occupies the low 12 bits
const HASH1: u64 = 0x9E37_79B9_7F4A_7C15; // index hash
const HASH2: u64 = 0xD1B5_4A32_D192_ED03; // independent tag hash (4-bit confirm)

impl Model for ContextModel {
    #[allow(clippy::cast_possible_truncation)]
    fn predict(&mut self, ctx: &Context) -> i32 {
        // context_value reads only byte-constant state (finalized history,
        // word_hash, num_*), never c0/bpos — so it is identical across the 8
        // bits of a byte. Compute it once at bpos==0 and reuse; only the c0 term
        // below varies per bit. (Same per-byte caching IndirectModel already uses.)
        if ctx.bpos == 0 {
            self.cv = self.context_value(ctx);
        }
        let raw = (self.cv << 8) | u64::from(ctx.c0 & 0xff);
        if self.direct {
            self.idx = raw as usize;
            return self.sm.predict(self.cells[self.idx] as usize);
        }
        self.idx = (raw.wrapping_mul(HASH1) >> self.shift) as usize;
        self.check = (raw.wrapping_mul(HASH2) >> 60) as u16;
        let cell = self.cells[self.idx];
        let state = if cell >> 12 == self.check {
            cell & STATE_MASK
        } else {
            0
        };
        self.sm.predict(state as usize)
    }

    fn update(&mut self, _ctx: &Context, bit: u8) {
        if self.direct {
            let s = self.cells[self.idx];
            self.sm.update(s as usize, bit);
            self.cells[self.idx] = transition(s, bit);
            return;
        }
        let cell = self.cells[self.idx];
        let state = if cell >> 12 == self.check {
            cell & STATE_MASK
        } else {
            0
        };
        self.sm.update(state as usize, bit);
        self.cells[self.idx] = (self.check << 12) | transition(state, bit);
    }
}
