//! Generic context model: bit-history states + a `StateMap`.
//!
//! Keyed on either the last `n` bytes (order-`n`) or a word-derived hash
//! (current-word spelling, or previous word + current prefix) — the bit-history
//! and `StateMap` machinery is shared; only the context value differs.
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
const HASH_BITS: u32 = 22; // hashed-table size for orders ≥ 3: 4M cells × 2 B = 8 MB
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
    /// Previous complete word combined with the current word's prefix.
    PrevWord,
}

/// A context model over bit-history states. The context is either the last `n`
/// bytes (orders ≤ 2 direct-indexed, ≥ 3 hashed) or a word-derived hash.
#[derive(Debug)]
pub(crate) struct ContextModel {
    kind: CtxKind,
    direct: bool,
    shift: u32,
    cells: Vec<u16>,
    sm: StateMap,
    idx: usize,
}

impl ContextModel {
    /// Keyed on the last `order` finalized bytes. Orders ≤ 2 are directly
    /// indexed (exact); higher orders hash into a fixed table.
    pub(crate) fn new(order: usize) -> Self {
        let (direct, shift, size) = if order <= 2 {
            (true, 0, 1usize << (8 * order + 8))
        } else {
            (false, 64 - HASH_BITS, 1usize << HASH_BITS)
        };
        Self {
            kind: CtxKind::Order(order),
            direct,
            shift,
            cells: vec![0; size],
            sm: StateMap::new(SM_STATES),
            idx: 0,
        }
    }

    /// A word-keyed model over a hashed table (word contexts are sparse).
    fn hashed(kind: CtxKind) -> Self {
        Self {
            kind,
            direct: false,
            shift: 64 - HASH_BITS,
            cells: vec![0; 1usize << HASH_BITS],
            sm: StateMap::new(SM_STATES),
            idx: 0,
        }
    }

    /// Keyed on the current word's letters so far.
    pub(crate) fn word() -> Self {
        Self::hashed(CtxKind::Word)
    }

    /// Keyed on the previous word together with the current word's prefix.
    pub(crate) fn prev_word() -> Self {
        Self::hashed(CtxKind::PrevWord)
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
            CtxKind::PrevWord => ctx.prev_word.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ctx.word_hash,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn slot(&self, ctx: &Context) -> usize {
        let raw = (self.context_value(ctx) << 8) | u64::from(ctx.c0 & 0xff);
        if self.direct {
            raw as usize
        } else {
            (raw.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
        }
    }
}

impl Model for ContextModel {
    fn predict(&mut self, ctx: &Context) -> i32 {
        self.idx = self.slot(ctx);
        self.sm.predict(self.cells[self.idx] as usize)
    }

    fn update(&mut self, _ctx: &Context, bit: u8) {
        let s = self.cells[self.idx];
        self.sm.update(s as usize, bit);
        self.cells[self.idx] = transition(s, bit);
    }
}
