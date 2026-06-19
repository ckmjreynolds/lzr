//! Generic order-`N` context model: bit-history states + a `StateMap`.
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

use super::{Context, Model};
use crate::mixer::stretch;

const MAX: u16 = 63; // count cap; with the 6-bit packing this bounds a state to 12 bits
const HASH_BITS: u32 = 22; // hashed-table size for orders ≥ 3: 4M cells × 2 B = 8 MB
const SM_STATES: usize = 1 << 12; // (n0 << 6) | n1, each ≤ 63
const SM_LIMIT: usize = 1023; // count cap in the StateMap's adaptive rate

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

/// Adaptive map from a bit-history state to a probability, calibrated globally
/// across all cells of one model. Probability is held in 16-bit precision; the
/// learning rate decays with the per-state observation count (with a floor, so
/// it stays responsive to nonstationarity).
#[derive(Debug)]
struct StateMap {
    p: Vec<i32>,  // 16-bit probability per state
    n: Vec<u16>,  // observation count per state (capped at SM_LIMIT)
    dt: Vec<i32>, // dt[k] = (1<<16) / (k + 2): the rate after k observations
}

impl StateMap {
    fn new() -> Self {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let dt = (0..=SM_LIMIT)
            .map(|k| (1i32 << 16) / (k as i32 + 2))
            .collect();
        Self {
            p: vec![1 << 15; SM_STATES],
            n: vec![0; SM_STATES],
            dt,
        }
    }

    fn predict(&self, s: usize) -> i32 {
        stretch(self.p[s] >> 4)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn update(&mut self, s: usize, bit: u8) {
        let target = i32::from(bit) * 65535;
        let rate = self.dt[(self.n[s] as usize).min(SM_LIMIT)];
        self.p[s] += ((i64::from(target - self.p[s]) * i64::from(rate)) >> 16) as i32;
        if (self.n[s] as usize) < SM_LIMIT {
            self.n[s] += 1;
        }
    }
}

/// Order-`N` context model over bit-history states.
#[derive(Debug)]
pub(crate) struct ContextModel {
    order: usize,
    direct: bool,
    shift: u32,
    cells: Vec<u16>,
    sm: StateMap,
    idx: usize,
}

impl ContextModel {
    /// New model keyed on the last `order` finalized bytes. Orders ≤ 2 are
    /// directly indexed (exact); higher orders hash into a fixed table.
    pub(crate) fn new(order: usize) -> Self {
        let (direct, shift, size) = if order <= 2 {
            (true, 0, 1usize << (8 * order + 8))
        } else {
            (false, 64 - HASH_BITS, 1usize << HASH_BITS)
        };
        Self {
            order,
            direct,
            shift,
            cells: vec![0; size],
            sm: StateMap::new(),
            idx: 0,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn slot(&self, ctx: &Context) -> usize {
        let mut cv = 0u64;
        for i in 1..=self.order {
            cv = (cv << 8) | u64::from(ctx.byte_back(i));
        }
        let raw = (cv << 8) | u64::from(ctx.c0 & 0xff);
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
