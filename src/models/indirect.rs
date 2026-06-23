//! Indirect context model (the paq/lpaq "ICM").
//!
//! Predicts from *what historically followed* a context, not from the context
//! bytes themselves. For each order-`n` context we remember the last two bytes
//! that followed it (a `u16` follower-history register). The bit predictor is
//! keyed on that follower history plus the previous byte `c1` and the bit-tree
//! node `c0` — so every context that tends to be followed by the same bytes
//! pools its statistics and generalizes. This is orthogonal to the direct order
//! models (which key on the context bytes): v7 measured it as the largest
//! deterministic lever after the recurrent arm.

use super::context::hashed_bits;
use super::statemap::StateMap;
use super::{Context, Model};

const MAX: u16 = 63;
const SM_STATES: usize = 1 << 12;
const HIST_BITS_CAP: u32 = 24; // follower-history table cap (per context)
const CELL_BITS_CAP: u32 = 24; // bit-history table cap (per (follower-hist, c1, node))

/// One observed bit moves a bit-history cell to its next state (same shape as the
/// direct context model: bump the seen count, soft-discount the opposite).
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

const fn discount(x: u16) -> u16 {
    if x > 3 { 3 + ((x - 3) >> 1) } else { x }
}

/// What context an [`IndirectModel`]'s follower-history table is keyed on.
#[derive(Debug, Clone, Copy)]
enum IndKind {
    /// The last `n` finalized bytes.
    Order(usize),
    /// The current word's spelling hash.
    Word,
}

/// Indirect context model over a single context kind.
#[derive(Debug)]
pub(crate) struct IndirectModel {
    kind: IndKind,
    hist: Vec<u16>, // follower-history register per context
    hist_shift: u32,
    cells: Vec<u16>, // bit-history per (follower-hist, c1, node) key
    cell_shift: u32,
    sm: StateMap,
    hist_idx: usize, // current byte's follower-history slot (refreshed at bpos 0)
    fh: u16,         // current byte's follower history
    idx: usize,      // current bit's cells slot
    check: u16,      // expected 4-bit confirm tag for the current cell
}

impl IndirectModel {
    /// Production constructor: tables sized to the input, capped so several
    /// indirect orders fit the RAM budget (they generalize, so do not need the
    /// full high-order table size the direct models use). `LZR_IHB`/`LZR_ICB`
    /// override the caps for offline sweeps only (production uses the consts).
    pub(crate) fn new(order: usize, capacity: usize) -> Self {
        let env = |k: &str, d: u32| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let bits = hashed_bits(capacity);
        let hcap = env("LZR_IHB", HIST_BITS_CAP);
        let ccap = env("LZR_ICB", CELL_BITS_CAP);
        Self::build(IndKind::Order(order), bits.min(hcap), bits.min(ccap))
    }

    /// An indirect model keyed on the current word's spelling hash.
    pub(crate) fn word(capacity: usize) -> Self {
        let bits = hashed_bits(capacity);
        Self::build(
            IndKind::Word,
            bits.min(HIST_BITS_CAP),
            bits.min(CELL_BITS_CAP),
        )
    }

    /// Explicit-size constructor (ablation sweeps only).
    #[cfg(test)]
    pub(crate) fn with_bits(order: usize, hist_bits: u32, cell_bits: u32) -> Self {
        Self::build(IndKind::Order(order), hist_bits, cell_bits)
    }

    fn build(kind: IndKind, hist_bits: u32, cell_bits: u32) -> Self {
        Self {
            kind,
            hist: vec![0u16; 1usize << hist_bits],
            hist_shift: 64 - hist_bits,
            cells: vec![0u16; 1usize << cell_bits],
            cell_shift: 64 - cell_bits,
            sm: StateMap::new(SM_STATES),
            hist_idx: 0,
            fh: 0,
            idx: 0,
            check: 0,
        }
    }

    fn ctx_value(&self, ctx: &Context) -> u64 {
        match self.kind {
            IndKind::Order(order) => {
                let mut cv = 0u64;
                for i in 1..=order {
                    cv = (cv << 8) | u64::from(ctx.byte_back(i));
                }
                cv
            }
            IndKind::Word => ctx.word_hash,
        }
    }
}

const MULT: u64 = 0x9E37_79B9_7F4A_7C15;
const MULT2: u64 = 0xD1B5_4A32_D192_ED03; // independent 4-bit confirm-tag hash
const STATE_MASK: u16 = 0x0FFF; // bit-history state occupies the low 12 bits

impl Model for IndirectModel {
    #[allow(clippy::cast_possible_truncation)]
    fn predict(&mut self, ctx: &Context) -> i32 {
        if ctx.bpos == 0 {
            let cv = self.ctx_value(ctx);
            self.hist_idx = (cv.wrapping_mul(MULT) >> self.hist_shift) as usize;
            self.fh = self.hist[self.hist_idx];
        }
        let key =
            u64::from(self.fh) | (u64::from(ctx.byte_back(1)) << 16) | (u64::from(ctx.c0) << 24);
        self.idx = (key.wrapping_mul(MULT) >> self.cell_shift) as usize;
        self.check = (key.wrapping_mul(MULT2) >> 60) as u16;
        let cell = self.cells[self.idx];
        let state = if cell >> 12 == self.check {
            cell & STATE_MASK
        } else {
            0
        };
        self.sm.predict(state as usize)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn update(&mut self, ctx: &Context, bit: u8) {
        let cell = self.cells[self.idx];
        let state = if cell >> 12 == self.check {
            cell & STATE_MASK
        } else {
            0
        };
        self.sm.update(state as usize, bit);
        self.cells[self.idx] = (self.check << 12) | transition(state, bit);
        if ctx.bpos == 7 {
            let b = (((ctx.c0 << 1) | u32::from(bit)) & 0xff) as u16;
            self.hist[self.hist_idx] = (self.fh << 8) | b;
        }
    }
}
