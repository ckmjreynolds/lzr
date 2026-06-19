//! Adaptive order-2 model.
//!
//! One [`Fenwick`] byte-frequency table per previous *two* bytes (65536
//! contexts, ~67 MB), each the same half-split predictor as the order-0 model.
//! The context is the last two finalized bytes (`ctx.c4 & 0xffff`), stable
//! across the eight bits of the byte being coded.

use super::{Context, Model};
use crate::fenwick::Fenwick;
use crate::mixer::stretch;

const ALPHABET: usize = 256; // byte symbols
const CONTEXTS: usize = 1 << 16; // one table per previous two bytes
const INC: u32 = 32; // count increment per observed byte
const RESCALE_LIMIT: u32 = 1 << 16; // halve a table's counts past this total

/// Order-2 byte-frequency model keyed on the previous two bytes.
#[derive(Debug)]
pub(crate) struct Order2 {
    tables: Vec<Fenwick>,
}

impl Order2 {
    /// New model with add-1 smoothing in every context (each byte starts at 1).
    pub(crate) fn new() -> Self {
        let mut table = Fenwick::new(ALPHABET);
        for i in 0..ALPHABET {
            table.add(i, 1);
        }
        Self {
            tables: vec![table; CONTEXTS],
        }
    }

    fn rescale(table: &mut Fenwick) {
        let mut halved = [0u32; ALPHABET];
        for (i, h) in halved.iter_mut().enumerate() {
            *h = (table.range_sum(i, i + 1) >> 1).max(1);
        }
        table.clear();
        for (i, &h) in halved.iter().enumerate() {
            table.add(i, h);
        }
    }
}

impl Model for Order2 {
    #[allow(clippy::cast_possible_truncation)]
    fn predict(&mut self, ctx: &Context) -> i32 {
        let table = &self.tables[(ctx.c4 & 0xffff) as usize];
        let bpos = u32::from(ctx.bpos);
        let prefix = (ctx.c0 & ((1 << bpos) - 1)) as usize;
        let span = 1usize << (8 - bpos);
        let base = prefix << (8 - bpos);
        let half = span / 2;
        let c1 = table.range_sum(base + half, base + span); // next bit == 1
        let c0 = table.range_sum(base, base + half); // next bit == 0
        let p = ((u64::from(c1) << 12) / u64::from(c0 + c1)) as i32;
        stretch(p)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn update(&mut self, ctx: &Context, bit: u8) {
        if ctx.bpos == 7 {
            let table = &mut self.tables[(ctx.c4 & 0xffff) as usize];
            let byte = (((ctx.c0 << 1) | u32::from(bit)) & 0xff) as usize;
            table.add(byte, INC);
            if table.total() > RESCALE_LIMIT {
                Self::rescale(table);
            }
        }
    }
}
