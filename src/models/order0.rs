//! Adaptive order-0 model.
//!
//! Maintains a 256-entry Fenwick table of byte frequencies. To predict bit
//! `bpos` of the current byte, the bytes consistent with the bits seen so far
//! form a contiguous range; the next bit splits that range in half, and the
//! probability is the count mass of the upper (bit == 1) half over the total.
//! This is the worked example for adding further models.

use super::{Context, Model};
use crate::fenwick::Fenwick;
use crate::mixer::stretch;

const ALPHABET: usize = 256; // byte symbols
const INC: u32 = 32; // count increment per observed byte
const RESCALE_LIMIT: u32 = 1 << 16; // halve counts past this total (adaptivity)

/// Order-0 (context-free) byte-frequency model.
#[derive(Debug)]
pub(crate) struct Order0 {
    counts: Fenwick,
}

impl Order0 {
    /// New model with add-1 smoothing (every byte starts at count 1).
    pub(crate) fn new() -> Self {
        let mut counts = Fenwick::new(ALPHABET);
        for i in 0..ALPHABET {
            counts.add(i, 1);
        }
        Self { counts }
    }

    fn rescale(&mut self) {
        let mut halved = [0u32; ALPHABET];
        for (i, h) in halved.iter_mut().enumerate() {
            *h = (self.counts.range_sum(i, i + 1) >> 1).max(1);
        }
        self.counts.clear();
        for (i, &h) in halved.iter().enumerate() {
            self.counts.add(i, h);
        }
    }
}

impl Model for Order0 {
    #[allow(clippy::cast_possible_truncation)]
    fn predict(&mut self, ctx: &Context) -> i32 {
        let bpos = u32::from(ctx.bpos);
        // The `bpos` bits coded so far are the high bits of the byte.
        let prefix = (ctx.c0 & ((1 << bpos) - 1)) as usize;
        let span = 1usize << (8 - bpos);
        let base = prefix << (8 - bpos);
        let half = span / 2;
        let c1 = self.counts.range_sum(base + half, base + span); // next bit == 1
        let c0 = self.counts.range_sum(base, base + half); // next bit == 0
        let p = ((u64::from(c1) << 12) / u64::from(c0 + c1)) as i32;
        stretch(p)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn update(&mut self, ctx: &Context, bit: u8) {
        if ctx.bpos == 7 {
            let byte = (((ctx.c0 << 1) | u32::from(bit)) & 0xff) as usize;
            self.counts.add(byte, INC);
            if self.counts.total() > RESCALE_LIMIT {
                self.rescale();
            }
        }
    }
}
