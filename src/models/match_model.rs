//! Match model: predict the next byte from the longest recent repeat.
//!
//! A [`Finder`] maps the last 8 finalized bytes to the position that followed
//! them last time. While the prediction keeps coming true we ride the match,
//! growing `len`; a [`StateMap`] keyed on `(match length, predicted bit)` turns
//! that into a calibrated probability, so the mixer weights it by how reliable
//! matches of that length have actually been. Reads history from [`Context`];
//! orders 0–6 already cover short contexts, so the 8-byte key is additive.

use super::finder::{Finder, FlatFinder, LruFinder};
use super::statemap::StateMap;
use super::{Context, Model};

const KEY_BYTES: u32 = 8; // context length feeding the finder (the u64 key)
const LEN_CAP: u32 = 63; // match-length bucket cap for the StateMap
const LRU_CAP: usize = 90_000_000; // exact baseline: ≈0 evictions on enwik9
const FLAT_BITS: u32 = 27; // flat table: 128M slots, ~768 MB at 6 B/slot
const USE_LRU: bool = false; // false → flat table; true → LRU exact baseline (kept for A/B)

/// Predicts the bits of the matched byte, confidence scaled by match length.
#[derive(Debug)]
pub(crate) struct MatchModel {
    finder: Box<dyn Finder>,
    last8: u64,    // rolling last 8 finalized bytes
    key_mask: u64, // masks `last8` to the low `key_bytes` bytes for the finder
    key_bytes: u32,
    ptr: usize, // history index of the predicted next byte
    len: u32,   // current match length
    sm: StateMap,
    predicted: bool, // did predict() consult the StateMap this bit?
    state: usize,    // StateMap state from the last predict()
    // tuning stats
    total: u64,
    covered: u64,
    lookups: u64,
    hits: u64,
}

impl MatchModel {
    pub(crate) fn new() -> Self {
        Self::with_key(KEY_BYTES)
    }

    /// A match model keyed on the last `key_bytes` finalized bytes (`key_bytes`
    /// ≤ 8). A shorter key acquires matches from shorter repeats; the [`StateMap`]
    /// (keyed on length) discounts the resulting shorter/less-reliable matches.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn with_key(key_bytes: u32) -> Self {
        let finder: Box<dyn Finder> = if USE_LRU {
            Box::new(LruFinder::new(LRU_CAP))
        } else {
            Box::new(FlatFinder::new(FLAT_BITS))
        };
        let key_mask = if key_bytes >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * key_bytes)) - 1
        };
        Self {
            finder,
            last8: 0,
            key_mask,
            key_bytes,
            ptr: 0,
            len: 0,
            sm: StateMap::new(2 * (LEN_CAP as usize + 1)),
            predicted: false,
            state: 0,
            total: 0,
            covered: 0,
            lookups: 0,
            hits: 0,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn byte_step(&mut self, ctx: &Context, b: u8) {
        let n = ctx.history().len(); // b will be appended at index n
        self.total += 1;
        if self.len > 0 {
            self.covered += 1;
        }
        let followed = self.len > 0 && self.ptr < n && ctx.history()[self.ptr] == b;
        if followed {
            self.ptr += 1;
            self.len += 1;
        } else {
            self.len = 0;
        }
        self.last8 = (self.last8 << 8) | u64::from(b);
        let key = self.last8 & self.key_mask;
        if self.len == 0 {
            self.lookups += 1;
            if let Some(q) = self.finder.lookup(key) {
                self.ptr = q as usize;
                self.len = 1;
                self.hits += 1;
            }
        }
        self.finder.insert(key, (n + 1) as u32);
    }
}

impl Model for MatchModel {
    #[allow(clippy::cast_possible_truncation)]
    fn predict(&mut self, ctx: &Context) -> i32 {
        self.predicted = false;
        let hist = ctx.history();
        if self.len == 0 || self.ptr >= hist.len() {
            return 0;
        }
        let pb = u32::from(hist[self.ptr]);
        let bpos = u32::from(ctx.bpos);
        let coded = ctx.c0 & ((1 << bpos) - 1);
        if coded != pb >> (8 - bpos) {
            return 0; // the partial byte already diverged from the prediction
        }
        let pbit = (pb >> (7 - bpos)) & 1;
        let bucket = self.len.min(LEN_CAP);
        self.state = ((bucket << 1) | pbit) as usize;
        self.predicted = true;
        self.sm.predict(self.state)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn update(&mut self, ctx: &Context, bit: u8) {
        if self.predicted {
            self.sm.update(self.state, bit);
        }
        if ctx.bpos == 7 {
            let b = (((ctx.c0 << 1) | u32::from(bit)) & 0xff) as u8;
            self.byte_step(ctx, b);
        }
    }

    /// Current match-length bucket (0 = no active match) — a decorrelated mixer
    /// selector: the blend can lean on the match in long-repeat regions.
    #[allow(clippy::cast_possible_truncation)]
    fn selector(&self) -> Option<usize> {
        Some(self.len.min(LEN_CAP) as usize)
    }
}

impl Drop for MatchModel {
    #[allow(clippy::cast_precision_loss)]
    fn drop(&mut self) {
        if self.total == 0 {
            return;
        }
        eprintln!(
            "match (key={}B): {} | coverage {:.1}%  acquire {}/{} ({:.1}%)",
            self.key_bytes,
            self.finder.report(),
            100.0 * self.covered as f64 / self.total as f64,
            self.hits,
            self.lookups,
            100.0 * self.hits as f64 / self.lookups.max(1) as f64,
        );
    }
}
