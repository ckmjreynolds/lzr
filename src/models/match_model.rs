//! Match model: predict the next byte from the longest recent repeat.
//!
//! A [`Finder`] maps the last 8 finalized bytes to the position that followed
//! them last time. While the prediction keeps coming true we ride the match,
//! growing `len`; a [`StateMap`] keyed on `(match length, predicted bit)` turns
//! that into a calibrated probability, so the mixer weights it by how reliable
//! matches of that length have actually been. Reads history from [`Context`];
//! orders 0–6 already cover short contexts, so the 8-byte key is additive.

#[cfg(test)]
use super::finder::SetFinder;
use super::finder::{Finder, FlatFinder, LruFinder, WAYS};
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
    hi: u64,       // rolling bytes 9..16 back (for keys longer than 8 bytes)
    key_mask: u64, // masks `last8` to the low `key_bytes` bytes for the finder
    hi_mask: u64,  // masks `hi` to the high `key_bytes - 8` bytes (0 if key ≤ 8)
    key_bytes: u32,
    vcap: u32,  // back-verify candidates up to this many context bytes (0 = off)
    ptr: usize, // history index of the predicted next byte
    len: u32,   // current match length
    pb: u8,     // predicted byte (history[ptr]) cached at bpos==0 (byte-constant)
    sm: StateMap,
    predicted: bool, // did predict() consult the StateMap this bit?
    state: usize,    // StateMap state from the last predict()
    // Offline break-exclusion probe (test-only): when set, the model abstains
    // (returns 0) on byte positions flagged as long-match breaks, simulating a
    // side channel that tells the decoder "this match breaks here". Measures the
    // gross gain of break exclusion before any real side channel is built.
    #[cfg(test)]
    abstain: Option<(std::rc::Rc<[bool]>, u32)>,
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
    /// ≤ 16). A shorter key acquires matches from shorter repeats; the [`StateMap`]
    /// (keyed on length) discounts the resulting shorter/less-reliable matches.
    /// Keys longer than 8 fold the bytes 9..16 word into the finder key — a
    /// higher-order match that only acquires from long repeats.
    pub(crate) fn with_key(key_bytes: u32) -> Self {
        Self::with_key_bits(key_bytes, FLAT_BITS)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn with_key_bits(key_bytes: u32, flat_bits: u32) -> Self {
        let finder: Box<dyn Finder> = if USE_LRU {
            Box::new(LruFinder::new(LRU_CAP))
        } else {
            Box::new(FlatFinder::new(flat_bits))
        };
        Self::with_finder(finder, key_bytes, 0)
    }

    /// A verified multi-candidate match: a [`SetFinder`] keeps the last [`WAYS`]
    /// occurrences per key; on acquisition each candidate is verified against
    /// real history (killing tag false-positives exactly) and the one with the
    /// longest verified context wins, seeding `len` from it — long-context
    /// acquisitions start already trusted by the length-keyed [`StateMap`].
    /// Test-only (see [`SetFinder`]): neutral as a replacement, redundant as an
    /// extra next to the key-12/16 higher-order matches.
    #[cfg(test)]
    pub(crate) fn verified(key_bytes: u32, bucket_bits: u32, vcap: u32) -> Self {
        Self::with_finder(Box::new(SetFinder::new(bucket_bits)), key_bytes, vcap)
    }

    fn with_finder(finder: Box<dyn Finder>, key_bytes: u32, vcap: u32) -> Self {
        let key_mask = if key_bytes >= 8 {
            u64::MAX
        } else {
            (1u64 << (8 * key_bytes)) - 1
        };
        let hi_mask = match key_bytes {
            0..=8 => 0,
            16.. => u64::MAX,
            k => (1u64 << (8 * (k - 8))) - 1,
        };
        Self {
            finder,
            last8: 0,
            hi: 0,
            key_mask,
            hi_mask,
            key_bytes,
            vcap,
            ptr: 0,
            len: 0,
            pb: 0,
            sm: StateMap::new(2 * (LEN_CAP as usize + 1)),
            predicted: false,
            state: 0,
            #[cfg(test)]
            abstain: None,
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
        self.hi = (self.hi << 8) | (self.last8 >> 56);
        self.last8 = (self.last8 << 8) | u64::from(b);
        let key = (self.last8 & self.key_mask)
            ^ (self.hi & self.hi_mask).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        if self.len == 0 {
            self.lookups += 1;
            if self.vcap == 0 {
                if let Some(q) = self.finder.lookup(key) {
                    self.ptr = q as usize;
                    self.len = 1;
                    self.hits += 1;
                }
            } else if let Some((q, l)) = self.verify_best(ctx, b, n, key) {
                self.ptr = q;
                self.len = l - self.key_bytes + 1; // key-length context ⇒ len 1
                self.hits += 1;
            }
        }
        self.finder.insert(key, (n + 1) as u32);
    }

    /// Among the finder's candidates for `key`, the one with the longest context
    /// verified against real history (newest wins ties), with that length.
    /// Verification checks ALL context bytes (not just beyond the key), so a tag
    /// false-positive verifies < `key_bytes` and is rejected — exact, no false
    /// matches. `b` is the current byte (not yet in history at index `n`).
    #[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
    fn verify_best(&mut self, ctx: &Context, b: u8, n: usize, key: u64) -> Option<(usize, u32)> {
        let mut cand = [0u32; WAYS];
        let m = self.finder.lookup_multi(key, &mut cand);
        let hist = ctx.history();
        let mut best = None;
        let mut bestl = 0u32;
        for &qr in &cand[..m] {
            let q = qr as usize;
            if q == 0 || q > n || hist[q - 1] != b {
                continue;
            }
            // context byte d back: candidate hist[q-d] vs current hist[n+1-d]
            // (d == 1 is `b`, checked above).
            let mut l = 1u32;
            let mut d = 2usize;
            while l < self.vcap && d <= q && d <= n + 1 && hist[q - d] == hist[n + 1 - d] {
                l += 1;
                d += 1;
            }
            if l >= self.key_bytes && l > bestl {
                bestl = l;
                best = Some(q);
            }
        }
        best.map(|q| (q, bestl))
    }
}

impl Model for MatchModel {
    #[allow(clippy::cast_possible_truncation)]
    fn predict(&mut self, ctx: &Context) -> i32 {
        self.predicted = false;
        #[cfg(test)]
        if let Some((flags, t)) = &self.abstain {
            if self.len >= *t && flags.get(ctx.history().len()).copied().unwrap_or(false) {
                return 0;
            }
        }
        let hist = ctx.history();
        if self.len == 0 || self.ptr >= hist.len() {
            return 0;
        }
        // ptr, len and hist (appended only at end_symbol) are byte-constant, so
        // the predicted byte is the same for all 8 bits; load it once at bpos==0.
        if ctx.bpos == 0 {
            self.pb = hist[self.ptr];
        }
        let pb = u32::from(self.pb);
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

impl MatchModel {
    /// Arm the offline break-exclusion probe: abstain on flagged positions whose
    /// active match length is at least `thresh` (test-only).
    #[cfg(test)]
    pub(crate) fn set_abstain(&mut self, flags: std::rc::Rc<[bool]>, thresh: u32) {
        self.abstain = Some((flags, thresh));
    }

    /// `(match-length bucket, predicted byte)` for warming heads to key on:
    /// `(0, 0)` when there is no active match, else the capped length and the byte
    /// the match predicts (`history[ptr]`). Byte-constant within a symbol.
    pub(crate) fn match_key(&self, ctx: &Context) -> (u32, u8) {
        let hist = ctx.history();
        if self.len == 0 || self.ptr >= hist.len() {
            (0, 0)
        } else {
            (self.len.min(LEN_CAP), hist[self.ptr])
        }
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
