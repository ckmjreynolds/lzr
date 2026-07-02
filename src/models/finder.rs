//! Match finder: maps a context key to the position that followed it last time.
//!
//! Pluggable so the match model can swap implementations. The first is an exact
//! [`LruCache`] (no false matches, eviction by recency) — it gives the best
//! achievable bpb for a given capacity, the baseline a cheaper flat hash table
//! will be sized against. Determinism holds despite the cache's random hasher
//! seed: LRU eviction is by access order, which encode and decode share, so the
//! sequence of lookups is identical.

use std::num::NonZeroUsize;

use lru::LruCache;

/// Candidates a multi-way finder can return per key.
pub(crate) const WAYS: usize = 4;

/// A `key -> position` store the match model queries and updates each byte.
pub(crate) trait Finder: std::fmt::Debug {
    /// Record that the context `key` was last followed by the byte at `pos`.
    fn insert(&mut self, key: u64, pos: u32);
    /// The position recorded for `key`, if any.
    fn lookup(&mut self, key: u64) -> Option<u32>;
    /// Up to [`WAYS`] recent positions recorded for `key`, newest first; returns
    /// the count. Single-slot finders yield at most one.
    fn lookup_multi(&mut self, key: u64, out: &mut [u32; WAYS]) -> usize {
        self.lookup(key).map_or(0, |q| {
            out[0] = q;
            1
        })
    }
    /// A one-line summary of fill/eviction, for tuning the table size.
    fn report(&self) -> String;
}

/// Exact LRU finder over `u64` keys.
#[derive(Debug)]
pub(crate) struct LruFinder {
    cache: LruCache<u64, u32>,
    capacity: usize,
    evictions: u64,
}

impl LruFinder {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity).expect("capacity > 0")),
            capacity,
            evictions: 0,
        }
    }
}

impl Finder for LruFinder {
    fn insert(&mut self, key: u64, pos: u32) {
        if self.cache.len() == self.capacity && self.cache.peek(&key).is_none() {
            self.evictions += 1;
        }
        self.cache.put(key, pos);
    }

    fn lookup(&mut self, key: u64) -> Option<u32> {
        self.cache.get(&key).copied()
    }

    #[allow(clippy::cast_precision_loss)]
    fn report(&self) -> String {
        format!(
            "lru: used {} / cap {} ({:.1}%), evictions {}",
            self.cache.len(),
            self.capacity,
            100.0 * self.cache.len() as f64 / self.capacity as f64,
            self.evictions,
        )
    }
}

/// Flat hashed-table finder: `hash(key)` indexes one slot holding the position
/// plus a check tag. A different key hashing to an occupied slot overwrites it
/// (most-recent-wins, no recency protection); the tag turns a hash collision on
/// lookup into a clean miss rather than a false match. Cheaper per entry than
/// the LRU (6 B/slot vs ~50 B/node) and far faster, at the cost of the matches
/// lost to collisions — the quantity this measures.
#[derive(Debug)]
pub(crate) struct FlatFinder {
    slots: Vec<u32>, // stored position (0 = empty; the model never stores 0)
    tags: Vec<u16>,  // collision check tag
    shift: u32,      // 64 - bits, selecting the high `bits` of the hash as index
    writes: u64,
    collisions: u64, // writes that overwrote a different key
    occupied: u64,   // slots filled for the first time
}

impl FlatFinder {
    pub(crate) fn new(bits: u32) -> Self {
        let size = 1usize << bits;
        Self {
            slots: vec![0; size],
            tags: vec![0; size],
            shift: 64 - bits,
            writes: 0,
            collisions: 0,
            occupied: 0,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    const fn locate(&self, key: u64) -> (usize, u16) {
        let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((h >> self.shift) as usize, h as u16)
    }
}

/// Set-associative multi-candidate finder: each bucket keeps the last [`WAYS`]
/// positions inserted for keys hashing there, newest first, tag-checked. Unlike
/// [`FlatFinder`] a key's older occurrences survive newer inserts, so the match
/// model can verify each candidate against real history and start from the one
/// with the longest verified context (the continuous form of a higher-order key).
/// Test-only: measured NEUTRAL as a primary-finder replacement and ~redundant
/// with the shipped key-12/16 higher-order matches as an extra (2026-07-01 lab);
/// kept for future `match_order_lab` experiments.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct SetFinder {
    pos: Vec<u32>, // WAYS entries per bucket, newest first (0 = empty)
    tags: Vec<u16>,
    shift: u32,
    writes: u64,
    occupied: u64,
}

#[cfg(test)]
impl SetFinder {
    pub(crate) fn new(bucket_bits: u32) -> Self {
        let size = (1usize << bucket_bits) * WAYS;
        Self {
            pos: vec![0; size],
            tags: vec![0; size],
            shift: 64 - bucket_bits,
            writes: 0,
            occupied: 0,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    const fn locate(&self, key: u64) -> (usize, u16) {
        let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (((h >> self.shift) as usize) * WAYS, h as u16)
    }
}

#[cfg(test)]
impl Finder for SetFinder {
    fn insert(&mut self, key: u64, pos: u32) {
        let (s, tag) = self.locate(key);
        self.writes += 1;
        if self.pos[s + WAYS - 1] == 0 {
            self.occupied += 1;
        }
        for w in (1..WAYS).rev() {
            self.pos[s + w] = self.pos[s + w - 1];
            self.tags[s + w] = self.tags[s + w - 1];
        }
        self.pos[s] = pos;
        self.tags[s] = tag;
    }

    fn lookup(&mut self, key: u64) -> Option<u32> {
        let mut out = [0u32; WAYS];
        (self.lookup_multi(key, &mut out) > 0).then(|| out[0])
    }

    fn lookup_multi(&mut self, key: u64, out: &mut [u32; WAYS]) -> usize {
        let (s, tag) = self.locate(key);
        let mut n = 0;
        for w in 0..WAYS {
            if self.pos[s + w] != 0 && self.tags[s + w] == tag {
                out[n] = self.pos[s + w];
                n += 1;
            }
        }
        n
    }

    #[allow(clippy::cast_precision_loss)]
    fn report(&self) -> String {
        let buckets = self.pos.len() / WAYS;
        format!(
            "set: {buckets} buckets x {WAYS} ways, full {:.1}%, {} writes",
            100.0 * self.occupied as f64 / buckets as f64,
            self.writes,
        )
    }
}

impl Finder for FlatFinder {
    fn insert(&mut self, key: u64, pos: u32) {
        let (slot, tag) = self.locate(key);
        self.writes += 1;
        if self.slots[slot] == 0 {
            self.occupied += 1;
        } else if self.tags[slot] != tag {
            self.collisions += 1;
        }
        self.slots[slot] = pos;
        self.tags[slot] = tag;
    }

    fn lookup(&mut self, key: u64) -> Option<u32> {
        let (slot, tag) = self.locate(key);
        if self.slots[slot] != 0 && self.tags[slot] == tag {
            Some(self.slots[slot])
        } else {
            None
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn report(&self) -> String {
        let size = self.slots.len();
        format!(
            "flat: {size} slots, occupancy {:.1}%, collisions {:.1}% of {} writes",
            100.0 * self.occupied as f64 / size as f64,
            100.0 * self.collisions as f64 / self.writes.max(1) as f64,
            self.writes,
        )
    }
}
