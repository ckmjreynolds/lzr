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

/// A `key -> position` store the match model queries and updates each byte.
pub(crate) trait Finder: std::fmt::Debug {
    /// Record that the context `key` was last followed by the byte at `pos`.
    fn insert(&mut self, key: u64, pos: u32);
    /// The position recorded for `key`, if any.
    fn lookup(&mut self, key: u64) -> Option<u32>;
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
