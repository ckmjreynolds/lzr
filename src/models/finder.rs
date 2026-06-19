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
