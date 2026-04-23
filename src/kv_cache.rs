//! Sliding-window KV cache implemented as a ring buffer.
//!
//! One `k` buffer and one `v` buffer per transformer layer, each sized
//! `N_HEADS * CONTEXT_LEN * HEAD_DIM` and pre-allocated at model
//! construction. Eviction is a pointer bump — no memmove.
//!
//! Walking the cache in logical time order means reading the two contiguous
//! chunks: `[head .. CONTEXT_LEN)` and `[0 .. head)` when the cache is full,
//! or just `[0 .. head)` while still warming up.

use crate::arch::{CONTEXT_LEN, HEAD_DIM, N_HEADS, N_LAYERS};

const SLOT_LEN: usize = N_HEADS * HEAD_DIM;
const BUFFER_LEN: usize = CONTEXT_LEN * SLOT_LEN;

/// One layer's KV ring buffer.
#[derive(Debug)]
pub(crate) struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    head: usize,
    filled: usize,
}

impl Default for LayerCache {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerCache {
    /// Allocate a fresh, zero-filled layer cache.
    pub(crate) fn new() -> Self {
        Self {
            k: vec![0.0; BUFFER_LEN],
            v: vec![0.0; BUFFER_LEN],
            head: 0,
            filled: 0,
        }
    }

    /// Reset to empty (does not deallocate).
    pub(crate) const fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
    }

    /// Number of valid entries currently in the ring.
    pub(crate) const fn len(&self) -> usize {
        self.filled
    }

    #[cfg(test)]
    pub(crate) const fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Append a new K / V slot (each of length `SLOT_LEN`). Overwrites the
    /// oldest entry when the ring is full.
    pub(crate) fn push(&mut self, k_slot: &[f32], v_slot: &[f32]) {
        debug_assert_eq!(k_slot.len(), SLOT_LEN);
        debug_assert_eq!(v_slot.len(), SLOT_LEN);
        let off = self.head * SLOT_LEN;
        self.k[off..off + SLOT_LEN].copy_from_slice(k_slot);
        self.v[off..off + SLOT_LEN].copy_from_slice(v_slot);
        self.head = (self.head + 1) % CONTEXT_LEN;
        if self.filled < CONTEXT_LEN {
            self.filled += 1;
        }
    }

    /// Iterate over the cached entries in logical time order (oldest first).
    /// Each iteration yields `(logical_position, k_slot, v_slot)`.
    ///
    /// `logical_position` runs from 0 to `filled - 1`; callers that want the
    /// `RoPE` position for each slot should treat that as the position.
    pub(crate) const fn entries(&self) -> CacheIter<'_> {
        let start = if self.filled < CONTEXT_LEN {
            0
        } else {
            self.head
        };
        CacheIter {
            cache: self,
            start,
            index: 0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CacheIter<'a> {
    cache: &'a LayerCache,
    start: usize,
    index: usize,
}

impl<'a> Iterator for CacheIter<'a> {
    type Item = (usize, &'a [f32], &'a [f32]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.cache.filled {
            return None;
        }
        let phys = (self.start + self.index) % CONTEXT_LEN;
        let off = phys * SLOT_LEN;
        let k = &self.cache.k[off..off + SLOT_LEN];
        let v = &self.cache.v[off..off + SLOT_LEN];
        let logical = self.index;
        self.index += 1;
        Some((logical, k, v))
    }
}

/// All-layers KV cache.
#[derive(Debug)]
pub(crate) struct KvCache {
    pub layers: [LayerCache; N_LAYERS],
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KvCache {
    pub(crate) fn new() -> Self {
        Self {
            layers: core::array::from_fn(|_| LayerCache::new()),
        }
    }

    pub(crate) fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }
}

#[cfg(test)]
// Tests use small `usize` counters for tag values — `usize -> f32` is exact
// for values well under 2^24.
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    #[test]
    fn ring_push_wrap() {
        let mut cache = LayerCache::new();
        assert!(cache.is_empty());

        let make_slot = |tag: f32| vec![tag; SLOT_LEN];
        for i in 0..CONTEXT_LEN {
            let s = make_slot(i as f32);
            cache.push(&s, &s);
        }
        assert_eq!(cache.len(), CONTEXT_LEN);

        // Overwrite the oldest entry — logical ordering now [1, 2, ..., CONTEXT_LEN].
        let new_slot = make_slot(CONTEXT_LEN as f32);
        cache.push(&new_slot, &new_slot);
        assert_eq!(cache.len(), CONTEXT_LEN);

        let ordered: Vec<f32> = cache.entries().map(|(_, k, _)| k[0]).collect();
        let expected: Vec<f32> = (1..=CONTEXT_LEN).map(|i| i as f32).collect();
        assert_eq!(ordered, expected);
    }

    #[test]
    fn partial_fill() {
        let mut cache = LayerCache::new();
        let s0 = vec![7.0; SLOT_LEN];
        let s1 = vec![9.0; SLOT_LEN];
        cache.push(&s0, &s0);
        cache.push(&s1, &s1);
        let ordered: Vec<f32> = cache.entries().map(|(_, k, _)| k[0]).collect();
        assert_eq!(ordered, vec![7.0, 9.0]);
    }

    #[test]
    fn reset_clears_without_dealloc() {
        let mut cache = LayerCache::new();
        let s = vec![1.0; SLOT_LEN];
        cache.push(&s, &s);
        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.entries().count(), 0);
    }
}
