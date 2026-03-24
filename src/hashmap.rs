// //! FIFO hash map for LZ77 match finding.
// //!
// //! A position-capacity-bounded multi-map that evicts out-of-range positions
// //! via a FIFO queue. Each key maps to a bucket of positions (most recent last).
// //! When the total number of stored positions exceeds `capacity`, the oldest
// //! out-of-range positions are evicted from the front of the queue.

// use std::collections::{HashMap, VecDeque};
// use std::hash::Hash;

// use smallvec::SmallVec;

// /// A bounded multi-map with FIFO eviction of out-of-range positions.
// ///
// /// - `capacity` limits the total number of positions stored across all buckets.
// /// - `bucket_capacity` limits how many positions a single key may hold.
// ///
// /// Positions are assumed to be monotonically increasing. On each insert the
// /// queue is drained from the front: entries whose oldest position satisfies
// /// `current_pos - pos >= capacity` are evicted; stale entries (whose position
// /// was already removed by bucket overflow) are skipped.
// ///
// /// # Examples
// ///
// /// ```text
// /// let mut map = FifoHashMap::<u32, usize>::with_capacity(1024, 4);
// ///
// /// // Insert hash → position mappings as the encoder slides forward.
// /// map.insert(0xDEAD, 0, 0);
// /// map.insert(0xBEEF, 1, 1);
// /// map.insert(0xDEAD, 5, 5);
// ///
// /// // Look up candidate match positions for a given hash.
// /// assert_eq!(map.get(&0xDEAD).unwrap(), &[0, 5]);
// /// assert_eq!(map.len(), 3);
// /// ```
// pub(crate) struct FifoHashMap<K, V> {
//     map: HashMap<K, SmallVec<[V; 4]>>,
//     queue: VecDeque<K>,
//     count: usize,
//     capacity: usize,
//     bucket_capacity: usize,
// }

// impl<K: Hash + Eq + Clone, V: Copy + Into<usize>> FifoHashMap<K, V> {
//     /// Creates an empty map.
//     ///
//     /// - `capacity` — maximum total positions stored across all buckets.
//     /// - `bucket_capacity` — maximum positions stored per key.
//     ///
//     /// # Examples
//     ///
//     /// ```text
//     /// let map = FifoHashMap::<u32, usize>::with_capacity(1024, 4);
//     /// assert!(map.is_empty());
//     /// assert_eq!(map.capacity(), 1024);
//     /// ```
//     pub(crate) fn with_capacity(capacity: usize, bucket_capacity: usize) -> Self {
//         Self {
//             map: HashMap::with_capacity(capacity),
//             queue: VecDeque::with_capacity(capacity),
//             count: 0,
//             capacity,
//             bucket_capacity,
//         }
//     }

//     /// Inserts `value` into the bucket for `key`.
//     ///
//     /// If the bucket already contains `bucket_capacity` positions, its oldest
//     /// position is dropped first. After insertion, out-of-range positions are
//     /// evicted from the queue front until `count <= capacity`.
//     ///
//     /// `current_pos` is the current stream position, used to determine whether
//     /// a queued position is out of range.
//     ///
//     /// # Examples
//     ///
//     /// ```text
//     /// let mut map = FifoHashMap::<u32, usize>::with_capacity(4, 2);
//     ///
//     /// map.insert(1, 0, 0);
//     /// map.insert(1, 1, 1);
//     /// map.insert(1, 2, 2); // bucket overflow: position 0 is evicted
//     /// assert_eq!(map.get(&1).unwrap(), &[1, 2]);
//     /// ```
//     pub(crate) fn insert(&mut self, key: K, value: V, current_pos: usize) {
//         // 1. Bucket overflow — cap per-key positions.
//         if let Some(bucket) = self.map.get_mut(&key) {
//             if bucket.len() >= self.bucket_capacity {
//                 bucket.remove(0);
//                 self.count -= 1;
//             }
//         }

//         // 2. Global eviction — drain stale/out-of-range entries from queue front.
//         while self.count >= self.capacity {
//             let Some(evict_key) = self.queue.pop_front() else {
//                 break;
//             };

//             let remove_key = if let Some(bucket) = self.map.get_mut(&evict_key) {
//                 let oldest = bucket[0].into();
//                 if current_pos - oldest >= self.capacity {
//                     bucket.remove(0);
//                     self.count -= 1;
//                     bucket.is_empty()
//                 } else {
//                     // Stale queue entry — position was already bucket-evicted.
//                     false
//                 }
//             } else {
//                 // Key fully removed — stale.
//                 false
//             };

//             if remove_key {
//                 self.map.remove(&evict_key);
//             }
//         }

//         // 3. Insert new position.
//         self.map.entry(key.clone()).or_default().push(value);
//         self.queue.push_back(key);
//         self.count += 1;
//     }

//     /// Returns the stored positions for `key` as a slice, or `None` if the key has no entries.
//     pub(crate) fn get(&self, key: &K) -> Option<&[V]> {
//         self.map.get(key).map(SmallVec::as_slice)
//     }

//     /// Returns the total number of positions stored across all buckets.
//     pub(crate) const fn len(&self) -> usize {
//         self.count
//     }

//     /// Returns `true` if no positions are stored.
//     pub(crate) const fn is_empty(&self) -> bool {
//         self.count == 0
//     }

//     /// Returns the maximum total positions this map will store.
//     pub(crate) const fn capacity(&self) -> usize {
//         self.capacity
//     }
// }

// #[cfg(test)]
// #[cfg_attr(coverage_nightly, coverage(off))]
// mod tests {
//     use pretty_assertions::assert_eq;
//     use proptest::prelude::*;

//     use super::*;

//     #[test]
//     fn basic_insert_and_get() {
//         let mut m = FifoHashMap::<u32, usize>::with_capacity(8, 4);
//         assert!(m.is_empty());
//         assert_eq!(m.capacity(), 8);

//         m.insert(10, 0, 0);
//         m.insert(20, 1, 1);
//         m.insert(10, 2, 2);

//         assert_eq!(m.len(), 3);
//         assert_eq!(m.get(&10).unwrap(), &[0, 2]);
//         assert_eq!(m.get(&20).unwrap(), &[1]);
//         assert!(m.get(&99).is_none());
//     }

//     #[test]
//     fn bucket_capacity_enforced() {
//         let mut m = FifoHashMap::<u32, usize>::with_capacity(100, 2);

//         m.insert(1, 10, 10);
//         m.insert(1, 20, 20);
//         m.insert(1, 30, 30);

//         // Bucket should only hold the 2 newest.
//         assert_eq!(m.get(&1).unwrap(), &[20, 30]);
//         assert_eq!(m.len(), 2);
//     }

//     #[test]
//     fn global_eviction_by_range() {
//         let mut m = FifoHashMap::<u32, usize>::with_capacity(4, 4);

//         // Fill to capacity.
//         m.insert(1, 0, 0);
//         m.insert(2, 1, 1);
//         m.insert(3, 2, 2);
//         m.insert(4, 3, 3);
//         assert_eq!(m.len(), 4);

//         // Insert at position 10 — position 0 is out of range (10 - 0 >= 4).
//         m.insert(5, 10, 10);
//         assert!(m.len() <= 4);
//         assert!(m.get(&1).is_none());
//     }

//     #[test]
//     fn stale_queue_entries_skipped() {
//         // bucket_capacity=1 means every re-insert to the same key creates a stale queue entry.
//         let mut m = FifoHashMap::<u32, usize>::with_capacity(4, 1);

//         m.insert(1, 0, 0);
//         m.insert(1, 1, 1); // Evicts position 0 from bucket, queue entry for key=1 at pos=0 is stale.
//         m.insert(2, 2, 2);
//         m.insert(3, 3, 3);
//         m.insert(4, 4, 4);

//         // count=4, queue has 5 entries (one stale). Insert one more to trigger eviction.
//         m.insert(5, 10, 10);
//         assert!(m.len() <= 4);
//     }

//     // ── Property-based tests ──────────────────────────────────────────────

//     /// Naive oracle that mirrors the FIFO map's queue-based eviction.
//     #[derive(Debug, Clone)]
//     struct Oracle {
//         entries: Vec<(u32, usize)>,
//         queue: VecDeque<u32>,
//         capacity: usize,
//         bucket_capacity: usize,
//     }

//     impl Oracle {
//         fn new(capacity: usize, bucket_capacity: usize) -> Self {
//             Self {
//                 entries: Vec::new(),
//                 queue: VecDeque::new(),
//                 capacity,
//                 bucket_capacity,
//             }
//         }

//         fn insert(&mut self, key: u32, value: usize, current_pos: usize) {
//             // Bucket overflow.
//             let bucket_count = self.entries.iter().filter(|(k, _)| *k == key).count();
//             if bucket_count >= self.bucket_capacity {
//                 let idx = self.entries.iter().position(|(k, _)| *k == key).unwrap();
//                 self.entries.remove(idx);
//             }

//             self.entries.push((key, value));
//             self.queue.push_back(key);

//             // Global eviction — mirror queue-based strategy.
//             while self.entries.len() > self.capacity {
//                 let Some(evict_key) = self.queue.pop_front() else {
//                     break;
//                 };
//                 if let Some(idx) = self.entries.iter().position(|(k, _)| *k == evict_key) {
//                     let (_, pos) = self.entries[idx];
//                     if current_pos - pos >= self.capacity {
//                         self.entries.remove(idx);
//                     }
//                     // else: stale queue entry, skip.
//                 }
//                 // else: key gone, stale, skip.
//             }
//         }

//         fn get(&self, key: u32) -> Vec<usize> {
//             self.entries.iter().filter(|(k, _)| *k == key).map(|(_, v)| *v).collect()
//         }

//         fn len(&self) -> usize {
//             self.entries.len()
//         }
//     }

//     #[derive(Debug, Clone)]
//     struct Op {
//         key: u32,
//         pos: usize,
//     }

//     fn ops_strategy() -> impl Strategy<Value = Vec<Op>> {
//         prop::collection::vec((0..16u32, 0..1000usize), 1..=200).prop_map(|pairs| {
//             let mut ops: Vec<Op> = pairs
//                 .into_iter()
//                 .map(|(key, pos)| Op {
//                     key,
//                     pos,
//                 })
//                 .collect();
//             // Positions must be monotonically increasing.
//             ops.sort_by_key(|op| op.pos);
//             // Deduplicate positions to keep the oracle simple.
//             ops.dedup_by_key(|op| op.pos);
//             ops
//         })
//     }

//     proptest! {
//         #[test]
//         fn count_never_exceeds_capacity(
//             ops in ops_strategy(),
//             capacity in 4..64usize,
//             bucket_capacity in 1..8usize,
//         ) {
//             let mut m = FifoHashMap::<u32, usize>::with_capacity(capacity, bucket_capacity);
//             for op in &ops {
//                 m.insert(op.key, op.pos, op.pos);
//                 prop_assert!(m.len() <= capacity, "count {} exceeded capacity {}", m.len(), capacity);
//             }
//         }

//         #[test]
//         fn bucket_len_never_exceeds_bucket_capacity(
//             ops in ops_strategy(),
//             capacity in 4..64usize,
//             bucket_capacity in 1..8usize,
//         ) {
//             let mut m = FifoHashMap::<u32, usize>::with_capacity(capacity, bucket_capacity);
//             for op in &ops {
//                 m.insert(op.key, op.pos, op.pos);
//                 for key in 0..16u32 {
//                     if let Some(bucket) = m.get(&key) {
//                         prop_assert!(
//                             bucket.len() <= bucket_capacity,
//                             "bucket for key {} has {} entries, exceeds {}",
//                             key, bucket.len(), bucket_capacity
//                         );
//                     }
//                 }
//             }
//         }

//         #[test]
//         fn matches_oracle(
//             ops in ops_strategy(),
//             capacity in 4..64usize,
//             bucket_capacity in 1..8usize,
//         ) {
//             let mut m = FifoHashMap::<u32, usize>::with_capacity(capacity, bucket_capacity);
//             let mut oracle = Oracle::new(capacity, bucket_capacity);

//             for op in &ops {
//                 m.insert(op.key, op.pos, op.pos);
//                 oracle.insert(op.key, op.pos, op.pos);
//             }

//             // Verify all keys.
//             for key in 0..16u32 {
//                 let actual: Vec<usize> = m.get(&key).map_or_else(Vec::new, <[usize]>::to_vec);
//                 let expected = oracle.get(key);
//                 prop_assert_eq!(&actual, &expected, "mismatch for key {}", key);
//             }
//             prop_assert_eq!(m.len(), oracle.len(), "count mismatch");
//         }
//     }
// }
