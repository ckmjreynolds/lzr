//! A packed queue of up to four `u16` values stored in a single `u64`.
//!
//! [`Queue`] packs up to four 16-bit values into a single 64-bit integer,
//! avoiding heap allocation and keeping recent history cache-friendly.
//! Values are enqueued into the least-significant bits, shifting existing
//! values toward the most-significant end.

/// Four `u16` values packed into a single `u64`.
///
/// Values are enqueued into the least-significant bits, shifting existing
/// values toward the most-significant end. Index 0 always refers to the
/// most recently enqueued value; index 3 is the oldest.
///
/// # Examples
///
/// ```text
/// let mut q = Queue::new();
/// q.enqueue(10);
/// q.enqueue(20);
///
/// assert_eq!(q.get(0), 20); // most recent
/// assert_eq!(q.get(1), 10); // previous
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(crate) struct Queue(u64);

impl Queue {
    /// Maximum number of values the queue can hold.
    pub(crate) const CAPACITY: usize = 4;

    /// Creates a new queue with all slots initialized to zero.
    ///
    /// # Examples
    ///
    /// ```text
    /// let q = Queue::new();
    /// assert_eq!(q.get(0), 0);
    /// ```
    pub(crate) const fn new() -> Self {
        Self(0)
    }

    /// Returns the value at `index`, where 0 is the most recently enqueued.
    ///
    /// # Examples
    ///
    /// ```text
    /// let mut q = Queue::new();
    /// q.enqueue(100);
    /// q.enqueue(200);
    /// q.enqueue(300);
    ///
    /// assert_eq!(q.get(0), 300);
    /// assert_eq!(q.get(1), 200);
    /// assert_eq!(q.get(2), 100);
    /// ```
    #[allow(clippy::cast_possible_truncation, clippy::trivially_copy_pass_by_ref)]
    pub(crate) const fn get(&self, index: usize) -> u16 {
        (self.0 >> (index * 16)) as u16
    }

    /// Pushes `value` into slot 0, shifting all existing values up by one slot.
    ///
    /// The oldest value (slot 3) is silently discarded when the queue is full.
    ///
    /// # Examples
    ///
    /// ```text
    /// let mut q = Queue::new();
    /// q.enqueue(0xAAAA);
    /// assert_eq!(q.get(0), 0xAAAA);
    ///
    /// q.enqueue(0xBBBB);
    /// assert_eq!(q.get(0), 0xBBBB); // newest
    /// assert_eq!(q.get(1), 0xAAAA); // shifted up
    /// ```
    pub(crate) const fn enqueue(&mut self, value: u16) {
        self.0 = (self.0 << 16) | (value as u64);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn enqueued_values_are_readable(values in proptest::collection::vec(any::<u16>(), 1..=Queue::CAPACITY)) {
            let mut queue = Queue::new();
            for &v in &values {
                queue.enqueue(v);
            }
            // Most recently enqueued value is at index 0; first enqueued is at len-1.
            for (i, &v) in values.iter().rev().enumerate() {
                assert_eq!(queue.get(i), v);
            }
        }
    }
}
