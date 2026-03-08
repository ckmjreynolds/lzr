//! Fixed-size ring buffer for streaming block compression.

use std::io::{self, Read};
use std::ops::{Index, Range};

const RING_SIZE: usize = 1 << 20; // 1 MiB
const RING_MASK: usize = RING_SIZE - 1;
const BLOCK_SIZE: usize = 1 << 16; // 64 KiB
const BATCH_SIZE: usize = RING_SIZE - BLOCK_SIZE; // 960 KiB (15 blocks)

static_assertions::const_assert!(RING_SIZE.is_power_of_two());
static_assertions::const_assert!(RING_SIZE.is_multiple_of(BLOCK_SIZE));

/// A 1 MiB ring buffer with block-aligned cursor for streaming compression.
///
/// The buffer size is a power of two, so all index wrapping uses bitwise AND.
/// The cursor starts at `BLOCK_SIZE` (64 KiB) so the initial lookback window
/// is the zeroed region `[0..BLOCK_SIZE)`.
pub(crate) struct RingBuffer {
    buf: Vec<u8>,
    cursor: usize,
}

impl RingBuffer {
    pub(crate) fn new() -> Self {
        Self {
            buf: vec![0u8; RING_SIZE],
            cursor: BLOCK_SIZE,
        }
    }

    /// Read up to `BATCH_SIZE` (960 KiB) bytes into the ring at the cursor,
    /// handling wrap-around with up to two reads.
    pub(crate) fn read_batch(&mut self, input: &mut impl Read) -> io::Result<usize> {
        let end = self.cursor + BATCH_SIZE;
        let mut total = 0;

        if end <= RING_SIZE {
            while total < BATCH_SIZE {
                let n = input.read(&mut self.buf[self.cursor + total..self.cursor + BATCH_SIZE])?;
                if n == 0 {
                    break;
                }
                total += n;
            }
        } else {
            let first_len = RING_SIZE - self.cursor;
            while total < first_len {
                let n = input.read(&mut self.buf[self.cursor + total..RING_SIZE])?;
                if n == 0 {
                    return Ok(total);
                }
                total += n;
            }
            let second_len = BATCH_SIZE - first_len;
            let mut second_total = 0;
            while second_total < second_len {
                let n = input.read(&mut self.buf[second_total..second_len])?;
                if n == 0 {
                    break;
                }
                second_total += n;
            }
            total += second_total;
        }

        Ok(total)
    }

    /// Build block-aligned `(start, len)` ranges for the last batch of `bytes_read` bytes.
    pub(crate) fn block_ranges(&self, bytes_read: usize) -> Vec<(usize, usize)> {
        let full_blocks = bytes_read / BLOCK_SIZE;
        let tail = bytes_read % BLOCK_SIZE;
        let mut ranges = Vec::with_capacity(full_blocks + usize::from(tail > 0));
        for i in 0..full_blocks {
            let start = (self.cursor + i * BLOCK_SIZE) & RING_MASK;
            ranges.push((start, BLOCK_SIZE));
        }
        if tail > 0 {
            let start = (self.cursor + full_blocks * BLOCK_SIZE) & RING_MASK;
            ranges.push((start, tail));
        }
        ranges
    }

    /// Advance the cursor by `n` bytes, wrapping around the ring.
    pub(crate) const fn advance(&mut self, n: usize) {
        self.cursor = (self.cursor + n) & RING_MASK;
    }
}

impl Index<usize> for RingBuffer {
    type Output = u8;

    fn index(&self, index: usize) -> &u8 {
        &self.buf[index & RING_MASK]
    }
}

impl Index<Range<usize>> for RingBuffer {
    type Output = [u8];

    fn index(&self, range: Range<usize>) -> &[u8] {
        let len = range.end - range.start;
        debug_assert!(len <= RING_SIZE, "range length {len} exceeds ring size");
        let start = range.start & RING_MASK;
        let end = start + len;
        debug_assert!(end <= RING_SIZE, "range [{start}..{end}) wraps around the ring buffer boundary");
        &self.buf[start..end]
    }
}
