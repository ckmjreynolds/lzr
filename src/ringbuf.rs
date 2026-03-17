//! Fixed-capacity ring buffer indexed by file position.
//!
//! Both the encoder and decoder need a sliding window over the byte stream.
//! [`RingBuf`] provides power-of-two capacity so that all position-to-index
//! mapping reduces to a single bitwise AND.

use std::io::Read;
use std::ops::{Index, Range};

/// A power-of-two ring buffer addressed by `isize` file positions.
///
/// Negative indices (positions before the stream) land in the zero-prefilled
/// region naturally via two's-complement masking.
#[derive(Debug)]
pub(crate) struct RingBuf {
    buf: Vec<u8>,
    mask: usize,
    write_pos: isize,
}

impl RingBuf {
    /// Creates a new zero-filled ring buffer.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is not a power of two.
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "capacity must be a power of two");
        Self {
            buf: vec![0; capacity],
            mask: capacity - 1,
            write_pos: 0,
        }
    }

    /// Returns the next file position that will be written.
    pub(crate) const fn write_pos(&self) -> isize {
        self.write_pos
    }

    /// Returns the buffer capacity in bytes.
    pub(crate) const fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Writes a single byte at the current write position and advances by one.
    pub(crate) fn push(&mut self, byte: u8) {
        self.buf[self.write_pos.cast_unsigned() & self.mask] = byte;
        self.write_pos += 1;
    }

    /// Returns 1–2 contiguous slices covering the given position range.
    ///
    /// If the range doesn't wrap around the buffer, the second slice is empty.
    /// This mirrors [`VecDeque::as_slices`] — callers iterate both slices for
    /// `write_all` / `adler.update`.
    pub(crate) fn slices(&self, range: Range<isize>) -> (&[u8], &[u8]) {
        let start = range.start.cast_unsigned() & self.mask;
        let end = range.end.cast_unsigned() & self.mask;
        if start <= end {
            (&self.buf[start..end], &[])
        } else {
            (&self.buf[start..], &self.buf[..end])
        }
    }

    /// Bulk-fills the buffer from `reader`, reading up to `len` bytes.
    ///
    /// Internally splits at the wrap boundary so each `read` call targets a
    /// contiguous slice. Returns the total number of bytes read (may be less
    /// than `len` if the reader is exhausted).
    ///
    /// # Errors
    ///
    /// Returns any I/O error from the underlying reader.
    pub(crate) fn fill_from_reader(&mut self, reader: &mut impl Read, len: usize) -> std::io::Result<usize> {
        let mut total = 0;
        let mut remaining = len;

        while remaining > 0 {
            let start = self.write_pos.cast_unsigned() & self.mask;
            let contiguous = (self.capacity() - start).min(remaining);
            let n = reader.read(&mut self.buf[start..start + contiguous])?;
            if n == 0 {
                break;
            }
            self.write_pos += n.cast_signed();
            total += n;
            remaining -= n;
        }

        Ok(total)
    }

    /// Returns the number of contiguous bytes available from `pos` before
    /// hitting the physical end of the buffer (wrap point).
    #[inline]
    pub(crate) const fn contiguous_len(&self, pos: isize) -> usize {
        self.buf.len() - (pos.cast_unsigned() & self.mask)
    }

    /// Reads up to 4 bytes at `pos` as a big-endian `u32`.
    ///
    /// Uses direct slice access when the bytes are contiguous in the buffer.
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn read_u32_be(&self, pos: isize, bytes_available: usize) -> u32 {
        let len = bytes_available.min(4);
        let start = pos.cast_unsigned() & self.mask;

        if start + 4 <= self.buf.len() && len == 4 {
            let bytes: [u8; 4] = self.buf[start..start + 4].try_into().unwrap();
            u32::from_be_bytes(bytes)
        } else if start + len <= self.buf.len() {
            let mut key: u32 = 0;
            for (i, &b) in self.buf[start..start + len].iter().enumerate() {
                key |= u32::from(b) << (24 - 8 * i);
            }
            key
        } else {
            let mut key: u32 = 0;
            for i in 0..len {
                key |= u32::from(self.buf[(start + i) & self.mask]) << (24 - 8 * i);
            }
            key
        }
    }

    /// Reads up to 8 bytes at `pos` as a big-endian `u64`.
    ///
    /// Uses direct slice access when the bytes are contiguous in the buffer.
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn read_u64_be(&self, pos: isize, bytes_available: usize) -> u64 {
        let len = bytes_available.min(8);
        let start = pos.cast_unsigned() & self.mask;

        if start + 8 <= self.buf.len() && len == 8 {
            let bytes: [u8; 8] = self.buf[start..start + 8].try_into().unwrap();
            u64::from_be_bytes(bytes)
        } else if start + len <= self.buf.len() {
            let mut key: u64 = 0;
            for (i, &b) in self.buf[start..start + len].iter().enumerate() {
                key |= u64::from(b) << (56 - 8 * i);
            }
            key
        } else {
            let mut key: u64 = 0;
            for i in 0..len {
                key |= u64::from(self.buf[(start + i) & self.mask]) << (56 - 8 * i);
            }
            key
        }
    }

    /// Reads up to 16 bytes at `pos` as a big-endian `u128`.
    ///
    /// Uses direct slice access when the bytes are contiguous in the buffer
    /// (99.97% of positions). Falls back to per-byte reads at the wrap point.
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn read_u128_be(&self, pos: isize, bytes_available: usize) -> u128 {
        let len = bytes_available.min(16);
        let start = pos.cast_unsigned() & self.mask;

        if start + 16 <= self.buf.len() && len == 16 {
            // Fast path: 16 contiguous bytes — single slice read.
            let bytes: [u8; 16] = self.buf[start..start + 16].try_into().unwrap();
            u128::from_be_bytes(bytes)
        } else if start + len <= self.buf.len() {
            // Contiguous but fewer than 16 bytes (near end of input).
            let mut key: u128 = 0;
            for (i, &b) in self.buf[start..start + len].iter().enumerate() {
                key |= u128::from(b) << (120 - 8 * i);
            }
            key
        } else {
            // Slow path: wraps around the buffer boundary.
            let mut key: u128 = 0;
            for i in 0..len {
                key |= u128::from(self.buf[(start + i) & self.mask]) << (120 - 8 * i);
            }
            key
        }
    }

    /// Returns the number of matching bytes between two positions (both going forward), up to `max_len`.
    ///
    /// Compares using contiguous slices for speed, avoiding per-byte index lookups.
    #[inline]
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn match_length(&self, pos1: isize, pos2: isize, max_len: usize) -> usize {
        let mut matched = 0;
        let mut remaining = max_len;

        while remaining > 0 {
            let idx1 = (pos1 + matched as isize).cast_unsigned() & self.mask;
            let idx2 = (pos2 + matched as isize).cast_unsigned() & self.mask;
            let chunk = remaining.min(self.buf.len() - idx1).min(self.buf.len() - idx2);
            let s1 = &self.buf[idx1..idx1 + chunk];
            let s2 = &self.buf[idx2..idx2 + chunk];
            let n = s1.iter().zip(s2).take_while(|(a, b)| a == b).count();
            matched += n;
            if n < chunk {
                break;
            }
            remaining -= chunk;
        }

        matched
    }
}

impl Index<isize> for RingBuf {
    type Output = u8;

    fn index(&self, index: isize) -> &u8 {
        &self.buf[index.cast_unsigned() & self.mask]
    }
}

impl Index<Range<isize>> for RingBuf {
    type Output = [u8];

    fn index(&self, range: Range<isize>) -> &[u8] {
        let start = range.start.cast_unsigned() & self.mask;
        let end = range.end.cast_unsigned() & self.mask;
        debug_assert!(start <= end, "range must not wrap around the buffer");
        &self.buf[start..end]
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
mod tests {
    use std::io::Cursor;

    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn zero_prefill() {
        let buf = RingBuf::new(64);
        for i in -64..0_isize {
            assert_eq!(buf[i], 0, "negative index {i} should be zero");
        }
    }

    #[test]
    fn push_and_read_back() {
        let mut buf = RingBuf::new(16);
        for b in 0..10_u8 {
            buf.push(b);
        }
        for i in 0..10_isize {
            assert_eq!(buf[i], i as u8);
        }
        assert_eq!(buf.write_pos(), 10);
    }

    #[test]
    fn wrap_around() {
        let mut buf = RingBuf::new(8);
        for b in 0..12_u8 {
            buf.push(b);
        }
        for i in 4..12_isize {
            assert_eq!(buf[i], i as u8);
        }
        assert_eq!(buf[0_isize], 8);
    }

    #[test]
    fn slice_access() {
        let mut buf = RingBuf::new(16);
        for b in 0..16_u8 {
            buf.push(b);
        }
        assert_eq!(buf[4..8_isize], [4, 5, 6, 7]);
    }

    #[test]
    fn fill_from_reader_basic() {
        let mut buf = RingBuf::new(16);
        let data: Vec<u8> = (0..10).collect();
        let mut cursor = Cursor::new(&data);

        let n = buf.fill_from_reader(&mut cursor, 10).unwrap();
        assert_eq!(n, 10);
        assert_eq!(buf.write_pos(), 10);
        for i in 0..10_isize {
            assert_eq!(buf[i], i as u8);
        }
    }

    #[test]
    fn fill_from_reader_short_read() {
        let mut buf = RingBuf::new(16);
        let data: Vec<u8> = (0..5).collect();
        let mut cursor = Cursor::new(&data);

        let n = buf.fill_from_reader(&mut cursor, 10).unwrap();
        assert_eq!(n, 5);
        assert_eq!(buf.write_pos(), 5);
    }

    #[test]
    fn fill_from_reader_wraps() {
        let mut buf = RingBuf::new(8);
        for _ in 0..6 {
            buf.push(0xFF);
        }
        assert_eq!(buf.write_pos(), 6);

        let data: Vec<u8> = (0..8).collect();
        let mut cursor = Cursor::new(&data);

        let n = buf.fill_from_reader(&mut cursor, 8).unwrap();
        assert_eq!(n, 8);
        assert_eq!(buf.write_pos(), 14);

        for i in 0..8_isize {
            assert_eq!(buf[6 + i], i as u8);
        }
    }

    #[test]
    #[should_panic(expected = "capacity must be a power of two")]
    fn non_power_of_two_panics() {
        let _ = RingBuf::new(7);
    }

    #[test]
    fn slices_no_wrap() {
        let mut buf = RingBuf::new(16);
        for b in 0..10_u8 {
            buf.push(b);
        }
        let (s1, s2) = buf.slices(2..6);
        assert_eq!(s1, &[2, 3, 4, 5]);
        assert!(s2.is_empty());
    }

    #[test]
    fn slices_with_wrap() {
        let mut buf = RingBuf::new(8);
        // Push 6 bytes, then 4 more → write_pos = 10, wraps around
        for b in 0..10_u8 {
            buf.push(b);
        }
        // Positions 6..10 map to indices 6,7,0,1
        let (s1, s2) = buf.slices(6..10);
        assert_eq!(s1, &[6, 7]);
        assert_eq!(s2, &[8, 9]);
    }

    #[test]
    fn slices_empty_range() {
        let mut buf = RingBuf::new(16);
        for b in 0..5_u8 {
            buf.push(b);
        }
        let (s1, s2) = buf.slices(3..3);
        assert!(s1.is_empty());
        assert!(s2.is_empty());
    }

    proptest! {
        #[test]
        fn push_matches_flat_vec(data in proptest::collection::vec(any::<u8>(), 0..=256)) {
            let cap = 256;
            let mut buf = RingBuf::new(cap);
            for &b in &data {
                buf.push(b);
            }
            for (i, &b) in data.iter().enumerate() {
                prop_assert_eq!(buf[i as isize], b);
            }
        }

        #[test]
        fn fill_matches_push(data in proptest::collection::vec(any::<u8>(), 1..=256)) {
            let cap = 256;

            let mut push_buf = RingBuf::new(cap);
            for &b in &data {
                push_buf.push(b);
            }

            let mut fill_buf = RingBuf::new(cap);
            let mut cursor = Cursor::new(&data);
            let n = fill_buf.fill_from_reader(&mut cursor, data.len()).unwrap();
            prop_assert_eq!(n, data.len());

            for i in 0..data.len() {
                prop_assert_eq!(push_buf[i as isize], fill_buf[i as isize]);
            }
        }

        #[test]
        fn negative_index_is_zero_before_any_write(idx in -1_000_000_isize..0) {
            let buf = RingBuf::new(64);
            prop_assert_eq!(buf[idx], 0);
        }
    }
}
