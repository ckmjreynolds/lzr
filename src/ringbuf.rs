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
