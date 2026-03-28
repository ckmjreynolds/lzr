//! Cursor types for sequential byte-level access to a [`Buffer`].
//!
//! [`ReadCursor`] and [`WriteCursor`] wrap a [`Buffer<N>`](crate::buffer::Buffer) reference and
//! track an advancing position. They implement the [`ReadBuf`] and [`WriteBuf`] traits
//! respectively, which erase the const-generic `N` at call sites — consumers only see
//! `&mut impl ReadBuf` or `&mut impl WriteBuf`.
//!
//! Both cursor types are zero-cost abstractions: the trait methods are monomorphized via
//! static dispatch and the per-byte operations inline to the same code as manual
//! `buf[offset]` / `offset += 1` patterns.

use crate::buffer::Buffer;

/// Sequential byte reader.
///
/// Implementors maintain an internal position that advances with each read.
pub(crate) trait ReadBuf {
    /// Reads the next byte and advances the position by one.
    fn read_u8(&mut self) -> u8;

    /// Reads a little-endian `u16` (2 bytes) and advances the position by two.
    fn read_u16_le(&mut self) -> u16;

    /// Reads a little-endian `u32` (4 bytes) and advances the position by four.
    fn read_u32_le(&mut self) -> u32;

    /// Reads `buf.len()` bytes into `buf` and advances the position accordingly.
    fn read_bytes(&mut self, buf: &mut [u8]);
}

/// Sequential byte writer.
///
/// Implementors maintain an internal position that advances with each write.
pub(crate) trait WriteBuf {
    /// Writes one byte and advances the position by one.
    fn write_u8(&mut self, byte: u8);

    /// Writes a `u16` in little-endian order (2 bytes) and advances the position by two.
    fn write_u16_le(&mut self, value: u16);

    /// Writes a `u32` in little-endian order (4 bytes) and advances the position by four.
    fn write_u32_le(&mut self, value: u32);

    /// Writes all bytes from `src` and advances the position accordingly.
    fn write_bytes(&mut self, src: &[u8]);

    /// Copies `len` bytes forward from `self.position() - distance` and advances the position.
    fn copy_within(&mut self, distance: usize, len: usize);

    /// Copies `len` bytes in reverse from `self.position() - distance` and advances the position.
    fn copy_within_rev(&mut self, distance: usize, len: usize);
}

/// A read cursor over a [`Buffer<N>`](crate::buffer::Buffer).
///
/// Wraps an immutable buffer reference and an advancing position. The
/// const-generic `N` is confined to the cursor — callers accepting
/// `&mut impl ReadBuf` never see it.
///
/// # Examples
///
/// ```text
/// let mut buf = Buffer::<256>::new();
/// buf.copy_from_slice(b"hello", 0);
///
/// let mut cur = ReadCursor::new(&buf, 0);
/// assert_eq!(cur.read_u8(), b'h');
/// assert_eq!(cur.read_u8(), b'e');
/// assert_eq!(cur.position(), 2);
/// ```
pub(crate) struct ReadCursor<'a, const N: usize> {
    buf: &'a Buffer<N>,
    pos: usize,
}

impl<'a, const N: usize> ReadCursor<'a, N> {
    /// Creates a read cursor starting at logical position `pos`.
    pub(crate) const fn new(buf: &'a Buffer<N>, pos: usize) -> Self {
        Self {
            buf,
            pos,
        }
    }

    /// Returns the current logical position.
    pub(crate) const fn position(&self) -> usize {
        self.pos
    }
}

impl<const N: usize> ReadBuf for ReadCursor<'_, N> {
    fn read_u8(&mut self) -> u8 {
        let b = self.buf[self.pos];
        self.pos += 1;
        b
    }

    fn read_u16_le(&mut self) -> u16 {
        let lo = self.buf[self.pos];
        let hi = self.buf[self.pos + 1];
        self.pos += 2;
        u16::from_le_bytes([lo, hi])
    }

    fn read_u32_le(&mut self) -> u32 {
        let b0 = self.buf[self.pos];
        let b1 = self.buf[self.pos + 1];
        let b2 = self.buf[self.pos + 2];
        let b3 = self.buf[self.pos + 3];
        self.pos += 4;
        u32::from_le_bytes([b0, b1, b2, b3])
    }

    fn read_bytes(&mut self, buf: &mut [u8]) {
        let (a, b) = self.buf.slices(self.pos, buf.len());
        buf[..a.len()].copy_from_slice(a);
        buf[a.len()..].copy_from_slice(b);
        self.pos += buf.len();
    }
}

/// A write cursor over a [`Buffer<N>`](crate::buffer::Buffer).
///
/// Wraps a mutable buffer reference and an advancing position. The
/// const-generic `N` is confined to the cursor — callers accepting
/// `&mut impl WriteBuf` never see it.
///
/// # Examples
///
/// ```text
/// let mut buf = Buffer::<256>::new();
/// let mut cur = WriteCursor::new(&mut buf, 0);
/// cur.write_u8(0xAA);
/// cur.write_u8(0xBB);
/// assert_eq!(cur.position(), 2);
/// ```
pub(crate) struct WriteCursor<'a, const N: usize> {
    buf: &'a mut Buffer<N>,
    pos: usize,
}

impl<'a, const N: usize> WriteCursor<'a, N> {
    /// Creates a write cursor starting at logical position `pos`.
    pub(crate) const fn new(buf: &'a mut Buffer<N>, pos: usize) -> Self {
        Self {
            buf,
            pos,
        }
    }

    /// Returns the current logical position.
    pub(crate) const fn position(&self) -> usize {
        self.pos
    }
}

impl<const N: usize> WriteBuf for WriteCursor<'_, N> {
    fn write_u8(&mut self, byte: u8) {
        self.buf[self.pos] = byte;
        self.pos += 1;
    }

    fn write_u16_le(&mut self, value: u16) {
        let [lo, hi] = value.to_le_bytes();
        self.buf[self.pos] = lo;
        self.buf[self.pos + 1] = hi;
        self.pos += 2;
    }

    fn write_u32_le(&mut self, value: u32) {
        let [b0, b1, b2, b3] = value.to_le_bytes();
        self.buf[self.pos] = b0;
        self.buf[self.pos + 1] = b1;
        self.buf[self.pos + 2] = b2;
        self.buf[self.pos + 3] = b3;
        self.pos += 4;
    }

    fn write_bytes(&mut self, src: &[u8]) {
        self.buf.copy_from_slice(src, self.pos);
        self.pos += src.len();
    }

    fn copy_within(&mut self, distance: usize, len: usize) {
        let src_start = self.pos - distance;
        self.buf.copy_within(src_start..src_start + len, self.pos);
        self.pos += len;
    }

    fn copy_within_rev(&mut self, distance: usize, len: usize) {
        let src_start = self.pos - distance;
        self.buf.copy_within_rev(src_start..src_start + len, self.pos);
        self.pos += len;
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn read_cursor_sequential() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(b"abcdef", 0);

        let mut cur = ReadCursor::new(&buf, 0);
        assert_eq!(cur.read_u8(), b'a');
        assert_eq!(cur.read_u8(), b'b');
        assert_eq!(cur.read_u8(), b'c');
        assert_eq!(cur.position(), 3);
    }

    #[test]
    fn read_bytes_bulk() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(b"hello world", 0);

        let mut cur = ReadCursor::new(&buf, 0);
        let mut dst = [0u8; 5];
        cur.read_bytes(&mut dst);
        assert_eq!(&dst, b"hello");
        assert_eq!(cur.position(), 5);
    }

    #[test]
    fn write_cursor_sequential() {
        let mut buf = Buffer::<256>::new();

        let mut cur = WriteCursor::new(&mut buf, 0);
        cur.write_u8(0xAA);
        cur.write_u8(0xBB);
        cur.write_u8(0xCC);
        assert_eq!(cur.position(), 3);

        assert_eq!(buf[0], 0xAA);
        assert_eq!(buf[1], 0xBB);
        assert_eq!(buf[2], 0xCC);
    }

    #[test]
    fn write_bytes_bulk() {
        let mut buf = Buffer::<256>::new();

        let mut cur = WriteCursor::new(&mut buf, 0);
        cur.write_bytes(b"hello");
        assert_eq!(cur.position(), 5);

        assert_eq!(buf[0], b'h');
        assert_eq!(buf[4], b'o');
    }

    #[test]
    fn cursor_wraps() {
        let mut buf = Buffer::<256>::new();

        // Write across the wrap boundary (starting at 254).
        let mut wc = WriteCursor::new(&mut buf, 254);
        wc.write_bytes(b"wrap!");
        assert_eq!(wc.position(), 259);

        // Read it back — same logical positions.
        let mut rc = ReadCursor::new(&buf, 254);
        let mut dst = [0u8; 5];
        rc.read_bytes(&mut dst);
        assert_eq!(&dst, b"wrap!");
        assert_eq!(rc.position(), 259);
    }

    #[test]
    fn copy_within_forward() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(b"ABCD", 0);

        let mut wc = WriteCursor::new(&mut buf, 4);
        wc.copy_within(4, 4); // copies from pos 0..4
        assert_eq!(wc.position(), 8);

        assert_eq!(buf[4], b'A');
        assert_eq!(buf[7], b'D');
    }

    #[test]
    fn copy_within_rev_backward() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(b"ABCD", 0);

        let mut wc = WriteCursor::new(&mut buf, 10);
        wc.copy_within_rev(7, 4); // reads from pos 3 backwards: D, C, B, A
        assert_eq!(wc.position(), 14);

        assert_eq!(buf[10], b'D');
        assert_eq!(buf[11], b'C');
        assert_eq!(buf[12], b'B');
        assert_eq!(buf[13], b'A');
    }

    proptest! {
        #[test]
        fn u16_le_roundtrip(value: u16, start in 0usize..=255) {
            let mut buf = Buffer::<256>::new();

            let mut wc = WriteCursor::new(&mut buf, start);
            wc.write_u16_le(value);
            prop_assert_eq!(wc.position(), start + 2);

            let mut rc = ReadCursor::new(&buf, start);
            prop_assert_eq!(rc.read_u16_le(), value);
            prop_assert_eq!(rc.position(), start + 2);
        }

        #[test]
        fn u32_le_roundtrip(value: u32, start in 0usize..=255) {
            let mut buf = Buffer::<256>::new();

            let mut wc = WriteCursor::new(&mut buf, start);
            wc.write_u32_le(value);
            prop_assert_eq!(wc.position(), start + 4);

            let mut rc = ReadCursor::new(&buf, start);
            prop_assert_eq!(rc.read_u32_le(), value);
            prop_assert_eq!(rc.position(), start + 4);
        }

        #[test]
        fn roundtrip(data in prop::collection::vec(any::<u8>(), 1..=256), start in 0usize..=255) {
            let mut buf = Buffer::<256>::new();

            let mut wc = WriteCursor::new(&mut buf, start);
            wc.write_bytes(&data);
            prop_assert_eq!(wc.position(), start + data.len());

            let mut rc = ReadCursor::new(&buf, start);
            let mut dst = vec![0u8; data.len()];
            rc.read_bytes(&mut dst);
            prop_assert_eq!(rc.position(), start + data.len());
            prop_assert_eq!(dst, data);
        }
    }
}
