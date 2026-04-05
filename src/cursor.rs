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

use std::io::{self, Read};

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
/// Wraps a mutable buffer reference, an advancing read position, and a fill
/// level (`end`). The fill level tracks how far the buffer has been populated
/// from an external source — bytes between `pos` and `end` are available for
/// reading.
///
/// For streaming use, create with `end = pos` (empty) and call [`refill`](Self::refill)
/// or [`ensure`](Self::ensure) to fill from an [`impl Read`](Read) source. For
/// non-streaming use (e.g. tests), set `end` to cover the pre-populated data.
///
/// # Examples
///
/// ```text
/// let mut buf = Buffer::<256>::new();
/// buf.copy_from_slice(b"hello", 0);
///
/// let mut cur = ReadCursor::new(&mut buf, 0, 5);
/// assert_eq!(cur.read_u8(), b'h');
/// assert_eq!(cur.read_u8(), b'e');
/// assert_eq!(cur.position(), 2);
/// assert_eq!(cur.remaining(), 3);
/// ```
pub(crate) struct ReadCursor<'a, const N: usize> {
    buf: &'a mut Buffer<N>,
    pos: usize,
    end: usize,
}

impl<'a, const N: usize> ReadCursor<'a, N> {
    /// Creates a read cursor starting at logical position `pos` with data
    /// available up to `end`.
    pub(crate) const fn new(buf: &'a mut Buffer<N>, pos: usize, end: usize) -> Self {
        Self {
            buf,
            pos,
            end,
        }
    }

    /// Returns the current logical position.
    pub(crate) const fn position(&self) -> usize {
        self.pos
    }

    /// Bytes available between the read position and the fill level.
    pub(crate) const fn remaining(&self) -> usize {
        self.end - self.pos
    }

    /// Reads from `src` to fill available space in the buffer.
    ///
    /// Returns the number of bytes read (0 means the source is exhausted).
    pub(crate) fn refill(&mut self, src: &mut impl Read) -> io::Result<usize> {
        let free = N - self.remaining();
        if free == 0 {
            return Ok(0);
        }
        let n = self.buf.read_from(self.end, free, src)?;
        self.end += n;
        Ok(n)
    }

    /// Loops [`refill`](Self::refill) until at least `need` bytes are available.
    ///
    /// Returns [`UnexpectedEof`](io::ErrorKind::UnexpectedEof) if the source is
    /// exhausted before `need` bytes are buffered.
    pub(crate) fn ensure(&mut self, src: &mut impl Read, need: usize) -> io::Result<()> {
        while self.remaining() < need {
            if self.refill(src)? == 0 {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
            }
        }
        Ok(())
    }

    /// Best-effort fill toward `target` bytes. Does not error on EOF.
    pub(crate) fn fill_toward(&mut self, src: &mut impl Read, target: usize) -> io::Result<()> {
        while self.remaining() < target {
            if self.refill(src)? == 0 {
                return Ok(());
            }
        }
        Ok(())
    }

    /// Peeks at a byte at `offset` from the current position without advancing.
    pub(crate) fn peek(&self, offset: usize) -> u8 {
        self.buf[self.pos + offset]
    }

    /// Consumes `len` bytes, returning them as one or two contiguous slices.
    ///
    /// Advances the position by `len`.
    pub(crate) fn consume(&mut self, len: usize) -> (&[u8], &[u8]) {
        let pos = self.pos;
        self.pos += len;
        self.buf.slices(pos, len)
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
/// Wraps a mutable buffer reference, an advancing write position, and a flush
/// watermark. Bytes between `flushed` and `pos` are pending — call
/// [`flush`](Self::flush) to drain them to an external sink.
///
/// # Examples
///
/// ```text
/// let mut buf = Buffer::<256>::new();
/// let mut cur = WriteCursor::new(&mut buf, 0);
/// cur.write_u8(0xAA);
/// cur.write_u8(0xBB);
/// assert_eq!(cur.position(), 2);
/// assert_eq!(cur.pending(), 2);
/// ```
pub(crate) struct WriteCursor<'a, const N: usize> {
    buf: &'a mut Buffer<N>,
    pos: usize,
    flushed: usize,
}

impl<'a, const N: usize> WriteCursor<'a, N> {
    /// Creates a write cursor starting at logical position `pos` with the
    /// flush watermark initialized to `pos` (nothing pending).
    pub(crate) const fn new(buf: &'a mut Buffer<N>, pos: usize) -> Self {
        Self {
            buf,
            pos,
            flushed: pos,
        }
    }

    /// Returns the current logical position.
    pub(crate) const fn position(&self) -> usize {
        self.pos
    }

    /// Bytes written but not yet flushed.
    pub(crate) const fn pending(&self) -> usize {
        self.pos - self.flushed
    }

    /// Flushes pending bytes through `sink`.
    ///
    /// Calls `sink` once per contiguous slice (at most twice if the range wraps
    /// the ring boundary). Updates the watermark after each successful call so
    /// partial progress is preserved on error.
    pub(crate) fn flush(&mut self, mut sink: impl FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
        let pending = self.pos - self.flushed;
        if pending == 0 {
            return Ok(());
        }
        let (a, b) = self.buf.slices(self.flushed, pending);
        sink(a)?;
        self.flushed += a.len();
        if !b.is_empty() {
            sink(b)?;
            self.flushed += b.len();
        }
        Ok(())
    }

    /// Resets position and flush watermark to zero for a new stream.
    pub(crate) const fn reset(&mut self) {
        self.pos = 0;
        self.flushed = 0;
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

        let mut cur = ReadCursor::new(&mut buf, 0, 6);
        assert_eq!(cur.read_u8(), b'a');
        assert_eq!(cur.read_u8(), b'b');
        assert_eq!(cur.read_u8(), b'c');
        assert_eq!(cur.position(), 3);
    }

    #[test]
    fn read_bytes_bulk() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(b"hello world", 0);

        let mut cur = ReadCursor::new(&mut buf, 0, 11);
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
        let mut rc = ReadCursor::new(&mut buf, 254, 259);
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
            let end = wc.position();
            prop_assert_eq!(end, start + 2);

            let mut rc = ReadCursor::new(&mut buf, start, end);
            prop_assert_eq!(rc.read_u16_le(), value);
            prop_assert_eq!(rc.position(), start + 2);
        }

        #[test]
        fn u32_le_roundtrip(value: u32, start in 0usize..=255) {
            let mut buf = Buffer::<256>::new();

            let mut wc = WriteCursor::new(&mut buf, start);
            wc.write_u32_le(value);
            let end = wc.position();
            prop_assert_eq!(end, start + 4);

            let mut rc = ReadCursor::new(&mut buf, start, end);
            prop_assert_eq!(rc.read_u32_le(), value);
            prop_assert_eq!(rc.position(), start + 4);
        }

        #[test]
        fn roundtrip(data in prop::collection::vec(any::<u8>(), 1..=256), start in 0usize..=255) {
            let mut buf = Buffer::<256>::new();

            let mut wc = WriteCursor::new(&mut buf, start);
            wc.write_bytes(&data);
            let end = wc.position();
            prop_assert_eq!(end, start + data.len());

            let mut rc = ReadCursor::new(&mut buf, start, end);
            let mut dst = vec![0u8; data.len()];
            rc.read_bytes(&mut dst);
            prop_assert_eq!(rc.position(), start + data.len());
            prop_assert_eq!(dst, data);
        }
    }

    // --- ReadCursor stream method tests ---

    #[test]
    fn remaining_tracks_fill_level() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(b"hello", 0);
        let cur = ReadCursor::new(&mut buf, 0, 5);
        assert_eq!(cur.remaining(), 5);
    }

    #[test]
    fn refill_fills_from_reader() {
        let mut buf = Buffer::<256>::new();
        let mut cur = ReadCursor::new(&mut buf, 0, 0);
        assert_eq!(cur.remaining(), 0);

        let n = cur.refill(&mut &b"hello"[..]).unwrap();
        assert_eq!(n, 5);
        assert_eq!(cur.remaining(), 5);
    }

    #[test]
    fn ensure_blocks_until_available() {
        let mut buf = Buffer::<256>::new();
        let mut cur = ReadCursor::new(&mut buf, 0, 0);
        cur.ensure(&mut &b"hello world"[..], 5).unwrap();
        assert!(cur.remaining() >= 5);
    }

    #[test]
    fn ensure_errors_on_eof() {
        let mut buf = Buffer::<256>::new();
        let mut cur = ReadCursor::new(&mut buf, 0, 0);
        let err = cur.ensure(&mut &b"hi"[..], 10).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn fill_toward_does_not_error_on_eof() {
        let mut buf = Buffer::<256>::new();
        let mut cur = ReadCursor::new(&mut buf, 0, 0);
        cur.fill_toward(&mut &b"hi"[..], 10).unwrap();
        assert_eq!(cur.remaining(), 2);
    }

    #[test]
    fn peek_does_not_advance() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(b"abc", 0);
        let cur = ReadCursor::new(&mut buf, 0, 3);
        assert_eq!(cur.peek(0), b'a');
        assert_eq!(cur.peek(2), b'c');
        assert_eq!(cur.position(), 0);
    }

    #[test]
    fn consume_returns_slices_and_advances() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(b"hello", 0);
        let mut cur = ReadCursor::new(&mut buf, 0, 5);
        let (a, b) = cur.consume(5);
        let mut v = Vec::from(a);
        v.extend_from_slice(b);
        assert_eq!(v, b"hello");
        assert_eq!(cur.position(), 5);
    }

    // --- WriteCursor flush method tests ---

    #[test]
    fn pending_tracks_unflushed() {
        let mut buf = Buffer::<256>::new();
        let mut cur = WriteCursor::new(&mut buf, 0);
        assert_eq!(cur.pending(), 0);
        cur.write_bytes(b"hello");
        assert_eq!(cur.pending(), 5);
    }

    #[test]
    fn flush_writes_pending_bytes() {
        let mut buf = Buffer::<256>::new();
        let mut cur = WriteCursor::new(&mut buf, 0);
        cur.write_bytes(b"hello");

        let mut out = Vec::new();
        cur.flush(|bytes| {
            out.extend_from_slice(bytes);
            Ok(())
        })
        .unwrap();
        assert_eq!(out, b"hello");
    }

    #[test]
    fn flush_noop_when_empty() {
        let mut buf = Buffer::<256>::new();
        let mut cur = WriteCursor::new(&mut buf, 0);
        let mut called = false;
        cur.flush(|_| {
            called = true;
            Ok(())
        })
        .unwrap();
        assert!(!called);
    }

    #[test]
    fn flush_updates_watermark() {
        let mut buf = Buffer::<256>::new();
        let mut cur = WriteCursor::new(&mut buf, 0);
        cur.write_bytes(b"hello");
        cur.flush(|_| Ok(())).unwrap();
        assert_eq!(cur.pending(), 0);
    }

    #[test]
    fn flush_wraps_around() {
        let mut buf = Buffer::<256>::new();
        let mut cur = WriteCursor::new(&mut buf, 254);
        cur.write_bytes(b"wrap!");

        let mut out = Vec::new();
        cur.flush(|bytes| {
            out.extend_from_slice(bytes);
            Ok(())
        })
        .unwrap();
        assert_eq!(out, b"wrap!");
        assert_eq!(cur.pending(), 0);
    }

    #[test]
    fn reset_zeroes_state() {
        let mut buf = Buffer::<256>::new();
        let mut cur = WriteCursor::new(&mut buf, 0);
        cur.write_bytes(b"hello");
        cur.flush(|_| Ok(())).unwrap();
        cur.reset();
        assert_eq!(cur.position(), 0);
        assert_eq!(cur.pending(), 0);
    }
}
