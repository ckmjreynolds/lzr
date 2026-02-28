use crate::error::{Error, Result};

/// Writes a stream of 4-bit nibbles, packing two per byte.
///
/// Packing convention (high-first):
/// - Nibble 2K   → byte K, bits 7–4
/// - Nibble 2K+1 → byte K, bits 3–0
/// - Odd total   → final byte's low nibble is zero padding
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct NibbleWriter {
    buf: Vec<u8>,
    len: usize,
}

#[allow(dead_code)]
impl NibbleWriter {
    /// Creates a new, empty writer.
    pub(crate) const fn new() -> Self {
        Self {
            buf: Vec::new(),
            len: 0,
        }
    }

    /// Creates a writer pre-allocated for the given number of nibbles.
    pub(crate) fn with_capacity(nibbles: usize) -> Self {
        Self {
            buf: Vec::with_capacity(nibbles.div_ceil(2)),
            len: 0,
        }
    }

    /// Returns the number of nibbles written so far.
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if no nibbles have been written.
    pub(crate) const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends a single nibble (must be 0x0–0xF).
    pub(crate) fn push(&mut self, nibble: u8) {
        debug_assert!(nibble <= 0xF, "nibble out of range: {nibble:#X}");
        let nibble = nibble & 0xF;

        if self.len.is_multiple_of(2) {
            // Even index → high nibble of a new byte.
            self.buf.push(nibble << 4);
        } else {
            // Odd index → low nibble of the last byte.
            *self.buf.last_mut().expect("odd len implies non-empty buf") |= nibble;
        }
        self.len += 1;
    }

    /// Appends a full byte as two nibbles: low nibble first, then high nibble.
    pub(crate) fn push_byte(&mut self, byte: u8) {
        self.push(byte & 0xF);
        self.push(byte >> 4);
    }

    /// Consumes the writer and returns the packed byte buffer.
    pub(crate) fn finish(self) -> Vec<u8> {
        self.buf
    }
}

/// Reads a stream of 4-bit nibbles from a packed byte slice.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct NibbleReader<'a> {
    data: &'a [u8],
    pos: usize,
}

#[allow(dead_code)]
impl<'a> NibbleReader<'a> {
    /// Creates a reader over the given byte slice.
    pub(crate) const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
        }
    }

    /// Returns the current nibble position.
    pub(crate) const fn position(&self) -> usize {
        self.pos
    }

    /// Returns the number of nibbles remaining (may include a padding nibble).
    pub(crate) const fn remaining(&self) -> usize {
        self.data.len() * 2 - self.pos
    }

    /// Returns `true` if all nibbles have been consumed.
    pub(crate) const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Reads the next nibble, or returns `Err(UnexpectedEnd)`.
    pub(crate) fn read(&mut self) -> Result<u8> {
        let byte = *self.data.get(self.pos / 2).ok_or(Error::UnexpectedEnd)?;
        let nibble = if self.pos.is_multiple_of(2) {
            byte >> 4
        } else {
            byte & 0xF
        };
        self.pos += 1;
        Ok(nibble)
    }

    /// Reads two nibbles and reassembles them into a byte (low nibble first).
    pub(crate) fn read_byte(&mut self) -> Result<u8> {
        let lo = self.read()?;
        let hi = self.read()?;
        Ok((hi << 4) | lo)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    // ── Deterministic: Writer ──────────────────────────────────────────

    #[test]
    fn empty_writer() {
        let w = NibbleWriter::new();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
        assert_eq!(w.finish(), Vec::<u8>::new());
    }

    #[test]
    fn single_nibble_pads() {
        let mut w = NibbleWriter::new();
        w.push(0xA);
        assert_eq!(w.len(), 1);
        assert!(!w.is_empty());
        assert_eq!(w.finish(), vec![0xA0]);
    }

    #[test]
    fn two_nibbles_no_pad() {
        let mut w = NibbleWriter::new();
        w.push(0xA);
        w.push(0x5);
        assert_eq!(w.finish(), vec![0xA5]);
    }

    #[test]
    fn three_nibbles_pads() {
        let mut w = NibbleWriter::new();
        w.push(0xA);
        w.push(0x5);
        w.push(0x3);
        assert_eq!(w.finish(), vec![0xA5, 0x30]);
    }

    #[test]
    fn push_byte_literal() {
        // 0x48 → low nibble 0x8 first, high nibble 0x4 second → [0x84]
        let mut w = NibbleWriter::new();
        w.push_byte(0x48);
        assert_eq!(w.finish(), vec![0x84]);
    }

    #[test]
    fn worked_example_hi() {
        // From FORMAT.md: nibbles 0,2,8,4,9,6,0,0 → [0x02, 0x84, 0x96, 0x00]
        let mut w = NibbleWriter::new();
        for &n in &[0x0, 0x2, 0x8, 0x4, 0x9, 0x6, 0x0, 0x0] {
            w.push(n);
        }
        assert_eq!(w.finish(), vec![0x02, 0x84, 0x96, 0x00]);
    }

    // ── Deterministic: Reader ──────────────────────────────────────────

    #[test]
    fn reader_empty() {
        let mut r = NibbleReader::new(&[]);
        assert!(r.is_empty());
        assert_eq!(r.remaining(), 0);
        assert_eq!(r.read(), Err(Error::UnexpectedEnd));
    }

    #[test]
    fn reader_single_byte() {
        let mut r = NibbleReader::new(&[0xA5]);
        assert_eq!(r.remaining(), 2);
        assert_eq!(r.read(), Ok(0xA));
        assert_eq!(r.read(), Ok(0x5));
        assert_eq!(r.read(), Err(Error::UnexpectedEnd));
    }

    #[test]
    fn reader_read_byte() {
        // [0x84] → lo=0x8, hi=0x4 → 0x48
        let mut r = NibbleReader::new(&[0x84]);
        assert_eq!(r.read_byte(), Ok(0x48));
    }

    #[test]
    fn reader_round_trip_worked_example() {
        let data = [0x02, 0x84, 0x96, 0x00];
        let mut r = NibbleReader::new(&data);
        let expected = [0x0, 0x2, 0x8, 0x4, 0x9, 0x6, 0x0, 0x0];
        for &want in &expected {
            assert_eq!(r.read(), Ok(want));
        }
        assert!(r.is_empty());
    }

    #[test]
    fn reader_copy_checkpoint() {
        let mut r = NibbleReader::new(&[0xAB, 0xCD]);
        assert_eq!(r.read(), Ok(0xA));
        assert_eq!(r.read(), Ok(0xB));

        let checkpoint = r; // Copy at position 2
        assert_eq!(r.read(), Ok(0xC));
        assert_eq!(r.read(), Ok(0xD));

        // Checkpoint is still at position 2.
        assert_eq!(checkpoint.position(), 2);
        assert_eq!(checkpoint.remaining(), 2);
    }

    // ── Property-based ─────────────────────────────────────────────────

    proptest! {
        #[test]
        fn round_trip_arbitrary_nibbles(nibbles in prop::collection::vec(0u8..=0xF, 0..256)) {
            let mut w = NibbleWriter::with_capacity(nibbles.len());
            for &n in &nibbles {
                w.push(n);
            }
            let packed = w.finish();
            let mut r = NibbleReader::new(&packed);
            for &want in &nibbles {
                assert_eq!(r.read().unwrap(), want);
            }
        }

        #[test]
        fn push_byte_equals_two_pushes(byte in proptest::num::u8::ANY) {
            let mut w1 = NibbleWriter::new();
            w1.push_byte(byte);

            let mut w2 = NibbleWriter::new();
            w2.push(byte & 0xF);
            w2.push(byte >> 4);

            assert_eq!(w1.finish(), w2.finish());
        }

        #[test]
        fn round_trip_arbitrary_bytes(bytes in prop::collection::vec(proptest::num::u8::ANY, 0..128)) {
            let mut w = NibbleWriter::new();
            for &b in &bytes {
                w.push_byte(b);
            }
            let packed = w.finish();
            let mut r = NibbleReader::new(&packed);
            for &want in &bytes {
                assert_eq!(r.read_byte().unwrap(), want);
            }
        }
    }
}
