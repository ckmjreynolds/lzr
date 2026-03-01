/// Writes a stream of 4-bit nibbles, packing two per byte.
///
/// Packing convention (high-first):
/// - Nibble 2K   → byte K, bits 7–4
/// - Nibble 2K+1 → byte K, bits 3–0
/// - Odd total   → final byte's low nibble is zero padding
#[derive(Debug, Clone)]
pub(crate) struct NibbleWriter {
    buf: Vec<u8>,
    len: usize,
}

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
        if self.len.is_multiple_of(2) {
            // Byte-aligned: pack both nibbles directly (nibble-swap).
            self.buf.push(byte.rotate_left(4));
            self.len += 2;
        } else {
            self.push(byte & 0xF);
            self.push(byte >> 4);
        }
    }

    /// Appends a slice of bytes, each as two nibbles: low nibble first, then high.
    pub(crate) fn push_bytes(&mut self, bytes: &[u8]) {
        if self.len.is_multiple_of(2) {
            // Byte-aligned: nibble-swap each byte and extend directly.
            self.buf.reserve(bytes.len());
            for &b in bytes {
                self.buf.push(b.rotate_left(4));
            }
            self.len += bytes.len() * 2;
        } else {
            for &b in bytes {
                self.push_byte(b);
            }
        }
    }

    /// Consumes the writer and returns the packed byte buffer.
    pub(crate) fn finish(self) -> Vec<u8> {
        self.buf
    }
}

/// Trait for types that can produce nibbles one at a time.
pub(crate) trait ReadNibble {
    /// The error type returned on failure.
    type Error;

    /// Reads the next 4-bit nibble (0x0–0xF).
    fn read_nibble(&mut self) -> Result<u8, Self::Error>;

    /// Reads two nibbles and reassembles them into a byte (low nibble first).
    fn read_byte(&mut self) -> Result<u8, Self::Error> {
        let lo = self.read_nibble()?;
        let hi = self.read_nibble()?;
        Ok((hi << 4) | lo)
    }
}

/// Reads a stream of 4-bit nibbles from a packed byte slice (test-only).
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct NibbleReader<'a> {
    data: &'a [u8],
    pos: usize,
}

#[cfg(test)]
impl<'a> NibbleReader<'a> {
    /// Creates a reader over the given byte slice.
    pub(crate) const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
        }
    }

    /// Reads the next nibble, or returns `Err(UnexpectedEnd)`.
    pub(crate) fn read(&mut self) -> Result<u8, crate::error::Error> {
        let byte = *self.data.get(self.pos / 2).ok_or(crate::error::Error::UnexpectedEnd)?;
        let nibble = if self.pos.is_multiple_of(2) {
            byte >> 4
        } else {
            byte & 0xF
        };
        self.pos += 1;
        Ok(nibble)
    }
}

#[cfg(test)]
impl ReadNibble for NibbleReader<'_> {
    type Error = crate::error::Error;

    fn read_nibble(&mut self) -> Result<u8, crate::error::Error> {
        self.read()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn worked_example_hi() {
        // From FORMAT.md: nibbles 0,2,8,4,9,6,0,0 → [0x02, 0x84, 0x96, 0x00]
        let mut w = NibbleWriter::new();
        for &n in &[0x0, 0x2, 0x8, 0x4, 0x9, 0x6, 0x0, 0x0] {
            w.push(n);
        }
        assert_eq!(w.finish(), vec![0x02, 0x84, 0x96, 0x00]);
    }

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
