//! Nibble-oriented I/O with BLEB8 variable-length integer encoding.

use std::io::{Read, Write};

use crate::error::Result;

/// Writes nibbles to an inner byte-oriented writer.
///
/// Nibbles pack into bytes high-first: nibble 2K occupies bits 7–4,
/// nibble 2K+1 occupies bits 3–0.
#[derive(Debug)]
pub(crate) struct NibbleWriter<W> {
    inner: W,
    /// High nibble waiting for its low-nibble pair.
    pending: Option<u8>,
}

impl<W: Write> NibbleWriter<W> {
    /// Creates a new nibble writer wrapping the given writer.
    #[must_use]
    pub(crate) const fn new(inner: W) -> Self {
        Self {
            inner,
            pending: None,
        }
    }

    /// Writes a single nibble (0x0..=0xF).
    #[inline]
    pub(crate) fn write_nibble(&mut self, nibble: u8) -> Result<()> {
        match self.pending.take() {
            None => self.pending = Some(nibble),
            Some(high) => self.inner.write_all(&[(high << 4) | nibble])?,
        }

        Ok(())
    }

    /// Writes a literal byte as two nibbles (low nibble first, high nibble second).
    #[inline]
    pub(crate) fn write_literal_byte(&mut self, byte: u8) -> Result<()> {
        if self.pending.is_none() {
            self.inner.write_all(&[byte.rotate_right(4)])?;
        } else {
            self.write_nibble(byte & 0xF)?;
            self.write_nibble(byte >> 4)?;
        }

        Ok(())
    }

    /// Encodes a `u16` as UBLEB8 (max 5 nibbles).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn write_ubleb8_u16(&mut self, mut value: u16) -> Result<()> {
        let nibbles = (16 - value.leading_zeros()).div_ceil(3).clamp(1, 5) as usize;

        for _ in 1..nibbles {
            self.write_nibble((value & 0x7) as u8 | 0x8)?;
            value >>= 3;
        }
        self.write_nibble(value as u8)?;
        Ok(())
    }

    /// Encodes a `u64` as UBLEB8 (max 21 nibbles).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn write_ubleb8_u64(&mut self, mut value: u64) -> Result<()> {
        let nibbles = (64 - value.leading_zeros()).div_ceil(3).clamp(1, 21) as usize;

        for _ in 1..nibbles {
            self.write_nibble((value & 0x7) as u8 | 0x8)?;
            value >>= 3;
        }
        self.write_nibble(value as u8)?;
        Ok(())
    }

    /// Encodes an `i16` as SBLEB8 (max 5 nibbles).
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub(crate) fn write_sbleb8_i16(&mut self, mut value: i16) -> Result<()> {
        let nibbles = (16 - ((value ^ (value >> 15)) as u16).leading_zeros() + 1).div_ceil(3).clamp(1, 5) as usize;
        let mask = if nibbles == 5 {
            0xF
        } else {
            0x7
        };

        for _ in 1..nibbles {
            self.write_nibble((value & 0x7) as u8 | 0x8)?;
            value >>= 3;
        }
        self.write_nibble((value & mask) as u8)?;
        Ok(())
    }

    /// Pads to byte boundary if needed and returns the inner writer.
    pub(crate) fn finish(mut self) -> Result<W> {
        if self.pending.is_some() {
            self.write_nibble(0)?;
        }
        Ok(self.inner)
    }
}

/// Reads nibbles from an inner byte-oriented reader.
///
/// Nibbles unpack from bytes high-first: nibble 2K from bits 7–4,
/// nibble 2K+1 from bits 3–0.
#[derive(Debug)]
pub(crate) struct NibbleReader<R> {
    inner: R,
    /// Low nibble from a previously read byte.
    pending: Option<u8>,
}

impl<R: Read> NibbleReader<R> {
    /// Creates a new nibble reader wrapping the given reader.
    #[must_use]
    pub(crate) const fn new(inner: R) -> Self {
        Self {
            inner,
            pending: None,
        }
    }

    /// Reads a single nibble (0x0..=0xF).
    #[inline]
    pub(crate) fn read_nibble(&mut self) -> Result<u8> {
        if let Some(nibble) = self.pending.take() {
            Ok(nibble)
        } else {
            let mut buf = [0];
            self.inner.read_exact(&mut buf)?;
            self.pending = Some(buf[0] & 0xF);
            Ok(buf[0] >> 4)
        }
    }

    /// Reads a literal byte from two nibbles (low nibble first, high nibble second).
    #[inline]
    pub(crate) fn read_literal_byte(&mut self) -> Result<u8> {
        if self.pending.is_none() {
            let mut buf = [0];
            self.inner.read_exact(&mut buf)?;
            Ok(buf[0].rotate_left(4))
        } else {
            let low = self.read_nibble()?;
            let high = self.read_nibble()?;
            Ok((high << 4) | low)
        }
    }

    /// Decodes a UBLEB8-encoded `u16` (max 5 nibbles).
    pub(crate) fn read_ubleb8_u16(&mut self) -> Result<u16> {
        let mut value: u16 = 0;
        let mut shift = 0;

        for _ in 0..4 {
            let nibble = self.read_nibble()?;
            value |= u16::from(nibble & 0x7) << shift;
            shift += 3;
            if nibble & 0x8 == 0 {
                return Ok(value);
            }
        }
        let nibble = self.read_nibble()?;
        value |= u16::from(nibble) << shift;
        Ok(value)
    }

    /// Decodes a UBLEB8-encoded `u64` (max 21 nibbles).
    pub(crate) fn read_ubleb8_u64(&mut self) -> Result<u64> {
        let mut value: u64 = 0;
        let mut shift = 0;

        for _ in 0..20 {
            let nibble = self.read_nibble()?;
            value |= u64::from(nibble & 0x7) << shift;
            shift += 3;
            if nibble & 0x8 == 0 {
                return Ok(value);
            }
        }
        let nibble = self.read_nibble()?;
        value |= u64::from(nibble) << shift;
        Ok(value)
    }

    /// Decodes an SBLEB8-encoded `i16` (max 5 nibbles).
    #[allow(clippy::cast_possible_wrap)]
    pub(crate) fn read_sbleb8_i16(&mut self) -> Result<i16> {
        let mut value: u16 = 0;
        let mut shift = 0;

        for _ in 0..4 {
            let nibble = self.read_nibble()?;
            value |= u16::from(nibble & 0x7) << shift;
            shift += 3;
            if nibble & 0x8 == 0 {
                if value & (1 << (shift - 1)) != 0 {
                    value |= !0 << shift;
                }
                return Ok(value as i16);
            }
        }
        let nibble = self.read_nibble()?;
        value |= u16::from(nibble) << shift;
        Ok(value as i16)
    }

    /// Discards any pending nibble and returns the inner reader at byte alignment.
    pub(crate) fn into_inner(self) -> R {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn round_trip_literal_byte(byte: u8) {
            let mut writer = NibbleWriter::new(Vec::new());
            writer.write_literal_byte(byte).unwrap();
            let buf = writer.finish().unwrap();
            let mut reader = NibbleReader::new(buf.as_slice());
            let result = reader.read_literal_byte().unwrap();
            prop_assert_eq!(result, byte);
        }

        #[test]
        fn round_trip_ubleb8_u16(value: u16) {
            let mut writer = NibbleWriter::new(Vec::new());
            writer.write_ubleb8_u16(value).unwrap();
            let buf = writer.finish().unwrap();
            let mut reader = NibbleReader::new(buf.as_slice());
            let result = reader.read_ubleb8_u16().unwrap();
            prop_assert_eq!(result, value);
        }

        #[test]
        fn round_trip_ubleb8_u64(value: u64) {
            let mut writer = NibbleWriter::new(Vec::new());
            writer.write_ubleb8_u64(value).unwrap();
            let buf = writer.finish().unwrap();
            let mut reader = NibbleReader::new(buf.as_slice());
            let result = reader.read_ubleb8_u64().unwrap();
            prop_assert_eq!(result, value);
        }

        #[test]
        fn round_trip_sbleb8_i16(value: i16) {
            let mut writer = NibbleWriter::new(Vec::new());
            writer.write_sbleb8_i16(value).unwrap();
            let buf = writer.finish().unwrap();
            let mut reader = NibbleReader::new(buf.as_slice());
            let result = reader.read_sbleb8_i16().unwrap();
            prop_assert_eq!(result, value);
        }
    }
}
