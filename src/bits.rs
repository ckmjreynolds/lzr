//! Bit-level I/O over byte buffers.
//!
//! `BitWriter` packs bits LSB-first into a `Vec<u8>`; `BitReader`
//! reads them back in the same order. Both are used by codecs that
//! emit fractional-bit symbols (dictionary indices, AC ranges, etc.)
//! and need bit-aligned packing to avoid 7-bit-per-byte rounding
//! losses.
//!
//! Padding: `BitWriter::finish` pads the final byte with zero bits.
//! The padding bit count (0-7) is returned so the codec can attribute
//! it to a `"padding"` decomposition bucket.

// `write_byte` / `read_byte` were used by v3 codecs we stripped from
// v4. They stay here as part of the bit-IO API for the moment; future
// v4 codecs (runtime n-gram framing, etc.) will exercise them again.
#![allow(dead_code)]

use anyhow::{Result, bail};

#[derive(Debug)]
pub(crate) struct BitWriter {
    buf: Vec<u8>,
    /// Number of valid bits currently in the in-progress byte at
    /// the end of `buf`. Always in `0..8`.
    bits_in_last: u8,
}

impl BitWriter {
    pub(crate) const fn new() -> Self {
        Self {
            buf: Vec::new(),
            bits_in_last: 0,
        }
    }

    /// Write `count` low-order bits of `value` (LSB-first).
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn write_bits(&mut self, value: u64, count: u8) {
        debug_assert!(count <= 64, "write_bits count must be ≤ 64");
        let mut remaining = count;
        let mut v = value;
        while remaining > 0 {
            if self.bits_in_last == 0 {
                self.buf.push(0);
            }
            let last_idx = self.buf.len() - 1;
            let space = 8 - self.bits_in_last;
            let take = remaining.min(space);
            let mask = if take == 64 {
                u64::MAX
            } else {
                (1u64 << take) - 1
            };
            let chunk = (v & mask) as u8;
            self.buf[last_idx] |= chunk << self.bits_in_last;
            v >>= take;
            self.bits_in_last = (self.bits_in_last + take) & 7;
            remaining -= take;
        }
    }

    /// Write the low 8 bits of `byte`.
    pub(crate) fn write_byte(&mut self, byte: u8) {
        self.write_bits(u64::from(byte), 8);
    }

    /// Flush to byte boundary and return `(buffer, padding_bits)`.
    /// `padding_bits` is in `0..8` — the number of zero bits added
    /// to fill the final byte.
    pub(crate) fn finish(self) -> (Vec<u8>, u8) {
        let pad = if self.bits_in_last == 0 {
            0
        } else {
            8 - self.bits_in_last
        };
        (self.buf, pad)
    }

    /// Total bits written so far (excluding any padding).
    ///
    /// `bits_in_last == 0` is ambiguous on its own — it can mean
    /// either "buffer is empty" or "last byte is full and no new byte
    /// is in progress yet." We disambiguate via `buf.is_empty()`.
    pub(crate) fn bits_written(&self) -> u64 {
        if self.buf.is_empty() {
            0
        } else if self.bits_in_last == 0 {
            // Every byte in `buf` is full.
            self.buf.len() as u64 * 8
        } else {
            (self.buf.len() - 1) as u64 * 8 + u64::from(self.bits_in_last)
        }
    }
}

#[derive(Debug)]
pub(crate) struct BitReader<'a> {
    buf: &'a [u8],
    /// Bit position within `buf`, LSB-first. `pos / 8` is the byte
    /// index; `pos % 8` is the bit offset within that byte.
    pos: u64,
}

impl<'a> BitReader<'a> {
    pub(crate) const fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn read_bits(&mut self, count: u8) -> Result<u64> {
        debug_assert!(count <= 64, "read_bits count must be ≤ 64");
        let mut result = 0u64;
        let mut shift = 0u8;
        let mut remaining = count;
        while remaining > 0 {
            let byte_idx = (self.pos / 8) as usize;
            let bit_off = (self.pos % 8) as u8;
            if byte_idx >= self.buf.len() {
                bail!(
                    "bit-stream underflow: tried to read past byte {byte_idx} (buffer is {} bytes)",
                    self.buf.len(),
                );
            }
            let space = 8 - bit_off;
            let take = remaining.min(space);
            let mask = if take == 8 { 0xFFu8 } else { (1u8 << take) - 1 };
            let chunk = (self.buf[byte_idx] >> bit_off) & mask;
            result |= u64::from(chunk) << shift;
            shift += take;
            self.pos += u64::from(take);
            remaining -= take;
        }
        Ok(result)
    }

    pub(crate) fn read_byte(&mut self) -> Result<u8> {
        let v = self.read_bits(8)?;
        Ok(u8::try_from(v).expect("read_bits(8) result fits u8"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_bytes_roundtrips() {
        let mut w = BitWriter::new();
        for &b in b"hello\x00\xFF\x80" {
            w.write_byte(b);
        }
        let (buf, pad) = w.finish();
        assert_eq!(pad, 0);
        assert_eq!(buf, b"hello\x00\xFF\x80");

        let mut r = BitReader::new(&buf);
        let out: Vec<u8> = (0..buf.len()).map(|_| r.read_byte().unwrap()).collect();
        assert_eq!(out, b"hello\x00\xFF\x80");
    }

    #[test]
    fn fractional_bits_pack_and_unpack() {
        let mut w = BitWriter::new();
        w.write_bits(0b1, 1);
        w.write_bits(0b10101, 5);
        w.write_bits(0b11, 2);
        w.write_bits(0xAB, 8);
        let (buf, _pad) = w.finish();

        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(1).unwrap(), 0b1);
        assert_eq!(r.read_bits(5).unwrap(), 0b10101);
        assert_eq!(r.read_bits(2).unwrap(), 0b11);
        assert_eq!(r.read_bits(8).unwrap(), 0xAB);
    }

    #[test]
    fn padding_reports_correct_bit_count() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        let (buf, pad) = w.finish();
        assert_eq!(buf.len(), 1);
        assert_eq!(pad, 5);
    }

    #[test]
    fn empty_writer_returns_empty_buffer() {
        let w = BitWriter::new();
        let (buf, pad) = w.finish();
        assert_eq!(buf, Vec::<u8>::new());
        assert_eq!(pad, 0);
    }

    #[test]
    fn underflow_is_caught() {
        let buf = [0x00u8; 1];
        let mut r = BitReader::new(&buf);
        let _ = r.read_bits(8).unwrap();
        assert!(r.read_bits(1).is_err());
    }

    #[test]
    fn cross_byte_bit_writes_pack_correctly() {
        let mut w = BitWriter::new();
        // 5 bits + 5 bits crosses a byte boundary.
        w.write_bits(0b10110, 5);
        w.write_bits(0b00111, 5);
        let (buf, _) = w.finish();
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_bits(5).unwrap(), 0b10110);
        assert_eq!(r.read_bits(5).unwrap(), 0b00111);
    }
}
