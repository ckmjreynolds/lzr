//! Nibble-oriented I/O with BLEB8 variable-length integer encoding.

#![allow(unused_variables, clippy::needless_pass_by_ref_mut)]

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
        todo!()
    }

    /// Encodes a `u16` as UBLEB8 (max 5 nibbles).
    pub(crate) fn write_ubleb8_u16(&mut self, value: u16) -> Result<()> {
        todo!()
    }

    /// Encodes a `u64` as UBLEB8 (max 21 nibbles).
    pub(crate) fn write_ubleb8_u64(&mut self, value: u64) -> Result<()> {
        todo!()
    }

    /// Encodes an `i16` as SBLEB8 (max 5 nibbles).
    pub(crate) fn write_sbleb8_i16(&mut self, value: i16) -> Result<()> {
        todo!()
    }

    /// Writes a literal byte as two nibbles (low nibble first, high nibble second).
    #[inline]
    pub(crate) fn write_literal_byte(&mut self, byte: u8) -> Result<()> {
        todo!()
    }

    /// Pads to byte boundary if needed and returns the inner writer.
    pub(crate) fn finish(self) -> Result<W> {
        todo!()
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
        todo!()
    }

    /// Decodes a UBLEB8-encoded `u16` (max 5 nibbles).
    pub(crate) fn read_ubleb8_u16(&mut self) -> Result<u16> {
        todo!()
    }

    /// Decodes a UBLEB8-encoded `u64` (max 21 nibbles).
    pub(crate) fn read_ubleb8_u64(&mut self) -> Result<u64> {
        todo!()
    }

    /// Decodes an SBLEB8-encoded `i16` (max 5 nibbles).
    pub(crate) fn read_sbleb8_i16(&mut self) -> Result<i16> {
        todo!()
    }

    /// Reads a literal byte from two nibbles (low nibble first, high nibble second).
    #[inline]
    pub(crate) fn read_literal_byte(&mut self) -> Result<u8> {
        todo!()
    }

    /// Discards any pending nibble and returns the inner reader at byte alignment.
    pub(crate) fn into_inner(self) -> R {
        self.inner
    }
}
