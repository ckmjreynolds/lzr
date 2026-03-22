//! ULEB128 encoding and decoding for `u64` values (max 9 bytes).
//!
//! Unsigned LEB128 (ULEB128) is a variable-length encoding for unsigned integers.
//! Each byte stores 7 payload bits and 1 continuation bit. Values up to 127 fit in
//! a single byte; `u64::MAX` requires at most 9 bytes (the 9th byte uses all 8 bits).
//!
//! See the [FORMAT.md](../docs/FORMAT.md) specification for details on the encoding.

use std::io::{Read, Write};

use crate::error::Result;

/// Encodes a `u64` as ULEB128 into the given writer (at most 9 bytes).
///
/// Each of the first 8 bytes carries 7 payload bits in bits `[6:0]` and a continuation
/// flag in bit 7. A 9th byte, if needed, uses all 8 bits for the remaining payload.
///
/// # Errors
///
/// Returns an error if the underlying writer fails.
///
/// # Examples
///
/// ```text
/// let mut out = Vec::new();
/// encode_uleb128_u64(300, &mut out).unwrap();
/// assert_eq!(out, vec![0xAC, 0x02]);
/// ```
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn encode_uleb128_u64<W: Write>(mut value: u64, w: &mut W) -> Result<()> {
    for _ in 0..8 {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            w.write_all(&[byte])?;
            return Ok(());
        }
        w.write_all(&[byte | 0x80])?;
    }
    // 9th byte: all 8 bits are payload.
    w.write_all(&[value as u8])?;
    Ok(())
}

/// Decodes a ULEB128-encoded `u64` from the given reader (at most 9 bytes).
///
/// Reads bytes one at a time until a byte without the continuation bit is encountered,
/// or until 9 bytes have been consumed.
///
/// # Errors
///
/// Returns an error if the underlying reader fails (e.g., unexpected EOF).
///
/// # Examples
///
/// ```text
/// let data: &[u8] = &[0xAC, 0x02];
/// let value = decode_uleb128_u64(&mut &data[..]).unwrap();
/// assert_eq!(value, 300);
/// ```
pub(crate) fn decode_uleb128_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut buf = [0u8];

    for _ in 0..8 {
        r.read_exact(&mut buf)?;
        value |= u64::from(buf[0] & 0x7F) << shift;
        if buf[0] & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
    // 9th byte: all 8 bits are payload.
    r.read_exact(&mut buf)?;
    value |= u64::from(buf[0]) << shift;
    Ok(value)
}
