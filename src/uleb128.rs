//! ULEB128 encoding and decoding for `u64` values (max 9 bytes).
//!
//! See the [FORMAT.md](../docs/FORMAT.md) specification for details on the encoding.

use std::io::{Read, Write};

use crate::error::Result;

/// Encodes a `u64` as ULEB128 (max 9 bytes).
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

/// Decodes a ULEB128-encoded `u64` (max 9 bytes).
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
