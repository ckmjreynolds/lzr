//! ULEB128 encoding and decoding for `u64` values (max 9 bytes).
//!
//! Unsigned LEB128 (ULEB128) is a variable-length encoding for unsigned integers.
//! Each byte stores 7 payload bits and 1 continuation bit. Values up to 127 fit in
//! a single byte; `u64::MAX` requires at most 9 bytes (the 9th byte uses all 8 bits).
//!
//! See the [FORMAT.md](../docs/FORMAT.md) specification for details on the encoding.

use crate::error::{Error, Result};

/// Encodes `value` as ULEB128 by appending bytes to `out`.
///
/// Writes at most 9 bytes. Values up to 127 require a single byte;
/// `u64::MAX` requires exactly 9 bytes.
///
/// # Examples
///
/// ```text
/// let mut out = Vec::new();
/// encode_uleb128_u64(128, &mut out);
/// assert_eq!(out, vec![0x80, 0x01]);
/// ```
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[expect(clippy::cast_possible_truncation, reason = "Truncation masked/intentional.")]
pub(crate) fn encode_uleb128_u64(mut value: u64, out: &mut Vec<u8>) {
    for _ in 0..8 {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
    // 9th byte: all 8 bits are payload.
    out.push(value as u8);
}

/// Decodes a ULEB128-encoded `u64` from `buf` starting at `*pos`, rejecting
/// non-canonical (overlong) encodings per FORMAT.md §7.
///
/// Advances `*pos` by the number of bytes consumed (1 to 9).
///
/// # Errors
///
/// Returns [`Error::NonCanonicalUleb128`] if the terminating 7-bit payload
/// byte is `0x00` yet a shorter encoding would have produced the same value.
///
/// # Examples
///
/// ```text
/// let buf = [0x80, 0x01];
/// let mut pos = 0;
/// assert_eq!(decode_uleb128_u64(&buf, &mut pos).unwrap(), 128);
/// assert_eq!(pos, 2);
/// ```
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) fn decode_uleb128_u64(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;

    for i in 0..8 {
        let byte = buf[*pos];
        *pos += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            // Canonical requires: either this is the first byte (i=0), or the
            // terminating byte has a nonzero payload. Otherwise a shorter
            // encoding would represent the same value.
            if i > 0 && byte == 0 {
                return Err(Error::NonCanonicalUleb128);
            }
            return Ok(value);
        }
        shift += 7;
    }
    // 9th byte: all 8 bits are payload; canonical requires it to be nonzero.
    let byte = buf[*pos];
    *pos += 1;
    if byte == 0 {
        return Err(Error::NonCanonicalUleb128);
    }
    value |= u64::from(byte) << shift;
    Ok(value)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    proptest! {
        #[test]
        fn roundtrip(value: u64) {
            let mut buf = Vec::new();
            encode_uleb128_u64(value, &mut buf);
            let mut pos = 0;
            let decoded = decode_uleb128_u64(&buf, &mut pos).unwrap();
            prop_assert_eq!(pos, buf.len());
            prop_assert_eq!(decoded, value);
        }
    }

    #[test]
    fn known_values() {
        let cases: &[(u64, &[u8])] = &[
            (0, &[0x00]),
            (1, &[0x01]),
            (127, &[0x7F]),
            (128, &[0x80, 0x01]),
            (16384, &[0x80, 0x80, 0x01]),
            (u64::MAX, &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
        ];

        for &(value, expected) in cases {
            // Verify encoding produces the expected bytes.
            let mut encoded = Vec::new();
            encode_uleb128_u64(value, &mut encoded);
            assert_eq!(&encoded[..], expected, "encode {value}");

            // Verify decoding reads the expected value.
            let mut pos = 0;
            let decoded = decode_uleb128_u64(expected, &mut pos).unwrap();
            assert_eq!(pos, expected.len(), "decode len {value}");
            assert_eq!(decoded, value, "decode {value}");
        }
    }

    #[test]
    fn rejects_non_canonical() {
        // `0x80 0x00` represents 0 but the canonical encoding of 0 is `0x00`.
        let mut pos = 0;
        assert!(matches!(decode_uleb128_u64(&[0x80, 0x00], &mut pos), Err(Error::NonCanonicalUleb128),));

        // Nine-byte encoding with a zero top byte is also non-canonical.
        let mut pos = 0;
        let bad = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        assert!(matches!(decode_uleb128_u64(&bad, &mut pos), Err(Error::NonCanonicalUleb128),));
    }
}
