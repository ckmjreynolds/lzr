//! ULEB128 encoding and decoding for `u64` values (max 9 bytes).
//!
//! Unsigned LEB128 (ULEB128) is a variable-length encoding for unsigned integers.
//! Each byte stores 7 payload bits and 1 continuation bit. Values up to 127 fit in
//! a single byte; `u64::MAX` requires at most 9 bytes (the 9th byte uses all 8 bits).
//!
//! See the [FORMAT.md](../docs/FORMAT.md) specification for details on the encoding.

use arbitrary_int::u15;

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
/// encode_u64(128, &mut out);
/// assert_eq!(out, vec![0x80, 0x01]);
/// ```
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[expect(clippy::cast_possible_truncation, reason = "Truncation masked/intentional.")]
pub(crate) fn encode_u64(mut value: u64, out: &mut Vec<u8>) {
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
/// assert_eq!(decode_u64(&buf, &mut pos).unwrap(), 128);
/// assert_eq!(pos, 2);
/// ```
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) fn decode_u64(buf: &[u8], pos: &mut usize) -> Result<u64> {
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

/// Encodes a 15-bit `value` as a 1- or 2-byte varint by appending to `out`.
///
/// This is a bespoke, fixed-width-bounded variant of ULEB128 for values that are
/// statically known to fit in 15 bits (e.g. token ids). Values below `0x80` take
/// a single byte; everything else takes exactly two, with the terminating byte
/// carrying all 8 remaining bits rather than 7. A `u15` therefore never needs the
/// 3 bytes that the general [`encode_uleb128_u64`] would spend on values ≥ 16384,
/// and the [`u15`] type makes an out-of-range input unrepresentable at the call
/// site, so the encoder needs no range check of its own.
///
/// # Examples
///
/// ```text
/// let mut out = Vec::new();
/// encode_u15(u15::new(16384), &mut out);
/// assert_eq!(out, vec![0x80, 0x80]);
/// ```
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[expect(clippy::cast_possible_truncation, reason = "Truncation masked/intentional.")]
pub(crate) fn encode_u15(value: u15, out: &mut Vec<u8>) {
    let v = value.value();
    if v < 0x80 {
        out.push(v as u8);
    } else {
        // Terminating byte uses all 8 bits: `v >> 7` is `< 256` for any 15-bit `v`.
        out.extend_from_slice(&[(v as u8 & 0x7F) | 0x80, (v >> 7) as u8]);
    }
}

/// Decodes a [`u15`] written by [`encode_u15`] from `buf` starting at `*pos`,
/// rejecting non-canonical (overlong) encodings per FORMAT.md §7.
///
/// Advances `*pos` by the number of bytes consumed (1 or 2).
///
/// # Errors
///
/// Returns [`Error::NonCanonicalUleb128`] if the 2-byte form's terminating byte
/// is `0x00`, since the same value would then fit in a single byte.
///
/// # Examples
///
/// ```text
/// let buf = [0x80, 0x80];
/// let mut pos = 0;
/// assert_eq!(decode_u15(&buf, &mut pos).unwrap(), u15::new(16384));
/// assert_eq!(pos, 2);
/// ```
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) fn decode_u15(buf: &[u8], pos: &mut usize) -> Result<u15> {
    let byte0 = buf[*pos];
    *pos += 1;
    if byte0 & 0x80 == 0 {
        // Single-byte form: the 7-bit payload is already a valid `u15`.
        return Ok(u15::new(u16::from(byte0)));
    }

    let byte1 = buf[*pos];
    *pos += 1;
    // Canonical requires the terminating 8-bit byte to be nonzero; otherwise the
    // single-byte form would encode the same value (mirrors the u64 decoder).
    if byte1 == 0 {
        return Err(Error::NonCanonicalUleb128);
    }

    // 7 payload bits + 8 terminating bits = 15 bits, so the result always fits a
    // `u15` (max `0xFF << 7 | 0x7F` == 32767) and `u15::new` cannot panic.
    let value = u16::from(byte0 & 0x7F) | (u16::from(byte1) << 7);
    Ok(u15::new(value))
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
            encode_u64(value, &mut buf);
            let mut pos = 0;
            let decoded = decode_u64(&buf, &mut pos).unwrap();
            prop_assert_eq!(pos, buf.len());
            prop_assert_eq!(decoded, value);
        }

        #[test]
        fn u15_roundtrip(raw in 0u16..=0x7FFF) {
            let value = u15::new(raw);
            let mut buf = Vec::new();
            encode_u15(value, &mut buf);
            let mut pos = 0;
            let decoded = decode_u15(&buf, &mut pos).unwrap();
            prop_assert_eq!(pos, buf.len());
            prop_assert!(buf.len() <= 2);
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
            encode_u64(value, &mut encoded);
            assert_eq!(&encoded[..], expected, "encode {value}");

            // Verify decoding reads the expected value.
            let mut pos = 0;
            let decoded = decode_u64(expected, &mut pos).unwrap();
            assert_eq!(pos, expected.len(), "decode len {value}");
            assert_eq!(decoded, value, "decode {value}");
        }
    }

    #[test]
    fn u15_known_values() {
        let cases: &[(u16, &[u8])] = &[
            (0, &[0x00]),
            (1, &[0x01]),
            (127, &[0x7F]),
            (128, &[0x80, 0x01]),
            (16384, &[0x80, 0x80]),
            (0x7FFF, &[0xFF, 0xFF]),
        ];

        for &(raw, expected) in cases {
            let value = u15::new(raw);

            // Verify encoding produces the expected bytes.
            let mut encoded = Vec::new();
            encode_u15(value, &mut encoded);
            assert_eq!(&encoded[..], expected, "encode {raw}");

            // Verify decoding reads the expected value.
            let mut pos = 0;
            let decoded = decode_u15(expected, &mut pos).unwrap();
            assert_eq!(pos, expected.len(), "decode len {raw}");
            assert_eq!(decoded, value, "decode {raw}");
        }
    }

    #[test]
    fn u15_rejects_non_canonical() {
        // `0x80 0x00` represents 0 but the canonical encoding of 0 is `0x00`.
        let mut pos = 0;
        assert!(matches!(decode_u15(&[0x80, 0x00], &mut pos), Err(Error::NonCanonicalUleb128)));
    }

    #[test]
    fn rejects_non_canonical() {
        // `0x80 0x00` represents 0 but the canonical encoding of 0 is `0x00`.
        let mut pos = 0;
        assert!(matches!(decode_u64(&[0x80, 0x00], &mut pos), Err(Error::NonCanonicalUleb128),));

        // Nine-byte encoding with a zero top byte is also non-canonical.
        let mut pos = 0;
        let bad = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        assert!(matches!(decode_u64(&bad, &mut pos), Err(Error::NonCanonicalUleb128),));
    }
}
