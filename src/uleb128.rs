//! ULEB128 encoding and decoding for `u64` values (max 9 bytes).
//!
//! Unsigned LEB128 (ULEB128) is a variable-length encoding for unsigned integers.
//! Each byte stores 7 payload bits and 1 continuation bit. Values up to 127 fit in
//! a single byte; `u64::MAX` requires at most 9 bytes (the 9th byte uses all 8 bits).
//!
//! Encodings must be **canonical**: an overlong encoding (a terminating `0x00` payload where a
//! shorter encoding exists) is rejected rather than accepted, so every value has one representation.

use anyhow::{Context as _, Result, bail};
use arbitrary_int::u22;

/// Reads the byte at `*pos` and advances it, erroring if the input is exhausted.
/// Shared by the ULEB128 decoders so each read is bounds-checked in one place.
fn take(buf: &[u8], pos: &mut usize) -> Result<u8> {
    let byte = *buf.get(*pos).context("unexpected end of input decoding ULEB128")?;
    *pos += 1;
    Ok(byte)
}

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
/// non-canonical (overlong) encodings.
///
/// Advances `*pos` by the number of bytes consumed (1 to 9).
///
/// # Errors
///
/// Returns an error if the input ends before the value is complete, or if the
/// encoding is non-canonical (overlong) — the terminating 7-bit payload byte is
/// `0x00` yet a shorter encoding would have produced the same value.
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
        let byte = take(buf, pos)?;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            // Canonical requires: either this is the first byte (i=0), or the
            // terminating byte has a nonzero payload. Otherwise a shorter
            // encoding would represent the same value.
            if i > 0 && byte == 0 {
                bail!("non-canonical ULEB128 encoding");
            }
            return Ok(value);
        }
        shift += 7;
    }
    // 9th byte: all 8 bits are payload; canonical requires it to be nonzero.
    let byte = take(buf, pos)?;
    if byte == 0 {
        bail!("non-canonical ULEB128 encoding");
    }
    value |= u64::from(byte) << shift;
    Ok(value)
}

/// Encodes a 22-bit `value` as a 1-, 2-, or 3-byte varint by appending to `out`.
///
/// A bespoke, fixed-width-bounded variant of ULEB128 for values statically known to
/// fit in 22 bits (e.g. token ids). The layout packs 7 + 7 + 8 = 22 bits: bytes 0
/// and 1 carry 7 payload bits plus a continuation bit, and the *terminating* third
/// byte carries all 8 remaining bits. Unlike the 15-bit variant, both leading bytes
/// must keep their continuation bit — three lengths (1/2/3 bytes) need two escapes
/// to stay unambiguous; only the final byte can spend all 8 bits. Values below
/// `0x80` take one byte, below `0x4000` two, otherwise three. The [`u22`] type makes
/// an out-of-range input unrepresentable at the call site, so no range check is
/// needed.
///
/// # Examples
///
/// ```text
/// let mut out = Vec::new();
/// encode_u22(u22::new(0x4000), &mut out);
/// assert_eq!(out, vec![0x80, 0x80, 0x01]);
/// ```
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
#[expect(clippy::cast_possible_truncation, reason = "Truncation masked/intentional.")]
pub(crate) fn encode_u22(value: u22, out: &mut Vec<u8>) {
    let v = value.value();
    if v < 0x80 {
        out.push(v as u8);
    } else if v < 0x4000 {
        // Terminating byte 1 carries the top 7 bits (its continuation bit stays clear).
        out.extend_from_slice(&[(v as u8 & 0x7F) | 0x80, (v >> 7) as u8]);
    } else {
        // Terminating byte 2 uses all 8 bits: `v >> 14` is `<= 0xFF` for any 22-bit `v`.
        out.extend_from_slice(&[(v as u8 & 0x7F) | 0x80, ((v >> 7) as u8 & 0x7F) | 0x80, (v >> 14) as u8]);
    }
}

/// The number of bytes [`encode_u22`] writes for `value`: `1` below `0x80`, `2` below
/// `0x4000`, otherwise `3`. Lets a caller price a `u22` varint without encoding it (the LZ77
/// stage uses it to decide whether a match is worth emitting).
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) const fn u22_len(value: u32) -> usize {
    if value < 0x80 {
        1
    } else if value < 0x4000 {
        2
    } else {
        3
    }
}

/// Decodes a [`u22`] written by [`encode_u22`] from `buf` starting at `*pos`,
/// rejecting non-canonical (overlong) encodings.
///
/// Advances `*pos` by the number of bytes consumed (1 to 3).
///
/// # Errors
///
/// Returns an error if the input ends before the value is complete, or if the
/// encoding is non-canonical — a terminating byte of `0x00` in the 2- or 3-byte
/// form, since a shorter encoding would represent the same value.
///
/// # Examples
///
/// ```text
/// let buf = [0x80, 0x80, 0x01];
/// let mut pos = 0;
/// assert_eq!(decode_u22(&buf, &mut pos).unwrap(), u22::new(0x4000));
/// assert_eq!(pos, 3);
/// ```
#[cfg_attr(feature = "bench-internals", visibility::make(pub))]
pub(crate) fn decode_u22(buf: &[u8], pos: &mut usize) -> Result<u22> {
    let byte0 = take(buf, pos)?;
    if byte0 & 0x80 == 0 {
        // Single-byte form: the 7-bit payload is already a valid `u22`.
        return Ok(u22::new(u32::from(byte0)));
    }

    let byte1 = take(buf, pos)?;
    if byte1 & 0x80 == 0 {
        // Two-byte form: byte1 is a 7-bit terminating payload. Canonical requires it
        // nonzero; otherwise the single-byte form would encode the same value.
        if byte1 == 0 {
            bail!("non-canonical ULEB128 encoding");
        }
        let value = u32::from(byte0 & 0x7F) | (u32::from(byte1) << 7);
        return Ok(u22::new(value));
    }

    // Three-byte form: byte2 carries all 8 bits, terminating. Canonical requires it
    // nonzero; otherwise the two-byte form suffices.
    let byte2 = take(buf, pos)?;
    if byte2 == 0 {
        bail!("non-canonical ULEB128 encoding");
    }

    // 7 + 7 + 8 = 22 bits, so the result always fits a `u22` (max
    // `0xFF << 14 | 0x7F << 7 | 0x7F` == 0x3F_FFFF) and `u22::new` cannot panic.
    let value = u32::from(byte0 & 0x7F) | (u32::from(byte1 & 0x7F) << 7) | (u32::from(byte2) << 14);
    Ok(u22::new(value))
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
        fn u22_roundtrip(raw in 0u32..=0x3F_FFFF) {
            let value = u22::new(raw);
            let mut buf = Vec::new();
            encode_u22(value, &mut buf);
            let mut pos = 0;
            let decoded = decode_u22(&buf, &mut pos).unwrap();
            prop_assert_eq!(pos, buf.len());
            prop_assert!(buf.len() <= 3);
            prop_assert_eq!(decoded, value);
        }

        /// `u22_len` must predict exactly how many bytes `encode_u22` writes.
        #[test]
        fn u22_len_matches_encoding(raw in 0u32..=0x3F_FFFF) {
            let mut buf = Vec::new();
            encode_u22(u22::new(raw), &mut buf);
            prop_assert_eq!(u22_len(raw), buf.len());
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
    fn u22_known_values() {
        let cases: &[(u32, &[u8])] = &[
            (0, &[0x00]),
            (1, &[0x01]),
            (127, &[0x7F]),
            (128, &[0x80, 0x01]),
            (0x3FFF, &[0xFF, 0x7F]),       // largest 2-byte value
            (0x4000, &[0x80, 0x80, 0x01]), // smallest 3-byte value
            (0x3F_FFFF, &[0xFF, 0xFF, 0xFF]),
        ];

        for &(raw, expected) in cases {
            let value = u22::new(raw);

            // Verify encoding produces the expected bytes.
            let mut encoded = Vec::new();
            encode_u22(value, &mut encoded);
            assert_eq!(&encoded[..], expected, "encode {raw}");

            // Verify decoding reads the expected value.
            let mut pos = 0;
            let decoded = decode_u22(expected, &mut pos).unwrap();
            assert_eq!(pos, expected.len(), "decode len {raw}");
            assert_eq!(decoded, value, "decode {raw}");
        }
    }

    #[test]
    fn u22_rejects_non_canonical() {
        // `0x80 0x00` represents 0 but the canonical encoding of 0 is `0x00`.
        let mut pos = 0;
        assert!(decode_u22(&[0x80, 0x00], &mut pos).is_err());
        // `0x80 0x80 0x00` represents a 2-byte value in three bytes.
        let mut pos = 0;
        assert!(decode_u22(&[0x80, 0x80, 0x00], &mut pos).is_err());
    }

    #[test]
    fn rejects_non_canonical() {
        // `0x80 0x00` represents 0 but the canonical encoding of 0 is `0x00`.
        let mut pos = 0;
        assert!(decode_u64(&[0x80, 0x00], &mut pos).is_err());

        // Nine-byte encoding with a zero top byte is also non-canonical.
        let mut pos = 0;
        let bad = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        assert!(decode_u64(&bad, &mut pos).is_err());
    }

    #[test]
    fn rejects_truncated() {
        // A trailing continuation bit with no successor byte must error, not panic.
        let mut pos = 0;
        assert!(decode_u64(&[0x80], &mut pos).is_err());
        let mut pos = 0;
        assert!(decode_u22(&[0x80], &mut pos).is_err());
        let mut pos = 0;
        assert!(decode_u22(&[0x80, 0x80], &mut pos).is_err());
        // An empty buffer must error, not panic.
        let mut pos = 0;
        assert!(decode_u64(&[], &mut pos).is_err());
        let mut pos = 0;
        assert!(decode_u22(&[], &mut pos).is_err());
    }
}
