//! ULEB128 encoding and decoding for `u64` values (max 9 bytes).
//!
//! Unsigned LEB128 (ULEB128) is a variable-length encoding for unsigned integers.
//! Each byte stores 7 payload bits and 1 continuation bit. Values up to 127 fit in
//! a single byte; `u64::MAX` requires at most 9 bytes (the 9th byte uses all 8 bits).
//!
//! See the [FORMAT.md](../docs/FORMAT.md) specification for details on the encoding.

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
#[allow(clippy::cast_possible_truncation)]
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

/// Decodes a ULEB128-encoded `u64` from `buf` starting at `*pos`.
///
/// Advances `*pos` by the number of bytes consumed (1 to 9).
///
/// # Examples
///
/// ```text
/// let buf = [0x80, 0x01];
/// let mut pos = 0;
/// assert_eq!(decode_uleb128_u64(&buf, &mut pos), 128);
/// assert_eq!(pos, 2);
/// ```
pub(crate) fn decode_uleb128_u64(buf: &[u8], pos: &mut usize) -> u64 {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;

    for _ in 0..8 {
        let byte = buf[*pos];
        *pos += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return value;
        }
        shift += 7;
    }
    // 9th byte: all 8 bits are payload.
    let byte = buf[*pos];
    *pos += 1;
    value |= u64::from(byte) << shift;
    value
}

/// Returns the number of bytes needed to encode `value` as ULEB128.
///
/// Uses `leading_zeros` to compute the result in constant time without
/// encoding. Returns a value in the range `1..=9`.
///
/// # Examples
///
/// ```text
/// assert_eq!(uleb128_u64_len(0), 1);
/// assert_eq!(uleb128_u64_len(127), 1);
/// assert_eq!(uleb128_u64_len(128), 2);
/// assert_eq!(uleb128_u64_len(u64::MAX), 9);
/// ```
pub(crate) fn uleb128_u64_len(value: u64) -> usize {
    // Each of the first 8 bytes carries 7 payload bits; the 9th byte carries
    // all 8 bits. Ceiling-divide significant bits by 7, then cap at 9.
    let bits = (64 - value.leading_zeros()).max(1);
    (bits as usize).div_ceil(7).min(9)
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
            let decoded = decode_uleb128_u64(&buf, &mut pos);
            prop_assert_eq!(pos, buf.len());
            prop_assert_eq!(decoded, value);
        }
    }

    proptest! {
        #[test]
        fn len_matches_encode(value: u64) {
            let mut buf = Vec::new();
            encode_uleb128_u64(value, &mut buf);
            prop_assert_eq!(uleb128_u64_len(value), buf.len());
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
            let decoded = decode_uleb128_u64(expected, &mut pos);
            assert_eq!(pos, expected.len(), "decode len {value}");
            assert_eq!(decoded, value, "decode {value}");

            // Verify length prediction matches actual encoding.
            assert_eq!(uleb128_u64_len(value), expected.len(), "len {value}");
        }
    }
}
