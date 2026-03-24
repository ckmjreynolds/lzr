//! ULEB128 encoding and decoding for `u64` values (max 9 bytes).
//!
//! Unsigned LEB128 (ULEB128) is a variable-length encoding for unsigned integers.
//! Each byte stores 7 payload bits and 1 continuation bit. Values up to 127 fit in
//! a single byte; `u64::MAX` requires at most 9 bytes (the 9th byte uses all 8 bits).
//!
//! See the [FORMAT.md](../docs/FORMAT.md) specification for details on the encoding.

/// Encodes `value` as ULEB128 into `buf` and returns the number of bytes written.
///
/// `buf` must be at least 9 bytes long.
///
/// # Examples
///
/// ```text
/// let mut buf = [0u8; 9];
/// assert_eq!(encode_uleb128_u64(0, &mut buf), 1);
/// assert_eq!(buf[0], 0x00);
///
/// assert_eq!(encode_uleb128_u64(128, &mut buf), 2);
/// assert_eq!(&buf[..2], &[0x80, 0x01]);
/// ```
#[allow(clippy::cast_possible_truncation, clippy::needless_range_loop)]
pub(crate) fn encode_uleb128_u64(mut value: u64, buf: &mut [u8]) -> usize {
    for i in 0..8 {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf[i] = byte;
            return i + 1;
        }
        buf[i] = byte | 0x80;
    }
    // 9th byte: all 8 bits are payload.
    buf[8] = value as u8;
    9
}

/// Decodes a ULEB128-encoded `u64` from `buf`, returning `(value, bytes_read)`.
///
/// # Examples
///
/// ```text
/// let (value, n) = decode_uleb128_u64(&[0x80, 0x01]);
/// assert_eq!(value, 128);
/// assert_eq!(n, 2);
/// ```
#[allow(clippy::needless_range_loop)]
pub(crate) fn decode_uleb128_u64(buf: &[u8]) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;

    for i in 0..8 {
        value |= u64::from(buf[i] & 0x7F) << shift;
        if buf[i] & 0x80 == 0 {
            return (value, i + 1);
        }
        shift += 7;
    }
    // 9th byte: all 8 bits are payload.
    value |= u64::from(buf[8]) << shift;
    (value, 9)
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
            let mut buf = [0u8; 9];
            let n = encode_uleb128_u64(value, &mut buf);
            let (decoded, consumed) = decode_uleb128_u64(&buf);
            prop_assert_eq!(consumed, n);
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
            let mut buf = [0u8; 9];
            let n = encode_uleb128_u64(value, &mut buf);
            assert_eq!(&buf[..n], expected, "encode {value}");

            let (decoded, consumed) = decode_uleb128_u64(expected);
            assert_eq!(consumed, expected.len(), "decode len {value}");
            assert_eq!(decoded, value, "decode {value}");
        }
    }
}
