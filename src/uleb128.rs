//! ULEB128 encoding and decoding for `u64` values (max 9 bytes).
//!
//! Unsigned LEB128 (ULEB128) is a variable-length encoding for unsigned integers.
//! Each byte stores 7 payload bits and 1 continuation bit. Values up to 127 fit in
//! a single byte; `u64::MAX` requires at most 9 bytes (the 9th byte uses all 8 bits).
//!
//! See the [FORMAT.md](../docs/FORMAT.md) specification for details on the encoding.

use crate::cursor::{ReadBuf, WriteBuf};

/// Encodes `value` as ULEB128 into `w`.
///
/// Writes at most 9 bytes. Values up to 127 require a single byte;
/// `u64::MAX` requires exactly 9 bytes.
///
/// # Examples
///
/// ```text
/// let mut buf = Buffer::<256>::new();
/// let mut cur = WriteCursor::new(&mut buf, 0);
/// encode_uleb128_u64(128, &mut cur);
/// assert_eq!(cur.position(), 2);
/// ```
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn encode_uleb128_u64(mut value: u64, output: &mut impl WriteBuf) {
    for _ in 0..8 {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            output.write_u8(byte);
            return;
        }
        output.write_u8(byte | 0x80);
    }
    // 9th byte: all 8 bits are payload.
    output.write_u8(value as u8);
}

/// Decodes a ULEB128-encoded `u64` from `r`.
///
/// Reads at most 9 bytes and returns the decoded value.
///
/// # Examples
///
/// ```text
/// let mut buf = Buffer::<256>::new();
/// // ... write [0x80, 0x01] into buf ...
/// let mut cur = ReadCursor::new(&buf, 0);
/// assert_eq!(decode_uleb128_u64(&mut cur), 128);
/// assert_eq!(cur.position(), 2);
/// ```
pub(crate) fn decode_uleb128_u64(input: &mut impl ReadBuf) -> u64 {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;

    for _ in 0..8 {
        let byte = input.read_u8();
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return value;
        }
        shift += 7;
    }
    // 9th byte: all 8 bits are payload.
    value |= u64::from(input.read_u8()) << shift;
    value
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;
    use crate::buffer::Buffer;
    use crate::cursor::{ReadCursor, WriteCursor};

    proptest! {
        #[test]
        fn roundtrip(value: u64) {
            let mut buf = Buffer::<256>::new();

            let mut wc = WriteCursor::new(&mut buf, 0);
            encode_uleb128_u64(value, &mut wc);
            let written = wc.position();

            let mut rc = ReadCursor::new(&buf, 0);
            let decoded = decode_uleb128_u64(&mut rc);
            prop_assert_eq!(rc.position(), written);
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
            let mut buf = Buffer::<256>::new();
            buf.copy_from_slice(expected, 0);

            // Verify encoding produces the expected bytes.
            let mut enc_buf = Buffer::<256>::new();
            let mut wc = WriteCursor::new(&mut enc_buf, 0);
            encode_uleb128_u64(value, &mut wc);
            let n = wc.position();

            let (a, b) = enc_buf.slices(0, n);
            let mut encoded = Vec::from(a);
            encoded.extend_from_slice(b);
            assert_eq!(&encoded[..], expected, "encode {value}");

            // Verify decoding reads the expected value.
            let mut rc = ReadCursor::new(&buf, 0);
            let decoded = decode_uleb128_u64(&mut rc);
            assert_eq!(rc.position(), expected.len(), "decode len {value}");
            assert_eq!(decoded, value, "decode {value}");
        }
    }
}
