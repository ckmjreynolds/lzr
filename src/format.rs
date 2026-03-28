//! LZR format codec — header, frame, EOS, and footer encode/decode.
//!
//! All functions operate on byte buffers and are independent of I/O. The caller
//! is responsible for reading from and writing to streams.
//!
//! See [`FORMAT.md`](../docs/FORMAT.md) for the canonical specification.

use arbitrary_int::{u3, u5};

use crate::cursor::{ReadBuf, WriteBuf};
use crate::error::{Error, Result};
use crate::uleb128::{decode_uleb128_u64, encode_uleb128_u64};

/// LZR stream header: magic bytes (`LZR`) + format version (`0x00`).
const HEADER: [u8; 4] = [0x4C, 0x5A, 0x52, 0x00];

/// LZR end-of-stream sentinel: token `0x00` + distance `0x0000`.
const EOS: [u8; 3] = [0x00, 0x00, 0x00];

/// Lookup table mapping the 5-bit match field (MMMMM, bits 4–0) of a token byte to its match length (i16).
///
/// | MMMMM | Match Length | Type             |
/// |-------|-------------|------------------|
/// | 0–5   | −9 to −4    | Reverse match    |
/// | 6     | 0           | Literal-only     |
/// | 7–31  | 4 to 28     | Forward match    |
///
/// MMMMM=0 and MMMMM=31 are extension sentinels (base values −9 and 28 respectively).
#[rustfmt::skip]
const MATCH_LENGTH_TABLE: [i16; 32] = [
    -9, -8, -7, -6, -5, -4,          // 0..=5:  reverse match
     0,                              // 6:      literal-only
     4,  5,  6,  7,  8,  9, 10, 11,  // 7..=14: forward match
    12, 13, 14, 15, 16, 17, 18, 19,  // 15..=22
    20, 21, 22, 23, 24, 25, 26, 27,  // 23..=30
    28,                              // 31:     forward match (extension sentinel)
];

/// Writes the 4-byte LZR stream header (magic `LZR` + version `0x00`).
pub(crate) fn encode_header(output: &mut impl WriteBuf) {
    output.write_bytes(&HEADER);
}

/// Reads and validates the 4-byte LZR stream header.
///
/// Returns [`Error::InvalidMagic`] if the first three bytes are not `LZR`, or
/// [`Error::UnsupportedVersion`] if the version byte is not `0x00`.
pub(crate) fn decode_header(input: &mut impl ReadBuf) -> Result<()> {
    let mut buf = [0u8; HEADER.len()];

    input.read_bytes(&mut buf);

    if buf[..3] != HEADER[..3] {
        return Err(Error::InvalidMagic);
    }
    if buf[3] != 0x00 {
        return Err(Error::UnsupportedVersion(buf[3]));
    }

    Ok(())
}

/// Encodes a single LZR frame (token + distance + extensions) into `output`.
///
/// Writes the token byte (`LLLMMMMM`), the 2-byte little-endian distance, and
/// any literal-length or match-length extension bytes. The caller is responsible
/// for appending the `literal_len` literal bytes after this call.
///
/// # Examples
///
/// ```text
/// // Literal-only frame: 2 literals, match length 0, distance ignored.
/// encode_frame(2, 0, 0, &mut output);
/// // Writes: [0x46, 0x00, 0x00] — token 010_00110, distance 0.
/// ```
pub(crate) fn encode_frame(literal_len: i16, match_len: i16, distance: u16, output: &mut impl WriteBuf) {
    let (mmmmm, match_ext) = match_len_to_mmmmm(match_len);

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let (lll, lit_ext) = if literal_len < 7 {
        (u3::new(literal_len as u8), 0i16)
    } else {
        (u3::new(7), literal_len - 7)
    };

    let token = (lll.value() << 5) | mmmmm.value();
    output.write_u8(token);
    output.write_u16_le(distance);

    if lll == u3::new(7) {
        encode_extension(lit_ext, output);
    }
    if mmmmm == u5::new(0) || mmmmm == u5::new(31) {
        encode_extension(match_ext, output);
    }
}

/// Decodes a single LZR frame from `input`.
///
/// Reads the token byte, distance, and any extension bytes. Returns
/// `(literal_len, match_len, distance)`. The caller is responsible for
/// reading `literal_len` literal bytes from `input` after this call.
///
/// When `distance` is 0 the frame is the end-of-stream sentinel — no
/// extensions are read and the caller should stop decoding frames.
///
/// Uses [`MATCH_LENGTH_TABLE`] for a direct lookup of the base match
/// length from the 5-bit MMMMM field.
///
/// # Examples
///
/// ```text
/// // Decode a literal-only frame: token 0x46, distance 0x0000.
/// let (lit, mat, dist) = decode_frame(&mut input);
/// assert_eq!((lit, mat, dist), (2, 0, 0));
/// ```
pub(crate) fn decode_frame(input: &mut impl ReadBuf) -> (i16, i16, u16) {
    let token = input.read_u8();
    let lll = token >> 5;
    let mmmmm = token & 0x1F;

    let distance = input.read_u16_le();

    // Distance 0 is reserved for EOS — no extensions follow.
    if distance == 0 {
        return (i16::from(lll), MATCH_LENGTH_TABLE[mmmmm as usize], 0);
    }

    let literal_len = if lll < 7 {
        i16::from(lll)
    } else {
        7 + decode_extension(input)
    };

    let match_len = match mmmmm {
        0 => MATCH_LENGTH_TABLE[0] - decode_extension(input),
        31 => MATCH_LENGTH_TABLE[31] + decode_extension(input),
        m => MATCH_LENGTH_TABLE[m as usize],
    };

    (literal_len, match_len, distance)
}

/// Writes the 3-byte end-of-stream sentinel (token `0x00`, distance `0x0000`).
pub(crate) fn encode_eos(output: &mut impl WriteBuf) {
    output.write_bytes(&EOS);
}

/// Encodes the stream footer: ULEB128-encoded uncompressed `length` followed
/// by a 4-byte little-endian Adler-32 `checksum`.
///
/// # Examples
///
/// ```text
/// // Footer for the 2-byte input "Hi":
/// encode_footer(2, 0x00FB_00B2, &mut output);
/// // Writes: [0x02, 0xB2, 0x00, 0xFB, 0x00]
/// ```
pub(crate) fn encode_footer(length: u64, checksum: u32, output: &mut impl WriteBuf) {
    encode_uleb128_u64(length, output);
    output.write_u32_le(checksum);
}

/// Decodes the stream footer from `input`.
///
/// Reads a ULEB128-encoded uncompressed length and a 4-byte little-endian
/// Adler-32 checksum. Returns `(length, checksum)`.
///
/// # Examples
///
/// ```text
/// let (length, checksum) = decode_footer(&mut input);
/// assert_eq!(length, 2);
/// assert_eq!(checksum, 0x00FB_00B2);
/// ```
pub(crate) fn decode_footer(input: &mut impl ReadBuf) -> (u64, u32) {
    let length = decode_uleb128_u64(input);
    let checksum = input.read_u32_le();
    (length, checksum)
}

/// Maps a match length to its 5-bit MMMMM token field and extension remainder.
///
/// Returns `(mmmmm, remainder)` where `mmmmm` is the 5-bit field packed into the
/// token byte and `remainder` is the non-negative value written as an extension
/// chain (only present when `mmmmm` is 0 or 31).
///
/// This is the inverse of the [`MATCH_LENGTH_TABLE`] lookup used during decoding.
fn match_len_to_mmmmm(match_len: i16) -> (u5, i16) {
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    match match_len {
        0 => (u5::new(6), 0),
        -8..=-4 => (u5::new((match_len + 9) as u8), 0),
        ..=-9 => (u5::new(0), -9 - match_len),
        4..=27 => (u5::new((match_len + 3) as u8), 0),
        28.. => (u5::new(31), match_len - 28),
        _ => unreachable!("invalid match length: {match_len}"),
    }
}

/// Writes an extension byte chain for the given non-negative remainder.
///
/// Emits one or more bytes: each `0xFF` byte adds 255 to the total, and a
/// final byte in `0x00..=0xFE` terminates the chain. Always writes at least
/// one byte (a zero remainder produces a single `0x00`).
fn encode_extension(mut remainder: i16, output: &mut impl WriteBuf) {
    loop {
        if remainder >= 255 {
            output.write_u8(0xFF);
            remainder -= 255;
        } else {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            output.write_u8(remainder as u8);
            break;
        }
    }
}

/// Reads an extension byte chain, returning the accumulated sum.
///
/// Reads bytes until a value less than 255 is encountered. The sum of all
/// bytes read (including the terminator) is returned.
fn decode_extension(input: &mut impl ReadBuf) -> i16 {
    let mut total: i16 = 0;
    loop {
        let byte = input.read_u8();
        total += i16::from(byte);
        if byte < 255 {
            return total;
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::cursor::{ReadCursor, WriteCursor};
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    /// Helper: encode into a buffer and return the written bytes.
    fn encode_frame_to_vec(literal_len: i16, match_len: i16, distance: u16) -> Vec<u8> {
        let mut buf = Buffer::<256>::new();
        let mut cursor = WriteCursor::<256>::new(&mut buf, 0);
        encode_frame(literal_len, match_len, distance, &mut cursor);
        let len = cursor.position();
        let (a, b) = buf.slices(0, len);
        let mut v = Vec::from(a);
        v.extend_from_slice(b);
        v
    }

    /// Helper: decode a frame from a byte slice.
    fn decode_frame_from_slice(data: &[u8]) -> (i16, i16, u16) {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(data, 0);
        let mut cursor = ReadCursor::<256>::new(&buf, 0);
        decode_frame(&mut cursor)
    }

    #[test]
    fn header_encode() {
        let mut buf = Buffer::<256>::new();
        let mut w = WriteCursor::<256>::new(&mut buf, 0);
        encode_header(&mut w);
        assert_eq!(w.position(), 4);
        let (a, _) = buf.slices(0, 4);
        assert_eq!(a, &[0x4C, 0x5A, 0x52, 0x00]);
    }

    #[test]
    fn header_decode_valid() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(&[0x4C, 0x5A, 0x52, 0x00], 0);
        let mut r = ReadCursor::<256>::new(&buf, 0);
        assert!(decode_header(&mut r).is_ok());
        assert_eq!(r.position(), 4);
    }

    #[test]
    fn header_decode_bad_magic() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(&[0x00, 0x00, 0x00, 0x00], 0);
        let mut r = ReadCursor::<256>::new(&buf, 0);
        let err = decode_header(&mut r).unwrap_err();
        assert!(matches!(err, Error::InvalidMagic));
    }

    #[test]
    fn header_decode_bad_version() {
        let mut buf = Buffer::<256>::new();
        buf.copy_from_slice(&[0x4C, 0x5A, 0x52, 0x01], 0);
        let mut r = ReadCursor::<256>::new(&buf, 0);
        let err = decode_header(&mut r).unwrap_err();
        assert!(matches!(err, Error::UnsupportedVersion(0x01)));
    }

    #[test]
    fn header_roundtrip() {
        let mut buf = Buffer::<256>::new();
        let mut w = WriteCursor::<256>::new(&mut buf, 0);
        encode_header(&mut w);

        let mut r = ReadCursor::<256>::new(&buf, 0);
        assert!(decode_header(&mut r).is_ok());
    }

    #[test]
    fn eos_encode() {
        let mut buf = Buffer::<256>::new();
        let mut w = WriteCursor::<256>::new(&mut buf, 0);
        encode_eos(&mut w);
        assert_eq!(w.position(), 3);
        let (a, _) = buf.slices(0, 3);
        assert_eq!(a, &[0x00, 0x00, 0x00]);
    }

    #[test]
    fn worked_example_literal_only() {
        // FORMAT.md: Token 0x46 = 010_00110: L=2, M=6 (literal-only), distance ignored.
        let bytes = encode_frame_to_vec(2, 0, 0);
        assert_eq!(bytes, vec![0x46, 0x00, 0x00]);
    }

    #[test]
    fn eos_decode() {
        let (lit, mat, dist) = decode_frame_from_slice(&[0x00, 0x00, 0x00]);
        assert_eq!((lit, mat, dist), (0, -9, 0));
    }

    #[test]
    fn boundary_match_neg9() {
        // match_len=-9 uses MMMMM=0 sentinel with extension byte 0x00.
        let bytes = encode_frame_to_vec(0, -9, 1);
        assert_eq!(bytes, vec![0x00, 0x01, 0x00, 0x00]); // token, dist_lo, dist_hi, ext=0
        let decoded = decode_frame_from_slice(&bytes);
        assert_eq!(decoded, (0, -9, 1));
    }

    #[test]
    fn boundary_match_neg4() {
        let bytes = encode_frame_to_vec(0, -4, 100);
        let decoded = decode_frame_from_slice(&bytes);
        assert_eq!(decoded, (0, -4, 100));
    }

    #[test]
    fn boundary_match_4() {
        let bytes = encode_frame_to_vec(0, 4, 500);
        let decoded = decode_frame_from_slice(&bytes);
        assert_eq!(decoded, (0, 4, 500));
    }

    #[test]
    fn boundary_match_28() {
        // match_len=28 uses MMMMM=31 sentinel with extension byte 0x00.
        let bytes = encode_frame_to_vec(0, 28, 1);
        assert_eq!(bytes[3], 0x00); // extension byte
        let decoded = decode_frame_from_slice(&bytes);
        assert_eq!(decoded, (0, 28, 1));
    }

    #[test]
    fn large_literal_extension() {
        // literal_len=262 = 7 + 255 + 0
        let bytes = encode_frame_to_vec(262, 4, 1);
        let decoded = decode_frame_from_slice(&bytes);
        assert_eq!(decoded, (262, 4, 1));
    }

    #[test]
    fn large_positive_match_extension() {
        // match_len=283 = 28 + 255 + 0
        let bytes = encode_frame_to_vec(0, 283, 1);
        let decoded = decode_frame_from_slice(&bytes);
        assert_eq!(decoded, (0, 283, 1));
    }

    #[test]
    fn large_negative_match_extension() {
        // match_len=-264 = -9 - 255 - 0
        let bytes = encode_frame_to_vec(0, -264, 1);
        let decoded = decode_frame_from_slice(&bytes);
        assert_eq!(decoded, (0, -264, 1));
    }

    #[test]
    fn footer_roundtrip_known() {
        let mut buf = Buffer::<64>::new();
        let mut w = WriteCursor::<64>::new(&mut buf, 0);
        encode_footer(42, 0x00FB_00B2, &mut w);
        let len = w.position();

        let mut r = ReadCursor::<64>::new(&buf, 0);
        let (length, checksum) = decode_footer(&mut r);
        assert_eq!(length, 42);
        assert_eq!(checksum, 0x00FB_00B2);
        assert_eq!(r.position(), len);
    }

    #[test]
    #[should_panic(expected = "invalid match length")]
    fn invalid_match_length() {
        let _ = match_len_to_mmmmm(-3);
    }

    /// Generates valid match lengths for property testing.
    fn valid_match_len() -> impl Strategy<Value = i16> {
        prop_oneof![Just(0i16), (-8..=-4i16), (4..=27i16), (-500..=-9i16), (28..=500i16),]
    }

    proptest! {
        #[test]
        fn frame_roundtrip(
            literal_len in 0..=500i16,
            match_len in valid_match_len(),
            distance in 1..=u16::MAX,
        ) {
            let bytes = encode_frame_to_vec(literal_len, match_len, distance);
            let decoded = decode_frame_from_slice(&bytes);
            prop_assert_eq!(decoded, (literal_len, match_len, distance));
        }

        #[test]
        fn footer_roundtrip(length: u64, checksum: u32) {
            let mut buf = Buffer::<64>::new();
            let mut w = WriteCursor::<64>::new(&mut buf, 0);
            encode_footer(length, checksum, &mut w);

            let mut r = ReadCursor::<64>::new(&buf, 0);
            let (dec_len, dec_cksum) = decode_footer(&mut r);
            prop_assert_eq!(dec_len, length);
            prop_assert_eq!(dec_cksum, checksum);
        }
    }
}
