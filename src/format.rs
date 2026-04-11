//! LZR format codec — header, frame, EOS, and footer encode/decode.
//!
//! All functions operate on byte buffers and are independent of I/O. Encoders
//! append bytes to a `Vec<u8>`; decoders read from a `&[u8]` and advance a
//! position cursor passed by `&mut`.
//!
//! See [`FORMAT.md`](../docs/FORMAT.md) for the canonical specification.

use arbitrary_int::{u3, u5};

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
pub(crate) fn encode_header(out: &mut Vec<u8>) {
    out.extend_from_slice(&HEADER);
}

/// Reads and validates the 4-byte LZR stream header from `buf` starting at `*pos`.
///
/// Returns [`Error::InvalidMagic`] if the first three bytes are not `LZR`, or
/// [`Error::UnsupportedVersion`] if the version byte is not `0x00`.
pub(crate) fn decode_header(buf: &[u8], pos: &mut usize) -> Result<()> {
    let p = *pos;
    if buf[p..p + 3] != HEADER[..3] {
        return Err(Error::InvalidMagic);
    }
    if buf[p + 3] != 0x00 {
        return Err(Error::UnsupportedVersion(buf[p + 3]));
    }
    *pos = p + 4;
    Ok(())
}

/// Encodes a single LZR frame (token + distance + extensions) by appending
/// bytes to `out`.
///
/// Writes the token byte (`LLLMMMMM`), the 2-byte little-endian distance, and
/// any literal-length or match-length extension bytes. The caller is
/// responsible for appending the `literal_len` literal bytes after this call.
///
/// # Examples
///
/// ```text
/// let mut out = Vec::new();
/// encode_frame(2, 0, 0, &mut out);
/// // out == [0x46, 0x00, 0x00] — token 010_00110, distance 0.
/// ```
pub(crate) fn encode_frame(literal_len: i16, match_len: i16, distance: u16, out: &mut Vec<u8>) {
    let (mmmmm, match_ext) = match_len_to_mmmmm(match_len);

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let (lll, lit_ext) = if literal_len < 7 {
        (u3::new(literal_len as u8), 0i16)
    } else {
        (u3::new(7), literal_len - 7)
    };

    let token = (lll.value() << 5) | mmmmm.value();
    out.push(token);
    out.extend_from_slice(&distance.to_le_bytes());

    if lll == u3::new(7) {
        encode_extension(lit_ext, out);
    }
    if mmmmm == u5::new(0) || mmmmm == u5::new(31) {
        encode_extension(match_ext, out);
    }
}

/// Decodes a single LZR frame from `buf` starting at `*pos`.
///
/// Reads the token byte, distance, and any extension bytes. Returns
/// `(literal_len, match_len, distance)` and advances `*pos` by the number of
/// bytes consumed (NOT including any literal payload bytes that follow).
///
/// When `distance` is 0 the frame is the end-of-stream sentinel — no
/// extensions are read and the caller should stop decoding frames.
pub(crate) fn decode_frame(buf: &[u8], pos: &mut usize) -> (i16, i16, u16) {
    let p = *pos;
    let token = buf[p];
    let lll = token >> 5;
    let mmmmm = token & 0x1F;

    let distance = u16::from_le_bytes([buf[p + 1], buf[p + 2]]);
    *pos = p + 3;

    // Distance 0 is reserved for EOS — no extensions follow.
    if distance == 0 {
        return (i16::from(lll), MATCH_LENGTH_TABLE[mmmmm as usize], 0);
    }

    let literal_len = if lll < 7 {
        i16::from(lll)
    } else {
        7 + decode_extension(buf, pos)
    };

    let match_len = match mmmmm {
        0 => MATCH_LENGTH_TABLE[0] - decode_extension(buf, pos),
        31 => MATCH_LENGTH_TABLE[31] + decode_extension(buf, pos),
        m => MATCH_LENGTH_TABLE[m as usize],
    };

    (literal_len, match_len, distance)
}

/// Writes the 3-byte end-of-stream sentinel (token `0x00`, distance `0x0000`).
pub(crate) fn encode_eos(out: &mut Vec<u8>) {
    out.extend_from_slice(&EOS);
}

/// Encodes the stream footer: ULEB128-encoded uncompressed `length` followed
/// by a 4-byte little-endian Adler-32 `checksum`.
pub(crate) fn encode_footer(length: u64, checksum: u32, out: &mut Vec<u8>) {
    encode_uleb128_u64(length, out);
    out.extend_from_slice(&checksum.to_le_bytes());
}

/// Decodes the stream footer from `buf` starting at `*pos`.
///
/// Reads a ULEB128-encoded uncompressed length and a 4-byte little-endian
/// Adler-32 checksum. Returns `(length, checksum)` and advances `*pos`.
pub(crate) fn decode_footer(buf: &[u8], pos: &mut usize) -> (u64, u32) {
    let length = decode_uleb128_u64(buf, pos);
    let p = *pos;
    let checksum = u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]);
    *pos = p + 4;
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
fn encode_extension(mut remainder: i16, out: &mut Vec<u8>) {
    loop {
        if remainder >= 255 {
            out.push(0xFF);
            remainder -= 255;
        } else {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            out.push(remainder as u8);
            break;
        }
    }
}

/// Reads an extension byte chain, returning the accumulated sum.
///
/// Reads bytes until a value less than 255 is encountered. The sum of all
/// bytes read (including the terminator) is returned.
fn decode_extension(buf: &[u8], pos: &mut usize) -> i16 {
    let mut total: i16 = 0;
    loop {
        let byte = buf[*pos];
        *pos += 1;
        total += i16::from(byte);
        if byte < 255 {
            return total;
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    /// Helper: encode a frame and return the written bytes.
    fn encode_frame_to_vec(literal_len: i16, match_len: i16, distance: u16) -> Vec<u8> {
        let mut out = Vec::new();
        encode_frame(literal_len, match_len, distance, &mut out);
        out
    }

    /// Helper: decode a frame from a byte slice.
    fn decode_frame_from_slice(data: &[u8]) -> (i16, i16, u16) {
        let mut pos = 0;
        decode_frame(data, &mut pos)
    }

    #[test]
    fn header_encode() {
        let mut out = Vec::new();
        encode_header(&mut out);
        assert_eq!(out, vec![0x4C, 0x5A, 0x52, 0x00]);
    }

    #[test]
    fn header_decode_valid() {
        let buf = [0x4C, 0x5A, 0x52, 0x00];
        let mut pos = 0;
        assert!(decode_header(&buf, &mut pos).is_ok());
        assert_eq!(pos, 4);
    }

    #[test]
    fn header_decode_bad_magic() {
        let buf = [0x00, 0x00, 0x00, 0x00];
        let mut pos = 0;
        let err = decode_header(&buf, &mut pos).unwrap_err();
        assert!(matches!(err, Error::InvalidMagic));
    }

    #[test]
    fn header_decode_bad_version() {
        let buf = [0x4C, 0x5A, 0x52, 0x01];
        let mut pos = 0;
        let err = decode_header(&buf, &mut pos).unwrap_err();
        assert!(matches!(err, Error::UnsupportedVersion(0x01)));
    }

    #[test]
    fn header_roundtrip() {
        let mut buf = Vec::new();
        encode_header(&mut buf);
        let mut pos = 0;
        assert!(decode_header(&buf, &mut pos).is_ok());
    }

    #[test]
    fn eos_encode() {
        let mut out = Vec::new();
        encode_eos(&mut out);
        assert_eq!(out, vec![0x00, 0x00, 0x00]);
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
        let mut out = Vec::new();
        encode_footer(42, 0x00FB_00B2, &mut out);
        let mut pos = 0;
        let (length, checksum) = decode_footer(&out, &mut pos);
        assert_eq!(length, 42);
        assert_eq!(checksum, 0x00FB_00B2);
        assert_eq!(pos, out.len());
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
            let mut out = Vec::new();
            encode_footer(length, checksum, &mut out);
            let mut pos = 0;
            let (dec_len, dec_cksum) = decode_footer(&out, &mut pos);
            prop_assert_eq!(dec_len, length);
            prop_assert_eq!(dec_cksum, checksum);
            prop_assert_eq!(pos, out.len());
        }
    }
}
