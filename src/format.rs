//! LZR format codec — header, frame, EOS, and footer encode/decode.
//!
//! All functions operate on byte buffers and are independent of I/O. The caller
//! is responsible for reading from and writing to streams.
//!
//! See [`FORMAT.md`](../docs/FORMAT.md) for the canonical specification.

use crate::HEADER;
use crate::error::{Error, Result};
use crate::uleb128::{decode_uleb128_u64, encode_uleb128_u64};

/// Maximum size of a frame header in bytes (token + distance + extensions, excluding literals).
///
/// Worst case: 1 (token) + 2 (distance) + 129 (literal extension for 32,767)
/// + 129 (match extension for 32,767 or -32,768) = 261.
pub(crate) const MAX_FRAME_HEADER: usize = 261;

// ── Header ──────────────────────────────────────────────────────────────

/// Writes the 4-byte LZR stream header into `buf` and returns 4.
pub(crate) fn encode_header(buf: &mut [u8]) -> usize {
    buf[..4].copy_from_slice(&HEADER);
    4
}

/// Validates the 4-byte LZR stream header in `buf` and returns 4.
///
/// # Errors
///
/// Returns [`Error::InvalidMagic`] if the first three bytes are not `LZR`.
/// Returns [`Error::UnsupportedVersion`] if the version byte is not `0x00`.
pub(crate) fn decode_header(buf: &[u8]) -> Result<usize> {
    if buf[..3] != HEADER[..3] {
        return Err(Error::InvalidMagic);
    }
    if buf[3] != 0x00 {
        return Err(Error::UnsupportedVersion(buf[3]));
    }
    Ok(4)
}

// ── Frame ───────────────────────────────────────────────────────────────

/// Encodes a frame into `buf` and returns the number of bytes written.
///
/// Writes the token byte, distance, any extension bytes, and the literal data.
/// `buf` must be large enough to hold the entire frame ([`MAX_FRAME_HEADER`]
/// + `literals.len()`).
///
/// # Parameters
///
/// - `literals`: literal bytes to embed in the frame (length 0..=32,767).
/// - `match_length`: signed match length. Negative = reverse copy, 0 = literal-only,
///   positive = forward copy. Valid range: -32,768..=-4 or 0 or 4..=32,767.
/// - `distance`: logical copy distance (1..=65,535). Ignored when `match_length == 0`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn encode_frame(literals: &[u8], match_length: i16, distance: u16, buf: &mut [u8]) -> usize {
    let lit_count = literals.len() as u16;
    let l = lit_count.min(7) as u8;

    let m: u8 = if match_length <= -9 {
        0
    } else if match_length < 0 {
        (match_length + 9) as u8 // -8→1, -7→2, ..., -4→5
    } else if match_length == 0 {
        6
    } else if match_length >= 28 {
        31
    } else {
        (match_length + 3) as u8 // 4→7, 5→8, ..., 27→30
    };

    buf[0] = (l << 5) | m;
    buf[1..3].copy_from_slice(&distance.to_le_bytes());
    let mut offset = 3;

    // Literal length extension.
    if l == 7 {
        offset += encode_extension(lit_count - 7, &mut buf[offset..]);
    }

    // Match length extension.
    if m == 0 {
        let extra = ((-9i16).wrapping_sub(match_length)) as u16;
        offset += encode_extension(extra, &mut buf[offset..]);
    } else if m == 31 {
        let extra = (match_length - 28) as u16;
        offset += encode_extension(extra, &mut buf[offset..]);
    }

    // Literal data.
    buf[offset..offset + literals.len()].copy_from_slice(literals);
    offset + literals.len()
}

/// Decodes a frame from `buf`.
///
/// Returns `None` for the end-of-stream sentinel. Otherwise returns
/// `Some((literal_count, distance, match_length, header_size))` where
/// `header_size` is the number of bytes consumed by the token, distance, and
/// extensions. The caller reads `literal_count` literal bytes starting at
/// `buf[header_size..]`.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub(crate) fn decode_frame(buf: &[u8]) -> Option<(i16, u16, i16, usize)> {
    let token = buf[0];
    let distance = u16::from_le_bytes([buf[1], buf[2]]);
    let mut offset = 3;

    // EOS sentinel: token 0x00, distance 0x0000.
    if token == 0x00 && distance == 0x0000 {
        return None;
    }

    let l = token >> 5;
    let m = token & 0x1F;

    // Decode literal count.
    let literal_count: i16 = if l == 7 {
        let (ext, n) = decode_extension(&buf[offset..]);
        offset += n;
        7 + ext as i16
    } else {
        i16::from(l)
    };

    // Decode match length.
    let match_length: i16 = if m == 0 {
        let (ext, n) = decode_extension(&buf[offset..]);
        offset += n;
        -9 - ext as i16
    } else if m <= 5 {
        i16::from(m) - 9 // 1→-8, 2→-7, ..., 5→-4
    } else if m == 6 {
        0
    } else if m == 31 {
        let (ext, n) = decode_extension(&buf[offset..]);
        offset += n;
        28 + ext as i16
    } else {
        i16::from(m) - 3 // 7→4, 8→5, ..., 30→27
    };

    Some((literal_count, distance, match_length, offset))
}

// ── EOS ─────────────────────────────────────────────────────────────────

/// Writes the 3-byte end-of-stream sentinel into `buf` and returns 3.
pub(crate) fn encode_eos(buf: &mut [u8]) -> usize {
    buf[0] = 0x00;
    buf[1] = 0x00;
    buf[2] = 0x00;
    3
}

// ── Footer ──────────────────────────────────────────────────────────────

/// Encodes the stream footer (ULEB128 length + Adler-32 checksum) into `buf`.
///
/// Returns the number of bytes written (at most 13: 9 for ULEB128 + 4 for checksum).
pub(crate) fn encode_footer(uncompressed_len: u64, checksum: u32, buf: &mut [u8]) -> usize {
    let n = encode_uleb128_u64(uncompressed_len, buf);
    buf[n..n + 4].copy_from_slice(&checksum.to_le_bytes());
    n + 4
}

/// Decodes the stream footer from `buf`.
///
/// Returns `(uncompressed_len, checksum, bytes_consumed)`. The caller is
/// responsible for verifying the length and checksum against the decoded data.
pub(crate) fn decode_footer(buf: &[u8]) -> (u64, u32, usize) {
    let (len, n) = decode_uleb128_u64(buf);
    let checksum = u32::from_le_bytes([buf[n], buf[n + 1], buf[n + 2], buf[n + 3]]);
    (len, checksum, n + 4)
}

// ── Extension helpers ───────────────────────────────────────────────────

/// Encodes a length extension chain into `buf` and returns bytes written.
///
/// The extension represents `extra` additional units beyond the base value.
/// Always writes at least one byte (a `0x00` byte when `extra == 0`).
#[allow(clippy::cast_possible_truncation)]
fn encode_extension(extra: u16, buf: &mut [u8]) -> usize {
    let mut remaining = extra;
    let mut offset = 0;
    loop {
        if remaining >= 255 {
            buf[offset] = 0xFF;
            offset += 1;
            remaining -= 255;
        } else {
            buf[offset] = remaining as u8;
            offset += 1;
            break;
        }
    }
    offset
}

/// Decodes a length extension chain from `buf`, returning `(sum, bytes_read)`.
fn decode_extension(buf: &[u8]) -> (u16, usize) {
    let mut sum: u16 = 0;
    let mut offset = 0;
    loop {
        let byte = u16::from(buf[offset]);
        sum += byte;
        offset += 1;
        if byte < 255 {
            break;
        }
    }
    (sum, offset)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;

    // ── Header ──────────────────────────────────────────────────────────

    #[test]
    fn header_roundtrip() {
        let mut buf = [0u8; 4];
        let n = encode_header(&mut buf);
        assert_eq!(n, 4);
        assert_eq!(&buf, &[0x4C, 0x5A, 0x52, 0x00]);
        assert_eq!(decode_header(&buf).unwrap(), 4);
    }

    #[test]
    fn header_bad_magic() {
        let buf = [0x00, 0x00, 0x00, 0x00];
        assert!(matches!(decode_header(&buf), Err(Error::InvalidMagic)));
    }

    #[test]
    fn header_bad_version() {
        let buf = [0x4C, 0x5A, 0x52, 0x01];
        assert!(matches!(decode_header(&buf), Err(Error::UnsupportedVersion(1))));
    }

    // ── EOS ─────────────────────────────────────────────────────────────

    #[test]
    fn eos_roundtrip() {
        let mut buf = [0xFFu8; 3];
        let n = encode_eos(&mut buf);
        assert_eq!(n, 3);
        assert_eq!(&buf, &[0x00, 0x00, 0x00]);
        assert!(decode_frame(&buf).is_none());
    }

    // ── Extension helpers ───────────────────────────────────────────────

    #[test]
    fn extension_known_values() {
        let cases: &[(u16, &[u8])] = &[
            (0, &[0x00]),
            (1, &[0x01]),
            (254, &[0xFE]),
            (255, &[0xFF, 0x00]),
            (256, &[0xFF, 0x01]),
            (510, &[0xFF, 0xFF, 0x00]),
            (511, &[0xFF, 0xFF, 0x01]),
        ];

        for &(extra, expected) in cases {
            let mut buf = [0u8; 130];
            let n = encode_extension(extra, &mut buf);
            assert_eq!(&buf[..n], expected, "encode {extra}");

            let (decoded, consumed) = decode_extension(expected);
            assert_eq!(consumed, expected.len(), "decode len {extra}");
            assert_eq!(decoded, extra, "decode {extra}");
        }
    }

    // ── Frame known values ──────────────────────────────────────────────

    #[test]
    fn frame_literal_only_hi() {
        // Worked example from FORMAT.md: "Hi" literal-only frame.
        // Token 0x46 = 010_00110: L=2, M=6 (literal-only).
        // Distance 0x0000 (ignored). Literals: 0x48 0x69.
        let mut buf = [0u8; 64];
        let n = encode_frame(b"Hi", 0, 0, &mut buf);
        assert_eq!(&buf[..n], &[0x46, 0x00, 0x00, 0x48, 0x69]);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 2);
        assert_eq!(distance, 0);
        assert_eq!(match_length, 0);
        assert_eq!(header_size, 3);
        assert_eq!(&buf[header_size..header_size + usize::from(lit_count.cast_unsigned())], b"Hi");
    }

    #[test]
    fn frame_forward_match_no_literals() {
        // match_length=4, distance=1 → M=7, L=0. Token = 000_00111 = 0x07.
        let mut buf = [0u8; 64];
        let n = encode_frame(&[], 4, 1, &mut buf);
        assert_eq!(&buf[..n], &[0x07, 0x01, 0x00]);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 0);
        assert_eq!(distance, 1);
        assert_eq!(match_length, 4);
        assert_eq!(header_size, 3);
    }

    #[test]
    fn frame_reverse_match() {
        // match_length=-4, distance=10 → M=5, L=0. Token = 000_00101 = 0x05.
        let mut buf = [0u8; 64];
        let n = encode_frame(&[], -4, 10, &mut buf);
        assert_eq!(&buf[..n], &[0x05, 0x0A, 0x00]);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 0);
        assert_eq!(distance, 10);
        assert_eq!(match_length, -4);
        assert_eq!(header_size, 3);
    }

    #[test]
    fn frame_negative_extension() {
        // match_length=-9, distance=5 → M=0, extension byte 0x00.
        let mut buf = [0u8; 64];
        let n = encode_frame(&[], -9, 5, &mut buf);
        assert_eq!(&buf[..n], &[0x00, 0x05, 0x00, 0x00]);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 0);
        assert_eq!(distance, 5);
        assert_eq!(match_length, -9);
        assert_eq!(header_size, 4);
    }

    #[test]
    fn frame_negative_extension_larger() {
        // match_length=-10, distance=5 → M=0, extension byte 0x01.
        let mut buf = [0u8; 64];
        let n = encode_frame(&[], -10, 5, &mut buf);
        assert_eq!(&buf[..n], &[0x00, 0x05, 0x00, 0x01]);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 0);
        assert_eq!(distance, 5);
        assert_eq!(match_length, -10);
        assert_eq!(header_size, 4);
    }

    #[test]
    fn frame_positive_extension() {
        // match_length=28, distance=100 → M=31, extension byte 0x00.
        let mut buf = [0u8; 64];
        let n = encode_frame(&[], 28, 100, &mut buf);
        assert_eq!(&buf[..n], &[0x1F, 0x64, 0x00, 0x00]);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 0);
        assert_eq!(distance, 100);
        assert_eq!(match_length, 28);
        assert_eq!(header_size, 4);
    }

    #[test]
    fn frame_positive_extension_larger() {
        // match_length=30, distance=100 → M=31, extension byte 0x02.
        let mut buf = [0u8; 64];
        let n = encode_frame(&[], 30, 100, &mut buf);
        assert_eq!(&buf[..n], &[0x1F, 0x64, 0x00, 0x02]);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 0);
        assert_eq!(distance, 100);
        assert_eq!(match_length, 30);
        assert_eq!(header_size, 4);
    }

    #[test]
    fn frame_literal_extension() {
        // 10 literal bytes, match_length=0 → L=7, extension byte 0x03.
        let literals = [0xAA; 10];
        let mut buf = [0u8; 64];
        let n = encode_frame(&literals, 0, 0, &mut buf);
        // Token: 111_00110 = 0xE6. Distance: 0x0000. Lit ext: 0x03. Then 10 literal bytes.
        assert_eq!(buf[0], 0xE6);
        assert_eq!(&buf[1..3], &[0x00, 0x00]);
        assert_eq!(buf[3], 0x03);
        assert_eq!(&buf[4..14], &[0xAA; 10]);
        assert_eq!(n, 14);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 10);
        assert_eq!(distance, 0);
        assert_eq!(match_length, 0);
        assert_eq!(header_size, 4);
    }

    #[test]
    fn frame_both_extensions() {
        // 10 literals + match_length=30, distance=500.
        // L=7, ext 0x03. M=31, ext 0x02.
        let literals = [0xBB; 10];
        let mut buf = [0u8; 64];
        let n = encode_frame(&literals, 30, 500, &mut buf);
        // Token: 111_11111 = 0xFF. Distance: 500 LE = [0xF4, 0x01].
        // Lit ext: 0x03. Match ext: 0x02. Then 10 literal bytes.
        assert_eq!(buf[0], 0xFF);
        assert_eq!(&buf[1..3], &[0xF4, 0x01]);
        assert_eq!(buf[3], 0x03); // lit ext
        assert_eq!(buf[4], 0x02); // match ext
        assert_eq!(&buf[5..15], &[0xBB; 10]);
        assert_eq!(n, 15);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 10);
        assert_eq!(distance, 500);
        assert_eq!(match_length, 30);
        assert_eq!(header_size, 5);
    }

    // ── Footer ──────────────────────────────────────────────────────────

    #[test]
    fn footer_worked_example() {
        // From FORMAT.md: uncompressed length 2, Adler-32 0x00FB00B2.
        let mut buf = [0u8; 16];
        let n = encode_footer(2, 0x00FB_00B2, &mut buf);
        assert_eq!(&buf[..n], &[0x02, 0xB2, 0x00, 0xFB, 0x00]);
        assert_eq!(n, 5);

        let (len, checksum, consumed) = decode_footer(&buf);
        assert_eq!(len, 2);
        assert_eq!(checksum, 0x00FB_00B2);
        assert_eq!(consumed, 5);
    }

    #[test]
    fn footer_large_length() {
        let mut buf = [0u8; 16];
        let n = encode_footer(u64::MAX, 0xDEAD_BEEF, &mut buf);
        assert_eq!(n, 13); // 9 ULEB128 bytes + 4 checksum bytes

        let (len, checksum, consumed) = decode_footer(&buf);
        assert_eq!(len, u64::MAX);
        assert_eq!(checksum, 0xDEAD_BEEF);
        assert_eq!(consumed, 13);
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn frame_max_literal_count() {
        let literals = vec![0x42u8; 32_767];
        let mut buf = vec![0u8; MAX_FRAME_HEADER + 32_767];
        let n = encode_frame(&literals, 0, 0, &mut buf);

        let (lit_count, _, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 32_767);
        assert_eq!(match_length, 0);
        assert_eq!(&buf[header_size..header_size + 32_767], &literals[..]);
        assert_eq!(n, header_size + 32_767);
    }

    #[test]
    fn frame_max_positive_match() {
        let mut buf = [0u8; 512];
        let n = encode_frame(&[], 32_767, 1000, &mut buf);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 0);
        assert_eq!(distance, 1000);
        assert_eq!(match_length, 32_767);
        assert_eq!(header_size, n);
    }

    #[test]
    fn frame_max_negative_match() {
        let mut buf = [0u8; 512];
        let n = encode_frame(&[], -32_768, 1000, &mut buf);

        let (lit_count, distance, match_length, header_size) = decode_frame(&buf).unwrap();
        assert_eq!(lit_count, 0);
        assert_eq!(distance, 1000);
        assert_eq!(match_length, -32_768);
        assert_eq!(header_size, n);
    }

    #[test]
    fn frame_max_distance() {
        let mut buf = [0u8; 64];
        let n = encode_frame(&[], 4, 65_535, &mut buf);
        assert_eq!(&buf[1..3], &[0xFF, 0xFF]);

        let (_, distance, _, _) = decode_frame(&buf).unwrap();
        assert_eq!(distance, 65_535);
        assert_eq!(n, 3);
    }

    #[test]
    fn frame_min_distance() {
        let mut buf = [0u8; 64];
        encode_frame(&[], 4, 1, &mut buf);

        let (_, distance, _, _) = decode_frame(&buf).unwrap();
        assert_eq!(distance, 1);
    }

    // ── Proptest ────────────────────────────────────────────────────────

    fn match_length_strategy() -> impl Strategy<Value = i16> {
        prop_oneof![
            (-32_768i16..=-4), // reverse match
            Just(0i16),        // literal-only
            (4i16..=32_767),   // forward match
        ]
    }

    proptest! {
        #[test]
        fn frame_roundtrip(
            lit_len in 0u16..=1000,
            match_length in match_length_strategy(),
            distance in 1u16..=65_535,
        ) {
            let literals: Vec<u8> = (0..lit_len).map(|i| u8::try_from(i % 256).unwrap()).collect();
            let mut buf = vec![0u8; MAX_FRAME_HEADER + literals.len()];
            let n = encode_frame(&literals, match_length, distance, &mut buf);

            let result = decode_frame(&buf);
            prop_assert!(result.is_some(), "should not decode as EOS");
            let (dec_lit, dec_dist, dec_match, header_size) = result.unwrap();

            prop_assert_eq!(dec_lit, lit_len.cast_signed());
            prop_assert_eq!(dec_match, match_length);
            prop_assert_eq!(dec_dist, distance);

            let lit_usize = usize::from(lit_len);
            prop_assert_eq!(&buf[header_size..header_size + lit_usize], &literals[..]);
            prop_assert_eq!(n, header_size + lit_usize);
        }

        #[test]
        fn extension_roundtrip(extra: u16) {
            let mut buf = [0u8; 258];
            let n = encode_extension(extra, &mut buf);
            let (decoded, consumed) = decode_extension(&buf);
            prop_assert_eq!(consumed, n);
            prop_assert_eq!(decoded, extra);
        }

        #[test]
        fn footer_roundtrip(len: u64, checksum: u32) {
            let mut buf = [0u8; 16];
            let n = encode_footer(len, checksum, &mut buf);
            let (dec_len, dec_checksum, consumed) = decode_footer(&buf);
            prop_assert_eq!(consumed, n);
            prop_assert_eq!(dec_len, len);
            prop_assert_eq!(dec_checksum, checksum);
        }
    }
}
