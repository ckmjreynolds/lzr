/// Shared format constants, helpers, and codec primitives for the LZR format.
///
/// This module is internal to the crate and used by both the encoder and decoder.
use std::io::{self, Read, Write};

use crate::{Error, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes at the start of every LZR stream.
pub(crate) const MAGIC: [u8; 3] = [0x4C, 0x5A, 0x52];

/// Current format version.
pub(crate) const VERSION: u8 = 0x00;

/// Total header size in bytes (magic + version + flags).
pub(crate) const HEADER_SIZE: usize = 5;

/// Default compression level (1 = fastest, 9 = best).
pub(crate) const DEFAULT_LEVEL: u8 = 9;

/// Minimum compression level.
pub(crate) const MIN_LEVEL: u8 = 1;

/// Maximum compression level.
pub(crate) const MAX_LEVEL: u8 = 9;

/// Converts a compression level (1-9) to a window exponent (7-15).
pub(crate) const fn level_to_window_exp(level: u8) -> u8 {
    level + 6
}

/// Maximum number of bytes in a bounded ULEB128 encoding.
const ULEB128_MAX_BYTES: usize = 9;

/// The end-of-stream token (LLL=0, DDDDD=0).
pub(crate) const EOS_TOKEN: u8 = 0x00;

/// Returns the window size for a given exponent: `2^(exp + 9)`.
pub(crate) const fn window_size(exp: u8) -> usize {
    1 << (exp as u32 + 9)
}

// ---------------------------------------------------------------------------
// Adler-32
// ---------------------------------------------------------------------------

/// Modulus for Adler-32.
const ADLER_MOD: u32 = 65_521;

/// Largest `n` such that `255*n*(n+1)/2 + (n+1)*(ADLER_MOD-1)` fits in u32.
const NMAX: usize = 5552;

/// Incremental Adler-32 checksum calculator.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Adler32 {
    a: u32,
    b: u32,
}

impl Adler32 {
    /// Creates a new Adler-32 with the initial value (a=1, b=0).
    pub(crate) const fn new() -> Self {
        Self {
            a: 1,
            b: 0,
        }
    }

    /// Feeds `data` into the checksum.
    pub(crate) fn update(&mut self, data: &[u8]) {
        for chunk in data.chunks(NMAX) {
            for &byte in chunk {
                self.a += u32::from(byte);
                self.b += self.a;
            }
            self.a %= ADLER_MOD;
            self.b %= ADLER_MOD;
        }
    }

    /// Returns the final Adler-32 value.
    pub(crate) const fn finish(self) -> u32 {
        (self.b << 16) | self.a
    }
}

// ---------------------------------------------------------------------------
// ULEB128 (bounded, N=9)
// ---------------------------------------------------------------------------

/// Encodes a `u64` as bounded ULEB128 (max 9 bytes).
///
/// Returns `(buffer, length)`.
#[allow(clippy::cast_possible_truncation)]
const fn encode_uleb128(mut value: u64) -> ([u8; ULEB128_MAX_BYTES], usize) {
    let mut buf = [0u8; ULEB128_MAX_BYTES];
    let mut i = 0;

    loop {
        if i == ULEB128_MAX_BYTES - 1 {
            // Last byte: use all 8 bits, no continuation.
            buf[i] = value as u8;
            i += 1;
            break;
        }

        let byte = (value & 0x7F) as u8;
        value >>= 7;

        if value == 0 {
            buf[i] = byte; // bit 7 clear → stop
            i += 1;
            break;
        }

        buf[i] = byte | 0x80; // bit 7 set → continue
        i += 1;
    }

    (buf, i)
}

/// Decodes a bounded ULEB128 value (max 9 bytes) from a reader.
fn decode_uleb128(reader: &mut impl Read) -> io::Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    for i in 0..ULEB128_MAX_BYTES {
        let mut byte = [0u8; 1];
        reader.read_exact(&mut byte)?;
        let b = byte[0];

        if i == ULEB128_MAX_BYTES - 1 {
            // Last byte: all 8 bits are data.
            result |= u64::from(b) << shift;
            break;
        }

        result |= u64::from(b & 0x7F) << shift;
        shift += 7;

        if b & 0x80 == 0 {
            break;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Header / Footer
// ---------------------------------------------------------------------------

/// Writes the 5-byte LZR header.
pub(crate) fn write_header(writer: &mut impl Write, window_exp: u8) -> io::Result<()> {
    let flags = window_exp & 0x0F;
    let header = [MAGIC[0], MAGIC[1], MAGIC[2], VERSION, flags];
    writer.write_all(&header)
}

/// Reads and validates the 5-byte LZR header. Returns the window exponent.
pub(crate) fn read_header(reader: &mut impl Read) -> Result<u8> {
    let mut buf = [0u8; HEADER_SIZE];
    reader.read_exact(&mut buf)?;

    if buf[0] != MAGIC[0] || buf[1] != MAGIC[1] || buf[2] != MAGIC[2] {
        return Err(Error::InvalidFormat);
    }
    if buf[3] != VERSION {
        return Err(Error::InvalidFormat);
    }

    Ok(buf[4] & 0x0F)
}

/// Writes the footer: ULEB128 uncompressed length + Adler-32 checksum (LE).
pub(crate) fn write_footer(writer: &mut impl Write, uncompressed_len: u64, checksum: u32) -> io::Result<()> {
    let (buf, len) = encode_uleb128(uncompressed_len);
    writer.write_all(&buf[..len])?;
    writer.write_all(&checksum.to_le_bytes())
}

/// Reads the footer. Returns `(uncompressed_len, checksum)`.
pub(crate) fn read_footer(reader: &mut impl Read) -> io::Result<(u64, u32)> {
    let length = decode_uleb128(reader)?;
    let mut checksum_buf = [0u8; 4];
    reader.read_exact(&mut checksum_buf)?;
    let checksum = u32::from_le_bytes(checksum_buf);
    Ok((length, checksum))
}

// ---------------------------------------------------------------------------
// Token helpers
// ---------------------------------------------------------------------------

/// Builds a token byte from LLL (3 bits) and DDDDD (5 bits).
pub(crate) const fn make_token(lll: u8, ddddd: u8) -> u8 {
    (lll << 5) | ddddd
}

/// Splits a token byte into `(LLL, DDDDD)`.
pub(crate) const fn split_token(token: u8) -> (u8, u8) {
    (token >> 5, token & 0x1F)
}

/// Encodes a signed length into `(lll, extension_bytes, extension_len)`.
///
/// - `lll` 0-5: length is inline (direct).
/// - `lll` 6: 1-byte signed extension.
/// - `lll` 7: 2-byte signed little-endian extension.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub(crate) fn encode_length(length: i32) -> (u8, [u8; 2], usize) {
    let mut ext = [0u8; 2];
    if (0..=5).contains(&length) {
        (length as u8, ext, 0)
    } else if (-128..=127).contains(&length) {
        ext[0] = length as i8 as u8;
        (6, ext, 1)
    } else {
        let bytes = (length as i16).to_le_bytes();
        ext[0] = bytes[0];
        ext[1] = bytes[1];
        (7, ext, 2)
    }
}

/// Decodes a length given the LLL code and a reader for extension bytes.
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn decode_length(lll: u8, reader: &mut impl Read) -> io::Result<i32> {
    match lll {
        0..=5 => Ok(i32::from(lll)),
        6 => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(i32::from(buf[0] as i8))
        }
        7 => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(i32::from(i16::from_le_bytes(buf)))
        }
        _ => unreachable!("LLL is 3 bits, max 7"),
    }
}

/// Encodes a distance into `(ddddd, extension_bytes, extension_len)`.
///
/// - `ddddd` 0-28: distance is inline (direct).
/// - `ddddd` 29: 1-byte unsigned extension, value = stored + 1.
/// - `ddddd` 30: 2-byte unsigned LE extension, value = stored + 1.
/// - `ddddd` 31: 3-byte unsigned LE extension, value = stored + 1.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn encode_distance(distance: u32) -> (u8, [u8; 3], usize) {
    let mut ext = [0u8; 3];
    if distance <= 28 {
        (distance as u8, ext, 0)
    } else if distance <= 256 {
        ext[0] = (distance - 1) as u8;
        (29, ext, 1)
    } else if distance <= 65_536 {
        let val = (distance - 1) as u16;
        let bytes = val.to_le_bytes();
        ext[0] = bytes[0];
        ext[1] = bytes[1];
        (30, ext, 2)
    } else {
        let val = distance - 1;
        ext[0] = val as u8;
        ext[1] = (val >> 8) as u8;
        ext[2] = (val >> 16) as u8;
        (31, ext, 3)
    }
}

/// Decodes a distance given the DDDDD code and a reader for extension bytes.
pub(crate) fn decode_distance(ddddd: u8, reader: &mut impl Read) -> io::Result<u32> {
    match ddddd {
        0..=28 => Ok(u32::from(ddddd)),
        29 => {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(u32::from(buf[0]) + 1)
        }
        30 => {
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(u32::from(u16::from_le_bytes(buf)) + 1)
        }
        31 => {
            let mut buf = [0u8; 3];
            reader.read_exact(&mut buf)?;
            let val = u32::from(buf[0]) | (u32::from(buf[1]) << 8) | (u32::from(buf[2]) << 16);
            Ok(val + 1)
        }
        _ => unreachable!("DDDDD is 5 bits, max 31"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- window_size --

    #[test]
    fn window_size_values() {
        assert_eq!(window_size(0), 512);
        assert_eq!(window_size(12), 1 << 21); // 2 MiB
        assert_eq!(window_size(15), 1 << 24); // 16 MiB
    }

    // -- Adler-32 --

    #[test]
    fn adler32_empty() {
        let a = Adler32::new();
        assert_eq!(a.finish(), 1);
    }

    #[test]
    fn adler32_wikipedia() {
        let mut a = Adler32::new();
        a.update(b"Wikipedia");
        assert_eq!(a.finish(), 0x11E6_0398);
    }

    #[test]
    fn adler32_chunked_consistency() {
        let data = b"Wikipedia";
        let mut one_shot = Adler32::new();
        one_shot.update(data);

        let mut chunked = Adler32::new();
        chunked.update(&data[..4]);
        chunked.update(&data[4..]);

        assert_eq!(one_shot.finish(), chunked.finish());
    }

    // -- ULEB128 --

    #[test]
    fn uleb128_round_trips() {
        for &val in &[0u64, 1, 127, 128, 16383, 16384, u64::MAX] {
            let (buf, len) = encode_uleb128(val);
            let mut cursor = io::Cursor::new(&buf[..len]);
            let decoded = decode_uleb128(&mut cursor).unwrap();
            assert_eq!(decoded, val, "ULEB128 round-trip failed for {val}");
        }
    }

    // -- Header --

    #[test]
    fn header_round_trip() {
        for exp in [0u8, 7, 12, 15] {
            let mut buf = Vec::new();
            write_header(&mut buf, exp).unwrap();
            assert_eq!(buf.len(), HEADER_SIZE);
            let decoded = read_header(&mut &buf[..]).unwrap();
            assert_eq!(decoded, exp);
        }
    }

    #[test]
    fn header_bad_magic() {
        let buf = [0x00, 0x00, 0x00, VERSION, 0x00];
        let err = read_header(&mut &buf[..]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat));
    }

    #[test]
    fn header_bad_version() {
        let buf = [MAGIC[0], MAGIC[1], MAGIC[2], 0x01, 0x00];
        let err = read_header(&mut &buf[..]).unwrap_err();
        assert!(matches!(err, Error::InvalidFormat));
    }

    #[test]
    fn header_reserved_bits_ignored() {
        // Set reserved upper 4 bits to 0xF, window exp = 5.
        let buf = [MAGIC[0], MAGIC[1], MAGIC[2], VERSION, 0xF5];
        let exp = read_header(&mut &buf[..]).unwrap();
        assert_eq!(exp, 5);
    }

    // -- Length encode/decode --

    #[test]
    fn length_round_trips() {
        for &val in &[0i32, 1, 5, -1, 127, -128, 128, 32767, -32768] {
            let (lll, ext, ext_len) = encode_length(val);
            let mut cursor = io::Cursor::new(&ext[..ext_len]);
            let decoded = decode_length(lll, &mut cursor).unwrap();
            assert_eq!(decoded, val, "length round-trip failed for {val}");
        }
    }

    // -- Distance encode/decode --

    #[test]
    fn distance_round_trips() {
        for &val in &[0u32, 1, 28, 29, 256, 257, 65536, 65537, 16_777_216] {
            let (ddddd, ext, ext_len) = encode_distance(val);
            let mut cursor = io::Cursor::new(&ext[..ext_len]);
            let decoded = decode_distance(ddddd, &mut cursor).unwrap();
            assert_eq!(decoded, val, "distance round-trip failed for {val}");
        }
    }

    // -- Token --

    #[test]
    fn token_round_trip() {
        for lll in 0..8u8 {
            for ddddd in 0..32u8 {
                let token = make_token(lll, ddddd);
                let (l, d) = split_token(token);
                assert_eq!((l, d), (lll, ddddd));
            }
        }
    }

    // -- Footer --

    #[test]
    fn footer_round_trip() {
        let len = 123_456_789u64;
        let checksum = 0xDEAD_BEEF;
        let mut buf = Vec::new();
        write_footer(&mut buf, len, checksum).unwrap();
        let (decoded_len, decoded_checksum) = read_footer(&mut &buf[..]).unwrap();
        assert_eq!(decoded_len, len);
        assert_eq!(decoded_checksum, checksum);
    }
}
