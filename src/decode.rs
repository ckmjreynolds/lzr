//! LZR decompression decoder.
//!
//! Reads one or more concatenated LZR streams from an [`impl Read`](Read) source,
//! decompresses each stream, and writes the result to an [`impl Write`](Write) sink.
//! Every stream is independently validated against its footer (Adler-32 checksum
//! and uncompressed length).
//!
//! See [`FORMAT.md`](../docs/FORMAT.md) for the canonical format specification.

use std::io::{Read, Write};

use crate::WINDOW_SIZE;
use crate::adler32::Adler32;
use crate::buffer::Buffer;
use crate::cursor::{ReadCursor, WriteBuf, WriteCursor};
use crate::error::{Error, Result};
use crate::format;

/// Decompresses one or more concatenated LZR streams from `input` into `output`.
///
/// Each stream is validated against its footer: the decompressed byte count must
/// match the stored length, and the Adler-32 checksum must match. If data
/// remains after a footer, it is treated as a new stream beginning with a header.
///
/// # Errors
///
/// Returns [`Error::Io`] on any underlying I/O failure (including unexpected EOF
/// during header or literal reads).
///
/// Returns [`Error::InvalidMagic`] or [`Error::UnsupportedVersion`] if a stream
/// header is malformed.
///
/// Returns [`Error::LengthMismatch`] if the decompressed byte count does not
/// match the footer.
///
/// Returns [`Error::ChecksumMismatch`] if the computed Adler-32 does not match
/// the footer.
pub fn decode(input: &mut impl Read, output: &mut impl Write) -> Result<()> {
    let mut in_buf = Buffer::<WINDOW_SIZE>::new();
    let mut reader = ReadCursor::new(&mut in_buf, 0, 0);

    loop {
        // Check for more data (handles concatenated streams and initial entry).
        if reader.remaining() == 0 && reader.refill(input)? == 0 {
            return Ok(());
        }

        let mut out_buf = Buffer::<WINDOW_SIZE>::new();
        decode_stream(&mut reader, input, &mut out_buf, output)?;
    }
}

/// Decodes a single LZR stream: header, frames, EOS, and footer.
fn decode_stream(
    reader: &mut ReadCursor<'_, WINDOW_SIZE>,
    input: &mut impl Read,
    out_buf: &mut Buffer<WINDOW_SIZE>,
    output: &mut impl Write,
) -> Result<()> {
    let mut writer = WriteCursor::new(out_buf, 0);
    let mut checksum = Adler32::new();
    let mut total_len: u64 = 0;

    // Header
    reader.ensure(input, 4)?;
    format::decode_header(reader)?;

    // Frame loop
    loop {
        reader.fill_toward(input, WINDOW_SIZE)?;

        let (literal_len, match_len, distance) = format::decode_frame(reader);

        // EOS sentinel: token 0x00 + distance 0x0000 → literal_len=0, distance=0.
        // Literal-only frames can also have distance=0, but with literal_len > 0.
        if distance == 0 && literal_len == 0 {
            break;
        }

        // Transfer literal bytes from input to output.
        #[allow(clippy::cast_sign_loss)]
        let lit_count = literal_len as usize;
        transfer_literals(reader, input, &mut writer, output, &mut checksum, lit_count)?;
        total_len += lit_count as u64;

        // Match copy.
        let abs_match = match_len.unsigned_abs() as usize;
        if match_len > 0 {
            flush_if_needed(&mut writer, output, &mut checksum, abs_match)?;
            writer.copy_within(distance as usize, abs_match);
            total_len += abs_match as u64;
        } else if match_len < 0 {
            flush_if_needed(&mut writer, output, &mut checksum, abs_match)?;
            writer.copy_within_rev(distance as usize, abs_match);
            total_len += abs_match as u64;
        }
    }

    // Flush remaining output.
    writer.flush(|bytes| {
        checksum.update(bytes);
        output.write_all(bytes)
    })?;

    // Footer
    reader.fill_toward(input, 13)?;
    let (expected_len, expected_checksum) = format::decode_footer(reader);

    if total_len != expected_len {
        return Err(Error::LengthMismatch {
            expected: expected_len,
            actual: total_len,
        });
    }
    if checksum.checksum() != expected_checksum {
        return Err(Error::ChecksumMismatch {
            expected: expected_checksum,
            actual: checksum.checksum(),
        });
    }

    Ok(())
}

/// Transfers `remaining` literal bytes from the input cursor to the output cursor.
fn transfer_literals(
    reader: &mut ReadCursor<'_, WINDOW_SIZE>,
    input: &mut impl Read,
    writer: &mut WriteCursor<'_, WINDOW_SIZE>,
    output: &mut impl Write,
    checksum: &mut Adler32,
    mut remaining: usize,
) -> Result<()> {
    while remaining > 0 {
        if reader.remaining() == 0 {
            reader.ensure(input, 1)?;
        }

        let chunk = remaining.min(reader.remaining());
        flush_if_needed(writer, output, checksum, chunk)?;

        let (a, b) = reader.consume(chunk);
        writer.write_bytes(a);
        if !b.is_empty() {
            writer.write_bytes(b);
        }

        remaining -= chunk;
    }
    Ok(())
}

/// Flushes the output buffer if writing `needed` more bytes would exceed capacity.
fn flush_if_needed(
    writer: &mut WriteCursor<'_, WINDOW_SIZE>,
    output: &mut impl Write,
    checksum: &mut Adler32,
    needed: usize,
) -> Result<()> {
    if writer.pending() + needed > WINDOW_SIZE {
        writer.flush(|bytes| {
            checksum.update(bytes);
            output.write_all(bytes)
        })?;
    }
    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// Helper: decode a byte slice and return the decompressed output.
    fn decode_bytes(compressed: &[u8]) -> std::result::Result<Vec<u8>, Error> {
        let mut output = Vec::new();
        decode(&mut &compressed[..], &mut output)?;
        Ok(output)
    }

    /// The "Hi" worked example from FORMAT.md.
    #[rustfmt::skip]
    const HI_STREAM: [u8; 17] = [
        0x4C, 0x5A, 0x52, 0x00,             // Header: LZR v0
        0x46, 0x00, 0x00, 0x48, 0x69,       // Frame: L=2, M=6, dist=0, literals "Hi"
        0x00, 0x00, 0x00,                    // EOS
        0x02,                                // Footer: ULEB128 length = 2
        0xB2, 0x00, 0xFB, 0x00,             // Footer: Adler-32 = 0x00FB00B2
    ];

    #[test]
    fn worked_example_hi() {
        let output = decode_bytes(&HI_STREAM).unwrap();
        assert_eq!(output, b"Hi");
    }

    #[test]
    fn empty_stream() {
        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            0x00, 0x00, 0x00,                // EOS
            0x00,                            // Footer: length = 0
            0x01, 0x00, 0x00, 0x00,          // Footer: Adler-32 of empty = 0x00000001
        ];
        let output = decode_bytes(&compressed).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn stream_concatenation() {
        let mut compressed = Vec::new();
        compressed.extend_from_slice(&HI_STREAM);
        compressed.extend_from_slice(&HI_STREAM);

        let output = decode_bytes(&compressed).unwrap();
        assert_eq!(output, b"HiHi");
    }

    #[test]
    fn invalid_magic() {
        let compressed = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let err = decode_bytes(&compressed).unwrap_err();
        assert!(matches!(err, Error::InvalidMagic));
    }

    #[test]
    fn unsupported_version() {
        let compressed = [0x4C, 0x5A, 0x52, 0x01, 0x00, 0x00, 0x00, 0x00];
        let err = decode_bytes(&compressed).unwrap_err();
        assert!(matches!(err, Error::UnsupportedVersion(0x01)));
    }

    #[test]
    fn checksum_mismatch() {
        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            0x46, 0x00, 0x00, 0x48, 0x69,   // Frame: "Hi"
            0x00, 0x00, 0x00,                // EOS
            0x02,                            // Footer: length = 2 (correct)
            0xFF, 0xFF, 0xFF, 0xFF,          // Footer: wrong checksum
        ];
        let err = decode_bytes(&compressed).unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }));
    }

    #[test]
    fn length_mismatch() {
        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            0x46, 0x00, 0x00, 0x48, 0x69,   // Frame: "Hi"
            0x00, 0x00, 0x00,                // EOS
            0x09,                            // Footer: length = 9 (wrong, should be 2)
            0xB2, 0x00, 0xFB, 0x00,          // Footer: correct checksum for "Hi"
        ];
        let err = decode_bytes(&compressed).unwrap_err();
        assert!(matches!(err, Error::LengthMismatch { .. }));
    }

    #[test]
    fn forward_match() {
        // Encode "ABCDABCD" as: 4 literals "ABCD" + forward match (distance=4, length=4).
        let mut checksum = Adler32::new();
        checksum.update(b"ABCDABCD");

        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            // Frame 1: L=4, M=7 (match_len=4), distance=4
            // Token: 100_00111 = 0x87
            0x87, 0x04, 0x00,
            0x41, 0x42, 0x43, 0x44,          // Literals: "ABCD"
            0x00, 0x00, 0x00,                // EOS
            0x08,                            // Footer: length = 8
            checksum.checksum().to_le_bytes()[0],
            checksum.checksum().to_le_bytes()[1],
            checksum.checksum().to_le_bytes()[2],
            checksum.checksum().to_le_bytes()[3],
        ];
        let output = decode_bytes(&compressed).unwrap();
        assert_eq!(output, b"ABCDABCD");
    }

    #[test]
    fn reverse_match() {
        // Encode "ABCDDCBA" as: 4 literals "ABCD" + reverse match (distance=1, length=-4).
        // Reverse copy from pos-1 backwards: D, C, B, A.
        let mut checksum = Adler32::new();
        checksum.update(b"ABCDDCBA");

        #[rustfmt::skip]
        let compressed = [
            0x4C, 0x5A, 0x52, 0x00,         // Header
            // Frame 1: L=4, M=5 (match_len=-4), distance=1
            // Token: 100_00101 = 0x85
            0x85, 0x01, 0x00,
            0x41, 0x42, 0x43, 0x44,          // Literals: "ABCD"
            0x00, 0x00, 0x00,                // EOS
            0x08,                            // Footer: length = 8
            checksum.checksum().to_le_bytes()[0],
            checksum.checksum().to_le_bytes()[1],
            checksum.checksum().to_le_bytes()[2],
            checksum.checksum().to_le_bytes()[3],
        ];
        let output = decode_bytes(&compressed).unwrap();
        assert_eq!(output, b"ABCDDCBA");
    }
}
