//! LZR decompression decoder.
//!
//! # Examples
//!
//! ```
//! use lzr::decode::decode;
//!
//! // The 13-byte "Hi" stream from the FORMAT.md worked example.
//! let compressed = [
//!     0x4C, 0x5A, 0x52, 0x00, 0xC4, 0x48, 0x69, 0xC0,
//!     0x02, 0xB2, 0x00, 0xFB, 0x00,
//! ];
//! let mut output = Vec::new();
//! decode(&mut &compressed[..], &mut output).unwrap();
//! assert_eq!(&output, b"Hi");
//! ```

use std::io::{Read, Write};

use crate::adler32::Adler32;
use crate::error::{Error, Result};
use crate::frame::{
    LONG_BASE_MAX, MEDIUM_FWD_BASE_MAX, MEDIUM_REV_BASE_MAX, SHORT_BASE_MAX, decode_extension, decode_uleb128_u64,
};
#[cfg(test)]
use crate::ringbuf::RingBuf;
use crate::{HEADER, WINDOW_SIZE};

/// Flat decode buffer: `[0..WINDOW_SIZE]` = history, `[WINDOW_SIZE..BUF_LEN]` = decode area.
const BUF_LEN: usize = WINDOW_SIZE * 2;

/// Reads and validates a 4-byte LZR header.
fn read_header(reader: &mut impl Read) -> Result<()> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    if buf[..3] != HEADER[..3] {
        return Err(Error::InvalidMagic);
    }
    if buf[3] != HEADER[3] {
        return Err(Error::UnsupportedVersion(buf[3]));
    }
    Ok(())
}

/// Flushes decoded data from the flat buffer and shifts the window.
#[inline(never)]
fn flush_decode(buf: &mut [u8], pos: &mut usize, adler: &mut Adler32, output: &mut impl Write) -> Result<()> {
    if *pos > WINDOW_SIZE {
        adler.update(&buf[WINDOW_SIZE..*pos]);
        output.write_all(&buf[WINDOW_SIZE..*pos])?;
        buf.copy_within(*pos - WINDOW_SIZE..*pos, 0);
        *pos = WINDOW_SIZE;
    }
    Ok(())
}

/// Replays a forward match in the flat decode buffer. Handles overlapping via
/// seed-and-double.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn replay_forward(buf: &mut [u8], pos: &mut usize, distance: u32, len: usize) {
    let dist = distance as usize;
    let src = *pos - dist;
    if dist >= len {
        buf.copy_within(src..src + len, *pos);
    } else {
        buf.copy_within(src..src + dist, *pos);
        let mut copied = dist;
        while copied < len {
            let chunk = copied.min(len - copied);
            buf.copy_within(*pos..*pos + chunk, *pos + copied);
            copied += chunk;
        }
    }
    *pos += len;
}

/// Replays a reverse match in the flat decode buffer (per-byte, rare path).
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn replay_reverse(buf: &mut [u8], pos: &mut usize, distance: u32, len: usize) {
    let start = *pos - distance as usize;
    for i in 0..len {
        buf[*pos + i] = buf[start - i];
    }
    *pos += len;
}

/// Decodes a single LZR stream (header already consumed).
///
/// Uses a flat 128 KiB buffer: the first 64 KiB is the history window and the
/// second 64 KiB is the active decode area. Literals read directly into the
/// buffer; matches resolve via `copy_within` on the same buffer — eliminating
/// the ring-buffer and output-buffer double copy.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn decode_stream(reader: &mut impl Read, output: &mut impl Write) -> Result<()> {
    let mut buf = vec![0u8; BUF_LEN];
    let mut pos = WINDOW_SIZE;
    let mut adler = Adler32::new();
    let mut total_len: u64 = 0;

    loop {
        let mut hdr = [0u8];
        reader.read_exact(&mut hdr)?;
        let b = hdr[0];

        // Medium frames (0x00–0xBF) are most common — check first.
        if b < 0xC0 {
            // Medium frame: SLLLLDDD
            let s = (b >> 7) & 1;
            let l_field = u32::from((b >> 3) & 0x0F);
            let d_low3 = u32::from(b & 0x07);
            let mut ext = [0u8];
            reader.read_exact(&mut ext)?;
            let distance = (u32::from(ext[0]) << 3 | d_low3) + 1;
            let base_max = if s == 0 {
                MEDIUM_FWD_BASE_MAX
            } else {
                MEDIUM_REV_BASE_MAX
            };
            let magnitude = decode_extension(l_field + 2, base_max, reader)? as usize;
            decode_dir(&mut buf, &mut pos, distance, s == 0, magnitude, &mut adler, output)?;
            total_len += magnitude as u64;
        } else if b == 0xC0 {
            break; // EOS
        } else if b < 0xE0 {
            // Short frame: 110LLLLD (0xC1–0xDF)
            let l_field = u32::from((b >> 1) & 0x0F);
            let d_flag = b & 0x01;
            let magnitude = decode_extension(l_field, SHORT_BASE_MAX, reader)? as usize;

            if d_flag == 0 {
                decode_literal(reader, &mut buf, &mut pos, magnitude, &mut adler, output)?;
                total_len += magnitude as u64;
            } else if magnitude == 0 {
                return Err(Error::InvalidFormat("reserved Short frame (L=0, D=1)".into()));
            } else {
                decode_dir(&mut buf, &mut pos, 1, true, magnitude, &mut adler, output)?;
                total_len += magnitude as u64;
            }
        } else {
            // Long frame: 111SLLLL (0xE0–0xFF)
            let s = (b >> 4) & 1;
            let l_field = u32::from(b & 0x0F);
            let mut dist_buf = [0u8; 2];
            reader.read_exact(&mut dist_buf)?;
            let distance = u32::from(u16::from_le_bytes(dist_buf)) + 1;
            let magnitude = decode_extension(l_field + 3, LONG_BASE_MAX, reader)? as usize;
            decode_dir(&mut buf, &mut pos, distance, s == 0, magnitude, &mut adler, output)?;
            total_len += magnitude as u64;
        }
    }

    // Flush remaining decoded data.
    flush_decode(&mut buf, &mut pos, &mut adler, output)?;

    // Footer: ULEB128 u64 length + 4-byte LE Adler-32.
    let expected_len = decode_uleb128_u64(reader)?;
    let mut checksum_buf = [0u8; 4];
    reader.read_exact(&mut checksum_buf)?;
    let expected_checksum = u32::from_le_bytes(checksum_buf);

    if total_len != expected_len {
        return Err(Error::LengthMismatch {
            expected: expected_len,
            actual: total_len,
        });
    }

    let actual_checksum = adler.checksum();
    if actual_checksum != expected_checksum {
        return Err(Error::ChecksumMismatch {
            expected: expected_checksum,
            actual: actual_checksum,
        });
    }

    Ok(())
}

/// Reads a literal of `len` bytes into the flat buffer, flushing as needed.
#[inline]
fn decode_literal(
    reader: &mut impl Read,
    buf: &mut [u8],
    pos: &mut usize,
    len: usize,
    adler: &mut Adler32,
    output: &mut impl Write,
) -> Result<()> {
    if *pos + len <= BUF_LEN {
        reader.read_exact(&mut buf[*pos..*pos + len])?;
        *pos += len;
    } else {
        decode_literal_large(reader, buf, pos, len, adler, output)?;
    }
    Ok(())
}

/// Slow path for literals that don't fit in the remaining decode area.
#[cold]
#[inline(never)]
fn decode_literal_large(
    reader: &mut impl Read,
    buf: &mut [u8],
    pos: &mut usize,
    len: usize,
    adler: &mut Adler32,
    output: &mut impl Write,
) -> Result<()> {
    let mut remaining = len;
    while remaining > 0 {
        if *pos >= BUF_LEN {
            flush_decode(buf, pos, adler, output)?;
        }
        let chunk = remaining.min(BUF_LEN - *pos);
        reader.read_exact(&mut buf[*pos..*pos + chunk])?;
        *pos += chunk;
        remaining -= chunk;
    }
    Ok(())
}

/// Replays a match in either direction, flushing as needed.
#[inline]
fn decode_dir(
    buf: &mut [u8],
    pos: &mut usize,
    distance: u32,
    is_forward: bool,
    len: usize,
    adler: &mut Adler32,
    output: &mut impl Write,
) -> Result<()> {
    if *pos + len <= BUF_LEN {
        if is_forward {
            replay_forward(buf, pos, distance, len);
        } else {
            replay_reverse(buf, pos, distance, len);
        }
    } else {
        decode_match_large(buf, pos, distance, is_forward, len, adler, output)?;
    }
    Ok(())
}

/// Slow path for matches that don't fit in the remaining decode area.
#[cold]
#[inline(never)]
fn decode_match_large(
    buf: &mut [u8],
    pos: &mut usize,
    distance: u32,
    is_forward: bool,
    len: usize,
    adler: &mut Adler32,
    output: &mut impl Write,
) -> Result<()> {
    let mut remaining = len;
    while remaining > 0 {
        if *pos >= BUF_LEN {
            flush_decode(buf, pos, adler, output)?;
        }
        let chunk = remaining.min(BUF_LEN - *pos);
        if is_forward {
            replay_forward(buf, pos, distance, chunk);
        } else {
            replay_reverse(buf, pos, distance, chunk);
        }
        remaining -= chunk;
    }
    Ok(())
}

/// Decompresses one or more concatenated LZR streams from `input` into `output`.
///
/// Reads frames from the bitstream, replays matches against a 64 KiB sliding
/// window, and verifies the footer checksum and length for each stream. After a
/// complete stream, if more data is available it is treated as a new stream.
///
/// # Errors
///
/// Returns an error if:
/// - The magic number is invalid ([`Error::InvalidMagic`]).
/// - The format version is unsupported ([`Error::UnsupportedVersion`]).
/// - The decompressed length doesn't match the footer ([`Error::LengthMismatch`]).
/// - The Adler-32 checksum doesn't match ([`Error::ChecksumMismatch`]).
/// - An I/O error occurs.
///
/// # Examples
///
/// ```
/// use lzr::decode::decode;
///
/// let compressed = [
///     0x4C, 0x5A, 0x52, 0x00, 0xC4, 0x48, 0x69, 0xC0,
///     0x02, 0xB2, 0x00, 0xFB, 0x00,
/// ];
/// let mut output = Vec::new();
/// decode(&mut compressed.as_slice(), &mut output).unwrap();
/// assert_eq!(&output, b"Hi");
/// ```
pub fn decode(input: &mut impl Read, output: &mut impl Write) -> Result<()> {
    // First stream: UnexpectedEof on header → InvalidMagic (empty/truncated input).
    match read_header(input) {
        Ok(()) => {}
        Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(Error::InvalidMagic);
        }
        Err(e) => return Err(e),
    }
    decode_stream(input, output)?;

    // Subsequent streams: UnexpectedEof on header → clean EOF.
    loop {
        match read_header(input) {
            Ok(()) => decode_stream(input, output)?,
            Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
mod tests {
    use pretty_assertions::assert_eq;

    use smallvec::SmallVec;

    use super::*;
    use crate::HEADER;
    use crate::adler32::Adler32;
    use crate::frame::{Frame, encode_uleb128_u64};

    /// Helper: builds a complete LZR stream from frames and returns the bytes.
    fn build_stream(frames: &[Frame]) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header.
        buf.extend_from_slice(&HEADER);

        // Frames.
        for frame in frames {
            frame.encode(&mut buf).unwrap();
        }
        Frame::EndOfStream.encode(&mut buf).unwrap();

        // Footer: compute expected output to get length and checksum.
        let expected_output = replay_frames(frames);
        let len = expected_output.len() as u64;
        let mut adler = Adler32::new();
        adler.update(&expected_output);

        encode_uleb128_u64(len, &mut buf).unwrap();
        buf.extend_from_slice(&adler.checksum().to_le_bytes());

        buf
    }

    /// Replays frames to get the expected decompressed output.
    fn replay_frames(frames: &[Frame]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut window = RingBuf::new(WINDOW_SIZE);

        for frame in frames {
            match frame {
                Frame::Literal(bytes) => {
                    for &b in bytes {
                        window.push(b);
                    }
                    output.extend_from_slice(bytes);
                }
                Frame::Match {
                    distance,
                    length,
                } => {
                    let abs_len = length.unsigned_abs() as usize;
                    let start = window.write_pos();
                    let d = *distance as isize;
                    let dir: isize = if *length > 0 {
                        1
                    } else {
                        -1
                    };

                    for i in 0..abs_len {
                        let b = window[start - d + dir * i as isize];
                        window.push(b);
                        output.push(b);
                    }
                }
                Frame::EndOfStream => {}
            }
        }
        output
    }

    #[test]
    fn decode_worked_example() {
        // The 13-byte "Hi" stream from FORMAT.md.
        let data: &[u8] = &[0x4C, 0x5A, 0x52, 0x00, 0xC4, 0x48, 0x69, 0xC0, 0x02, 0xB2, 0x00, 0xFB, 0x00];
        let mut output = Vec::new();
        decode(&mut &data[..], &mut output).unwrap();
        assert_eq!(&output, b"Hi");
    }

    #[test]
    fn decode_empty_stream() {
        let stream = build_stream(&[]);
        let mut output = Vec::new();
        decode(&mut stream.as_slice(), &mut output).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn decode_forward_match() {
        let frames = vec![
            Frame::Literal(SmallVec::from_slice(b"abc")),
            Frame::Match {
                distance: 3,
                length: 3,
            },
        ];
        let stream = build_stream(&frames);
        let mut output = Vec::new();
        decode(&mut stream.as_slice(), &mut output).unwrap();
        assert_eq!(&output, b"abcabc");
    }

    #[test]
    fn decode_reverse_match() {
        let frames = vec![
            Frame::Literal(SmallVec::from_slice(b"abc")),
            Frame::Match {
                distance: 1,
                length: -3,
            },
        ];
        let stream = build_stream(&frames);
        let mut output = Vec::new();
        decode(&mut stream.as_slice(), &mut output).unwrap();
        assert_eq!(&output, b"abccba");
    }

    #[test]
    fn decode_overlapping_match() {
        // D=1, L=10: RLE expansion of the last byte.
        let frames = vec![
            Frame::Literal(SmallVec::from_slice(b"x")),
            Frame::Match {
                distance: 1,
                length: 10,
            },
        ];
        let stream = build_stream(&frames);
        let mut output = Vec::new();
        decode(&mut stream.as_slice(), &mut output).unwrap();
        assert_eq!(&output, b"xxxxxxxxxxx");
    }

    #[test]
    fn decode_extended_length_match() {
        // D=1, L=100: RLE via length extension.
        let frames = vec![
            Frame::Literal(SmallVec::from_slice(b"x")),
            Frame::Match {
                distance: 1,
                length: 100,
            },
        ];
        let stream = build_stream(&frames);
        let mut output = Vec::new();
        decode(&mut stream.as_slice(), &mut output).unwrap();
        assert_eq!(output.len(), 101);
        assert!(output.iter().all(|&b| b == b'x'));
    }

    #[test]
    fn decode_concatenated_streams() {
        let stream1 = build_stream(&[Frame::Literal(SmallVec::from_slice(b"Hello"))]);
        let stream2 = build_stream(&[Frame::Literal(SmallVec::from_slice(b"World"))]);
        let mut combined = stream1;
        combined.extend_from_slice(&stream2);

        let mut output = Vec::new();
        decode(&mut combined.as_slice(), &mut output).unwrap();
        assert_eq!(&output, b"HelloWorld");
    }

    #[test]
    fn invalid_magic() {
        let data: &[u8] = &[0x00, 0x00, 0x00, 0x00];
        let err = decode(&mut &data[..], &mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::InvalidMagic));
    }

    #[test]
    fn invalid_magic_empty_input() {
        let data: &[u8] = &[];
        let err = decode(&mut &data[..], &mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::InvalidMagic));
    }

    #[test]
    fn unsupported_version() {
        let data: &[u8] = &[0x4C, 0x5A, 0x52, 0x01];
        let err = decode(&mut &data[..], &mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::UnsupportedVersion(1)));
    }

    #[test]
    fn checksum_mismatch() {
        let mut stream = build_stream(&[Frame::Literal(SmallVec::from_slice(b"test"))]);
        // Corrupt the last byte (part of the Adler-32 checksum).
        let last = stream.len() - 1;
        stream[last] ^= 0xFF;

        let err = decode(&mut stream.as_slice(), &mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }));
    }

    #[test]
    fn length_mismatch() {
        let mut stream = build_stream(&[Frame::Literal(SmallVec::from_slice(b"test"))]);
        // Footer length is a single ULEB128 byte (value 4). It sits 5 bytes from the end.
        let footer_len_pos = stream.len() - 5;
        stream[footer_len_pos] = 0x05; // Change length from 4 to 5

        let err = decode(&mut stream.as_slice(), &mut Vec::new()).unwrap_err();
        assert!(matches!(err, Error::LengthMismatch { .. }), "expected LengthMismatch, got {err:?}");
    }
}
