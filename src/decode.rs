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

use std::io::{BufRead, Write};

use crate::adler32::Adler32;
use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::uleb128::decode_uleb128_u64;
use crate::{HEADER, WINDOW_SIZE};

use arbitrary_int::{u1, u4, u11};
use smallvec::{SmallVec, smallvec};

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
pub fn decode(input: &mut impl BufRead, output: &mut impl Write) -> Result<()> {
    // Repeat for each stream in the input.
    while input.fill_buf().is_ok_and(|buf| !buf.is_empty()) {
        let mut window = Buffer::<WINDOW_SIZE>::new();
        let mut adler = Adler32::new();
        let mut pos = 0usize;
        let mut pending = 0usize;

        // 1. Read and validate the header.
        let mut header = [0u8; HEADER.len()];
        input.read_exact(&mut header)?;

        if header != HEADER {
            return Err(Error::InvalidMagic);
        }

        // 2. Process and expand frames.
        loop {
            let (length, distance) = decode_frame(input)?;

            // Expand the frame into our sliding window.
            let len = length.unsigned_abs();

            match (length, distance) {
                // Forward match (including RLE when distance=1)
                (1.., 1..) => {
                    window.copy_within(pos - distance..pos - distance + len, pos);
                }
                // Reverse match.
                (..=-1, _) => {
                    window.copy_within_rev(pos - distance..pos - distance + len, pos);
                }
                // Literals
                (1.., 0) => {
                    let mut buf: SmallVec<[u8; 15]> = smallvec![0u8; len];
                    input.read_exact(&mut buf)?;
                    window.copy_from_slice(&buf, pos);
                }
                // EOS (or reserved, we treat them the same)
                (0, _) => break,
            }

            // Write the frame from our window to the output.
            let slices = window.slices(pos, len);

            output.write_all(slices.0)?;

            if !slices.1.is_empty() {
                output.write_all(slices.1)?;
            }

            // Update our position.
            pos += len;
            pending += len;

            // Update the checksum.
            if pending >= i16::MAX as usize {
                let slices = window.slices(pos - pending, pending);
                adler.update(slices.0);

                if !slices.1.is_empty() {
                    adler.update(slices.1);
                }

                pending = 0;
            }
        }

        // 3. Flush any remaining pending bytes to the checksum.
        if pending > 0 {
            let slices = window.slices(pos - pending, pending);
            adler.update(slices.0);
            if !slices.1.is_empty() {
                adler.update(slices.1);
            }
        }

        // 4. Read and verify the footer.
        let expected_len = decode_uleb128_u64(input)?;
        let mut checksum_buf = [0u8; 4];
        input.read_exact(&mut checksum_buf)?;
        let expected_checksum = u32::from_le_bytes(checksum_buf);

        if pos as u64 != expected_len {
            return Err(Error::LengthMismatch {
                expected: expected_len,
                actual: pos as u64,
            });
        }

        let actual_checksum = adler.checksum();
        if actual_checksum != expected_checksum {
            return Err(Error::ChecksumMismatch {
                expected: expected_checksum,
                actual: actual_checksum,
            });
        }
    }
    Ok(())
}

#[allow(clippy::cast_lossless)]
fn decode_frame(input: &mut impl BufRead) -> Result<(isize, usize)> {
    let sign: u1;
    let mut length: i16;
    let distance: u16;

    // Decode the frame (length and distance).
    match input.fill_buf()?[0] {
        0x00..=0xBF => {
            // Medium Frame
            let mut buf = [0u8; 2];
            input.read_exact(&mut buf)?;

            let frame = u16::from_be_bytes(buf);
            sign = u1::extract_u16(frame, 15);
            length = u4::extract_u16(frame, 11).value() as i16 + 2;
            distance = u11::extract_u16(frame, 0).value();

            if sign.value() == 1 {
                length = -length;
            }

            if length == -9 || length == 17 {
                length = decode_extension(input, length)?;
            }

            Ok((length as isize, distance as usize + 1))
        }
        0xC0..=0xDF => {
            // Short Frame (EOS is implicit) L/D = 0.
            let mut buf = [0u8; 1];
            input.read_exact(&mut buf)?;

            length = u4::extract_u8(buf[0], 1).value() as i16;
            distance = u1::extract_u8(buf[0], 0).value() as u16;

            if length == 15 {
                length = decode_extension(input, length)?;
            }

            Ok((length as isize, distance as usize))
        }
        0xE0..=0xFF => {
            // Long Frame
            let mut buf = [0u8; 3];
            input.read_exact(&mut buf)?;

            sign = u1::extract_u8(buf[0], 4);
            length = u4::extract_u8(buf[0], 0).value() as i16 + 3;
            distance = u16::from_le_bytes([buf[1], buf[2]]);

            if sign.value() == 1 {
                length = -length;
            }

            if length.abs() == 18 {
                length = decode_extension(input, length)?;
            }

            Ok((length as isize, distance as usize + 1))
        }
    }
}

#[allow(clippy::cast_lossless)]
fn decode_extension(input: &mut impl BufRead, mut length: i16) -> Result<i16> {
    loop {
        let value = input.fill_buf()?[0] as i16;
        input.consume(1);

        if length >= 0 {
            length = length.saturating_add(value);
        } else {
            length = length.saturating_sub(value);
        }

        if value < 0xFF {
            break;
        }
    }

    Ok(length)
}
