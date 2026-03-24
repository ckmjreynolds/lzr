//! LZR decompression decoder.
//!
//! # Examples
//!
//! ```ignore
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

// use crate::adler32::Adler32;
// use crate::buffer::Buffer;
use crate::error::Result;
// use crate::{HEADER, WINDOW_SIZE};

// const FLUSH_THRESHOLD: usize = i16::MAX as usize;

// /// Compressed-input ring buffer size (must be power of two, ≥ 2 × max frame lookahead).
// const INPUT_BUF_SIZE: usize = 1 << 16; // 64 KiB

// /// Refill when fewer than this many bytes remain in the input buffer.
// /// Must cover the longest possible frame: 3-byte Long + 128 extension bytes +
// /// 32 KiB literal payload.
// const REFILL_THRESHOLD: usize = INPUT_BUF_SIZE / 2;

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
/// ```ignore
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
pub fn decode(_input: &mut impl Read, _output: &mut impl Write) -> Result<()> {
    todo!()
    // let mut inp = InputBuf::new();

    // while inp.refill(input)? > 0 {
    //     let mut window = Buffer::<WINDOW_SIZE>::new();
    //     let mut adler = Adler32::new();
    //     let mut pos = 0usize;
    //     let mut pending = 0usize;

    //     if inp.read_array::<4>() != HEADER {
    //         return Err(Error::InvalidMagic);
    //     }

    //     loop {
    //         if inp.available() < REFILL_THRESHOLD {
    //             inp.refill(input)?;
    //         }

    //         let len = decode_frame(&mut inp, &mut window, pos);
    //         if len == 0 {
    //             break; // EOS
    //         }

    //         pos += len;
    //         pending += len;

    //         if pending >= FLUSH_THRESHOLD {
    //             flush(&window, output, &mut adler, pos - pending, pending)?;
    //             pending = 0;
    //         }
    //     }

    //     if pending > 0 {
    //         flush(&window, output, &mut adler, pos - pending, pending)?;
    //     }

    //     if inp.available() < REFILL_THRESHOLD {
    //         inp.refill(input)?;
    //     }

    //     let expected_len = read_uleb128(&mut inp);
    //     if pos as u64 != expected_len {
    //         return Err(Error::LengthMismatch {
    //             expected: expected_len,
    //             actual: pos as u64,
    //         });
    //     }

    //     let expected_checksum = u32::from_le_bytes(inp.read_array::<4>());
    //     let actual_checksum = adler.checksum();
    //     if actual_checksum != expected_checksum {
    //         return Err(Error::ChecksumMismatch {
    //             expected: expected_checksum,
    //             actual: actual_checksum,
    //         });
    //     }
    // }
    // Ok(())
}

// /// Decodes one frame, expanding it into the window. Returns the number of
// /// output bytes produced (0 = EOS).
// #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
// fn decode_frame(inp: &mut InputBuf, window: &mut Buffer<WINDOW_SIZE>, pos: usize) -> usize {
//     let b0 = inp.read_u8();

//     if b0 <= 0xBF {
//         // Medium Frame
//         let b1 = inp.read_u8();
//         let frame = u16::from_be_bytes([b0, b1]);
//         let sign = frame >> 15;
//         let mut mag = ((frame >> 11) & 0xF) as i16 + 2;
//         let distance = (frame & 0x7FF) as usize + 1;

//         if sign == 1 {
//             mag = -mag;
//         }
//         if mag == -9 || mag == 17 {
//             mag = read_extension(inp, mag);
//         }

//         let len = mag.unsigned_abs() as usize;
//         if mag > 0 {
//             window.copy_fwd(pos.wrapping_sub(distance), len, pos);
//         } else {
//             window.copy_rev(pos.wrapping_sub(distance), len, pos);
//         }
//         len
//     } else if b0 <= 0xDF {
//         // Short Frame
//         let l = i16::from((b0 >> 1) & 0xF);
//         let d = b0 & 1;

//         if l == 0 && d == 0 {
//             return 0; // EOS
//         }

//         let mut len = l;
//         if len == 15 {
//             len = read_extension(inp, len);
//         }
//         let len = len as usize;

//         if d == 0 {
//             for i in 0..len {
//                 window[pos + i] = inp.read_u8();
//             }
//         } else {
//             window.copy_fwd(pos.wrapping_sub(1), len, pos);
//         }
//         len
//     } else {
//         // Long Frame
//         let b1 = inp.read_u8();
//         let b2 = inp.read_u8();
//         let sign = (b0 >> 4) & 1;
//         let mut mag = i16::from(b0 & 0xF) + 3;
//         let distance = u16::from_le_bytes([b1, b2]) as usize + 1;

//         if sign == 1 {
//             mag = -mag;
//         }
//         if mag.abs() == 18 {
//             mag = read_extension(inp, mag);
//         }

//         let len = mag.unsigned_abs() as usize;
//         if mag > 0 {
//             window.copy_fwd(pos.wrapping_sub(distance), len, pos);
//         } else {
//             window.copy_rev(pos.wrapping_sub(distance), len, pos);
//         }
//         len
//     }
// }

// // ── Input buffer ────────────────────────────────────────────────────────

// /// A flat ring buffer for compressed input. Frame parsing reads directly from
// /// the buffer via index, avoiding per-frame `fill_buf` / `read_exact` overhead.
// struct InputBuf {
//     buf: Vec<u8>,
//     pos: usize,
//     len: usize,
// }

// impl InputBuf {
//     fn new() -> Self {
//         Self {
//             buf: vec![0u8; INPUT_BUF_SIZE],
//             pos: 0,
//             len: 0,
//         }
//     }

//     /// How many unread bytes remain in the buffer.
//     const fn available(&self) -> usize {
//         self.len - self.pos
//     }

//     /// Refills the buffer from `reader`. Shifts unconsumed data to the front
//     /// first. Returns total available bytes after refill.
//     fn refill(&mut self, reader: &mut impl Read) -> Result<usize> {
//         let remaining = self.available();
//         if remaining > 0 {
//             self.buf.copy_within(self.pos..self.len, 0);
//         }
//         self.pos = 0;
//         self.len = remaining;

//         while self.len < INPUT_BUF_SIZE {
//             let n = reader.read(&mut self.buf[self.len..])?;
//             if n == 0 {
//                 break;
//             }
//             self.len += n;
//         }

//         Ok(self.available())
//     }

//     /// Reads one byte, advancing the position.
//     #[inline]
//     fn read_u8(&mut self) -> u8 {
//         let b = self.buf[self.pos];
//         self.pos += 1;
//         b
//     }

//     /// Reads a fixed-size array.
//     fn read_array<const N: usize>(&mut self) -> [u8; N] {
//         let mut arr = [0u8; N];
//         arr.copy_from_slice(&self.buf[self.pos..self.pos + N]);
//         self.pos += N;
//         arr
//     }
// }

// // ── Helpers ─────────────────────────────────────────────────────────────

// /// Reads a length extension chain from the input buffer.
// fn read_extension(inp: &mut InputBuf, mut length: i16) -> i16 {
//     loop {
//         let value = i16::from(inp.read_u8());
//         if length >= 0 {
//             length = length.saturating_add(value);
//         } else {
//             length = length.saturating_sub(value);
//         }
//         if value < 0xFF {
//             break;
//         }
//     }
//     length
// }

// /// Reads a ULEB128-encoded u64 from the input buffer (max 9 bytes).
// fn read_uleb128(inp: &mut InputBuf) -> u64 {
//     let mut result: u64 = 0;
//     for i in 0..9 {
//         let byte = inp.read_u8();
//         if i < 8 {
//             result |= u64::from(byte & 0x7F) << (i * 7);
//             if byte & 0x80 == 0 {
//                 return result;
//             }
//         } else {
//             result |= u64::from(byte) << 56;
//             return result;
//         }
//     }
//     result
// }

// /// Flushes pending decoded bytes to output and updates the Adler-32 checksum.
// fn flush(
//     window: &Buffer<WINDOW_SIZE>,
//     output: &mut impl Write,
//     adler: &mut Adler32,
//     start: usize,
//     len: usize,
// ) -> Result<()> {
//     let slices = window.slices(start, len);
//     output.write_all(slices.0)?;
//     adler.update(slices.0);
//     if !slices.1.is_empty() {
//         output.write_all(slices.1)?;
//         adler.update(slices.1);
//     }
//     Ok(())
// }
