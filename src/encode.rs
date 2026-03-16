//! LZR compression encoder.
//!
//! # Examples
//!
//! ```
//! use lzr::encode::encode;
//! use lzr::decode::decode;
//! use lzr::options::EncodeOptions;
//!
//! let input = b"Hello, world!";
//! let mut compressed = Vec::new();
//! encode(&mut &input[..], &mut compressed, &EncodeOptions::new()).unwrap();
//!
//! let mut decompressed = Vec::new();
//! decode(&mut compressed.as_slice(), &mut decompressed).unwrap();
//! assert_eq!(decompressed, input);
//! ```

use std::io::{Read, Write};

use crate::HEADER;
use crate::adler32::Adler32;
use crate::error::Result;
use crate::frame::Frame;
use crate::matchfinder::MatchFinder;
use crate::nibble::NibbleWriter;
use crate::options::EncodeOptions;
use crate::ringbuf::RingBuf;

/// Maximum literal chunk size per frame.
const CHUNK_SIZE: usize = 4096;

/// Window size (must match format spec: 64 KiB).
const WINDOW_SIZE: usize = 65_536;

/// How many bytes to read into the ring buffer at a time.
const FILL_SIZE: usize = 32_768;

/// Flushes the literal buffer as a literal frame.
fn flush_literals<W: Write>(lit_buf: &mut Vec<u8>, writer: &mut NibbleWriter<W>) -> Result<()> {
    if lit_buf.is_empty() {
        return Ok(());
    }
    for chunk in lit_buf.chunks(CHUNK_SIZE) {
        Frame::encode_literal(chunk, writer)?;
    }
    lit_buf.clear();
    Ok(())
}

/// Compresses data from `input` and writes it to `output`.
///
/// Uses an LZ77 hash-chain match finder to find back-references in the sliding
/// window. The compression level (from `options`) controls chain depth, minimum
/// match length, and lazy matching behavior.
///
/// # Errors
///
/// Returns an error if an I/O error occurs while reading or writing.
///
/// # Examples
///
/// ```
/// use lzr::encode::encode;
/// use lzr::decode::decode;
/// use lzr::options::EncodeOptions;
///
/// let input = b"Hello, world!";
/// let mut compressed = Vec::new();
/// encode(&mut &input[..], &mut compressed, &EncodeOptions::new()).unwrap();
///
/// let mut decompressed = Vec::new();
/// decode(&mut compressed.as_slice(), &mut decompressed).unwrap();
/// assert_eq!(decompressed, input);
/// ```
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::too_many_lines)]
pub fn encode(input: &mut impl Read, output: &mut impl Write, options: &EncodeOptions) -> Result<()> {
    // Header.
    output.write_all(&HEADER)?;

    let mut writer = NibbleWriter::new(&mut *output);
    let mut adler = Adler32::new();
    let mut total_len: u64 = 0;

    let mut window = RingBuf::new(WINDOW_SIZE);
    let mut mf = MatchFinder::new(options);
    let mut lit_buf: Vec<u8> = Vec::with_capacity(CHUNK_SIZE);

    // `loaded` tracks the write_pos boundary of data loaded into the window.
    // `pos` is the current encoding position.
    let mut pos: isize = 0;
    let mut loaded: isize = 0;
    let mut eof = false;

    loop {
        // Ensure we have data ahead of pos.
        while !eof && loaded - pos < FILL_SIZE as isize {
            let n = window.fill_from_reader(input, FILL_SIZE)?;
            if n == 0 {
                eof = true;
                break;
            }
            loaded += n as isize;
        }

        if pos >= loaded {
            break;
        }

        let remaining = (loaded - pos) as usize;

        mf.insert(&window, pos);

        if let Some((dist, len)) = mf.find_best_match(&window, pos, remaining, loaded) {
            let abs_len = len.unsigned_abs() as isize;

            // Lazy matching: check if pos+1 yields a better match.
            let lazy_thresh = mf.lazy_threshold();
            if lazy_thresh > 0 && abs_len < 128 && remaining > 1 {
                mf.insert(&window, pos + 1);
                let remaining_next = remaining - 1;
                if let Some((dist2, len2)) = mf.find_best_match(&window, pos + 1, remaining_next, loaded) {
                    let gain1 = Frame::match_gain(dist, len);
                    let gain2 = Frame::match_gain(dist2, len2);
                    if gain2 > gain1 + lazy_thresh {
                        // Emit current byte as literal, use the pos+1 match instead.
                        let byte = window[pos];
                        adler.update(&[byte]);
                        total_len += 1;
                        lit_buf.push(byte);
                        if lit_buf.len() >= CHUNK_SIZE {
                            flush_literals(&mut lit_buf, &mut writer)?;
                        }
                        pos += 1;

                        // Emit the better match at pos (which is now the old pos+1).
                        flush_literals(&mut lit_buf, &mut writer)?;
                        let abs_len2 = len2.unsigned_abs() as isize;
                        Frame::Match {
                            distance: dist2,
                            length: len2,
                        }
                        .encode(&mut writer)?;

                        // Update adler with matched bytes.
                        let (s1, s2) = window.slices(pos..pos + abs_len2);
                        adler.update(s1);
                        adler.update(s2);
                        total_len += abs_len2 as u64;

                        // Bulk insert skipped positions (pos+1 through pos+abs_len2-1).
                        mf.bulk_insert(&window, pos + 1, pos + abs_len2);
                        pos += abs_len2;
                        continue;
                    }
                }
            }

            // Emit the match.
            flush_literals(&mut lit_buf, &mut writer)?;
            Frame::Match {
                distance: dist,
                length: len,
            }
            .encode(&mut writer)?;

            // Update adler with matched bytes.
            let (s1, s2) = window.slices(pos..pos + abs_len);
            adler.update(s1);
            adler.update(s2);
            total_len += abs_len as u64;

            // Bulk insert skipped positions.
            mf.bulk_insert(&window, pos + 1, pos + abs_len);
            pos += abs_len;
        } else {
            // No profitable match — accumulate literal.
            let byte = window[pos];
            adler.update(&[byte]);
            total_len += 1;
            lit_buf.push(byte);
            if lit_buf.len() >= CHUNK_SIZE {
                flush_literals(&mut lit_buf, &mut writer)?;
            }
            pos += 1;
        }
    }

    // Flush remaining literals.
    flush_literals(&mut lit_buf, &mut writer)?;

    // Pad to byte boundary if needed, then emit EOS.
    if writer.has_pending() {
        Frame::noop().encode(&mut writer)?;
    }
    Frame::EndOfStream.encode(&mut writer)?;
    let output = writer.finish()?;

    // Footer: UBLEB8 u64 length + 4-byte LE Adler-32.
    let mut footer_writer = NibbleWriter::new(&mut *output);
    footer_writer.write_ubleb8_u64(total_len)?;
    let output = footer_writer.finish()?;
    output.write_all(&adler.checksum().to_le_bytes())?;

    Ok(())
}
