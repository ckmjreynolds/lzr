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

use crate::adler32::Adler32;
use crate::error::Result;
use crate::frame::Frame;
use crate::matchfinder::HashMapMatchFinder;
use crate::nibble::{NibbleWriter, sbleb8_i16_nibbles, ubleb8_u16_nibbles};
use crate::options::EncodeOptions;
use crate::ringbuf::RingBuf;
use crate::{HEADER, WINDOW_SIZE};

/// Maximum literal chunk size per frame.
const CHUNK_SIZE: usize = 4096;

/// How many bytes to read into the ring buffer at a time.
const FILL_SIZE: usize = 32_768;

/// Flushes the literal buffer as literal frames and updates the Adler-32 checksum.
fn flush_literals<W: Write>(
    lit_buf: &mut Vec<u8>,
    writer: &mut NibbleWriter<W>,
    adler: &mut Adler32,
    total_len: &mut u64,
) -> Result<()> {
    if lit_buf.is_empty() {
        return Ok(());
    }
    adler.update(lit_buf);
    *total_len += lit_buf.len() as u64;
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
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::manual_div_ceil,
    clippy::uninlined_format_args
)]
pub fn encode(input: &mut impl Read, output: &mut impl Write, options: &EncodeOptions) -> Result<()> {
    // Header.
    output.write_all(&HEADER)?;

    let mut writer = NibbleWriter::new(&mut *output);
    let mut adler = Adler32::new();
    let mut total_len: u64 = 0;

    let mut window = RingBuf::new(WINDOW_SIZE);
    let mut mf = HashMapMatchFinder::new(options);
    let mut lit_buf: Vec<u8> = Vec::with_capacity(CHUNK_SIZE);

    // `pos` is the current encoding position.
    // `batch_end` marks the end of the current indexed batch.
    let mut pos: isize = 0;
    let mut batch_end: isize = 0;
    let mut eof = false;

    // Diagnostics.
    let mut stat_literals: u64 = 0;
    let mut stat_matches: u64 = 0;
    let mut stat_match_bytes: u64 = 0;
    let mut stat_match_len_hist: [u64; 8] = [0; 8]; // 1-2, 3, 4-7, 8-15, 16-31, 32-63, 64-127, 128+
    let mut stat_match_gain_total: i64 = 0;
    let mut stat_lit_frame_overhead_nibbles: u64 = 0;
    let mut stat_max_fwd_match: i16 = 0;
    let mut stat_max_rev_match: i16 = 0;
    let mut stat_dist_nibbles_total: u64 = 0;
    let mut stat_len_nibbles_total: u64 = 0;
    let mut stat_dist_hist: [u64; 5] = [0; 5]; // 1-7, 8-63, 64-511, 512-4095, 4096+
    let mut stat_lit_runs: u64 = 0;
    let mut stat_repeat_dist_count: u64 = 0;
    let mut stat_last_dist: u16 = 0;

    loop {
        // When we've consumed the current batch, load and index a new one.
        if pos >= batch_end {
            if eof {
                break;
            }
            // Fill the ring buffer.
            while !eof && window.write_pos() - pos < FILL_SIZE as isize {
                let n = window.fill_from_reader(input, FILL_SIZE)?;
                if n == 0 {
                    eof = true;
                }
            }
            batch_end = window.write_pos();
            if pos >= batch_end {
                break;
            }
            // Build index for all positions that are live in the window.
            // The earliest valid position is write_pos - WINDOW_SIZE.
            let index_start = (batch_end - WINDOW_SIZE as isize).max(0);
            mf.build(&window, index_start, batch_end);
        }

        let remaining = (batch_end - pos) as usize;

        if let Some((dist, len)) = mf.find_best_match(&window, pos, remaining) {
            let abs_len = len.unsigned_abs() as isize;

            // Lazy matching: check if pos+1 yields a better match.
            // Skip for long matches — the gain from a slightly better match is negligible.
            let lazy_thresh = mf.lazy_threshold();
            if lazy_thresh > 0 && abs_len < 32 && remaining > 1 {
                mf.insert(&window, pos + 1);
                let remaining_next = remaining - 1;
                if let Some((dist2, len2)) = mf.find_best_match(&window, pos + 1, remaining_next) {
                    let gain1 = Frame::match_gain(dist, len);
                    let gain2 = Frame::match_gain(dist2, len2);
                    if gain2 > gain1 + lazy_thresh {
                        // Emit current byte as literal, use the pos+1 match instead.
                        lit_buf.push(window[pos]);
                        stat_literals += 1;
                        if lit_buf.len() >= CHUNK_SIZE {
                            stat_lit_frame_overhead_nibbles += 1 + ubleb8_u16_nibbles(CHUNK_SIZE as u16) as u64;
                            flush_literals(&mut lit_buf, &mut writer, &mut adler, &mut total_len)?;
                        }
                        pos += 1;

                        // Emit the better match at pos (which is now the old pos+1).
                        if !lit_buf.is_empty() {
                            stat_lit_frame_overhead_nibbles += 1 + ubleb8_u16_nibbles(lit_buf.len() as u16) as u64;
                        }
                        flush_literals(&mut lit_buf, &mut writer, &mut adler, &mut total_len)?;
                        let abs_len2 = len2.unsigned_abs() as isize;

                        stat_matches += 1;
                        stat_match_bytes += abs_len2 as u64;
                        stat_match_gain_total += Frame::match_gain(dist2, len2) as i64;
                        if len2 > 0 { stat_max_fwd_match = stat_max_fwd_match.max(len2); }
                        else { stat_max_rev_match = stat_max_rev_match.min(len2); }
                        stat_dist_nibbles_total += ubleb8_u16_nibbles(dist2) as u64;
                        stat_len_nibbles_total += sbleb8_i16_nibbles(len2) as u64;
                        let dbucket = match dist2 {
                            1..=7 => 0,
                            8..=63 => 1,
                            64..=511 => 2,
                            512..=4095 => 3,
                            _ => 4,
                        };
                        stat_dist_hist[dbucket] += 1;
                        if dist2 == stat_last_dist {
                            stat_repeat_dist_count += 1;
                        }
                        stat_last_dist = dist2;
                        if !lit_buf.is_empty() {
                            stat_lit_runs += 1;
                        }
                        let bucket = match abs_len2 as usize {
                            0..=2 => 0,
                            3 => 1,
                            4..=7 => 2,
                            8..=15 => 3,
                            16..=31 => 4,
                            32..=63 => 5,
                            64..=127 => 6,
                            _ => 7,
                        };
                        stat_match_len_hist[bucket] += 1;

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
            if !lit_buf.is_empty() {
                stat_lit_frame_overhead_nibbles += 1 + ubleb8_u16_nibbles(lit_buf.len() as u16) as u64;
            }
            flush_literals(&mut lit_buf, &mut writer, &mut adler, &mut total_len)?;

            stat_matches += 1;
            stat_match_bytes += abs_len as u64;
            stat_match_gain_total += Frame::match_gain(dist, len) as i64;
            if len > 0 { stat_max_fwd_match = stat_max_fwd_match.max(len); }
            else { stat_max_rev_match = stat_max_rev_match.min(len); }
            stat_dist_nibbles_total += ubleb8_u16_nibbles(dist) as u64;
            stat_len_nibbles_total += sbleb8_i16_nibbles(len) as u64;
            let dbucket = match dist {
                1..=7 => 0,
                8..=63 => 1,
                64..=511 => 2,
                512..=4095 => 3,
                _ => 4,
            };
            stat_dist_hist[dbucket] += 1;
            if dist == stat_last_dist {
                stat_repeat_dist_count += 1;
            }
            stat_last_dist = dist;
            if !lit_buf.is_empty() {
                stat_lit_runs += 1;
            }
            let bucket = match abs_len as usize {
                0..=2 => 0,
                3 => 1,
                4..=7 => 2,
                8..=15 => 3,
                16..=31 => 4,
                32..=63 => 5,
                64..=127 => 6,
                _ => 7,
            };
            stat_match_len_hist[bucket] += 1;

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
            stat_literals += 1;
            lit_buf.push(window[pos]);
            if lit_buf.len() >= CHUNK_SIZE {
                stat_lit_frame_overhead_nibbles += 1 + ubleb8_u16_nibbles(CHUNK_SIZE as u16) as u64;
                flush_literals(&mut lit_buf, &mut writer, &mut adler, &mut total_len)?;
            }
            pos += 1;
        }
    }

    // Flush remaining literals.
    flush_literals(&mut lit_buf, &mut writer, &mut adler, &mut total_len)?;

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

    // Log peak SmallVec lengths for tuning.
    let (max_fwd, max_rev) = mf.max_positions_used();
    eprintln!("lzr: max SmallVec lengths: forward={max_fwd}, reverse={max_rev}");

    // Remaining literal frame overhead.
    if !lit_buf.is_empty() {
        stat_lit_frame_overhead_nibbles += 1 + ubleb8_u16_nibbles(lit_buf.len() as u16) as u64;
    }

    let matched_pct = stat_match_bytes as f64 / total_len as f64 * 100.0;
    let literal_pct = stat_literals as f64 / total_len as f64 * 100.0;
    let lit_overhead_bytes = (stat_lit_frame_overhead_nibbles + 1) / 2;
    eprintln!(
        "lzr: {total_len} bytes: {stat_match_bytes} matched ({matched_pct:.1}%), {stat_literals} literal ({literal_pct:.1}%)"
    );
    eprintln!(
        "lzr: {stat_matches} matches, avg len {:.1}, total gain {stat_match_gain_total} nibbles ({:.0} bytes)",
        stat_match_bytes as f64 / stat_matches.max(1) as f64,
        stat_match_gain_total as f64 / 2.0
    );
    eprintln!(
        "lzr: match len histogram: 1-2={} 3={} 4-7={} 8-15={} 16-31={} 32-63={} 64-127={} 128+={}",
        stat_match_len_hist[0],
        stat_match_len_hist[1],
        stat_match_len_hist[2],
        stat_match_len_hist[3],
        stat_match_len_hist[4],
        stat_match_len_hist[5],
        stat_match_len_hist[6],
        stat_match_len_hist[7]
    );
    eprintln!(
        "lzr: literal frame overhead: {lit_overhead_bytes} bytes ({stat_lit_runs} runs, {stat_lit_frame_overhead_nibbles} nibbles)"
    );
    eprintln!(
        "lzr: distance nibbles: {stat_dist_nibbles_total} ({:.0} bytes), length nibbles: {stat_len_nibbles_total} ({:.0} bytes)",
        stat_dist_nibbles_total as f64 / 2.0,
        stat_len_nibbles_total as f64 / 2.0
    );
    eprintln!(
        "lzr: dist histogram: 1-7={} 8-63={} 64-511={} 512-4095={} 4096+={}",
        stat_dist_hist[0], stat_dist_hist[1], stat_dist_hist[2], stat_dist_hist[3], stat_dist_hist[4]
    );
    eprintln!(
        "lzr: repeat-distance matches: {stat_repeat_dist_count} ({:.1}%)",
        stat_repeat_dist_count as f64 / stat_matches.max(1) as f64 * 100.0
    );
    eprintln!("lzr: max match lengths: forward={stat_max_fwd_match}, reverse={stat_max_rev_match}");

    Ok(())
}
