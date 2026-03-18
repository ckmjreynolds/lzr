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
use crate::frame::{Frame, encode_uleb128_u64};
use crate::matchfinder::MatchFinder;
use crate::options::EncodeOptions;
use crate::ringbuf::RingBuf;
use crate::{HEADER, WINDOW_SIZE};

/// Maximum literal buffer size before flushing.
const CHUNK_SIZE: usize = 4096;

/// How many bytes to read into the ring buffer at a time.
const FILL_SIZE: usize = 32_768;

/// Flushes the literal buffer as a single literal frame and updates the Adler-32 checksum.
fn flush_literals<W: Write>(
    lit_buf: &mut Vec<u8>,
    output: &mut W,
    adler: &mut Adler32,
    total_len: &mut u64,
    frame_count: &mut u64,
) -> Result<()> {
    if lit_buf.is_empty() {
        return Ok(());
    }
    adler.update(lit_buf);
    *total_len += lit_buf.len() as u64;
    Frame::encode_literal(lit_buf, output)?;
    *frame_count += 1;
    lit_buf.clear();
    Ok(())
}

/// Compresses data from `input` and writes it to `output`.
///
/// Uses a brute-force match finder to find back-references in the sliding
/// window. The compression level (from `options`) controls lazy matching
/// behavior and whether reverse matches are attempted.
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
    unused_assignments
)]
pub fn encode(input: &mut impl Read, output: &mut impl Write, options: &EncodeOptions) -> Result<()> {
    // Header.
    output.write_all(&HEADER)?;

    let mut adler = Adler32::new();
    let mut total_len: u64 = 0;

    let mut window = RingBuf::new(WINDOW_SIZE);
    let mf = MatchFinder::new(options);
    let mut lit_buf: Vec<u8> = Vec::with_capacity(CHUNK_SIZE);

    let mut pos: isize = 0;
    let mut batch_end: isize = 0;
    let mut eof = false;

    // Diagnostics.
    let mut stat_literal_bytes: u64 = 0;
    let mut stat_short_literal_frames: u64 = 0;
    let mut stat_short_rle_frames: u64 = 0;
    let mut stat_medium_frames: u64 = 0;
    let mut stat_long_frames: u64 = 0;
    let mut stat_fwd_matches: u64 = 0;
    let mut stat_rev_matches: u64 = 0;
    let mut stat_match_bytes: f64 = 0.0;
    let mut stat_lazy_wins: u64 = 0;
    // Cross-tabulation: distance bins × length bins.
    // Distance: [1, 2-256, 257-1024, 1025-2048, 2049-4096, 4097-16384, 16385+]
    // Length:   [1-5, 6-15,  16-17,     18-270,    271+]
    #[allow(clippy::items_after_statements)]
    const D_BINS: usize = 7;
    #[allow(clippy::items_after_statements)]
    const L_BINS: usize = 5;
    let mut cross_tab: [[u64; L_BINS]; D_BINS] = [[0; L_BINS]; D_BINS];

    /// Record a match in the stats.
    macro_rules! record_match {
        ($dist:expr, $len:expr) => {{
            let d: u32 = $dist;
            let l: i32 = $len;
            let abs = l.unsigned_abs();
            let is_fwd = l > 0;
            stat_match_bytes += f64::from(abs);
            if is_fwd {
                stat_fwd_matches += 1;
            } else {
                stat_rev_matches += 1;
            }
            match Frame::match_type(d, abs, is_fwd) {
                0 => stat_short_rle_frames += 1,
                1 => stat_medium_frames += 1,
                _ => stat_long_frames += 1,
            }
            let dbin = match d {
                1 => 0,
                2..=256 => 1,
                257..=1024 => 2,
                1025..=2048 => 3,
                2049..=4096 => 4,
                4097..=16384 => 5,
                _ => 6,
            };
            let lbin = match abs {
                1..=5 => 0,
                6..=15 => 1,
                16..=17 => 2,
                18..=270 => 3,
                _ => 4,
            };
            cross_tab[dbin][lbin] += 1;
        }};
    }

    loop {
        // Fill the ring buffer when we've consumed the current batch.
        if pos >= batch_end {
            if eof {
                break;
            }
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
        }

        let remaining = (batch_end - pos) as usize;

        if let Some((dist, len)) = mf.find_best_match(&window, pos, remaining) {
            let abs_len = len.unsigned_abs() as isize;

            // Lazy matching: check if pos+1 yields a better match.
            let lazy_thresh = mf.lazy_threshold();
            if lazy_thresh > 0 && abs_len < 32 && remaining > 1 {
                let remaining_next = remaining - 1;
                if let Some((dist2, len2)) = mf.find_best_match(&window, pos + 1, remaining_next) {
                    let gain1 = Frame::match_gain(dist, len);
                    let gain2 = Frame::match_gain(dist2, len2);
                    if gain2 > gain1 + lazy_thresh {
                        // Emit current byte as literal, use the pos+1 match instead.
                        lit_buf.push(window[pos]);
                        stat_literal_bytes += 1;
                        if lit_buf.len() >= CHUNK_SIZE {
                            flush_literals(
                                &mut lit_buf,
                                output,
                                &mut adler,
                                &mut total_len,
                                &mut stat_short_literal_frames,
                            )?;
                        }
                        pos += 1;

                        flush_literals(
                            &mut lit_buf,
                            output,
                            &mut adler,
                            &mut total_len,
                            &mut stat_short_literal_frames,
                        )?;

                        let abs_len2 = len2.unsigned_abs() as isize;
                        Frame::Match {
                            distance: dist2,
                            length: len2,
                        }
                        .encode(output)?;

                        record_match!(dist2, len2);
                        stat_lazy_wins += 1;

                        let (s1, s2) = window.slices(pos..pos + abs_len2);
                        adler.update(s1);
                        adler.update(s2);
                        total_len += abs_len2 as u64;

                        pos += abs_len2;
                        continue;
                    }
                }
            }

            // Emit the match.
            flush_literals(&mut lit_buf, output, &mut adler, &mut total_len, &mut stat_short_literal_frames)?;

            Frame::Match {
                distance: dist,
                length: len,
            }
            .encode(output)?;

            record_match!(dist, len);

            let (s1, s2) = window.slices(pos..pos + abs_len);
            adler.update(s1);
            adler.update(s2);
            total_len += abs_len as u64;

            pos += abs_len;
        } else {
            // No profitable match — accumulate literal.
            lit_buf.push(window[pos]);
            stat_literal_bytes += 1;
            if lit_buf.len() >= CHUNK_SIZE {
                flush_literals(&mut lit_buf, output, &mut adler, &mut total_len, &mut stat_short_literal_frames)?;
            }
            pos += 1;
        }
    }

    // Flush remaining literals.
    flush_literals(&mut lit_buf, output, &mut adler, &mut total_len, &mut stat_short_literal_frames)?;

    // End of stream.
    Frame::EndOfStream.encode(output)?;

    // Footer: ULEB128 u64 length + 4-byte LE Adler-32.
    encode_uleb128_u64(total_len, output)?;
    output.write_all(&adler.checksum().to_le_bytes())?;

    // ── Diagnostics (verbose only) ──────────────────────────────────────
    if options.get_verbose() {
        let total_matches: u64 = cross_tab.iter().flat_map(|r| r.iter()).sum();

        eprintln!();
        eprintln!("── summary ─────────────────────────────────────────────────");
        eprintln!("input:       {total_len:>10} bytes");
        eprintln!(
            "literals:    {:>10} bytes  ({stat_short_literal_frames} frames, {:.1}% of input)",
            stat_literal_bytes,
            stat_literal_bytes as f64 / total_len.max(1) as f64 * 100.0
        );
        eprintln!(
            "matches:     {total_matches:>10}       (fwd={stat_fwd_matches}, rev={stat_rev_matches}, lazy={stat_lazy_wins})"
        );
        eprintln!("avg |len|:   {:>10.1}", stat_match_bytes / total_matches.max(1) as f64);
        eprintln!(
            "frames:      short-lit={stat_short_literal_frames} rle={stat_short_rle_frames} medium={stat_medium_frames} long={stat_long_frames}"
        );
        eprintln!();

        // ── Cross-tabulation ─────────────────────────────────────────────────
        let dist_labels = ["d=1     ", "d≤256   ", "d≤1024  ", "d≤2048  ", "d≤4096  ", "d≤16384 ", "d>16384 "];
        let len_labels = ["|L|≤5", "|L|≤15", "|L|≤17", "|L|≤270", "|L|>270"];

        eprintln!("── distance × length cross-tab ─────────────────────────────");
        eprint!("            ");
        for ll in &len_labels {
            eprint!("{ll:>8}");
        }
        eprint!("    total");
        eprintln!();

        for (di, dl) in dist_labels.iter().enumerate() {
            eprint!("  {dl}");
            let row_total: u64 = cross_tab[di].iter().sum();
            for &cell in &cross_tab[di] {
                eprint!("{cell:>8}");
            }
            let pct = row_total as f64 / total_matches.max(1) as f64 * 100.0;
            eprintln!("  {row_total:>7} ({pct:>5.1}%)");
        }
        eprintln!();
    }

    Ok(())
}
