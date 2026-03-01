//! LZR stream encoder.

use std::io::{self, BufReader, BufWriter, Read, Write};

use rayon::prelude::*;

use crate::adler32::Adler32;
use crate::bleb8::{Sleb8, Ubleb8};
use crate::nibble::NibbleWriter;
use crate::{MAGIC, VERSION, WINDOW_SIZE};

/// Ring buffer size (1 MiB).
const RING_SIZE: usize = 1 << 20;

/// Bytes of new data per batch (ring minus one window of context).
#[cfg(test)]
const BATCH_SIZE: usize = RING_SIZE - WINDOW_SIZE;

/// Minimum match length that is always profitable.
const MIN_MATCH_LEN: usize = 3;

/// Maximum literal count per frame.
const MAX_LITERAL_LEN: usize = u16::MAX as usize;

/// Maximum forward match length.
const MAX_MATCH_FWD: usize = i16::MAX as usize;

/// Maximum reverse match length.
const MAX_MATCH_REV: usize = i16::MIN.unsigned_abs() as usize;

/// Encodes data from `input` as an LZR stream written to `output`.
///
/// # Errors
///
/// Returns an error on I/O failure.
pub fn encode(input: impl Read, output: impl Write) -> io::Result<()> {
    let mut enc = Encoder::new(input, output);
    enc.run()?;
    enc.writer.flush()
}

/// Internal encoder state.
struct Encoder<R: Read, W: Write> {
    reader: BufReader<R>,
    writer: BufWriter<W>,
    adler: Adler32,
    uncompressed_len: u64,
}

impl<R: Read, W: Write> Encoder<R, W> {
    fn new(input: R, output: W) -> Self {
        Self {
            reader: BufReader::new(input),
            writer: BufWriter::new(output),
            adler: Adler32::new(),
            uncompressed_len: 0,
        }
    }

    fn run(&mut self) -> io::Result<()> {
        self.write_header()?;

        let mut ring = vec![0u8; RING_SIZE];

        loop {
            let bytes_read = read_batch(&mut self.reader, &mut ring[WINDOW_SIZE..])?;
            if bytes_read == 0 {
                break;
            }

            self.adler.update(&ring[WINDOW_SIZE..WINDOW_SIZE + bytes_read]);
            self.uncompressed_len += bytes_read as u64;

            let compressed_jobs = compress_batch(&ring, bytes_read);
            for job_bytes in &compressed_jobs {
                self.writer.write_all(job_bytes)?;
            }

            ring.copy_within(bytes_read..bytes_read + WINDOW_SIZE, 0);
        }

        self.write_eos()?;
        self.write_footer()
    }

    fn write_header(&mut self) -> io::Result<()> {
        self.writer.write_all(&MAGIC)?;
        self.writer.write_all(&[VERSION])
    }

    fn write_eos(&mut self) -> io::Result<()> {
        let mut nw = NibbleWriter::new();
        emit_eos(&mut nw);
        self.writer.write_all(&nw.finish())
    }

    fn write_footer(&mut self) -> io::Result<()> {
        let mut nw = NibbleWriter::new();
        self.uncompressed_len.encode_ubleb8(&mut nw);
        // Pad to byte boundary.
        if !nw.len().is_multiple_of(2) {
            nw.push(0);
        }
        self.writer.write_all(&nw.finish())?;
        self.writer.write_all(&self.adler.finish().to_le_bytes())
    }
}

// ── Batch I/O ─────────────────────────────────────────────────────────

/// Fills `buf` with up to `buf.len()` bytes, looping until full or EOF.
fn read_batch(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        let n = reader.read(&mut buf[total..])?;
        if n == 0 {
            break;
        }
        total += n;
    }
    Ok(total)
}

// ── Parallel Compression ──────────────────────────────────────────────

/// Divides `bytes_read` bytes (starting at `ring[WINDOW_SIZE]`) into 64 KiB
/// jobs, compresses each in parallel, and returns the byte-aligned output.
fn compress_batch(ring: &[u8], bytes_read: usize) -> Vec<Vec<u8>> {
    let full_jobs = bytes_read / WINDOW_SIZE;
    let remainder = bytes_read % WINDOW_SIZE;
    let job_count = full_jobs + usize::from(remainder > 0);

    (0..job_count)
        .into_par_iter()
        .map(|i| {
            let data_len = if i < full_jobs {
                WINDOW_SIZE
            } else {
                remainder
            };
            let start = i * WINDOW_SIZE;
            let end = start + WINDOW_SIZE + data_len;
            compress_job(&ring[start..end], WINDOW_SIZE, data_len)
        })
        .collect()
}

/// Compresses one job: `buf[..window_size]` is context,
/// `buf[window_size..window_size + data_len]` is data to encode.
/// Returns byte-aligned packed nibbles.
fn compress_job(buf: &[u8], window_size: usize, data_len: usize) -> Vec<u8> {
    let mut nw = NibbleWriter::new();
    let data_end = window_size + data_len;
    let mut pos = window_size;
    let mut literal_start = pos;

    while pos < data_end {
        if let Some((distance, length, is_reverse)) = find_best_match(buf, pos, data_end) {
            // Flush pending literals.
            if literal_start < pos {
                emit_literals(&mut nw, &buf[literal_start..pos]);
            }
            emit_match(&mut nw, distance, length, is_reverse);
            pos += length;
            literal_start = pos;
        } else {
            pos += 1;
        }
    }

    // Flush remaining literals.
    if literal_start < data_end {
        emit_literals(&mut nw, &buf[literal_start..data_end]);
    }

    // Pad to byte alignment with a no-op if needed.
    if !nw.len().is_multiple_of(2) {
        emit_noop(&mut nw);
    }

    nw.finish()
}

// ── Match Finder ──────────────────────────────────────────────────────

/// Exhaustive (brute-force) match finder. Scans all valid distances and
/// returns the best match, if any meets `MIN_MATCH_LEN`.
#[allow(clippy::cast_possible_truncation)]
fn find_best_match(buf: &[u8], pos: usize, data_end: usize) -> Option<(u16, usize, bool)> {
    let max_dist = pos.min(usize::from(u16::MAX));
    let mut best_len = MIN_MATCH_LEN - 1;
    let mut best_dist: u16 = 0;
    let mut best_reverse = false;

    for d in 1..=max_dist {
        // Forward match.
        let max_fwd = (data_end - pos).min(MAX_MATCH_FWD);
        let mut fwd_len = 0;
        while fwd_len < max_fwd && buf[pos - d + fwd_len] == buf[pos + fwd_len] {
            fwd_len += 1;
        }
        if fwd_len > best_len {
            best_len = fwd_len;
            // d <= u16::MAX by max_dist bound.
            best_dist = d as u16;
            best_reverse = false;
        }

        // Reverse match.
        let max_rev = (data_end - pos).min(MAX_MATCH_REV).min(pos - d + 1);
        let mut rev_len = 0;
        while rev_len < max_rev && buf[pos - d - rev_len] == buf[pos + rev_len] {
            rev_len += 1;
        }
        if rev_len > best_len {
            best_len = rev_len;
            best_dist = d as u16;
            best_reverse = true;
        }
    }

    if best_len >= MIN_MATCH_LEN {
        Some((best_dist, best_len, best_reverse))
    } else {
        None
    }
}

// ── Frame Encoding ────────────────────────────────────────────────────

/// Emits one or more literal frames (chunking at 65 535 bytes).
#[allow(clippy::cast_possible_truncation)]
fn emit_literals(nw: &mut NibbleWriter, bytes: &[u8]) {
    for chunk in bytes.chunks(MAX_LITERAL_LEN) {
        // D = 0
        0u16.encode_ubleb8(nw);
        // L = count (UBLEB8); chunk.len() <= MAX_LITERAL_LEN = u16::MAX.
        (chunk.len() as u16).encode_ubleb8(nw);
        for &b in chunk {
            nw.push_byte(b);
        }
    }
}

/// Emits a match frame.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn emit_match(nw: &mut NibbleWriter, distance: u16, length: usize, is_reverse: bool) {
    distance.encode_ubleb8(nw);
    // length <= MAX_MATCH_FWD (i16::MAX) or MAX_MATCH_REV (i16::MIN.unsigned_abs()).
    // For reverse, -(length as i32) as i16 handles length=32768 → i16::MIN.
    let l: i16 = if is_reverse {
        (-(length as i32)) as i16
    } else {
        length as i16
    };
    l.encode_sleb8(nw);
}

/// Emits an end-of-stream marker (D=0, L=0).
fn emit_eos(nw: &mut NibbleWriter) {
    0u16.encode_ubleb8(nw);
    0u16.encode_ubleb8(nw);
}

/// Emits a 3-nibble no-op (D=8, L=0) for byte alignment padding.
fn emit_noop(nw: &mut NibbleWriter) {
    8u16.encode_ubleb8(nw); // 2 nibbles: [0x8, 0x1]
    0i16.encode_sleb8(nw); // 1 nibble: [0x0]
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;
    use crate::decoder;

    fn round_trip(data: &[u8]) -> Vec<u8> {
        let mut compressed = Vec::new();
        encode(data, &mut compressed).unwrap();
        let mut decompressed = Vec::new();
        decoder::decode(compressed.as_slice(), &mut decompressed).unwrap();
        decompressed
    }

    // ── Worked Example ──────────────────────────────────────────────

    #[test]
    fn worked_example_hi() {
        let mut compressed = Vec::new();
        encode(b"Hi".as_slice(), &mut compressed).unwrap();
        // FORMAT.md: 13-byte stream encoding "Hi"
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x4C, 0x5A, 0x52, 0x00, // header
            0x02, 0x84, 0x96, 0x00, // bitstream: lit(2) "Hi", EOS
            0x20,                   // footer length: UBLEB8(2)
            0xB2, 0x00, 0xFB, 0x00, // footer checksum: Adler-32("Hi")
        ];
        assert_eq!(compressed, expected);
    }

    // ── Empty Input ─────────────────────────────────────────────────

    #[test]
    fn empty_input() {
        let mut compressed = Vec::new();
        encode(b"".as_slice(), &mut compressed).unwrap();
        #[rustfmt::skip]
        let expected: &[u8] = &[
            0x4C, 0x5A, 0x52, 0x00, // header
            0x00,                   // bitstream: EOS (D=0, L=0)
            0x00,                   // footer length: UBLEB8(0)
            0x01, 0x00, 0x00, 0x00, // footer checksum: Adler-32("")
        ];
        assert_eq!(compressed, expected);
    }

    // ── Round-Trip: Repeated Bytes (RLE) ────────────────────────────

    #[test]
    fn repeated_bytes_rle() {
        let data = vec![b'a'; 1000];
        assert_eq!(round_trip(&data), data);
    }

    // ── Round-Trip: Reverse Match Detection ─────────────────────────

    #[test]
    fn reverse_match_detection() {
        let data = b"abcddcba";
        assert_eq!(round_trip(data), data.as_slice());
    }

    // ── Round-Trip: Random Data ─────────────────────────────────────

    proptest! {
        #[test]
        fn round_trip_random(data in prop::collection::vec(any::<u8>(), 0..4096)) {
            prop_assert_eq!(round_trip(&data), data);
        }
    }

    // ── Round-Trip: Multi-Batch ─────────────────────────────────────

    #[test]
    fn multi_batch() {
        // >983 KiB to exercise batch loop and context shift.
        let data: Vec<u8> = (0u8..=255).cycle().take(BATCH_SIZE + WINDOW_SIZE + 1000).collect();
        assert_eq!(round_trip(&data), data);
    }

    // ── Match Finder Direct ─────────────────────────────────────────

    #[test]
    fn match_finder_forward() {
        // buf = [a, b, c, a, b, c], window_size=3, data at [3..6]
        let buf = b"abcabc";
        let result = find_best_match(buf, 3, 6);
        assert!(result.is_some());
        let (dist, len, is_rev) = result.unwrap();
        assert_eq!(dist, 3);
        assert_eq!(len, 3);
        assert!(!is_rev);
    }

    #[test]
    fn match_finder_reverse() {
        // buf = [a, b, c, c, b, a], window_size=3, data at [3..6]
        let buf = b"abccba";
        let result = find_best_match(buf, 3, 6);
        assert!(result.is_some());
        let (dist, len, is_rev) = result.unwrap();
        assert_eq!(dist, 1);
        assert_eq!(len, 3);
        assert!(is_rev);
    }

    #[test]
    fn match_finder_no_match() {
        let buf = b"\x00\x00\x00\x01\x02\x03";
        let result = find_best_match(buf, 3, 6);
        assert!(result.is_none());
    }

    // ── Literal Chunking ────────────────────────────────────────────

    #[test]
    fn literal_chunking() {
        // Data larger than MAX_LITERAL_LEN should produce multiple literal frames.
        let data: Vec<u8> = (0u8..=255).cycle().take(MAX_LITERAL_LEN + 100).collect();
        let mut nw = NibbleWriter::new();
        emit_literals(&mut nw, &data);
        // Verify by encoding the full thing and round-tripping.
        let full_data: Vec<u8> = (0u8..=255).cycle().take(MAX_LITERAL_LEN + 100).collect();
        assert_eq!(round_trip(&full_data), full_data);
    }

    // ── No-op Alignment ─────────────────────────────────────────────

    #[test]
    fn noop_is_3_nibbles() {
        let mut nw = NibbleWriter::new();
        let before = nw.len();
        emit_noop(&mut nw);
        assert_eq!(nw.len() - before, 3);
    }
}
