//! LZR stream encoder.

use std::io::{self, BufReader, BufWriter, Read, Write};

use rayon::prelude::*;

use crate::adler32::Adler32;
use crate::bleb8::{self, Sleb8, Ubleb8};
use crate::nibble::NibbleWriter;
use crate::{MAGIC, VERSION, WINDOW_SIZE};

/// Ring buffer size (1 MiB).
const RING_SIZE: usize = 1 << 20;

/// Bytes of new data per batch (ring minus one window of context).
#[cfg(test)]
const BATCH_SIZE: usize = RING_SIZE - WINDOW_SIZE;

/// Maximum literal count per frame.
const MAX_LITERAL_LEN: usize = u16::MAX as usize;

/// Maximum forward match length.
const MAX_MATCH_FWD: usize = i16::MAX as usize;

/// Maximum reverse match length.
const MAX_MATCH_REV: usize = i16::MIN.unsigned_abs() as usize;

// ── Hash Chain Constants ─────────────────────────────────────────────

const HASH_BITS: usize = 15;
const HASH_TABLE_SIZE: usize = 1 << HASH_BITS;
const MAX_CHAIN_DEPTH: usize = 256;
const NIL: u32 = u32::MAX;

/// Maximum distance where a short (1-2 byte) match can break even.
const SHORT_MATCH_RANGE: usize = 511;

/// Minimum savings from hash chains to skip the short-range linear search.
const SHORT_SEARCH_THRESHOLD: i32 = 2;

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

    let active_end = WINDOW_SIZE + bytes_read;
    let chains = HashChains::build(ring, active_end);
    let ring_slice = &ring[..active_end];

    (0..job_count)
        .into_par_iter()
        .map(|i| {
            let data_len = if i < full_jobs {
                WINDOW_SIZE
            } else {
                remainder
            };
            let data_start = WINDOW_SIZE + i * WINDOW_SIZE;
            let data_end = data_start + data_len;
            compress_job(ring_slice, &chains, data_start, data_end)
        })
        .collect()
}

/// Compresses one job: `buf` is the full active ring, data to encode spans
/// `buf[data_start..data_end]`. Returns byte-aligned packed nibbles.
fn compress_job(buf: &[u8], chains: &HashChains, data_start: usize, data_end: usize) -> Vec<u8> {
    let mut nw = NibbleWriter::new();
    let mut pos = data_start;
    let mut literal_start = pos;

    while pos < data_end {
        if let Some((distance, length, is_reverse)) = find_best_match(buf, chains, pos, data_end) {
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

// ── Savings Function ──────────────────────────────────────────────────

/// Nibbles saved by encoding `length` bytes at `distance` as a match
/// instead of literals. Negative means the match costs more.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn nibbles_saved(distance: u16, length: i16) -> i32 {
    let literal_cost = 2 * i32::from(length.unsigned_abs());
    let match_cost = bleb8::ubleb8_len(distance) as i32 + bleb8::sleb8_len(length) as i32;
    literal_cost - match_cost
}

/// Encodes match length as the signed i16 used in the format.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
const fn encode_length(length: usize, is_reverse: bool) -> i16 {
    if is_reverse {
        (-(length as i32)) as i16
    } else {
        length as i16
    }
}

// ── Hash Chains ──────────────────────────────────────────────────────

/// Multiplicative hash of 3 bytes.
fn hash3(a: u8, b: u8, c: u8) -> usize {
    let h = u32::from(a) | (u32::from(b) << 8) | (u32::from(c) << 16);
    (h.wrapping_mul(0x1E35_A7BD) >> (32 - HASH_BITS)) as usize
}

/// Two hash chains (forward + reverse 3-byte hashes) for match finding.
struct HashChains {
    fwd_head: Vec<u32>,
    fwd_prev: Vec<u32>,
    rev_head: Vec<u32>,
    rev_prev: Vec<u32>,
}

impl HashChains {
    fn new(ring_len: usize) -> Self {
        Self {
            fwd_head: vec![NIL; HASH_TABLE_SIZE],
            fwd_prev: vec![NIL; ring_len],
            rev_head: vec![NIL; HASH_TABLE_SIZE],
            rev_prev: vec![NIL; ring_len],
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn build(ring: &[u8], end: usize) -> Self {
        let mut chains = Self::new(ring.len());
        for pos in 0..end {
            // Forward: hash(buf[p], buf[p+1], buf[p+2])
            if pos + 2 < end {
                let h = hash3(ring[pos], ring[pos + 1], ring[pos + 2]);
                chains.fwd_prev[pos] = chains.fwd_head[h];
                chains.fwd_head[h] = pos as u32;
            }
            // Reverse: hash(buf[p], buf[p-1], buf[p-2])
            if pos >= 2 {
                let h = hash3(ring[pos], ring[pos - 1], ring[pos - 2]);
                chains.rev_prev[pos] = chains.rev_head[h];
                chains.rev_head[h] = pos as u32;
            }
        }
        chains
    }
}

// ── Match Finder ──────────────────────────────────────────────────────

/// Two-phase match finder: hash chains first, then short-range linear fallback.
#[allow(clippy::cast_possible_truncation)]
fn find_best_match(buf: &[u8], chains: &HashChains, pos: usize, data_end: usize) -> Option<(u16, usize, bool)> {
    // Phase 1: Hash chain lookup (3+ byte matches).
    let best = find_hash_match(buf, chains, pos, data_end);

    let best_savings = best.map_or(0, |(d, l, r)| nibbles_saved(d, encode_length(l, r)));

    // Phase 2: If hash chain result < threshold, try linear short-range search.
    if best_savings < SHORT_SEARCH_THRESHOLD {
        let linear = find_short_match(buf, pos, data_end);
        return pick_best(best, linear);
    }

    best
}

/// Walks both forward and reverse hash chains to find the best 3+ byte match.
#[allow(clippy::cast_possible_truncation)]
fn find_hash_match(buf: &[u8], chains: &HashChains, pos: usize, data_end: usize) -> Option<(u16, usize, bool)> {
    let mut best: Option<(u16, usize, bool)> = None;
    let mut best_savings: i32 = -1;
    let max_dist = pos.min(usize::from(u16::MAX));

    // Forward chain: find positions sharing hash3(buf[pos], buf[pos+1], buf[pos+2]).
    if pos + 2 < data_end {
        let h = hash3(buf[pos], buf[pos + 1], buf[pos + 2]);
        let mut entry = chains.fwd_head[h];
        let mut depth = 0;
        while entry != NIL && depth < MAX_CHAIN_DEPTH {
            let candidate = entry as usize;
            entry = chains.fwd_prev[candidate];
            depth += 1;
            if candidate >= pos {
                continue;
            }
            let d = pos - candidate;
            if d > max_dist {
                break;
            }
            // Extend forward match.
            let max_fwd = (data_end - pos).min(MAX_MATCH_FWD);
            let mut fwd_len = 0;
            while fwd_len < max_fwd && buf[candidate + fwd_len] == buf[pos + fwd_len] {
                fwd_len += 1;
            }
            if fwd_len > 0 {
                let s = nibbles_saved(d as u16, encode_length(fwd_len, false));
                if s > best_savings {
                    best_savings = s;
                    best = Some((d as u16, fwd_len, false));
                }
            }
        }
    }

    // Reverse chain: entry at position C has hash3(buf[C], buf[C-1], buf[C-2]).
    // We want C where buf[C..C-2] matches buf[pos..pos+2] reversed,
    // so we look up hash3(buf[pos], buf[pos+1], buf[pos+2]) in rev_head.
    if pos + 2 < data_end {
        let h = hash3(buf[pos], buf[pos + 1], buf[pos + 2]);
        let mut entry = chains.rev_head[h];
        let mut depth = 0;
        while entry != NIL && depth < MAX_CHAIN_DEPTH {
            let candidate = entry as usize;
            entry = chains.rev_prev[candidate];
            depth += 1;
            if candidate >= pos {
                continue;
            }
            let d = pos - candidate;
            if d > max_dist {
                break;
            }
            // Extend reverse match: buf[candidate - i] == buf[pos + i].
            let max_rev = (data_end - pos).min(MAX_MATCH_REV).min(candidate + 1);
            let mut rev_len = 0;
            while rev_len < max_rev && buf[candidate - rev_len] == buf[pos + rev_len] {
                rev_len += 1;
            }
            if rev_len > 0 {
                let s = nibbles_saved(d as u16, encode_length(rev_len, true));
                if s > best_savings {
                    best_savings = s;
                    best = Some((d as u16, rev_len, true));
                }
            }
        }
    }

    // Only return matches that at least break even.
    best.filter(|_| best_savings >= 0)
}

/// Linear scan of distances `1..=SHORT_MATCH_RANGE`, trying all match lengths.
#[allow(clippy::cast_possible_truncation)]
fn find_short_match(buf: &[u8], pos: usize, data_end: usize) -> Option<(u16, usize, bool)> {
    let max_dist = pos.min(SHORT_MATCH_RANGE).min(usize::from(u16::MAX));
    let mut best: Option<(u16, usize, bool)> = None;
    let mut best_savings: i32 = -1;

    for d in 1..=max_dist {
        // Forward match.
        let max_fwd = (data_end - pos).min(MAX_MATCH_FWD);
        let mut fwd_len = 0;
        while fwd_len < max_fwd && buf[pos - d + fwd_len] == buf[pos + fwd_len] {
            fwd_len += 1;
        }
        if fwd_len > 0 {
            let s = nibbles_saved(d as u16, encode_length(fwd_len, false));
            if s > best_savings {
                best_savings = s;
                best = Some((d as u16, fwd_len, false));
            }
        }

        // Reverse match.
        let max_rev = (data_end - pos).min(MAX_MATCH_REV).min(pos - d + 1);
        let mut rev_len = 0;
        while rev_len < max_rev && buf[pos - d - rev_len] == buf[pos + rev_len] {
            rev_len += 1;
        }
        if rev_len > 0 {
            let s = nibbles_saved(d as u16, encode_length(rev_len, true));
            if s > best_savings {
                best_savings = s;
                best = Some((d as u16, rev_len, true));
            }
        }
    }

    // Only return matches that at least break even.
    best.filter(|_| best_savings >= 0)
}

/// Picks the match with higher nibble savings, or `a` on tie.
fn pick_best(a: Option<(u16, usize, bool)>, b: Option<(u16, usize, bool)>) -> Option<(u16, usize, bool)> {
    match (a, b) {
        (Some(am), Some(bm)) => {
            let sa = nibbles_saved(am.0, encode_length(am.1, am.2));
            let sb = nibbles_saved(bm.0, encode_length(bm.1, bm.2));
            if sb > sa {
                Some(bm)
            } else {
                Some(am)
            }
        }
        (Some(_), None) => a,
        (None, Some(_)) => b,
        (None, None) => None,
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
fn emit_match(nw: &mut NibbleWriter, distance: u16, length: usize, is_reverse: bool) {
    distance.encode_ubleb8(nw);
    encode_length(length, is_reverse).encode_sleb8(nw);
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
        let chains = HashChains::build(buf, buf.len());
        let result = find_best_match(buf, &chains, 3, 6);
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
        let chains = HashChains::build(buf, buf.len());
        let result = find_best_match(buf, &chains, 3, 6);
        assert!(result.is_some());
        let (dist, len, is_rev) = result.unwrap();
        assert_eq!(dist, 1);
        assert_eq!(len, 3);
        assert!(is_rev);
    }

    #[test]
    fn match_finder_no_match() {
        let buf = b"\x00\x00\x00\x01\x02\x03";
        let chains = HashChains::build(buf, buf.len());
        let result = find_best_match(buf, &chains, 3, 6);
        assert!(result.is_none());
    }

    // ── Savings Function ─────────────────────────────────────────────

    #[test]
    fn nibbles_saved_cases() {
        // 3-byte forward match at distance 1: saves 2*3 - 1 - 1 = 4 nibbles
        assert_eq!(nibbles_saved(1, 3), 4);
        // 1-byte forward match at distance 1: saves 2*1 - 1 - 1 = 0 (break even)
        assert_eq!(nibbles_saved(1, 1), 0);
        // 1-byte forward match at distance 8: saves 2*1 - 2 - 1 = -1 (not worth it)
        assert_eq!(nibbles_saved(8, 1), -1);
        // 2-byte match at distance 511: saves 2*2 - 3 - 1 = 0 (break even)
        assert_eq!(nibbles_saved(511, 2), 0);
        // 2-byte match at distance 63: saves 2*2 - 2 - 1 = 1
        assert_eq!(nibbles_saved(63, 2), 1);
        // 2-byte match at distance 7: saves 2*2 - 1 - 1 = 2
        assert_eq!(nibbles_saved(7, 2), 2);
        // Reverse: 3-byte at distance 1: saves 2*3 - 1 - 1 = 4
        assert_eq!(nibbles_saved(1, -3), 4);
    }

    // ── Hash Chain Build ─────────────────────────────────────────────

    #[test]
    fn hash_chain_build_basic() {
        let buf = b"abcabc";
        let chains = HashChains::build(buf, buf.len());
        // Forward chain for hash3('a','b','c') should link position 3 → 0.
        let h = hash3(b'a', b'b', b'c');
        assert_eq!(chains.fwd_head[h], 3);
        assert_eq!(chains.fwd_prev[3], 0);
        assert_eq!(chains.fwd_prev[0], NIL);
    }

    #[test]
    fn hash3_basic_sanity() {
        // Different inputs should generally produce different hashes.
        let h1 = hash3(0, 0, 0);
        let h2 = hash3(1, 0, 0);
        let h3 = hash3(0, 1, 0);
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        // All within table size.
        assert!(h1 < HASH_TABLE_SIZE);
        assert!(h2 < HASH_TABLE_SIZE);
        assert!(h3 < HASH_TABLE_SIZE);
    }

    // ── Short Match Finder ───────────────────────────────────────────

    #[test]
    fn short_match_at_distance_1() {
        // "aa" — 1 byte repeated. Short match at d=1, len=1 should break even (savings=0).
        let buf = b"\x00aa";
        let result = find_short_match(buf, 2, 3);
        assert!(result.is_some());
        let (dist, len, is_rev) = result.unwrap();
        assert_eq!(dist, 1);
        // Should find the longest match at the best distance.
        assert_eq!(len, 1);
        assert!(!is_rev);
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
