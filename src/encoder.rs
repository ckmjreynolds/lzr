//! LZR stream encoder.

use std::io::{self, BufReader, BufWriter, Read, Write};

use rayon::prelude::*;

use crate::adler32::Adler32;
use crate::bleb8::{self, Sleb8, Ubleb8};
use crate::nibble::NibbleWriter;
use crate::{MAGIC, VERSION, WINDOW_SIZE};

/// Ring buffer size (1 MiB).
const RING_SIZE: usize = 1 << 20;

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

// ── Match Representation ─────────────────────────────────────────────

/// A back-reference match found by the match finder.
#[derive(Debug, Clone, Copy)]
struct Match {
    distance: u16,
    length: usize,
    is_reverse: bool,
}

impl Match {
    /// Encodes the match length as the signed i16 used in the format.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    const fn encoded_length(&self) -> i16 {
        if self.is_reverse {
            (-(self.length as i32)) as i16
        } else {
            self.length as i16
        }
    }

    /// Nibbles saved by encoding this match instead of literals.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn savings(&self) -> i32 {
        let literal_cost = 2 * i32::from(self.encoded_length().unsigned_abs());
        let match_cost = bleb8::ubleb8_len(self.distance) as i32 + bleb8::sleb8_len(self.encoded_length()) as i32;
        literal_cost - match_cost
    }
}

// ── Best Match Tracker ───────────────────────────────────────────────

/// Tracks the best match found during a search, keeping only profitable matches.
struct BestTracker {
    best: Option<Match>,
    savings: i32,
}

impl BestTracker {
    const fn new() -> Self {
        Self {
            best: None,
            savings: -1,
        }
    }

    /// Considers a candidate match, keeping it if it improves on the current best.
    fn consider(&mut self, m: Match) {
        let s = m.savings();
        if s > self.savings {
            self.savings = s;
            self.best = Some(m);
        }
    }

    /// Returns the best match, if it at least breaks even (savings >= 0).
    fn finish(self) -> Option<Match> {
        self.best.filter(|_| self.savings >= 0)
    }
}

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
    let data_len = data_end - data_start;
    let mut nw = NibbleWriter::with_capacity(2 * data_len);
    let mut pos = data_start;
    let mut literal_start = pos;

    while pos < data_end {
        if let Some(m) = find_best_match(buf, chains, pos, data_end) {
            // Flush pending literals.
            if literal_start < pos {
                emit_literals(&mut nw, &buf[literal_start..pos]);
            }
            emit_match(&mut nw, &m);
            pos += m.length;
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

// ── Match Extension ──────────────────────────────────────────────────

/// Extends a forward match: counts how many bytes match starting at
/// `buf[candidate]` vs `buf[pos]`, up to `max_len`.
fn extend_forward(buf: &[u8], candidate: usize, pos: usize, max_len: usize) -> usize {
    let mut len = 0;
    while len < max_len && buf[candidate + len] == buf[pos + len] {
        len += 1;
    }
    len
}

/// Extends a reverse match: counts how many bytes match with `candidate`
/// decrementing and `pos` incrementing, up to `max_len`.
fn extend_reverse(buf: &[u8], candidate: usize, pos: usize, max_len: usize) -> usize {
    let mut len = 0;
    while len < max_len && buf[candidate - len] == buf[pos + len] {
        len += 1;
    }
    len
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
    fn new(len: usize) -> Self {
        Self {
            fwd_head: vec![NIL; HASH_TABLE_SIZE],
            fwd_prev: vec![NIL; len],
            rev_head: vec![NIL; HASH_TABLE_SIZE],
            rev_prev: vec![NIL; len],
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn build(ring: &[u8], end: usize) -> Self {
        let mut chains = Self::new(end);
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
fn find_best_match(buf: &[u8], chains: &HashChains, pos: usize, data_end: usize) -> Option<Match> {
    // Phase 1: Hash chain lookup (3+ byte matches).
    let hash_best = find_hash_match(buf, chains, pos, data_end);

    let hash_savings = hash_best.as_ref().map_or(0, Match::savings);

    // Phase 2: If hash chain result < threshold, try linear short-range search.
    if hash_savings < SHORT_SEARCH_THRESHOLD {
        let linear_best = find_short_match(buf, pos, data_end);
        return pick_best(hash_best, linear_best);
    }

    hash_best
}

/// Walks both forward and reverse hash chains to find the best 3+ byte match.
#[allow(clippy::cast_possible_truncation)]
fn find_hash_match(buf: &[u8], chains: &HashChains, pos: usize, data_end: usize) -> Option<Match> {
    let mut tracker = BestTracker::new();
    let max_dist = pos.min(usize::from(u16::MAX));

    if pos + 2 >= data_end {
        return None;
    }

    let h = hash3(buf[pos], buf[pos + 1], buf[pos + 2]);
    let max_fwd = (data_end - pos).min(MAX_MATCH_FWD);

    // Forward chain: find positions sharing hash3(buf[pos], buf[pos+1], buf[pos+2]).
    {
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
            let fwd_len = extend_forward(buf, candidate, pos, max_fwd);
            if fwd_len > 0 {
                tracker.consider(Match {
                    distance: d as u16,
                    length: fwd_len,
                    is_reverse: false,
                });
            }
        }
    }

    // Reverse chain: entry at position C has hash3(buf[C], buf[C-1], buf[C-2]).
    // We want C where buf[C..C-2] matches buf[pos..pos+2] reversed,
    // so we look up hash3(buf[pos], buf[pos+1], buf[pos+2]) in rev_head.
    {
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
            // Cap length so the decoder's converging write/read pointers never collide.
            let max_rev = (data_end - pos).min(MAX_MATCH_REV).min(candidate + 1).min((WINDOW_SIZE - d) / 2 + 1);
            let rev_len = extend_reverse(buf, candidate, pos, max_rev);
            if rev_len > 0 {
                tracker.consider(Match {
                    distance: d as u16,
                    length: rev_len,
                    is_reverse: true,
                });
            }
        }
    }

    tracker.finish()
}

/// Linear scan of distances `1..=SHORT_MATCH_RANGE`, trying all match lengths.
#[allow(clippy::cast_possible_truncation)]
fn find_short_match(buf: &[u8], pos: usize, data_end: usize) -> Option<Match> {
    let max_dist = pos.min(SHORT_MATCH_RANGE).min(usize::from(u16::MAX));
    let mut tracker = BestTracker::new();
    let max_fwd = (data_end - pos).min(MAX_MATCH_FWD);

    for d in 1..=max_dist {
        let candidate = pos - d;

        // Forward match.
        let fwd_len = extend_forward(buf, candidate, pos, max_fwd);
        if fwd_len > 0 {
            tracker.consider(Match {
                distance: d as u16,
                length: fwd_len,
                is_reverse: false,
            });
        }

        // Reverse match: cap length so the decoder's converging write/read
        // pointers never collide: max_len = (WINDOW_SIZE - d) / 2 + 1.
        let max_rev = (data_end - pos).min(MAX_MATCH_REV).min(candidate + 1).min((WINDOW_SIZE - d) / 2 + 1);
        let rev_len = extend_reverse(buf, candidate, pos, max_rev);
        if rev_len > 0 {
            tracker.consider(Match {
                distance: d as u16,
                length: rev_len,
                is_reverse: true,
            });
        }
    }

    tracker.finish()
}

/// Picks the match with higher nibble savings, or `a` on tie.
fn pick_best(a: Option<Match>, b: Option<Match>) -> Option<Match> {
    match (a, b) {
        (Some(am), Some(bm)) => {
            if bm.savings() > am.savings() {
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
fn emit_match(nw: &mut NibbleWriter, m: &Match) {
    m.distance.encode_ubleb8(nw);
    m.encoded_length().encode_sleb8(nw);
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

    proptest! {
        #[test]
        fn round_trip_random(data in prop::collection::vec(any::<u8>(), 0..4096)) {
            prop_assert_eq!(round_trip(&data), data);
        }
    }
}
