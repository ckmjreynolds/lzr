//! LZR compression encoder.
//!
//! Compresses raw input into the LZR format using LZ77-style matching driven
//! by a flat zlib-style hash chain (`head` + `prev` arrays). Per-level
//! parameters tune chain depth, lazy matching, reverse-match search, and
//! lz4-style skip-on-miss for the fast levels.
//!
//! Single-threaded and multi-threaded paths share the same per-block compress
//! function. When `EncodeOptions::threads == 1`, the encoder bypasses rayon
//! entirely and reuses one output buffer across blocks; otherwise blocks are
//! compressed in parallel via rayon.

use std::cell::RefCell;
use std::io::{self, Read, Write};

use rayon::prelude::*;

use crate::adler32::Adler32;
use crate::error::Result;
use crate::format;
use crate::options::EncodeOptions;
use crate::wild;
use crate::{BATCH_SIZE, BLOCK_SIZE, WINDOW_SIZE};

/// Maximum literal count per frame (`i16::MAX`).
const MAX_LIT_LEN: usize = i16::MAX as usize;

/// Maximum match length per frame (`i16::MAX`).
const MAX_MATCH_LEN: usize = i16::MAX as usize;

/// Minimum match length the encoder will emit. The format requires ≥ 4 bytes
/// because match-length 0..3 cannot be encoded in the token field.
const MIN_MATCH: usize = 4;

/// Maximum forward distance representable in the 16-bit `distance` frame field.
const MAX_FWD_DIST: usize = u16::MAX as usize;

/// Maximum *raw* reverse distance: `frame_distance = raw - 3` must fit in u16,
/// so the raw decoder-side distance can be one of `4..=u16::MAX + 3`.
const MAX_REV_RAW_DIST: usize = MAX_FWD_DIST + 3;

/// Number of hash bits — 32 KiB head table.
const HASH_BITS: u32 = 15;

/// Hash table size.
const HASH_SIZE: usize = 1 << HASH_BITS;

/// `prev` table size: spans the full prefix + block extent so absolute offsets
/// from `prefix_start` index it directly without aliasing.
const PREV_SIZE: usize = 2 * WINDOW_SIZE;

/// Empty-chain sentinel.
const NIL: u32 = u32::MAX;

/// Skip-on-miss exponent (matches lz4-fast `SKIP_LOG = 6`).
const SKIP_LOG: u32 = 6;

/// Per-level encoder tunables.
#[derive(Clone, Copy)]
struct LevelParams {
    /// Maximum forward hash chain walk depth at one position.
    chain_depth: u32,
    /// Maximum reverse hash chain walk depth at one position. Reverse matches
    /// are rarer than forward matches in most data, so a shallower walk is
    /// usually a good speed/ratio tradeoff. `0` disables reverse search.
    reverse_chain_depth: u32,
    /// Stop chain walking once a match this long is found.
    nice_len: usize,
    /// `good_match` heuristic threshold: once `best_len >= good_match`, the
    /// remaining chain depth is divided by 4. Disabled when 0.
    good_match: usize,
    /// Enable single-step lazy matching.
    lazy: bool,
    /// Enable lz4-fast skip-on-miss for incompressible regions.
    skip_on_miss: bool,
}

impl LevelParams {
    const fn reverse(&self) -> bool {
        self.reverse_chain_depth > 0
    }
}

#[rustfmt::skip]
const PARAMS: [LevelParams; 10] = [
    // Index 0 unused — levels are 1..=9.
    LevelParams { chain_depth: 0,    reverse_chain_depth: 0,  nice_len: 0,   good_match: 0,  lazy: false, skip_on_miss: false },
    LevelParams { chain_depth: 1,    reverse_chain_depth: 0,  nice_len: 16,  good_match: 0,  lazy: false, skip_on_miss: true  }, // L1
    LevelParams { chain_depth: 4,    reverse_chain_depth: 0,  nice_len: 24,  good_match: 16, lazy: false, skip_on_miss: true  }, // L2
    LevelParams { chain_depth: 8,    reverse_chain_depth: 0,  nice_len: 32,  good_match: 16, lazy: false, skip_on_miss: false }, // L3
    LevelParams { chain_depth: 16,   reverse_chain_depth: 0,  nice_len: 32,  good_match: 24, lazy: true,  skip_on_miss: false }, // L4
    LevelParams { chain_depth: 32,   reverse_chain_depth: 0,  nice_len: 64,  good_match: 24, lazy: true,  skip_on_miss: false }, // L5
    LevelParams { chain_depth: 64,   reverse_chain_depth: 0,  nice_len: 96,  good_match: 32, lazy: true,  skip_on_miss: false }, // L6
    LevelParams { chain_depth: 96,   reverse_chain_depth: 16, nice_len: 96,  good_match: 32, lazy: true,  skip_on_miss: false }, // L7
    LevelParams { chain_depth: 128,  reverse_chain_depth: 32, nice_len: 128, good_match: 24, lazy: true,  skip_on_miss: false }, // L8
    LevelParams { chain_depth: 256,  reverse_chain_depth: 64, nice_len: 258, good_match: 32, lazy: true,  skip_on_miss: false }, // L9
];

/// Flat hash chain. `head[h]` stores the most recent absolute position with
/// 4-byte hash `h`; `prev[pos - prefix_start]` stores the previous position in
/// the same hash bucket. Storing absolute u32 positions sidesteps the
/// aliasing of a `(pos & WINDOW_MASK)` scheme when prefix and block together
/// span more than `WINDOW_SIZE`.
struct HashChain {
    head: Box<[u32]>,
    prev: Box<[u32]>,
}

impl HashChain {
    fn new() -> Self {
        Self {
            head: vec![NIL; HASH_SIZE].into_boxed_slice(),
            prev: vec![NIL; PREV_SIZE].into_boxed_slice(),
        }
    }

    /// Reset between blocks. `prev` is intentionally NOT cleared — every slot
    /// is overwritten by an `insert` before any chain walk reads it.
    fn reset(&mut self) {
        self.head.fill(NIL);
    }
}

/// Per-thread reusable scratch space.
struct Scratch {
    fwd: HashChain,
    rev: HashChain,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch {
        fwd: HashChain::new(),
        rev: HashChain::new(),
    });
}

/// Knuth multiplicative hash, folded to `HASH_BITS`.
#[inline]
const fn hash4(x: u32) -> usize {
    (x.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
}

/// 8-byte-at-a-time match length over an in-memory buffer.
#[inline]
fn match_len_forward(input: &[u8], pos: usize, cand: usize, max_len: usize) -> usize {
    let mut len = 0;
    while len + 8 <= max_len {
        let lhs = wild::read_u64_ne(input, cand + len);
        let rhs = wild::read_u64_ne(input, pos + len);
        let diff = lhs ^ rhs;
        if diff != 0 {
            return len + (diff.trailing_zeros() as usize >> 3);
        }
        len += 8;
    }
    while len < max_len && input[cand + len] == input[pos + len] {
        len += 1;
    }
    len
}

/// Inserts `pos` into `chain` under hash key `key`.
///
/// `STORE_PREV = false` skips the `prev` write, used at L1 (`chain_depth = 1`)
/// where chain walking only reads `head` and never follows `prev`. This is the
/// L1 cache-footprint win measured in Phase E.
#[allow(clippy::inline_always)]
#[inline(always)]
fn insert<const STORE_PREV: bool>(chain: &mut HashChain, pos: usize, key: u32, prefix_start: usize) {
    let h = hash4(key);
    #[allow(clippy::cast_possible_truncation)]
    let pos32 = pos as u32;
    if STORE_PREV {
        chain.prev[pos - prefix_start] = chain.head[h];
    }
    chain.head[h] = pos32;
}

/// Forward chain walker.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn find_forward_match(
    input: &[u8],
    chain: &HashChain,
    key: u32,
    pos: usize,
    block_end: usize,
    prefix_start: usize,
    params: LevelParams,
) -> Option<(usize, u16)> {
    let max_len = (block_end - pos).min(MAX_MATCH_LEN);
    if max_len < MIN_MATCH {
        return None;
    }

    let h = hash4(key);
    let mut cur = chain.head[h];
    if cur == NIL {
        return None;
    }

    let min_valid = prefix_start.max(pos.saturating_sub(MAX_FWD_DIST));
    let mut best_len = MIN_MATCH - 1;
    let mut best_dist: u16 = 0;
    let mut depth: u32 = 0;
    let mut chain_limit = params.chain_depth;

    while cur != NIL && depth < chain_limit {
        let cand = cur as usize;
        if cand >= pos || cand < min_valid {
            break;
        }

        if wild::read_u32_le(input, cand) == key {
            let len = match_len_forward(input, pos, cand, max_len);
            if len > best_len {
                best_len = len;
                best_dist = (pos - cand) as u16;
                if len >= params.nice_len {
                    break;
                }
                // good_match heuristic: once a sufficiently long match has
                // been found, cap the remaining chain depth.
                if params.good_match > 0 && len >= params.good_match {
                    let cap = depth + (params.chain_depth >> 2).max(1);
                    if cap < chain_limit {
                        chain_limit = cap;
                    }
                }
            }
        }

        cur = chain.prev[cand - prefix_start];
        depth += 1;
    }

    if best_len >= MIN_MATCH {
        Some((best_len, best_dist))
    } else {
        None
    }
}

/// Reverse chain walker.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn find_reverse_match(
    input: &[u8],
    chain: &HashChain,
    forward_key: u32,
    pos: usize,
    block_end: usize,
    prefix_start: usize,
    params: LevelParams,
) -> Option<(usize, u16)> {
    let max_forward = (block_end - pos).min(MAX_MATCH_LEN);
    if max_forward < MIN_MATCH {
        return None;
    }

    let h = hash4(forward_key);
    let mut cur = chain.head[h];
    if cur == NIL {
        return None;
    }

    let min_valid = prefix_start.max(pos.saturating_sub(MAX_REV_RAW_DIST));
    let mut best_len = MIN_MATCH - 1;
    let mut best_dist: u16 = 0;
    let mut depth: u32 = 0;
    let mut chain_limit = params.reverse_chain_depth;

    while cur != NIL && depth < chain_limit {
        let cand = cur as usize;
        if cand >= pos || cand < min_valid {
            break;
        }
        let raw_dist = pos - cand;
        if raw_dist < 4 {
            cur = chain.prev[cand - prefix_start];
            depth += 1;
            continue;
        }

        if wild::read_u32_be(input, cand) == forward_key {
            let frame_distance = (raw_dist - 3) as u16;
            // Decoder constraint: distance + 2*length <= WINDOW_SIZE + 1.
            let window_limit = (WINDOW_SIZE + 1 - frame_distance as usize) / 2;
            if window_limit >= MIN_MATCH {
                // Backward expansion bound: input[cand + 3 - len] must remain >= prefix_start.
                let max_back = (cand + 4).saturating_sub(prefix_start);
                let max_len = max_forward.min(max_back).min(window_limit);
                if max_len >= MIN_MATCH {
                    let mut len = MIN_MATCH;
                    while len < max_len && input[cand + 3 - len] == input[pos + len] {
                        len += 1;
                    }
                    if len > best_len {
                        best_len = len;
                        best_dist = frame_distance;
                        if len >= params.nice_len {
                            break;
                        }
                        if params.good_match > 0 && len >= params.good_match {
                            let cap = depth + (params.reverse_chain_depth >> 2).max(1);
                            if cap < chain_limit {
                                chain_limit = cap;
                            }
                        }
                    }
                }
            }
        }

        cur = chain.prev[cand - prefix_start];
        depth += 1;
    }

    if best_len >= MIN_MATCH {
        Some((best_len, best_dist))
    } else {
        None
    }
}

/// Returns the longer of forward and reverse matches at `pos` (forward wins ties).
#[inline]
#[allow(clippy::too_many_arguments)]
fn find_best(
    input: &[u8],
    fwd: &HashChain,
    rev: &HashChain,
    key: u32,
    pos: usize,
    block_end: usize,
    prefix_start: usize,
    params: LevelParams,
) -> Option<(usize, u16, bool)> {
    let f = find_forward_match(input, fwd, key, pos, block_end, prefix_start, params).map(|(l, d)| (l, d, false));

    if !params.reverse() {
        return f;
    }

    let r = find_reverse_match(input, rev, key, pos, block_end, prefix_start, params).map(|(l, d)| (l, d, true));

    match (f, r) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (Some(fx), Some(rx)) => {
            if fx.0 >= rx.0 {
                Some(fx)
            } else {
                Some(rx)
            }
        }
    }
}

/// Compresses one block into `out`, returning the block's Adler-32 checksum.
fn compress_block(
    input: &[u8],
    prefix_start: usize,
    block_start: usize,
    block_end: usize,
    level: usize,
    out: &mut Vec<u8>,
) -> Adler32 {
    let block_len = block_end - block_start;
    let mut checksum = Adler32::new();
    checksum.update(&input[block_start..block_end]);

    if block_len < MIN_MATCH {
        if block_len > 0 {
            emit_literals(input, out, block_start, block_end);
        }
        return checksum;
    }

    let params = PARAMS[level];

    SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        s.fwd.reset();
        if params.reverse() {
            s.rev.reset();
        }
        let Scratch {
            fwd,
            rev,
        } = &mut *s;
        if params.chain_depth > 1 {
            compress_block_inner::<true>(input, out, prefix_start, block_start, block_end, params, fwd, rev);
        } else {
            compress_block_inner::<false>(input, out, prefix_start, block_start, block_end, params, fwd, rev);
        }
    });

    checksum
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::too_many_lines
)]
fn compress_block_inner<const STORE_PREV: bool>(
    input: &[u8],
    out: &mut Vec<u8>,
    prefix_start: usize,
    block_start: usize,
    block_end: usize,
    params: LevelParams,
    fwd: &mut HashChain,
    rev: &mut HashChain,
) {
    let block_len = block_end - block_start;
    let initial_out_len = out.len();

    // Phase 1: insert prefix positions (no emission).
    if block_start > prefix_start {
        let prefix_end = block_start;
        let mut p = prefix_start;
        while p + MIN_MATCH <= prefix_end {
            let key = wild::read_u32_le(input, p);
            insert::<STORE_PREV>(fwd, p, key, prefix_start);
            if params.reverse() {
                let rkey = wild::read_u32_be(input, p);
                insert::<STORE_PREV>(rev, p, rkey, prefix_start);
            }
            p += 1;
        }
    }

    // Phase 2: main scan with optional lazy matching and skip-on-miss.
    let mut pos = block_start;
    let mut literal_start = block_start;
    let mut non_matches: u32 = 0;

    while pos + MIN_MATCH <= block_end {
        // Escape: if compressed bytes already exceed input bytes, stop matching
        // and emit the rest as literals.
        if out.len() - initial_out_len > block_len {
            break;
        }

        let key = wild::read_u32_le(input, pos);
        let m0 = find_best(input, fwd, rev, key, pos, block_end, prefix_start, params);

        // Insert pos AFTER searching so we never match against ourselves.
        insert::<STORE_PREV>(fwd, pos, key, prefix_start);
        if params.reverse() {
            let rkey = wild::read_u32_be(input, pos);
            insert::<STORE_PREV>(rev, pos, rkey, prefix_start);
        }

        let Some((mut blen, mut bdist, mut brev)) = m0 else {
            non_matches = non_matches.saturating_add(1);
            let step = if params.skip_on_miss {
                ((non_matches >> SKIP_LOG) + 1) as usize
            } else {
                1
            };
            pos += step;
            continue;
        };

        let mut bpos = pos;
        let mut next_to_insert = pos + 1;

        // Lazy matching: probe pos+1 for a strictly longer match.
        if params.lazy && blen < params.nice_len {
            let probe = pos + 1;
            if probe + MIN_MATCH <= block_end {
                let key1 = wild::read_u32_le(input, probe);
                let m1 = find_best(input, fwd, rev, key1, probe, block_end, prefix_start, params);
                insert::<STORE_PREV>(fwd, probe, key1, prefix_start);
                if params.reverse() {
                    let rkey1 = wild::read_u32_be(input, probe);
                    insert::<STORE_PREV>(rev, probe, rkey1, prefix_start);
                }
                next_to_insert = probe + 1;

                if let Some((len1, dist1, rev1)) = m1 {
                    if len1 > blen {
                        blen = len1;
                        bdist = dist1;
                        brev = rev1;
                        bpos = probe;
                    }
                }
            }
        }

        // Emit committed match (with any leading literals).
        let match_len_signed = if brev {
            -(blen as i16)
        } else {
            blen as i16
        };
        emit_match_frame(input, out, literal_start, bpos, match_len_signed, bdist);

        // Insert positions inside the matched span that we have not yet inserted.
        let match_end = bpos + blen;
        let skip_start = (bpos + 1).max(next_to_insert);
        let mut skip = skip_start;
        while skip < match_end {
            if skip + MIN_MATCH > block_end {
                break;
            }
            let fkey = wild::read_u32_le(input, skip);
            insert::<STORE_PREV>(fwd, skip, fkey, prefix_start);
            if params.reverse() {
                let bkey = wild::read_u32_be(input, skip);
                insert::<STORE_PREV>(rev, skip, bkey, prefix_start);
            }
            skip += 1;
        }

        pos = match_end;
        literal_start = pos;
        non_matches = 0;
    }

    // Phase 3: trailing literals.
    if literal_start < block_end {
        emit_literals(input, out, literal_start, block_end);
    }
}

/// Emits a match frame, flushing excess literals as standalone frames first.
#[allow(clippy::cast_possible_truncation)]
fn emit_match_frame(input: &[u8], out: &mut Vec<u8>, literal_start: usize, pos: usize, match_len: i16, distance: u16) {
    let mut lit_pos = literal_start;
    let mut remaining = pos - literal_start;

    // Flush literals that exceed the per-frame i16 limit. Distance 1 is used
    // for literal-only frames because distance 0 is reserved for EOS.
    while remaining > MAX_LIT_LEN {
        format::encode_frame(i16::MAX, 0, 1, out);
        out.extend_from_slice(&input[lit_pos..lit_pos + MAX_LIT_LEN]);
        lit_pos += MAX_LIT_LEN;
        remaining -= MAX_LIT_LEN;
    }

    #[allow(clippy::cast_possible_wrap)]
    let lit_len = remaining as i16;
    format::encode_frame(lit_len, match_len, distance, out);
    out.extend_from_slice(&input[lit_pos..pos]);
}

/// Emits literal-only frames, splitting into chunks if the count exceeds `i16`.
fn emit_literals(input: &[u8], out: &mut Vec<u8>, start: usize, end: usize) {
    let mut p = start;
    while p < end {
        let chunk = (end - p).min(MAX_LIT_LEN);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        format::encode_frame(chunk as i16, 0, 1, out);
        out.extend_from_slice(&input[p..p + chunk]);
        p += chunk;
    }
}

/// Compresses data from `input` into LZR format, writing to `output`.
///
/// # Errors
///
/// Returns [`Error::Io`](crate::error::Error::Io) on any underlying I/O failure.
///
/// # Panics
///
/// Panics if the rayon thread pool cannot be created when an explicit thread
/// count is requested.
pub fn encode(input: &mut impl Read, output: &mut impl Write, options: &EncodeOptions) -> Result<()> {
    let level = options.get_level();
    let threads = options.get_threads();

    // Header.
    {
        let mut hdr = Vec::with_capacity(4);
        format::encode_header(&mut hdr);
        output.write_all(&hdr)?;
    }

    let mut total_checksum = Adler32::new();
    let mut total_len: u64 = 0;

    // Each batch consists of up to WINDOW_SIZE bytes of carryover prefix
    // followed by READ_SIZE fresh bytes. Buf has BATCH_SIZE total capacity.
    let read_size = BATCH_SIZE - WINDOW_SIZE;
    let mut buf = vec![0u8; BATCH_SIZE];
    let mut prefix_len = 0usize;
    let single_threaded = threads == 1;

    let pool = if threads > 1 {
        Some(rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(io::Error::other)?)
    } else {
        None
    };

    loop {
        // Refill: read READ_SIZE bytes after the carried-over prefix.
        let target = prefix_len + read_size;
        let mut filled = prefix_len;
        while filled < target {
            let n = input.read(&mut buf[filled..target])?;
            if n == 0 {
                break;
            }
            filled += n;
        }

        if filled == prefix_len {
            // No new data — done with input.
            break;
        }

        let data_start = prefix_len;
        let data_end = filled;

        // Build the block list for this batch.
        let mut blocks: Vec<(usize, usize, usize)> = Vec::with_capacity((data_end - data_start) / BLOCK_SIZE + 1);
        let mut bs = data_start;
        while bs < data_end {
            let be = (bs + BLOCK_SIZE).min(data_end);
            let bp = bs.saturating_sub(WINDOW_SIZE);
            blocks.push((bp, bs, be));
            bs = be;
        }

        // Compress and write.
        if single_threaded {
            let mut block_out: Vec<u8> = Vec::with_capacity(2 * BLOCK_SIZE);
            for &(bp, bs, be) in &blocks {
                block_out.clear();
                let cks = compress_block(&buf, bp, bs, be, level, &mut block_out);
                output.write_all(&block_out)?;
                total_checksum = total_checksum.combine(&cks);
                total_len += (be - bs) as u64;
            }
        } else {
            let compress_one = |&(bp, bs, be): &(usize, usize, usize)| {
                let mut block_out: Vec<u8> = Vec::with_capacity(2 * BLOCK_SIZE);
                let cks = compress_block(&buf, bp, bs, be, level, &mut block_out);
                (block_out, cks, be - bs)
            };
            let results: Vec<(Vec<u8>, Adler32, usize)> = pool.as_ref().map_or_else(
                || blocks.par_iter().map(compress_one).collect(),
                |p| p.install(|| blocks.par_iter().map(compress_one).collect()),
            );
            for (block_out, cks, len) in results {
                output.write_all(&block_out)?;
                total_checksum = total_checksum.combine(&cks);
                total_len += len as u64;
            }
        }

        // Carry over the last WINDOW_SIZE bytes for the next batch.
        let carry_len = (data_end - data_start).min(WINDOW_SIZE).min(data_end);
        let carry_start = data_end - carry_len;
        if carry_start > 0 {
            buf.copy_within(carry_start..data_end, 0);
        }
        prefix_len = carry_len;

        if data_end - data_start < read_size {
            // Short read — input is exhausted.
            break;
        }
    }

    // Footer.
    {
        let mut ftr = Vec::with_capacity(13);
        format::encode_eos(&mut ftr);
        format::encode_footer(total_len, total_checksum.checksum(), &mut ftr);
        output.write_all(&ftr)?;
    }

    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;
    use proptest::prelude::*;

    use super::*;
    use crate::decode;

    fn roundtrip(data: &[u8]) {
        let opts = EncodeOptions::default();
        let mut compressed = Vec::new();
        encode(&mut &data[..], &mut compressed, &opts).unwrap();

        let mut decompressed = Vec::new();
        decode::decode(&mut &compressed[..], &mut decompressed).unwrap();

        assert_eq!(decompressed, data);
    }

    fn roundtrip_level(data: &[u8], level: usize) {
        let opts = EncodeOptions::new().level(level);
        let mut compressed = Vec::new();
        encode(&mut &data[..], &mut compressed, &opts).unwrap();

        let mut decompressed = Vec::new();
        decode::decode(&mut &compressed[..], &mut decompressed).unwrap();

        assert_eq!(decompressed, data);
    }

    fn roundtrip_threads(data: &[u8], level: usize, threads: usize) {
        let opts = EncodeOptions::new().level(level).threads(threads);
        let mut compressed = Vec::new();
        encode(&mut &data[..], &mut compressed, &opts).unwrap();

        let mut decompressed = Vec::new();
        decode::decode(&mut &compressed[..], &mut decompressed).unwrap();

        assert_eq!(decompressed, data);
    }

    #[test]
    fn empty() {
        roundtrip(b"");
    }

    #[test]
    fn single_byte() {
        roundtrip(b"A");
    }

    #[test]
    fn three_bytes() {
        roundtrip(b"ABC");
    }

    #[test]
    fn hi() {
        roundtrip(b"Hi");
    }

    #[test]
    fn forward_match() {
        roundtrip(b"ABCDABCD");
    }

    #[test]
    fn repeated_byte() {
        roundtrip(&vec![0xAA; 1000]);
    }

    #[test]
    fn patterned_sizes() {
        #[allow(clippy::unreadable_literal)]
        for &size in &[100usize, 1000, 10000, 65535, 65536, 65537, 70000, 100000, 131072] {
            #[allow(clippy::cast_possible_truncation)]
            let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            roundtrip(&data);
        }
    }

    #[test]
    fn long_repeated() {
        roundtrip(&vec![0x42; BLOCK_SIZE + 100]);
    }

    #[test]
    fn all_levels() {
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
        for level in 1..=9 {
            roundtrip_level(data, level);
        }
    }

    #[test]
    fn level_9_with_patterns() {
        let mut data = Vec::new();
        data.extend_from_slice(b"ABCDEFGH");
        data.extend_from_slice(b"ABCDEFGH"); // forward match
        data.extend_from_slice(b"HGFEDCBA"); // reverse match candidate
        roundtrip_level(&data, 9);
    }

    #[test]
    fn lazy_and_reverse_interleaved() {
        let mut data = Vec::new();
        for _ in 0..32 {
            data.extend_from_slice(b"abcdefghijklmnop");
            data.extend_from_slice(b"ponmlkjihgfedcba");
        }
        for level in 1..=9 {
            roundtrip_level(&data, level);
        }
    }

    #[test]
    fn threading_st_and_mt_match() {
        // Across-threads determinism for the same data and level.
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(2000);
        for level in 1..=9 {
            roundtrip_threads(&data, level, 1);
            roundtrip_threads(&data, level, 0);
        }
    }

    proptest! {
        #[test]
        fn roundtrip_random(data in prop::collection::vec(any::<u8>(), 0..BLOCK_SIZE * 2)) {
            let mut compressed = Vec::new();
            encode(&mut data.as_slice(), &mut compressed, &EncodeOptions::default()).unwrap();
            let mut decompressed = Vec::new();
            decode::decode(&mut compressed.as_slice(), &mut decompressed).unwrap();
            prop_assert_eq!(decompressed, data);
        }

        #[test]
        fn roundtrip_level_1(data in prop::collection::vec(any::<u8>(), 0..BLOCK_SIZE)) {
            let opts = EncodeOptions::new().level(1);
            let mut compressed = Vec::new();
            encode(&mut data.as_slice(), &mut compressed, &opts).unwrap();
            let mut decompressed = Vec::new();
            decode::decode(&mut compressed.as_slice(), &mut decompressed).unwrap();
            prop_assert_eq!(decompressed, data);
        }
    }
}
