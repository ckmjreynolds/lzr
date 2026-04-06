//! LZR compression encoder.
//!
//! Compresses raw data into the LZR format using LZ77-based matching with
//! hash-table-driven match finding. Supports compression levels 1–9; the
//! number of forward and reverse match candidates checked at each level is
//! defined by [`candidates_for_level`].

use std::io::{Read, Write};

use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::adler32::Adler32;
use crate::buffer::Buffer;
use crate::cursor::{WriteBuf, WriteCursor};
use crate::error::Result;
use crate::format;
use crate::options::EncodeOptions;
use crate::packed::Queue;
use crate::{BATCH_SIZE, BLOCK_SIZE, WINDOW_SIZE};

/// Output buffer size per block: 2× block size to accommodate incompressible data.
const OUTPUT_BUF_SIZE: usize = 2 * BLOCK_SIZE;

/// Maximum literal count per frame (limited by i16).
const MAX_LIT_LEN: usize = i16::MAX as usize;

/// Maximum match length per frame (limited by i16).
const MAX_MATCH_LEN: usize = i16::MAX as usize;

/// Per-level number of hash-bucket candidates to inspect.
///
/// Index 0 is unused. L1 always inspects only the most recent candidate so the
/// fast path stays fast; L2..=L9 scale linearly up to [`Queue::CAPACITY`].
/// Reverse matching (L9 only today) uses the same depth as forward.
const fn candidates_for_level(level: usize) -> usize {
    // 4-slot Queue: L1 keeps the fast path at 1 candidate; L4..=L9 saturate at 4.
    const TABLE: [usize; 10] = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4];
    TABLE[level]
}

/// Compresses data from `input` into LZR format, writing to `output`.
///
/// # Errors
///
/// Returns [`Error::Io`](crate::error::Error::Io) on any underlying I/O failure.
///
/// # Panics
///
/// Panics if the rayon thread pool cannot be created.
pub fn encode(input: &mut impl Read, output: &mut impl Write, options: &EncodeOptions) -> Result<()> {
    let level = options.get_level();
    let threads = options.get_threads();

    let mut in_buf = Buffer::<BATCH_SIZE>::new();
    let mut pos: usize = 0;

    let mut hdr_buf = Buffer::<WINDOW_SIZE>::new();
    let mut hdr = WriteCursor::new(&mut hdr_buf, 0);
    format::encode_header(&mut hdr);
    hdr.flush(|bytes| output.write_all(bytes))?;

    let mut total_checksum = Adler32::new();

    let pool = if threads > 0 {
        Some(rayon::ThreadPoolBuilder::new().num_threads(threads).build().map_err(std::io::Error::other)?)
    } else {
        None
    };

    loop {
        let n = read_batch(&mut in_buf, pos, BATCH_SIZE - WINDOW_SIZE, input)?;
        if n == 0 {
            break;
        }

        let data_end = pos + n;

        let mut blocks = Vec::with_capacity(n / BLOCK_SIZE + 1);
        let mut block_start = pos;
        while block_start < data_end {
            let block_end = (block_start + BLOCK_SIZE).min(data_end);
            let prefix_start = block_start.saturating_sub(WINDOW_SIZE);
            blocks.push((prefix_start, block_start, block_end));
            block_start = block_end;
        }

        let compress = |&(prefix_start, block_start, block_end): &(usize, usize, usize)| {
            let mut out_buf = Buffer::<OUTPUT_BUF_SIZE>::new();
            let (mut cursor, checksum) =
                compress_block(&in_buf, &mut out_buf, prefix_start, block_start, block_end, level);
            let mut compressed = Vec::with_capacity(OUTPUT_BUF_SIZE);
            cursor
                .flush(|bytes| {
                    compressed.extend_from_slice(bytes);
                    Ok(())
                })
                .expect("flush to Vec cannot fail");
            (compressed, checksum)
        };

        let results: Vec<(Vec<u8>, Adler32)> = pool.as_ref().map_or_else(
            || blocks.par_iter().map(compress).collect(),
            |pool| pool.install(|| blocks.par_iter().map(compress).collect()),
        );

        for (compressed, checksum) in &results {
            output.write_all(compressed)?;
            total_checksum = total_checksum.combine(checksum);
        }

        pos = data_end;
    }

    let mut ftr_buf = Buffer::<WINDOW_SIZE>::new();
    let mut ftr = WriteCursor::new(&mut ftr_buf, 0);
    format::encode_eos(&mut ftr);
    format::encode_footer(pos as u64, total_checksum.checksum(), &mut ftr);
    ftr.flush(|bytes| output.write_all(bytes))?;

    Ok(())
}

/// Reads up to `max` bytes from `input` into `buf` at logical position `start`.
fn read_batch(buf: &mut Buffer<BATCH_SIZE>, start: usize, max: usize, input: &mut impl Read) -> std::io::Result<usize> {
    let mut total = 0;
    while total < max {
        let n = buf.read_from(start + total, max - total, input)?;
        if n == 0 {
            break;
        }
        total += n;
    }
    Ok(total)
}

/// Reads a little-endian `u32` from four consecutive buffer positions.
fn read_u32_le(buf: &Buffer<BATCH_SIZE>, pos: usize) -> u32 {
    u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
}

/// Reads a big-endian `u32` from four consecutive buffer positions.
fn read_u32_be(buf: &Buffer<BATCH_SIZE>, pos: usize) -> u32 {
    u32::from_be_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]])
}

/// Compresses a single block, returning the output cursor and block checksum.
///
/// Input data is read from the shared `input` buffer via circular indexing.
/// `prefix_start..block_start` is context for match references (not emitted).
/// `block_start..block_end` is the data to compress.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn compress_block<'a>(
    input: &Buffer<BATCH_SIZE>,
    out_buf: &'a mut Buffer<OUTPUT_BUF_SIZE>,
    prefix_start: usize,
    block_start: usize,
    block_end: usize,
    level: usize,
) -> (WriteCursor<'a, OUTPUT_BUF_SIZE>, Adler32) {
    let mut cursor = WriteCursor::new(out_buf, 0);
    let block_len = block_end - block_start;

    let mut checksum = Adler32::new();
    let (a, b) = input.slices(block_start, block_len);
    checksum.update(a);
    checksum.update(b);

    // Too small for any match — emit as literals.
    if block_len < 4 {
        if block_len > 0 {
            emit_literals(input, &mut cursor, block_start, block_end);
        }
        return (cursor, checksum);
    }

    // L1 specialization: a much smaller value type (`u16` instead of `Queue`)
    // halves the FxHashMap bucket footprint, which Phase D measurements showed
    // is the dominant cost driver for the L1 fast path.
    if level == 1 {
        compress_block_l1_inner(input, &mut cursor, prefix_start, block_start, block_end);
        return (cursor, checksum);
    }

    let use_reverse = level >= 9;
    let mut forward_map: FxHashMap<u32, Queue> = FxHashMap::with_capacity_and_hasher(WINDOW_SIZE, FxBuildHasher);
    let mut reverse_map: FxHashMap<u32, Queue> = if use_reverse {
        FxHashMap::with_capacity_and_hasher(WINDOW_SIZE, FxBuildHasher)
    } else {
        FxHashMap::default()
    };

    // Phase 1: populate hash maps with prefix positions.
    for p in prefix_start..block_start {
        let key = read_u32_le(input, p);
        forward_map.entry(key).or_insert_with(Queue::new).enqueue(p as u16);
        if use_reverse {
            let rkey = read_u32_be(input, p);
            reverse_map.entry(rkey).or_insert_with(Queue::new).enqueue(p as u16);
        }
    }

    // Phase 2: scan block bytes, find matches, emit frames.
    let mut pos = block_start;
    let mut literal_start = block_start;

    while pos + 4 <= block_end {
        // Escape: stop match-finding when output exceeds input size, emit remainder as literals.
        if cursor.position() > block_len {
            break;
        }

        let key = read_u32_le(input, pos);

        let best =
            find_best_match(input, &forward_map, &reverse_map, key, pos, block_end, prefix_start, level, use_reverse);

        // Insert current position AFTER searching.
        forward_map.entry(key).or_insert_with(Queue::new).enqueue(pos as u16);
        if use_reverse {
            let rkey = read_u32_be(input, pos);
            reverse_map.entry(rkey).or_insert_with(Queue::new).enqueue(pos as u16);
        }

        if let Some((match_len, distance, is_reverse)) = best {
            let match_len_signed = if is_reverse {
                -(match_len as i16)
            } else {
                match_len as i16
            };

            emit_match_frame(input, &mut cursor, literal_start, pos, match_len_signed, distance);

            // Insert skipped positions into hash maps.
            let match_end = pos + match_len;
            for skip in (pos + 1)..match_end {
                if skip + 4 > block_end {
                    break;
                }
                let skey = read_u32_le(input, skip);
                forward_map.entry(skey).or_insert_with(Queue::new).enqueue(skip as u16);
                if use_reverse {
                    let skip_rkey = read_u32_be(input, skip);
                    reverse_map.entry(skip_rkey).or_insert_with(Queue::new).enqueue(skip as u16);
                }
            }

            pos = match_end;
            literal_start = pos;
        } else {
            pos += 1;
        }
    }

    // Phase 3: trailing literals.
    if literal_start < block_end {
        emit_literals(input, &mut cursor, literal_start, block_end);
    }

    (cursor, checksum)
}

/// L1-only main scan with a `FxHashMap<u32, u16>` (no `Queue`, no reverse, no
/// candidate walking). Caller has already initialized the cursor and checksum
/// and verified `block_len >= 4`.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn compress_block_l1_inner(
    input: &Buffer<BATCH_SIZE>,
    cursor: &mut WriteCursor<'_, OUTPUT_BUF_SIZE>,
    prefix_start: usize,
    block_start: usize,
    block_end: usize,
) {
    let block_len = block_end - block_start;
    let mut map: FxHashMap<u32, u16> = FxHashMap::with_capacity_and_hasher(WINDOW_SIZE, FxBuildHasher);

    // Phase 1: prefix walk — store the most recent position per hash key.
    for p in prefix_start..block_start {
        let key = read_u32_le(input, p);
        map.insert(key, p as u16);
    }

    // Phase 2: main scan.
    let mut pos = block_start;
    let mut literal_start = block_start;

    while pos + 4 <= block_end {
        if cursor.position() > block_len {
            break;
        }

        let key = read_u32_le(input, pos);
        let best = find_l1_match(input, &map, key, pos, block_end);

        // Insert AFTER searching so we never match against ourselves.
        map.insert(key, pos as u16);

        if let Some((match_len, distance)) = best {
            emit_match_frame(input, cursor, literal_start, pos, match_len as i16, distance);

            // Insert skipped positions inside the matched span.
            let match_end = pos + match_len;
            for skip in (pos + 1)..match_end {
                if skip + 4 > block_end {
                    break;
                }
                let skey = read_u32_le(input, skip);
                map.insert(skey, skip as u16);
            }

            pos = match_end;
            literal_start = pos;
        } else {
            pos += 1;
        }
    }

    // Phase 3: trailing literals.
    if literal_start < block_end {
        emit_literals(input, cursor, literal_start, block_end);
    }
}

/// L1-only single-candidate forward match: look up the most recent position
/// for `key`, verify the 4-byte hash, and expand forward.
#[allow(clippy::cast_possible_truncation)]
fn find_l1_match(
    input: &Buffer<BATCH_SIZE>,
    map: &FxHashMap<u32, u16>,
    key: u32,
    pos: usize,
    block_end: usize,
) -> Option<(usize, u16)> {
    let candidate_low = *map.get(&key)?;
    let distance = (pos as u16).wrapping_sub(candidate_low);
    if distance == 0 || (distance as usize) > pos {
        return None;
    }
    let cand = pos - distance as usize;
    if read_u32_le(input, cand) != key {
        return None;
    }

    let max_len = (block_end - pos).min(MAX_MATCH_LEN);
    let mut len = 4;
    while len < max_len && input[cand + len] == input[pos + len] {
        len += 1;
    }

    Some((len, distance))
}

/// Selects the best forward or reverse match at `pos`.
#[allow(clippy::too_many_arguments)]
fn find_best_match(
    input: &Buffer<BATCH_SIZE>,
    forward_map: &FxHashMap<u32, Queue>,
    reverse_map: &FxHashMap<u32, Queue>,
    key: u32,
    pos: usize,
    block_end: usize,
    prefix_start: usize,
    level: usize,
    use_reverse: bool,
) -> Option<(usize, u16, bool)> {
    let fwd = find_best_forward_match(input, forward_map, key, pos, block_end, level).map(|(l, d)| (l, d, false));

    if !use_reverse {
        return fwd;
    }

    let rev = find_best_reverse_match(input, reverse_map, key, pos, block_end, prefix_start).map(|(l, d)| (l, d, true));

    // On ties, prefer forward (the last element wins in `max_by_key`).
    [rev, fwd].into_iter().flatten().max_by_key(|&(len, _, _)| len)
}

/// Finds the best forward match at `pos`.
#[allow(clippy::cast_possible_truncation)]
fn find_best_forward_match(
    input: &Buffer<BATCH_SIZE>,
    forward_map: &FxHashMap<u32, Queue>,
    key: u32,
    pos: usize,
    block_end: usize,
    level: usize,
) -> Option<(usize, u16)> {
    let queue = forward_map.get(&key)?;

    // Early exit: if the newest entry is stale, all entries are stale.
    let newest = queue.get(0);
    let newest_dist = (pos as u16).wrapping_sub(newest);
    if newest_dist == 0 || (newest_dist as usize) > pos {
        return None;
    }
    if read_u32_le(input, pos - newest_dist as usize) != key {
        return None;
    }

    let candidates = candidates_for_level(level);
    let max_len = (block_end - pos).min(MAX_MATCH_LEN);
    let mut best_len: usize = 3;
    let mut best_dist: u16 = 0;

    for i in 0..candidates {
        let candidate_low = queue.get(i);
        let distance = (pos as u16).wrapping_sub(candidate_low);
        if distance == 0 || (distance as usize) > pos {
            continue;
        }

        let cand = pos - distance as usize;

        // Newest already verified above; skip redundant check for index 0.
        if i > 0 && read_u32_le(input, cand) != key {
            continue;
        }

        // Expand match forward.
        let mut len = 4;
        while len < max_len && input[cand + len] == input[pos + len] {
            len += 1;
        }

        if len > best_len {
            best_len = len;
            best_dist = distance;
        }
    }

    if best_len >= 4 {
        Some((best_len, best_dist))
    } else {
        None
    }
}

/// Finds the best reverse match at `pos`.
///
/// The reverse map is keyed by `u32::from_be_bytes` at each position.
/// Searching with the forward key (`u32::from_le_bytes` at `pos`) finds
/// candidates where `data[cand..cand+4]` reversed equals `data[pos..pos+4]`.
///
/// Frame distance = `pos - cand - 3` (decoder reads backwards from `cand+3`).
#[allow(clippy::cast_possible_truncation)]
fn find_best_reverse_match(
    input: &Buffer<BATCH_SIZE>,
    reverse_map: &FxHashMap<u32, Queue>,
    forward_key: u32,
    pos: usize,
    block_end: usize,
    prefix_start: usize,
) -> Option<(usize, u16)> {
    let queue = reverse_map.get(&forward_key)?;

    // Early exit: newest stale means all stale.
    let newest = queue.get(0);
    let newest_raw = (pos as u16).wrapping_sub(newest);
    // raw_dist >= 4 ensures frame distance = raw_dist - 3 >= 1.
    if newest_raw < 4 || (newest_raw as usize) > pos {
        return None;
    }
    if read_u32_be(input, pos - newest_raw as usize) != forward_key {
        return None;
    }

    let max_forward = (block_end - pos).min(MAX_MATCH_LEN);
    let mut best_len: usize = 3;
    let mut best_dist: u16 = 0;

    let candidates = candidates_for_level(9);
    for i in 0..candidates {
        let candidate_low = queue.get(i);
        let raw_dist = (pos as u16).wrapping_sub(candidate_low);
        if raw_dist < 4 || (raw_dist as usize) > pos {
            continue;
        }

        let cand = pos - raw_dist as usize;
        if i > 0 && read_u32_be(input, cand) != forward_key {
            continue;
        }

        let frame_distance = raw_dist - 3;

        // Limit length so the decoder's reverse copy doesn't read from buffer
        // slots that have been overwritten by earlier writes in the same copy.
        // The write set {(pos+i) mod N} and read set {(pos-distance-i) mod N}
        // must be disjoint in the N-byte circular buffer, which requires
        // distance + 2*length <= N + 1.
        let window_limit = (WINDOW_SIZE + 1 - frame_distance as usize) / 2;
        if window_limit < 4 {
            continue;
        }

        // Expand: data[cand+3-i] == data[pos+i] for i = 4, 5, ...
        let max_back = cand + 4 - prefix_start.min(cand + 4);
        let max_len = max_forward.min(max_back).min(window_limit);
        if max_len < 4 {
            continue;
        }
        let mut len = 4;
        while len < max_len && input[cand + 3 - len] == input[pos + len] {
            len += 1;
        }

        if len > best_len {
            best_len = len;
            best_dist = frame_distance;
        }
    }

    if best_len >= 4 {
        Some((best_len, best_dist))
    } else {
        None
    }
}

/// Emits a match frame, flushing excess literals as standalone frames first.
#[allow(clippy::cast_possible_truncation)]
fn emit_match_frame(
    input: &Buffer<BATCH_SIZE>,
    cursor: &mut WriteCursor<'_, OUTPUT_BUF_SIZE>,
    literal_start: usize,
    pos: usize,
    match_len: i16,
    distance: u16,
) {
    let mut lit_pos = literal_start;
    let mut remaining = pos - literal_start;

    // Flush literals that exceed the per-frame i16 limit.
    // Distance 1 is used for literal-only frames because distance 0 is
    // reserved for the EOS sentinel (the decoder skips extensions when
    // distance is 0). The distance field is ignored for literal-only frames.
    while remaining > MAX_LIT_LEN {
        format::encode_frame(i16::MAX, 0, 1, cursor);
        for i in lit_pos..lit_pos + MAX_LIT_LEN {
            cursor.write_u8(input[i]);
        }
        lit_pos += MAX_LIT_LEN;
        remaining -= MAX_LIT_LEN;
    }

    // Emit the match frame with remaining literals.
    #[allow(clippy::cast_possible_wrap)]
    let lit_len = remaining as i16;
    format::encode_frame(lit_len, match_len, distance, cursor);
    for i in lit_pos..pos {
        cursor.write_u8(input[i]);
    }
}

/// Emits literal-only frames, splitting into chunks if the count exceeds i16.
///
/// Uses distance 1 (not 0) because distance 0 is reserved for the EOS
/// sentinel — the decoder skips reading extensions when distance is 0.
fn emit_literals(input: &Buffer<BATCH_SIZE>, cursor: &mut WriteCursor<'_, OUTPUT_BUF_SIZE>, start: usize, end: usize) {
    let mut p = start;
    while p < end {
        let chunk = (end - p).min(MAX_LIT_LEN);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        format::encode_frame(chunk as i16, 0, 1, cursor);
        for i in p..p + chunk {
            cursor.write_u8(input[i]);
        }
        p += chunk;
    }
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
        // Data with both forward and reverse patterns.
        let mut data = Vec::new();
        data.extend_from_slice(b"ABCDEFGH");
        data.extend_from_slice(b"ABCDEFGH"); // forward match
        data.extend_from_slice(b"HGFEDCBA"); // reverse match candidate
        roundtrip_level(&data, 9);
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
