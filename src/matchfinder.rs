//! LZ77 match finder for the LZR encoder.
//!
//! Maintains four [`FifoHashMap`] tables — medium/long × forward/reverse — and
//! provides a single [`MatchFinder::find_match`] method that returns the best
//! (longest) match at a given position.

use std::ops::Range;

use arbitrary_int::u24;

use crate::buffer::Buffer;
use crate::hashmap::FifoHashMap;
use crate::{BATCH_SIZE, WINDOW_SIZE};

/// Maximum distance encodable in a Medium frame (11 bits → 1..=2,048).
const MEDIUM_CAPACITY: usize = 1 << 11;

/// Maximum distance encodable in a Long frame (16 bits → 1..=65,536).
const LONG_CAPACITY: usize = WINDOW_SIZE;

/// Per-key bucket depth in each hash map.
const BUCKET_CAPACITY: usize = 4;

/// A match found by the match finder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Match {
    /// Positive = forward copy, negative = reverse copy.
    pub length: i16,
    /// Copy distance. `0` means RLE (repeat last byte).
    pub distance: u16,
}

/// LZ77 match finder backed by four [`FifoHashMap`] tables.
///
/// - `medium_fwd` / `medium_rev`: keyed on 2 raw bytes (`u16`), capacity 2,048.
/// - `long_fwd` / `long_rev`: keyed on 3 raw bytes (`u24`), capacity 65,536.
///
/// Reverse maps store keys with bytes in reverse order so that a forward-bytes
/// query at the current position finds source positions whose reversed bytes
/// match.
pub(crate) struct MatchFinder {
    medium_fwd: FifoHashMap<u16, usize>,
    medium_rev: FifoHashMap<u16, usize>,
    long_fwd: FifoHashMap<u24, usize>,
    long_rev: FifoHashMap<u24, usize>,
}

impl MatchFinder {
    /// Creates a new match finder. `level` is accepted for future use but currently ignored.
    pub(crate) fn new(_level: usize) -> Self {
        Self {
            medium_fwd: FifoHashMap::with_capacity(MEDIUM_CAPACITY, BUCKET_CAPACITY),
            medium_rev: FifoHashMap::with_capacity(MEDIUM_CAPACITY, BUCKET_CAPACITY),
            long_fwd: FifoHashMap::with_capacity(LONG_CAPACITY, BUCKET_CAPACITY),
            long_rev: FifoHashMap::with_capacity(LONG_CAPACITY, BUCKET_CAPACITY),
        }
    }

    /// Inserts position `pos` into all four maps using prefix keys derived from `buf`.
    pub(crate) fn insert(&mut self, buf: &Buffer<BATCH_SIZE>, pos: usize) {
        let b0 = buf[pos];
        let b1 = buf[pos + 1];
        let b2 = buf[pos + 2];

        self.medium_fwd.insert(u16::from_le_bytes([b0, b1]), pos, pos);
        self.long_fwd.insert(u24::extract_u32(u32::from_le_bytes([b0, b1, b2, 0]), 0), pos, pos);

        if pos >= 1 {
            let bm1 = buf[pos - 1];
            self.medium_rev.insert(u16::from_le_bytes([b0, bm1]), pos, pos);
            if pos >= 2 {
                self.long_rev.insert(u24::extract_u32(u32::from_le_bytes([b0, bm1, buf[pos - 2], 0]), 0), pos, pos);
            }
        }
    }

    /// Inserts position `pos` into only the long-distance maps.
    fn insert_long(&mut self, buf: &Buffer<BATCH_SIZE>, pos: usize) {
        let b0 = buf[pos];
        let b1 = buf[pos + 1];
        let b2 = buf[pos + 2];

        self.long_fwd.insert(u24::extract_u32(u32::from_le_bytes([b0, b1, b2, 0]), 0), pos, pos);

        if pos >= 2 {
            self.long_rev.insert(
                u24::extract_u32(u32::from_le_bytes([b0, buf[pos - 1], buf[pos - 2], 0]), 0),
                pos,
                pos,
            );
        }
    }

    /// Populates all maps from positions in `range` (typically the window context).
    ///
    /// Only the last [`MEDIUM_CAPACITY`] positions are inserted into the medium
    /// maps, since earlier positions would be immediately evicted.
    pub(crate) fn bulk_load(&mut self, buf: &Buffer<BATCH_SIZE>, range: Range<usize>) {
        let medium_start = range.end.saturating_sub(MEDIUM_CAPACITY).max(range.start);

        for pos in range.start..medium_start {
            self.insert_long(buf, pos);
        }
        for pos in medium_start..range.end {
            self.insert(buf, pos);
        }
    }

    /// Returns the best (longest) match at `pos`, or `None` if no match is found.
    ///
    /// Search order:
    /// 1. RLE (repeat of last byte) — cheapest encoding at any length.
    /// 2. Long maps (3-byte prefix, distance 1–65,536).
    /// 3. Medium maps (2-byte prefix, distance 1–2,048) — only if no 3+ byte match.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    pub(crate) fn find_match(&self, buf: &Buffer<BATCH_SIZE>, pos: usize, end: usize) -> Option<Match> {
        let remaining = end - pos;
        if remaining < 2 {
            return None;
        }
        let max_len = remaining.min(i16::MAX as usize);
        let b0 = buf[pos];
        let b1 = buf[pos + 1];

        // 1. RLE — if current byte equals the last output byte, count the run.
        let (mut best, mut best_len) = if pos >= 1 && b0 == buf[pos - 1] {
            let mut len: usize = 1;
            while len < max_len && buf[pos + len] == b0 {
                len += 1;
            }
            (
                Some(Match {
                    length: len as i16,
                    distance: 0,
                }),
                len,
            )
        } else {
            (None, 0)
        };

        // 2. Long maps (3-byte prefix).
        if remaining >= 3 {
            let b2 = buf[pos + 2];
            let key = u24::extract_u32(u32::from_le_bytes([b0, b1, b2, 0]), 0);
            best_len = scan(self.long_fwd.get(&key), buf, pos, max_len, LONG_CAPACITY, 3, false, best_len, &mut best);
            best_len = scan(self.long_rev.get(&key), buf, pos, max_len, LONG_CAPACITY, 3, true, best_len, &mut best);
        }

        // 3. Medium maps — only if no 3+ byte match found.
        if best_len < 3 {
            let key = u16::from_le_bytes([b0, b1]);
            scan(self.medium_fwd.get(&key), buf, pos, max_len, MEDIUM_CAPACITY, 2, false, best_len, &mut best);
            scan(self.medium_rev.get(&key), buf, pos, max_len, MEDIUM_CAPACITY, 2, true, best_len, &mut best);
        }

        best
    }
}

/// Scans candidate positions from a hash bucket, extending each match and
/// updating `best` when a longer match is found. When `reverse` is true,
/// the source bytes are read backwards (for reverse-copy matches).
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn scan(
    candidates: Option<&[usize]>,
    buf: &Buffer<BATCH_SIZE>,
    pos: usize,
    max_len: usize,
    max_dist: usize,
    prefix: usize,
    reverse: bool,
    mut best_len: usize,
    best: &mut Option<Match>,
) -> usize {
    let Some(bucket) = candidates else {
        return best_len;
    };
    for &cand in bucket {
        let dist = pos - cand;
        if dist == 0 || dist > max_dist {
            continue;
        }
        let mut len = prefix;
        if reverse {
            while len < max_len && buf[pos + len] == buf[cand - len] {
                len += 1;
            }
        } else {
            while len < max_len && buf[pos + len] == buf[cand + len] {
                len += 1;
            }
        }
        if len > best_len {
            let length = if reverse {
                -(len as i16)
            } else {
                len as i16
            };
            *best = Some(Match {
                length,
                distance: dist as u16,
            });
            best_len = len;
        }
    }
    best_len
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn make_buf(data: &[u8], offset: usize) -> Buffer<BATCH_SIZE> {
        let mut buf = Buffer::<BATCH_SIZE>::new();
        buf.copy_from_slice(data, offset);
        buf
    }

    #[test]
    fn rle_detection() {
        let buf = make_buf(&[0xAA; 10], 10);
        let mut mf = MatchFinder::new(9);
        for pos in 10..15 {
            mf.insert(&buf, pos);
        }
        let m = mf.find_match(&buf, 15, 20).unwrap();
        assert_eq!(m.distance, 0);
        assert_eq!(m.length, 5);
    }

    #[test]
    fn forward_match_medium() {
        let data = [0x41, 0x42, 0x43, 0x44, 0x45];
        let mut buf = Buffer::<BATCH_SIZE>::new();
        buf.copy_from_slice(&data, 100);
        buf.copy_from_slice(&data, 200);
        let mut mf = MatchFinder::new(9);
        for pos in 100..105 {
            mf.insert(&buf, pos);
        }
        let m = mf.find_match(&buf, 200, 205).unwrap();
        assert_eq!(m.distance, 100);
        assert_eq!(m.length, 5);
    }

    #[test]
    fn reverse_match() {
        let mut buf = Buffer::<BATCH_SIZE>::new();
        buf.copy_from_slice(&[0x45, 0x44, 0x43, 0x42, 0x41], 96);
        buf.copy_from_slice(&[0x41, 0x42, 0x43, 0x44, 0x45], 200);

        let mut mf = MatchFinder::new(9);
        for pos in 96..=100 {
            mf.insert(&buf, pos);
        }
        let m = mf.find_match(&buf, 200, 205).unwrap();
        assert_eq!(m.distance, 100);
        assert_eq!(m.length, -5);
    }

    #[test]
    fn long_match_at_far_distance() {
        let data = [0x10, 0x20, 0x30, 0x40];
        let mut buf = Buffer::<BATCH_SIZE>::new();
        buf.copy_from_slice(&data, 1000);
        buf.copy_from_slice(&data, 5000);
        let mut mf = MatchFinder::new(9);
        for pos in 1000..1004 {
            mf.insert(&buf, pos);
        }
        let m = mf.find_match(&buf, 5000, 5004).unwrap();
        assert_eq!(m.distance, 4000);
        assert_eq!(m.length, 4);
    }

    #[test]
    fn no_match_unique_bytes() {
        let mut buf = Buffer::<BATCH_SIZE>::new();
        #[allow(clippy::cast_possible_truncation)]
        for i in 0..20_usize {
            buf[100 + i] = i as u8;
            buf[200 + i] = (i + 100) as u8;
        }
        let mut mf = MatchFinder::new(9);
        for pos in 100..110 {
            mf.insert(&buf, pos);
        }
        assert!(mf.find_match(&buf, 200, 210).is_none());
    }

    #[test]
    fn bulk_load_populates_maps() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA];
        let mut buf = Buffer::<BATCH_SIZE>::new();
        buf.copy_from_slice(&data, 500);
        buf.copy_from_slice(&data, 1000);
        let mut mf = MatchFinder::new(9);
        mf.bulk_load(&buf, 500..505);
        let m = mf.find_match(&buf, 1000, 1005).unwrap();
        assert_eq!(m.distance, 500);
        assert_eq!(m.length, 5);
    }

    #[test]
    fn boundary_pos_zero_and_one() {
        let buf = make_buf(&[0x01, 0x02, 0x03], 0);
        let mut mf = MatchFinder::new(9);
        mf.insert(&buf, 0);
        mf.insert(&buf, 1);
        mf.insert(&buf, 2);
    }

    #[test]
    fn rle_wins_over_equal_length_match() {
        let mut buf = Buffer::<BATCH_SIZE>::new();
        buf.copy_from_slice(&[0xBB; 3], 49);
        buf.copy_from_slice(&[0xBB; 4], 99);
        let mut mf = MatchFinder::new(9);
        for pos in 49..52 {
            mf.insert(&buf, pos);
        }
        let m = mf.find_match(&buf, 100, 103).unwrap();
        assert_eq!(m.distance, 0);
        assert_eq!(m.length, 3);
    }
}
