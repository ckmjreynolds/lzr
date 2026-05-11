//! Hash-chain LZ77 longest-match finder.
//!
//! Used by the Phase-4+ codecs to detect long-range repetition that
//! the per-byte Order-1 model can't see. The matcher is byte-stream
//! oriented: it indexes raw bytes regardless of XML mode, so a
//! Content-mode match record can reference an earlier byte position
//! in the same stream even if that earlier position fell in a
//! different mode.
//!
//! Parameters chosen to match LZR v1's converged settings:
//! - Window: 4 MiB (`1 << 22`). Lookback covers the panel's pre-warm
//!   region with headroom.
//! - Min match: 6 bytes — below this the LZ overhead beats the
//!   per-byte Order-1 cost (~3.8 bpb on Content ≈ 23 bits / 6 bytes,
//!   roughly the cost of a bucketed match record).
//! - Chain depth: 32 — zlib's "fast" preset territory. Optimal-parse
//!   refinement deferred until the basic structure is in place.

#[allow(dead_code)] // Re-exported through public helpers; symbol kept for documentation.
pub(crate) const WINDOW_LOG: usize = 22;
pub(crate) const WINDOW_SIZE: usize = 1 << WINDOW_LOG;
pub(crate) const WINDOW_MASK: usize = WINDOW_SIZE - 1;

const HASH_LOG: usize = 16;
const HASH_SIZE: usize = 1 << HASH_LOG;

pub(crate) const MIN_MATCH: usize = 6;
pub(crate) const MAX_MATCH: usize = MIN_MATCH + 255;
const CHAIN_DEPTH: usize = 32;

const NIL: u32 = u32::MAX;

/// Sentinel for the hash table. Used to detect both "empty bucket"
/// and "end of chain."
pub(crate) struct Matcher {
    hash: Vec<u32>,
    prev: Vec<u32>,
}

impl Matcher {
    pub(crate) fn new() -> Self {
        Self {
            hash: vec![NIL; HASH_SIZE],
            prev: vec![NIL; WINDOW_SIZE],
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn hash3(buf: &[u8], pos: usize) -> usize {
        let h =
            u32::from(buf[pos]) | (u32::from(buf[pos + 1]) << 8) | (u32::from(buf[pos + 2]) << 16);
        let h = h.wrapping_mul(2_654_435_761);
        (h >> (32 - HASH_LOG)) as usize
    }

    /// Record `pos` as the most-recent occurrence of its 3-byte
    /// prefix. Caller guarantees `pos + 3 <= buf.len()`.
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn insert(&mut self, buf: &[u8], pos: usize) {
        let h = Self::hash3(buf, pos);
        self.prev[pos & WINDOW_MASK] = self.hash[h];
        self.hash[h] = pos as u32;
    }

    /// Find the longest in-window match for `buf[pos..]` that clears
    /// [`MIN_MATCH`]. Returns `(offset, length)` or `None`. Offsets
    /// are 1-based: `offset == 1` means "match at position `pos - 1`."
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn find_match(&self, buf: &[u8], pos: usize) -> Option<(u32, u32)> {
        if pos + MIN_MATCH > buf.len() {
            return None;
        }
        let max_len = (buf.len() - pos).min(MAX_MATCH);
        let h = Self::hash3(buf, pos);
        let mut candidate = self.hash[h];
        let mut best_offset: u32 = 0;
        let mut best_length: usize = 0;

        for _ in 0..CHAIN_DEPTH {
            if candidate == NIL || best_length >= max_len {
                break;
            }
            let cpos = candidate as usize;
            if cpos >= pos || pos - cpos > WINDOW_SIZE {
                break;
            }
            // Quick reject: candidate must agree at `best_length` to
            // have any chance of being a new best.
            if best_length > 0 && buf[cpos + best_length] != buf[pos + best_length] {
                candidate = self.prev[cpos & WINDOW_MASK];
                continue;
            }
            let mut len = 0;
            while len < max_len && buf[cpos + len] == buf[pos + len] {
                len += 1;
            }
            if len >= MIN_MATCH && len > best_length {
                best_length = len;
                best_offset = (pos - cpos) as u32;
            }
            candidate = self.prev[cpos & WINDOW_MASK];
        }

        if best_length >= MIN_MATCH {
            Some((best_offset, best_length as u32))
        } else {
            None
        }
    }
}

impl Default for Matcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_finds_repeated_prefix() {
        let buf = b"hello world, hello world!"; // second "hello world" at offset 13
        let mut m = Matcher::new();
        for i in 0..13 - 2 {
            m.insert(buf, i);
        }
        // Looking for a match at position 13 ("hello world!").
        let result = m.find_match(buf, 13);
        let (offset, length) = result.expect("expected a match");
        assert_eq!(offset, 13);
        assert!(length >= u32::try_from(MIN_MATCH).unwrap());
    }

    #[test]
    fn matcher_finds_no_match_for_unique_prefix() {
        let buf = b"abcdefghijklmnopqrstuvwxyz"; // every 3-byte prefix unique
        let mut m = Matcher::new();
        for i in 0..buf.len() - 3 {
            m.insert(buf, i);
        }
        // Looking past where we've inserted; should miss because no
        // 3-byte prefix repeats.
        let result = m.find_match(buf, buf.len() - MIN_MATCH);
        assert!(result.is_none());
    }
}
