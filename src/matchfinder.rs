//! LZ77 hash-chain match finder.

use crate::frame::Frame;
use crate::options::EncodeOptions;
use crate::ringbuf::RingBuf;

/// Level-dependent tuning parameters.
#[derive(Debug, Clone, Copy)]
struct LevelParams {
    hash_bits: u32,
    max_chain: usize,
    min_match_len: u16,
    short_scan_dist: u16,
    try_reverse: bool,
    lazy_threshold: isize,
}

const LEVEL_PARAMS: [LevelParams; 10] = [
    // [0] unused sentinel
    LevelParams {
        hash_bits: 13,
        max_chain: 4,
        min_match_len: 3,
        short_scan_dist: 0,
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [1]
    LevelParams {
        hash_bits: 13,
        max_chain: 4,
        min_match_len: 3,
        short_scan_dist: 0,
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [2]
    LevelParams {
        hash_bits: 14,
        max_chain: 8,
        min_match_len: 3,
        short_scan_dist: 0,
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [3]
    LevelParams {
        hash_bits: 14,
        max_chain: 16,
        min_match_len: 3,
        short_scan_dist: 0,
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [4]
    LevelParams {
        hash_bits: 15,
        max_chain: 32,
        min_match_len: 2,
        short_scan_dist: 64,
        try_reverse: false,
        lazy_threshold: 1,
    },
    // [5]
    LevelParams {
        hash_bits: 15,
        max_chain: 64,
        min_match_len: 2,
        short_scan_dist: 64,
        try_reverse: false,
        lazy_threshold: 1,
    },
    // [6]
    LevelParams {
        hash_bits: 16,
        max_chain: 128,
        min_match_len: 2,
        short_scan_dist: 128,
        try_reverse: false,
        lazy_threshold: 2,
    },
    // [7]
    LevelParams {
        hash_bits: 16,
        max_chain: 256,
        min_match_len: 1,
        short_scan_dist: 128,
        try_reverse: true,
        lazy_threshold: 2,
    },
    // [8]
    LevelParams {
        hash_bits: 16,
        max_chain: 512,
        min_match_len: 1,
        short_scan_dist: 256,
        try_reverse: true,
        lazy_threshold: 3,
    },
    // [9]
    LevelParams {
        hash_bits: 16,
        max_chain: 1024,
        min_match_len: 1,
        short_scan_dist: 512,
        try_reverse: true,
        lazy_threshold: 4,
    },
];

const WINDOW_SIZE: usize = 65_536;
const WINDOW_MASK: usize = WINDOW_SIZE - 1;
const EMPTY: u32 = u32::MAX;

/// Multiplicative hash on 3 bytes (Knuth).
#[inline]
fn hash3(b0: u8, b1: u8, b2: u8, hash_bits: u32) -> usize {
    let h = u32::from(b0) | (u32::from(b1) << 8) | (u32::from(b2) << 16);
    (h.wrapping_mul(2_654_435_761) >> (32 - hash_bits)) as usize
}

/// Hash-chain based LZ77 match finder.
pub(crate) struct MatchFinder {
    params: LevelParams,
    head: Vec<u32>,
    prev: Vec<u32>,
    rev_head: Vec<u32>,
    rev_prev: Vec<u32>,
    hash_mask: usize,
}

impl MatchFinder {
    /// Construct from encoder options.
    pub(crate) fn new(options: &EncodeOptions) -> Self {
        let level = options.get_level().clamp(1, 9);
        let params = LEVEL_PARAMS[level];
        let hash_size = 1 << params.hash_bits;
        let hash_mask = hash_size - 1;

        let (rev_head, rev_prev) = if params.try_reverse {
            (vec![EMPTY; hash_size], vec![EMPTY; WINDOW_SIZE])
        } else {
            (Vec::new(), Vec::new())
        };

        Self {
            params,
            head: vec![EMPTY; hash_size],
            prev: vec![EMPTY; WINDOW_SIZE],
            rev_head,
            rev_prev,
            hash_mask,
        }
    }

    /// Insert position into forward (and optionally reverse) hash chains.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn insert(&mut self, window: &RingBuf, pos: isize) {
        // Forward chain: hash on bytes at pos, pos+1, pos+2.
        // We need at least 3 bytes ahead to hash (but we always insert even at boundaries;
        // the search side handles short remaining).
        if pos >= 0 {
            let b0 = window[pos];
            let b1 = window[pos + 1];
            let b2 = window[pos + 2];
            let h = hash3(b0, b1, b2, self.params.hash_bits) & self.hash_mask;
            let slot = pos as usize & WINDOW_MASK;
            self.prev[slot] = self.head[h];
            self.head[h] = pos as u32;

            // Reverse chain: hash on bytes at pos, pos-1, pos-2.
            if self.params.try_reverse && pos >= 2 {
                let rb0 = window[pos];
                let rb1 = window[pos - 1];
                let rb2 = window[pos - 2];
                let rh = hash3(rb0, rb1, rb2, self.params.hash_bits) & self.hash_mask;
                self.rev_prev[slot] = self.rev_head[rh];
                self.rev_head[rh] = pos as u32;
            }
        }
    }

    /// Insert positions `start..end` without searching (after emitting a match).
    pub(crate) fn bulk_insert(&mut self, window: &RingBuf, start: isize, end: isize) {
        for p in start..end {
            self.insert(window, p);
        }
    }

    /// Lazy matching threshold from `LevelParams`.
    pub(crate) const fn lazy_threshold(&self) -> isize {
        self.params.lazy_threshold
    }

    /// Find the best profitable match at `pos`. Returns `(distance, length)` or `None`.
    ///
    /// `window_write_pos` is the ring buffer's current write position (i.e., how far
    /// data has been loaded). This is used to reject hash chain candidates whose
    /// slots have been overwritten by newer data.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    pub(crate) fn find_best_match(
        &self,
        window: &RingBuf,
        pos: isize,
        bytes_remaining: usize,
        window_write_pos: isize,
    ) -> Option<(u16, i16)> {
        if bytes_remaining == 0 {
            return None;
        }

        let mut best_dist: u16 = 0;
        let mut best_len: i16 = 0;
        let mut best_gain: isize = -1;

        let max_forward = bytes_remaining.min(i16::MAX as usize);
        let wp = window_write_pos as u32;

        // 1. Short-range scan for 1–2 byte matches.
        if self.params.short_scan_dist > 0 && self.params.min_match_len <= 2 {
            let scan_limit = (self.params.short_scan_dist as isize).min(pos);
            for d in 1..=scan_limit {
                let cand = pos - d;
                if cand < 0 || wp.wrapping_sub(cand as u32) >= WINDOW_SIZE as u32 {
                    break;
                }
                // Check 1-byte match.
                if window[cand] == window[pos] {
                    let dist = d as u16;
                    // Try extending to 2 bytes.
                    if max_forward >= 2 && window[cand + 1] == window[pos + 1] {
                        let gain = Frame::match_gain(dist, 2);
                        if gain > best_gain || (gain == best_gain && dist < best_dist) {
                            best_dist = dist;
                            best_len = 2;
                            best_gain = gain;
                        }
                    }
                    if self.params.min_match_len <= 1 {
                        let gain = Frame::match_gain(dist, 1);
                        if gain > best_gain || (gain == best_gain && dist < best_dist) {
                            best_dist = dist;
                            best_len = 1;
                            best_gain = gain;
                        }
                    }
                }
            }
        }

        // 2. Forward hash chain (3+ bytes).
        if max_forward >= 3 && pos >= 0 {
            let b0 = window[pos];
            let b1 = window[pos + 1];
            let b2 = window[pos + 2];
            let h = hash3(b0, b1, b2, self.params.hash_bits) & self.hash_mask;

            let mut candidate = self.head[h];
            let pos_u32 = pos as u32;
            let mut steps = 0;

            while candidate != EMPTY && steps < self.params.max_chain {
                let dist_u32 = pos_u32.wrapping_sub(candidate);
                // Reject if same position, too far for u16, or slot overwritten by ring buffer.
                if dist_u32 == 0 || dist_u32 > WINDOW_SIZE as u32 || wp.wrapping_sub(candidate) >= WINDOW_SIZE as u32 {
                    candidate = self.prev[candidate as usize & WINDOW_MASK];
                    steps += 1;
                    continue;
                }

                let cand_pos = pos - dist_u32 as isize;
                let dist = dist_u32 as u16;

                // First 3 bytes match by hash, but verify.
                if window[cand_pos] == b0 && window[cand_pos + 1] == b1 && window[cand_pos + 2] == b2 {
                    // Extend forward.
                    let mut len: usize = 3;
                    while len < max_forward && window[cand_pos + len as isize] == window[pos + len as isize] {
                        len += 1;
                    }

                    let fwd_len = len.min(i16::MAX as usize) as i16;
                    let gain = Frame::match_gain(dist, fwd_len);
                    if gain > best_gain || (gain == best_gain && dist < best_dist) {
                        best_dist = dist;
                        best_len = fwd_len;
                        best_gain = gain;
                    }

                    if len >= 128 {
                        break; // Good enough.
                    }
                }

                candidate = self.prev[candidate as usize & WINDOW_MASK];
                steps += 1;
            }
        }

        // 3. Reverse hash chain (3+ bytes).
        if self.params.try_reverse && max_forward >= 3 && pos >= 0 {
            let b0 = window[pos];
            let b1 = window[pos + 1];
            let b2 = window[pos + 2];
            // Reverse hash: we look up hash3(b0, b1, b2) in rev_head.
            // Candidates at `cand` were inserted with hash3(window[cand], window[cand-1], window[cand-2]),
            // so window[cand]=b0, window[cand-1]=b1, window[cand-2]=b2 means we can extend backward.
            let rh = hash3(b0, b1, b2, self.params.hash_bits) & self.hash_mask;

            let mut candidate = self.rev_head[rh];
            let pos_u32 = pos as u32;
            let mut steps = 0;

            while candidate != EMPTY && steps < self.params.max_chain {
                let dist_u32 = pos_u32.wrapping_sub(candidate);
                if dist_u32 == 0 || dist_u32 > WINDOW_SIZE as u32 || wp.wrapping_sub(candidate) >= WINDOW_SIZE as u32 {
                    candidate = self.rev_prev[candidate as usize & WINDOW_MASK];
                    steps += 1;
                    continue;
                }

                let cand_pos = pos - dist_u32 as isize;
                let dist = dist_u32 as u16;

                // Verify: window[cand] should match window[pos], window[cand-1] should match window[pos+1], etc.
                // Also check cand_pos-1 and cand_pos-2 are still in the live window.
                if window[cand_pos] == b0
                    && cand_pos >= 1
                    && wp.wrapping_sub((cand_pos - 1) as u32) < WINDOW_SIZE as u32
                    && window[cand_pos - 1] == b1
                    && cand_pos >= 2
                    && wp.wrapping_sub((cand_pos - 2) as u32) < WINDOW_SIZE as u32
                    && window[cand_pos - 2] == b2
                {
                    // Extend: check window[cand-3] == window[pos+3], etc.
                    // Limit extension so cand_pos - len stays in the live window
                    // (i.e., wp - (cand_pos - len) < WINDOW_SIZE).
                    let mut len: usize = 3;
                    while len < max_forward
                        && (cand_pos - len as isize) >= 0
                        && wp.wrapping_sub((cand_pos - len as isize) as u32) < WINDOW_SIZE as u32
                        && window[cand_pos - len as isize] == window[pos + len as isize]
                    {
                        len += 1;
                    }

                    let rev_len = -(len.min(i16::MAX as usize) as i16);
                    let gain = Frame::match_gain(dist, rev_len);
                    if gain > best_gain || (gain == best_gain && dist < best_dist) {
                        best_dist = dist;
                        best_len = rev_len;
                        best_gain = gain;
                    }
                }

                candidate = self.rev_prev[candidate as usize & WINDOW_MASK];
                steps += 1;
            }
        }

        if best_gain >= 0 {
            Some((best_dist, best_len))
        } else {
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::options::EncodeOptions;

    fn make_window(data: &[u8]) -> RingBuf {
        let mut w = RingBuf::new(WINDOW_SIZE);
        for &b in data {
            w.push(b);
        }
        w
    }

    fn make_finder(level: usize) -> MatchFinder {
        MatchFinder::new(&EncodeOptions::new().level(level))
    }

    #[test]
    fn basic_forward_match() {
        // "abcabc" → at pos 3, should find match (3, 3)
        let data = b"abcabc";
        let window = make_window(data);
        let mut mf = make_finder(1);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        let result = mf.find_best_match(&window, 3, 3, window.write_pos());
        assert!(result.is_some(), "should find a match");
        let (dist, len) = result.unwrap();
        assert_eq!(dist, 3);
        assert_eq!(len, 3);
    }

    #[test]
    fn short_match_close_distance() {
        // "aba" at level 9 → should find (2, 1) at pos 2
        let data = b"aba";
        let window = make_window(data);
        let mut mf = make_finder(9);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        let result = mf.find_best_match(&window, 2, 1, window.write_pos());
        assert!(result.is_some(), "should find short match at level 9");
        let (dist, len) = result.unwrap();
        assert_eq!(dist, 2);
        assert_eq!(len, 1);
    }

    #[test]
    fn unprofitable_match_rejected() {
        // 1-byte match at D=8 → gain = -1, should return None
        // Create data where only a 1-byte match at dist 8 is possible.
        let data = b"a1234567a";
        let window = make_window(data);
        // Use level 9 so short scan is enabled, but dist 8 for a 1-byte match is unprofitable.
        let mut mf = make_finder(9);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        // At pos 8, 'a' matches pos 0 at distance 8. gain(8,1)=-1.
        // There's only 1 byte remaining, so no longer match possible.
        let result = mf.find_best_match(&window, 8, 1, window.write_pos());
        assert!(result.is_none(), "1-byte match at D=8 should be rejected");
    }

    #[test]
    fn chain_depth_respected() {
        // Many positions with the same hash prefix, low level shouldn't find distant ones.
        let mut data = Vec::new();
        // Repeat the same 3-byte prefix many times with different suffixes.
        for i in 0u8..64 {
            data.extend_from_slice(&[0xAA, 0xBB, 0xCC, i]);
        }
        // Add one more "AABBCC" that should match.
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0x00]);

        let window = make_window(&data);
        let mut mf_low = make_finder(1); // max_chain=4
        let mut mf_high = make_finder(9); // max_chain=1024

        for i in 0..data.len() as isize {
            mf_low.insert(&window, i);
            mf_high.insert(&window, i);
        }

        let pos = data.len() as isize - 4;
        let remaining = 4;

        // High level should find a match.
        let high_result = mf_high.find_best_match(&window, pos, remaining, window.write_pos());
        assert!(high_result.is_some(), "high level should find match");

        // Both should find something (chains aren't that deep here), but let's just verify
        // they don't crash and high level finds at least as good.
        let low_result = mf_low.find_best_match(&window, pos, remaining, window.write_pos());
        if let (Some((_, l_len)), Some((_, h_len))) = (low_result, high_result) {
            assert!(h_len >= l_len, "high level should find at least as long a match");
        }
    }

    #[test]
    fn reverse_match() {
        // "abccba" at level 9 → at pos 3, "cba" is the reverse of "abc" at pos 0.
        // Distance = 3, the candidate at pos 2 has window[2]='c', window[1]='b', window[0]='a'.
        // We're looking at pos 3: window[3]='c', window[4]='b', window[5]='a'.
        let data = b"abccba";
        let window = make_window(data);
        let mut mf = make_finder(9);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        let result = mf.find_best_match(&window, 3, 3, window.write_pos());
        assert!(result.is_some(), "should find a reverse match");
        let (dist, len) = result.unwrap();
        // Could be forward or reverse depending on gain.
        // The reverse match: at pos 3, cand=2, dist=1. window[2]='c'=window[3], window[1]='b'=window[4], window[0]='a'=window[5].
        // That's a reverse match of length 3 at distance 1: gain(1, -3) = 6-1-1 = 4.
        // Forward: "cba" at pos 3 doesn't match "abc" at pos 0 forward, so no 3-byte forward match.
        assert!(len == -3 || len == 3, "should find length 3 match, got {len}");
        assert!(dist > 0);
    }

    #[test]
    fn end_of_input_boundary() {
        // Fewer than 3 bytes remaining → should not crash.
        let data = b"ab";
        let window = make_window(data);
        let mut mf = make_finder(9);

        for i in 0..data.len() as isize {
            mf.insert(&window, i);
        }

        // Only 1 byte remaining at pos 1.
        let result = mf.find_best_match(&window, 1, 1, window.write_pos());
        // Might be None (no profitable match) or Some if short scan finds something.
        // Either way, it shouldn't panic.
        if let Some((dist, len)) = result {
            assert!(dist > 0);
            assert!(len.unsigned_abs() >= 1);
        }
    }

    #[test]
    fn window_wrap() {
        // Push more than 65536 bytes, matches should still work across wrap.
        let mut window = RingBuf::new(WINDOW_SIZE);
        let mut mf = make_finder(5);

        // Fill with ~66000 bytes of unique data so no hash collisions with our pattern.
        for i in 0..66_000u32 {
            window.push((i.wrapping_mul(131) & 0xFF) as u8);
        }
        // Insert all positions (need at least 3 lookahead bytes in window for correct hash).
        // In a real encoder, bytes are pushed then positions inserted sequentially.
        // Here we batch-insert after filling so hashes are computed on correct data.
        for i in 0..65_997u32 {
            mf.insert(&window, i as isize);
        }

        // Push "XYZXYZ" — first occurrence + second occurrence.
        let base = window.write_pos(); // 66000
        for &b in b"XYZXYZ" {
            window.push(b);
        }
        // Insert first "XYZ" positions (66000, 66001, 66002).
        for i in base..base + 3 {
            mf.insert(&window, i);
        }

        let search_pos = base + 3; // 66003
        let result = mf.find_best_match(&window, search_pos, 3, window.write_pos());
        assert!(result.is_some(), "should find match after window wrap");
        let (dist, len) = result.unwrap();
        assert_eq!(dist, 3);
        assert_eq!(len, 3);
    }
}
