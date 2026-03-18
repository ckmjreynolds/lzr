//! LZ77 match finder.

use crate::WINDOW_SIZE;
use crate::frame::Frame;
use crate::options::EncodeOptions;
use crate::ringbuf::RingBuf;

/// Level-dependent tuning parameters.
#[derive(Debug, Clone, Copy)]
struct LevelParams {
    try_reverse: bool,
    lazy_threshold: isize,
}

const LEVEL_PARAMS: [LevelParams; 10] = [
    // [0] unused sentinel
    LevelParams {
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [1]
    LevelParams {
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [2]
    LevelParams {
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [3]
    LevelParams {
        try_reverse: false,
        lazy_threshold: 0,
    },
    // [4]
    LevelParams {
        try_reverse: false,
        lazy_threshold: 1,
    },
    // [5]
    LevelParams {
        try_reverse: false,
        lazy_threshold: 1,
    },
    // [6]
    LevelParams {
        try_reverse: false,
        lazy_threshold: 2,
    },
    // [7]
    LevelParams {
        try_reverse: true,
        lazy_threshold: 2,
    },
    // [8]
    LevelParams {
        try_reverse: true,
        lazy_threshold: 3,
    },
    // [9]
    LevelParams {
        try_reverse: true,
        lazy_threshold: 4,
    },
];

/// Exhaustive brute-force match finder.
///
/// Checks every position in the window for matches. `O(n × window_size)`.
pub(crate) struct MatchFinder {
    params: LevelParams,
}

impl MatchFinder {
    /// Construct from encoder options.
    pub(crate) fn new(options: &EncodeOptions) -> Self {
        let level = options.get_level().clamp(1, 9);
        Self {
            params: LEVEL_PARAMS[level],
        }
    }

    /// Lazy matching threshold from level params.
    pub(crate) const fn lazy_threshold(&self) -> isize {
        self.params.lazy_threshold
    }

    /// Find the best profitable match at `pos`. Returns `(distance, length)` or `None`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::too_many_lines
    )]
    pub(crate) fn find_best_match(&self, window: &RingBuf, pos: isize, bytes_remaining: usize) -> Option<(u32, i32)> {
        if bytes_remaining == 0 || pos <= 0 {
            return None;
        }

        let max_forward = bytes_remaining;
        let wp = window.write_pos();
        let min_cand = 0isize.max(pos - WINDOW_SIZE as isize + 1).max(wp - WINDOW_SIZE as isize);

        let mut best_dist: u32 = 0;
        let mut best_len: i32 = 0;
        let mut best_gain: isize = -1;

        let target_byte = window[pos];

        for cand_pos in (min_cand..pos).rev() {
            if window[cand_pos] != target_byte {
                continue;
            }

            let dist = (pos - cand_pos) as u32;

            // Forward match.
            let fwd_len = 1 + window.match_length(cand_pos + 1, pos + 1, max_forward - 1);
            if fwd_len >= 2 {
                let fwd_len_i32 = fwd_len as i32;
                let fwd_gain = Frame::match_gain(dist, fwd_len_i32);
                if fwd_gain > best_gain || (fwd_gain == best_gain && dist < best_dist) {
                    best_dist = dist;
                    best_len = fwd_len_i32;
                    best_gain = fwd_gain;
                }
            }

            // Reverse match.
            if self.params.try_reverse {
                let mut rev_len: usize = 1;
                while rev_len < max_forward
                    && (cand_pos - rev_len as isize) >= min_cand
                    && window[cand_pos - rev_len as isize] == window[pos + rev_len as isize]
                {
                    rev_len += 1;
                }

                if rev_len >= 2 {
                    let rev_len_i32 = -(rev_len as i32);
                    let rev_gain = Frame::match_gain(dist, rev_len_i32);
                    if rev_gain > best_gain || (rev_gain == best_gain && dist < best_dist) {
                        best_dist = dist;
                        best_len = rev_len_i32;
                        best_gain = rev_gain;
                    }
                }
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
        let window = make_window(b"abcabc");
        let mf = make_finder(1);
        let result = mf.find_best_match(&window, 3, 3);
        assert!(result.is_some());
        let (dist, len) = result.unwrap();
        assert_eq!(dist, 3);
        assert_eq!(len, 3);
    }

    #[test]
    fn no_match_for_single_byte_remaining() {
        let window = make_window(b"aba");
        let mf = make_finder(9);
        let result = mf.find_best_match(&window, 2, 1);
        assert!(result.is_none());
    }

    #[test]
    fn reverse_match() {
        let window = make_window(b"abccba");
        let mf = make_finder(9);
        let result = mf.find_best_match(&window, 3, 3);
        assert!(result.is_some());
        let (dist, len) = result.unwrap();
        assert_eq!(dist, 1);
        assert_eq!(len, -3);
    }

    #[test]
    fn uncapped_match_length() {
        let data = vec![0xAA; 100];
        let window = make_window(&data);
        let mf = make_finder(1);
        let result = mf.find_best_match(&window, 1, 99);
        assert!(result.is_some());
        let (_, len) = result.unwrap();
        assert_eq!(len, 99);
    }

    #[test]
    fn end_of_input_boundary() {
        let window = make_window(b"ab");
        let mf = make_finder(9);
        // Only 1 byte remaining — can't form a 2-byte match.
        let result = mf.find_best_match(&window, 1, 1);
        assert!(result.is_none());
    }

    #[test]
    fn window_wrap() {
        let mut window = RingBuf::new(WINDOW_SIZE);
        let mf = make_finder(5);
        for i in 0..66_000u32 {
            window.push((i.wrapping_mul(131) & 0xFF) as u8);
        }
        let base = window.write_pos();
        for &b in b"XYZXYZ" {
            window.push(b);
        }
        let search_pos = base + 3;
        let result = mf.find_best_match(&window, search_pos, 3);
        assert!(result.is_some());
        let (dist, len) = result.unwrap();
        assert_eq!(dist, 3);
        assert_eq!(len, 3);
    }
}
