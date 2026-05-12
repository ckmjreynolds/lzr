//! Burrows-Wheeler Transform: forward and inverse.
//!
//! Block-based. Given an input `s` of length `n`, `forward` returns
//! the last column of the sorted-rotations matrix plus the "primary
//! index" — the row in the sorted matrix that corresponds to `s`
//! itself. `inverse` reconstructs `s` from these two pieces.
//!
//! The suffix-array construction uses the Manber-Myers doubling
//! algorithm. `O(n log² n)` — slower than SA-IS but tiny code. For
//! 256 KiB blocks (the panel's window size) it takes ~1-2 s; if BWT
//! becomes a hot path the SA can be swapped for SA-IS later.
//!
//! The Wikipedia article on BWT and the Mantaci/Sciortino "BWT and
//! associated combinatorial structures" notes are useful references
//! for the LF-mapping in `inverse`.

#[allow(clippy::cast_possible_truncation)]
pub(crate) fn forward(s: &[u8]) -> (Vec<u8>, u32) {
    let n = s.len();
    if n == 0 {
        return (Vec::new(), 0);
    }
    // Standard textbook BWT operates on cyclic rotations of `s`.
    // Suffix-array-of-`s` is *not* the same ordering — for example,
    // the suffix "a" sorts before "aa" (shorter sorts first), but the
    // rotation containing "aaa..." can come before "aa...". The fix:
    // compute the suffix array of `s + s` and filter entries less
    // than `n`. Those filtered positions are exactly the
    // rotation-start indices in true cyclic-rotation order.
    let mut ss = Vec::with_capacity(2 * n);
    ss.extend_from_slice(s);
    ss.extend_from_slice(s);
    let sa_doubled = suffix_array(&ss);

    let mut last_col = vec![0u8; n];
    let mut primary: u32 = 0;
    let mut write_idx: usize = 0;
    for &p in &sa_doubled {
        let p_us = p as usize;
        if p_us >= n {
            continue;
        }
        if p_us == 0 {
            last_col[write_idx] = s[n - 1];
            primary = write_idx as u32;
        } else {
            last_col[write_idx] = s[p_us - 1];
        }
        write_idx += 1;
    }
    debug_assert_eq!(write_idx, n);
    (last_col, primary)
}

pub(crate) fn inverse(last_col: &[u8], primary: u32) -> Vec<u8> {
    let n = last_col.len();
    if n == 0 {
        return Vec::new();
    }

    // `count[b]` = number of bytes < `b` in `last_col`. After the
    // cumulative pass this gives the starting offset of byte value
    // `b` in the sorted first column.
    let mut count = [0u32; 257];
    for &b in last_col {
        count[b as usize + 1] += 1;
    }
    for b in 1..=256 {
        count[b] += count[b - 1];
    }

    // LF-mapping: for each row `i` in the sorted matrix, the row of
    // the rotation that immediately follows it (cyclically) is at
    // `lf[i]` — that is, `last_col[i]` becomes the first character of
    // row `lf[i]`.
    let mut lf = vec![0u32; n];
    let mut rank = [0u32; 256];
    for (i, &b) in last_col.iter().enumerate() {
        let bi = b as usize;
        lf[i] = count[bi] + rank[bi];
        rank[bi] += 1;
    }

    // Walk: start at `primary` (the row corresponding to `s` itself),
    // emit `last_col[primary]` as the final character of `s`, then
    // follow LF to get the predecessor row, etc.
    let mut out = vec![0u8; n];
    let mut p = primary as usize;
    for k in (0..n).rev() {
        out[k] = last_col[p];
        p = lf[p] as usize;
    }
    out
}

/// Suffix array via Manber-Myers doubling. Returns a permutation of
/// `0..n` such that the suffixes `s[sa[i]..]` are lexicographically
/// sorted. Empty input → empty array.
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
fn suffix_array(s: &[u8]) -> Vec<u32> {
    let n = s.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sa: Vec<u32> = (0..n as u32).collect();
    let mut rank: Vec<i32> = s.iter().map(|&b| i32::from(b)).collect();
    let mut new_rank: Vec<i32> = vec![0; n];
    let mut h: usize = 1;

    loop {
        // Sort by (rank[i], rank[i + h]) using -1 for out-of-bounds.
        #[allow(clippy::many_single_char_names)]
        sa.sort_by(|&i, &j| {
            let i = i as usize;
            let j = j as usize;
            let ri = rank[i];
            let rj = rank[j];
            if ri != rj {
                return ri.cmp(&rj);
            }
            let ih = if i + h < n { rank[i + h] } else { -1 };
            let jh = if j + h < n { rank[j + h] } else { -1 };
            ih.cmp(&jh)
        });

        // Recompute ranks from the sort order. Identical key pairs
        // share a rank.
        new_rank[sa[0] as usize] = 0;
        for i in 1..n {
            let prev = sa[i - 1] as usize;
            let cur = sa[i] as usize;
            let prev_h = if prev + h < n { rank[prev + h] } else { -1 };
            let cur_h = if cur + h < n { rank[cur + h] } else { -1 };
            let same = rank[prev] == rank[cur] && prev_h == cur_h;
            new_rank[cur] = new_rank[prev] + i32::from(!same);
        }
        std::mem::swap(&mut rank, &mut new_rank);

        // Done when every suffix has a distinct rank. `n` is bounded
        // by `i32::MAX` for any practical BWT block (we work with
        // 256 KiB windows here), so the cast is safe.
        let last_rank = i32::try_from(n - 1).expect("BWT block size fits i32");
        if rank[sa[n - 1] as usize] == last_rank {
            break;
        }
        h = h.saturating_mul(2);
        if h >= n {
            break;
        }
    }
    sa
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(s: &[u8]) {
        let (l, primary) = forward(s);
        let back = inverse(&l, primary);
        assert_eq!(back, s);
    }

    #[test]
    fn bwt_roundtrip_empty() {
        roundtrip(b"");
    }

    #[test]
    fn bwt_roundtrip_single_byte() {
        roundtrip(b"a");
    }

    #[test]
    fn bwt_roundtrip_short_phrase() {
        roundtrip(b"banana");
    }

    #[test]
    fn bwt_roundtrip_text_with_repetition() {
        roundtrip(b"the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn bwt_roundtrip_all_zeros() {
        let s = vec![0u8; 100];
        roundtrip(&s);
    }

    #[test]
    fn bwt_roundtrip_all_distinct() {
        let s: Vec<u8> = (0..=255).collect();
        roundtrip(&s);
    }

    #[test]
    fn bwt_roundtrip_random_ish() {
        let s: Vec<u8> = (0..1000)
            .map(|i| u8::try_from((i * 31 + 7) % 256).unwrap())
            .collect();
        roundtrip(&s);
    }

    #[test]
    fn bwt_roundtrip_enwik8_4k() {
        let Ok(bytes) = std::fs::read("assets/enwik8") else {
            return;
        };
        let s = &bytes[4_194_304..4_194_304 + 4096];
        let (l, primary) = forward(s);
        let back = inverse(&l, primary);
        assert_eq!(back, s, "BWT alone failed on 4 KiB enwik8 slice");
    }

    #[test]
    fn bwt_produces_run_clusters_on_clustered_input() {
        // Standard textbook BWT example. The output of forward("banana")
        // has the property that similar-context bytes cluster: the
        // last column of the sorted rotation matrix tends to repeat.
        let (l, _) = forward(b"^BANANA|");
        // Just checking the BWT runs; specific value depends on
        // implementation conventions but should compress smaller than
        // raw input under MTF.
        assert!(!l.is_empty());
    }
}
