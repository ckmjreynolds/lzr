//! LZ-factoring preprocessor (optional, off by default).
//!
//! Replaces sufficiently long repeats with a compact `(length, distance)` token
//! so the downstream byte stream is shorter — fewer bytes for the CM coder (and
//! the online arm) to process. Whether this is a net win is corpus- and
//! placement-dependent: in a codec whose match model already prices long-range
//! redundancy at near-zero bits, factoring it out re-exposes the distance
//! entropy the match model hides, so this is kept as a pluggable stage
//! (`Pipeline`) rather than shipped on. The 2026-06-24 journal entry records the
//! measured trade.
//!
//! ## Token format
//!
//! A single escape byte `esc` introduces every token; a literal `esc` in the
//! data is stuffed as `esc, STUFF`. The byte after `esc` is an op selecting the
//! distance source and the byte-widths of the length/distance fields (so the
//! decoder knows how many bytes to read — the "different codes for different L/D
//! length combinations" scheme):
//!
//! - `op == STUFF (0)` — a literal `esc`.
//! - `op in 1..=8` — repeat-distance match: `rep_index = (op-1) >> 1` selects one
//!   of four cached recent distances (no distance bytes), `len_bytes =
//!   ((op-1) & 1) + 1`. Followed by the length field.
//! - `op in 9..=16` — explicit-distance match: `k = op-9`, `len_bytes = (k>>2)+1`
//!   (1..2), `dist_bytes = (k&3)+1` (1..4). Followed by the length field then the
//!   distance field, both little-endian.
//!
//! The length field holds `match_len - min_match` (so it starts at 0). The
//! distance cache is move-to-front on a hit and push-front on a new distance,
//! maintained identically on both sides, so recurring distances collapse to a
//! one-or-two-byte token.

// A pluggable stage not currently in `Pipeline::default_pipeline` (the 2026-06-24
// measurements show it is not a net win on enwik8); kept ready to add/remove, so
// its surface is "unused" in the production build until then.
#![allow(dead_code)]

use super::Preprocessor;

const STUFF: u8 = 0;
const REP_SLOTS: usize = 4;
const HASH_BITS: u32 = 22;
const MAX_CHAIN: usize = 64; // chain-search depth cap (match quality vs. speed)
const LEN_VALUE_CAP: usize = 0xFFFF; // max `len - min_match` (fits two bytes)

/// Reversible LZ-factoring stage. `esc` must be chosen so stuffing is rare;
/// `min_match` is the length threshold below which repeats stay literal.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lz {
    esc: u8,
    min_match: usize,
}

impl Lz {
    pub(crate) const fn new(esc: u8, min_match: usize) -> Self {
        Self { esc, min_match }
    }
}

const fn hash4(input: &[u8], p: usize) -> usize {
    let x = u32::from_le_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
    (x.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
}

/// Longest match for the data at `p` against earlier positions, via the hash
/// chain. Returns `(len, dist)` with the smallest distance among equal lengths
/// (cheaper to encode), or `(0, 0)` if none reaches `min_match`.
fn find_match(
    input: &[u8],
    p: usize,
    head: &[u32],
    prev: &[u32],
    min_match: usize,
) -> (usize, usize) {
    let n = input.len();
    if p + 4 > n {
        return (0, 0);
    }
    let max_len = (n - p).min(min_match + LEN_VALUE_CAP);
    let (mut best_len, mut best_dist) = (0usize, 0usize);
    let mut cand = head[hash4(input, p)];
    let mut depth = 0;
    while cand != u32::MAX && depth < MAX_CHAIN {
        let j = cand as usize;
        // Only bother if this candidate can beat the incumbent at its boundary.
        if best_len == 0 || (p + best_len < n && input[j + best_len] == input[p + best_len]) {
            let mut l = 0;
            while l < max_len && input[j + l] == input[p + l] {
                l += 1;
            }
            if l > best_len {
                best_len = l;
                best_dist = p - j;
                if l == max_len {
                    break;
                }
            }
        }
        cand = prev[j];
        depth += 1;
    }
    if best_len >= min_match {
        (best_len, best_dist)
    } else {
        (0, 0)
    }
}

#[allow(clippy::cast_possible_truncation)]
const fn insert(input: &[u8], p: usize, head: &mut [u32], prev: &mut [u32]) {
    if p + 4 <= input.len() {
        let h = hash4(input, p);
        prev[p] = head[h];
        head[h] = p as u32;
    }
}

const fn dist_bytes(dist: usize) -> usize {
    match dist {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        0x1_0000..=0x00FF_FFFF => 3,
        _ => 4,
    }
}

#[allow(clippy::cast_possible_truncation)]
fn push_le(out: &mut Vec<u8>, value: u32, nbytes: usize) {
    for b in 0..nbytes {
        out.push((value >> (8 * b)) as u8);
    }
}

fn read_le(buf: &[u8], nbytes: usize) -> u32 {
    let mut v = 0u32;
    for (b, &byte) in buf[..nbytes].iter().enumerate() {
        v |= u32::from(byte) << (8 * b);
    }
    v
}

fn mtf(rep: &mut [u32; REP_SLOTS], k: usize) {
    let d = rep[k];
    for j in (1..=k).rev() {
        rep[j] = rep[j - 1];
    }
    rep[0] = d;
}

fn rep_push(rep: &mut [u32; REP_SLOTS], d: u32) {
    for j in (1..REP_SLOTS).rev() {
        rep[j] = rep[j - 1];
    }
    rep[0] = d;
}

impl Preprocessor for Lz {
    #[allow(clippy::cast_possible_truncation)]
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        let n = input.len();
        let (esc, mm) = (self.esc, self.min_match);
        let mut out = Vec::with_capacity(n);
        let mut head = vec![u32::MAX; 1 << HASH_BITS];
        let mut prev = vec![u32::MAX; n.max(1)];
        let mut rep = [0u32; REP_SLOTS];

        let mut i = 0;
        while i < n {
            let (len, dist) = find_match(input, i, &head, &prev, mm);
            if len >= mm {
                let value = (len - mm) as u32;
                let len_bytes = if value <= 0xFF { 1 } else { 2 };
                out.push(esc);
                let rep_idx = rep.iter().position(|&d| d == dist as u32);
                if let Some(k) = rep_idx {
                    out.push(1 + (k as u8) * 2 + (len_bytes as u8 - 1));
                    push_le(&mut out, value, len_bytes);
                    mtf(&mut rep, k);
                } else {
                    let db = dist_bytes(dist);
                    out.push(9 + (len_bytes as u8 - 1) * 4 + (db as u8 - 1));
                    push_le(&mut out, value, len_bytes);
                    push_le(&mut out, dist as u32, db);
                    rep_push(&mut rep, dist as u32);
                }
                let end = (i + len).min(n);
                for p in i..end {
                    insert(input, p, &mut head, &mut prev);
                }
                i = end;
            } else {
                insert(input, i, &mut head, &mut prev);
                if input[i] == esc {
                    out.push(esc);
                    out.push(STUFF);
                } else {
                    out.push(input[i]);
                }
                i += 1;
            }
        }
        out
    }

    fn inverse(&self, input: &[u8]) -> Vec<u8> {
        let (esc, mm) = (self.esc, self.min_match);
        let mut out = Vec::with_capacity(input.len() * 2);
        let mut rep = [0u32; REP_SLOTS];
        let mut i = 0;
        while i < input.len() {
            let b = input[i];
            if b != esc {
                out.push(b);
                i += 1;
                continue;
            }
            i += 1;
            let op = input[i];
            i += 1;
            if op == STUFF {
                out.push(esc);
                continue;
            }
            let (len, dist) = if op <= 8 {
                let k = ((op - 1) >> 1) as usize;
                let len_bytes = ((op - 1) & 1) as usize + 1;
                let value = read_le(&input[i..], len_bytes);
                i += len_bytes;
                let dist = rep[k];
                mtf(&mut rep, k);
                (value as usize + mm, dist as usize)
            } else {
                let k = op - 9;
                let len_bytes = (k >> 2) as usize + 1;
                let db = (k & 3) as usize + 1;
                let value = read_le(&input[i..], len_bytes);
                i += len_bytes;
                let dist = read_le(&input[i..], db);
                i += db;
                rep_push(&mut rep, dist);
                (value as usize + mm, dist as usize)
            };
            let start = out.len() - dist;
            for k in 0..len {
                let c = out[start + k];
                out.push(c);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8], esc: u8, mm: usize) {
        let lz = Lz::new(esc, mm);
        let fwd = lz.forward(data);
        assert_eq!(lz.inverse(&fwd), data, "lz round-trip (mm={mm})");
    }

    #[test]
    fn roundtrips_basic() {
        roundtrip(b"", 0x02, 8);
        roundtrip(b"abc", 0x02, 8);
        roundtrip(&b"the quick brown fox ".repeat(50), 0x02, 8);
        // a literal escape byte must survive via stuffing
        roundtrip(b"a\x02b\x02\x02c the the the the the the", 0x02, 4);
        // long repeat exercises rep-distance + multi-byte length/distance
        let mut data = b"Lorem ipsum dolor sit amet ".repeat(200);
        data.extend_from_slice(&b"Lorem ipsum dolor sit amet ".repeat(200));
        roundtrip(&data, 0x02, 16);
    }

    #[test]
    fn roundtrips_enwik8_slice() {
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        for mm in [8usize, 16, 32, 64] {
            roundtrip(&e8[3_000_000..3_400_000], 0x1F, mm);
        }
    }
}
