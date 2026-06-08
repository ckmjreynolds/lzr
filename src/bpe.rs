//! Deterministic online byte-pair encoding.
//!
//! Schedule: the first 1 MB of data establishes byte statistics with no merges;
//! then at each 1 MB boundary `MERGES_PER_MB` most-frequent pairs are merged,
//! until the vocab reaches `target` (or the data runs out). Merges are tied to
//! absolute data volume, and the first merges are chosen from a full 1 MB rather
//! than a sliver — a steadier curriculum than ramping over a fixed corpus. It is
//! a *learning* schedule, fixed for the run: the merge list is shipped in the
//! weight blob and the decoder replays it, so determinism is all that matters.
//!
//! Both learning and encoding are near-linear. Encoding is rank-greedy over a
//! linked list with a min-rank heap; the learner maintains adjacent-pair counts
//! and per-pair occurrence lists incrementally so each merge costs only its
//! occurrences, not a full pass. The two agree: greedily applying the
//! lowest-rank (then leftmost) adjacent merge is equivalent to applying merges
//! in rank order as full left-to-right passes, because a merge only ever creates
//! pairs of strictly higher rank than the one just applied.
//!
//! Tokens `0..256` are the raw bytes; merge `i` creates token id `256 + i`.

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
#![allow(clippy::many_single_char_names, clippy::similar_names)]
// Shared verbatim between the submission codec (read path: from_bytes/encode/
// expand) and the trainer example (learn/to_bytes via #[path]); each consumer
// leaves the other's entry points unused.
#![allow(dead_code)]

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

pub(crate) const BASE: usize = 256;
const MB: usize = 1 << 20;
const MERGES_PER_MB: usize = 1000;
const MIN_FREQ: i64 = 2;
const NONE: usize = usize::MAX;

pub(crate) struct Bpe {
    /// merge `i` (→ token id `256+i`) combines this ordered pair of token ids.
    merges: Vec<(u32, u32)>,
    /// pair → merge index (= rank = `id - 256`); the encoder's priority.
    rank: HashMap<(u32, u32), u32>,
    /// token id → its byte expansion (built once at construction).
    expand: Vec<Vec<u8>>,
}

fn build_rank(merges: &[(u32, u32)]) -> HashMap<(u32, u32), u32> {
    merges
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i as u32))
        .collect()
}

fn build_expand(merges: &[(u32, u32)]) -> Vec<Vec<u8>> {
    let mut expand: Vec<Vec<u8>> = (0..BASE as u32).map(|b| vec![b as u8]).collect();
    for &(a, b) in merges {
        let mut e = expand[a as usize].clone();
        e.extend_from_slice(&expand[b as usize]);
        expand.push(e);
    }
    expand
}

/// Near-linear rank-greedy tokenizer: repeatedly merge the lowest-rank (then
/// leftmost) adjacent pair. `O(n + applied·log n)`.
fn encode_with(rank: &HashMap<(u32, u32), u32>, bytes: &[u8]) -> Vec<u32> {
    let n = bytes.len();
    if n == 0 {
        return Vec::new();
    }
    let mut tok: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
    let mut next: Vec<usize> = (1..=n).collect();
    next[n - 1] = NONE;
    let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect(); // prev[0] = NONE
    let mut alive = vec![true; n];
    let mut heap: BinaryHeap<(Reverse<u32>, Reverse<usize>)> = BinaryHeap::new();
    for i in 0..n - 1 {
        if let Some(&r) = rank.get(&(tok[i], tok[i + 1])) {
            heap.push((Reverse(r), Reverse(i)));
        }
    }
    while let Some((Reverse(r), Reverse(i))) = heap.pop() {
        if !alive[i] {
            continue;
        }
        let j = next[i];
        if j == NONE || rank.get(&(tok[i], tok[j])) != Some(&r) {
            continue; // stale: neighbor changed under us
        }
        let id = BASE as u32 + r;
        let (h, k) = (prev[i], next[j]);
        tok[i] = id;
        alive[j] = false;
        next[i] = k;
        if k != NONE {
            prev[k] = i;
        }
        if h != NONE {
            if let Some(&r2) = rank.get(&(tok[h], id)) {
                heap.push((Reverse(r2), Reverse(h)));
            }
        }
        if k != NONE {
            if let Some(&r2) = rank.get(&(id, tok[k])) {
                heap.push((Reverse(r2), Reverse(i)));
            }
        }
    }
    let mut out = Vec::new();
    let mut i = 0usize; // index 0 is never deleted (merges keep the left token)
    loop {
        out.push(tok[i]);
        let nx = next[i];
        if nx == NONE {
            break;
        }
        i = nx;
    }
    out
}

/// Perform up to `quota` merge rounds on `seq`, appending each to `merges`/`rank`.
/// Stops early at `target` vocab or when the best pair falls below `MIN_FREQ`.
/// Incremental: per-pair counts and occurrence lists are updated per merge, so a
/// merge costs only its occurrences. Best pair = max count, ties → smallest pair.
fn run_rounds(
    seq: Vec<u32>,
    merges: &mut Vec<(u32, u32)>,
    rank: &mut HashMap<(u32, u32), u32>,
    quota: usize,
    target: usize,
) {
    let n = seq.len();
    if n < 2 {
        return;
    }
    let mut tok = seq;
    let mut next: Vec<usize> = (1..=n).collect();
    next[n - 1] = NONE;
    let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
    let mut alive = vec![true; n];
    let mut counts: HashMap<(u32, u32), i64> = HashMap::new();
    let mut occ: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let mut heap: BinaryHeap<(i64, Reverse<(u32, u32)>)> = BinaryHeap::new();
    for i in 0..n - 1 {
        let p = (tok[i], tok[i + 1]);
        *counts.entry(p).or_insert(0) += 1;
        occ.entry(p).or_default().push(i);
    }
    for (&p, &c) in &counts {
        heap.push((c, Reverse(p)));
    }

    let mut done = 0;
    while done < quota && BASE + merges.len() < target {
        let pair = loop {
            let Some((c, Reverse(p))) = heap.pop() else {
                return;
            };
            match counts.get(&p) {
                Some(&cur) if cur == c => {
                    if c < MIN_FREQ {
                        return; // true max is below the floor — nothing left worth merging
                    }
                    break p;
                }
                _ => {} // stale snapshot, keep popping
            }
        };
        let id = (BASE + merges.len()) as u32;
        rank.insert(pair, merges.len() as u32);
        merges.push(pair);
        done += 1;

        counts.remove(&pair);
        for i in occ.remove(&pair).unwrap_or_default() {
            if !alive[i] {
                continue;
            }
            let j = next[i];
            if j == NONE || !alive[j] || tok[i] != pair.0 || tok[j] != pair.1 {
                continue; // stale occurrence
            }
            let (h, k) = (prev[i], next[j]);
            if h != NONE {
                if let Some(c) = counts.get_mut(&(tok[h], tok[i])) {
                    *c -= 1;
                }
            }
            if k != NONE {
                if let Some(c) = counts.get_mut(&(tok[j], tok[k])) {
                    *c -= 1;
                }
            }
            tok[i] = id;
            alive[j] = false;
            next[i] = k;
            if k != NONE {
                prev[k] = i;
            }
            if h != NONE {
                let np = (tok[h], id);
                let c = counts.entry(np).or_insert(0);
                *c += 1;
                heap.push((*c, Reverse(np)));
                occ.entry(np).or_default().push(h);
            }
            if k != NONE {
                let np = (id, tok[k]);
                let c = counts.entry(np).or_insert(0);
                *c += 1;
                heap.push((*c, Reverse(np)));
                occ.entry(np).or_default().push(i);
            }
        }
    }
}

impl Bpe {
    /// Learn merges from `data` on the 1 MB / `MERGES_PER_MB` schedule, growing
    /// the vocab to (at most) `target_vocab`.
    pub(crate) fn learn(data: &[u8], target_vocab: usize) -> Self {
        let mut merges: Vec<(u32, u32)> = Vec::new();
        let mut rank: HashMap<(u32, u32), u32> = HashMap::new();
        let mut b = 1;
        while BASE + merges.len() < target_vocab {
            let end = (b * MB).min(data.len());
            let seq = encode_with(&rank, &data[..end]);
            run_rounds(seq, &mut merges, &mut rank, MERGES_PER_MB, target_vocab);
            if end == data.len() {
                break;
            }
            b += 1;
        }
        let expand = build_expand(&merges);
        Self {
            merges,
            rank,
            expand,
        }
    }

    pub(crate) fn vocab_size(&self) -> usize {
        BASE + self.merges.len()
    }

    /// Tokenize bytes → token ids (near-linear).
    pub(crate) fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        encode_with(&self.rank, bytes)
    }

    /// Byte expansion of one token id.
    pub(crate) fn expand(&self, id: u32) -> &[u8] {
        &self.expand[id as usize]
    }

    /// Serialize the merge list: `[u32 n][ (u32 a, u32 b) × n ]`.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.merges.len() * 8);
        out.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        for &(a, b) in &self.merges {
            out.extend_from_slice(&a.to_le_bytes());
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }

    /// Deserialize; returns the `Bpe` and the number of bytes consumed.
    pub(crate) fn from_bytes(blob: &[u8]) -> (Self, usize) {
        let n = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        let mut merges = Vec::with_capacity(n);
        let mut p = 4;
        for _ in 0..n {
            let a = u32::from_le_bytes(blob[p..p + 4].try_into().unwrap());
            let b = u32::from_le_bytes(blob[p + 4..p + 8].try_into().unwrap());
            merges.push((a, b));
            p += 8;
        }
        let rank = build_rank(&merges);
        let expand = build_expand(&merges);
        (
            Self {
                merges,
                rank,
                expand,
            },
            p,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<u8> {
        "the quick brown fox the lazy dog the the the quick fox jumps "
            .repeat(2000)
            .into_bytes()
    }

    /// Reference tokenizer: apply merges in rank order as full left-to-right
    /// non-overlapping passes. The near-linear `encode` must match this.
    fn encode_reference(merges: &[(u32, u32)], bytes: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
        for (mi, &pair) in merges.iter().enumerate() {
            let id = (BASE + mi) as u32;
            let (mut w, mut r) = (0usize, 0usize);
            while r < ids.len() {
                if r + 1 < ids.len() && ids[r] == pair.0 && ids[r + 1] == pair.1 {
                    ids[w] = id;
                    r += 2;
                } else {
                    ids[w] = ids[r];
                    r += 1;
                }
                w += 1;
            }
            ids.truncate(w);
        }
        ids
    }

    #[test]
    fn roundtrips_and_grows() {
        let data = corpus();
        let bpe = Bpe::learn(&data, 512);
        assert!(bpe.vocab_size() > BASE);
        assert!(bpe.vocab_size() <= 512);
        let ids = bpe.encode(&data);
        let mut back = Vec::new();
        for id in &ids {
            back.extend_from_slice(bpe.expand(*id));
        }
        assert_eq!(back, data, "encode∘expand must be identity");
        assert!(ids.len() < data.len());
    }

    #[test]
    fn encode_matches_in_order_reference() {
        let data = corpus();
        let bpe = Bpe::learn(&data, 400);
        // on the training data and on novel bytes
        assert_eq!(bpe.encode(&data), encode_reference(&bpe.merges, &data));
        let novel = b"the lazy quick fox\x00\xff jumps the the dog brown".repeat(7);
        assert_eq!(bpe.encode(&novel), encode_reference(&bpe.merges, &novel));
    }

    #[test]
    fn deterministic() {
        let data = corpus();
        let a = Bpe::learn(&data, 400);
        let b = Bpe::learn(&data, 400);
        assert_eq!(a.merges, b.merges);
        assert_eq!(a.encode(&data), b.encode(&data));
    }

    #[test]
    fn serialize_roundtrips() {
        let data = corpus();
        let bpe = Bpe::learn(&data, 400);
        let blob = bpe.to_bytes();
        let (restored, used) = Bpe::from_bytes(&blob);
        assert_eq!(used, blob.len());
        assert_eq!(restored.merges, bpe.merges);
        assert_eq!(restored.encode(&data), bpe.encode(&data));
    }

    #[test]
    fn encodes_unseen_bytes() {
        let bpe = Bpe::learn(&corpus(), 400);
        let novel: Vec<u8> = (0..=255u8).collect();
        let ids = bpe.encode(&novel);
        let mut back = Vec::new();
        for id in &ids {
            back.extend_from_slice(bpe.expand(*id));
        }
        assert_eq!(back, novel);
    }
}
