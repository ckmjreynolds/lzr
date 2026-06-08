//! Deterministic online byte-pair encoding.
//!
//! The vocabulary is grown *incrementally*: the input is swept in `CHUNKS`
//! passes (1% of the data each), and after each pass the most-frequent adjacent
//! pairs are merged until the vocab reaches a per-pass quota, ramping linearly
//! to `target_vocab`. This is a learning *curriculum* (early data shapes the
//! first merges), not a submission-time adaptation: the learned merge list is
//! fixed, used to tokenize the training data, and shipped in the weight blob so
//! the decoder tokenizes identically. Determinism is the whole point — the
//! neural net sees the same token stream at train and at submission.
//!
//! Tokens `0..256` are the raw bytes; merge `i` creates token id `256 + i` from
//! an ordered pair of existing tokens. Encoding applies the merges in learned
//! order (each a single left-to-right non-overlapping pass), which is the same
//! procedure the learner uses to build its running sequence — so encoding any
//! byte string reproduces the learner's tokenization exactly.

#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
// Shared verbatim between the submission codec (read path: from_bytes/encode/
// expand) and the trainer example (learn/to_bytes via #[path]); each consumer
// leaves the other's entry points unused.
#![allow(dead_code)]

use std::collections::HashMap;

pub(crate) const BASE: usize = 256;
/// 1% passes.
const CHUNKS: usize = 100;
/// Don't create a merge for a pair seen fewer than this many times.
const MIN_FREQ: u32 = 2;

pub(crate) struct Bpe {
    /// merge `i` (→ token id `256+i`) combines this ordered pair of token ids.
    merges: Vec<(u32, u32)>,
    /// token id → its byte expansion (built once at construction).
    expand: Vec<Vec<u8>>,
}

/// Merge every non-overlapping `pair` occurrence into `id`, left to right.
fn apply_merge(seq: &mut Vec<u32>, pair: (u32, u32), id: u32) {
    let (mut w, mut r) = (0usize, 0usize);
    while r < seq.len() {
        if r + 1 < seq.len() && seq[r] == pair.0 && seq[r + 1] == pair.1 {
            seq[w] = id;
            r += 2;
        } else {
            seq[w] = seq[r];
            r += 1;
        }
        w += 1;
    }
    seq.truncate(w);
}

/// Most-frequent adjacent pair, ties broken by smallest `(a,b)` for determinism.
fn best_pair(seq: &[u32]) -> Option<((u32, u32), u32)> {
    if seq.len() < 2 {
        return None;
    }
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    for w in seq.windows(2) {
        *counts.entry((w[0], w[1])).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
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

impl Bpe {
    /// Learn merges from `data`, growing the vocab to (at most) `target_vocab`.
    pub(crate) fn learn(data: &[u8], target_vocab: usize) -> Self {
        let mut merges: Vec<(u32, u32)> = Vec::new();
        let mut seq: Vec<u32> = Vec::new();
        let mut prev_w = 0usize;
        for c in 0..CHUNKS {
            let w = (((c + 1) * data.len()) / CHUNKS).max(prev_w);
            // tokenize the new bytes under the current merges, then append
            let mut chunk: Vec<u32> = data[prev_w..w].iter().map(|&b| u32::from(b)).collect();
            for (mi, &pair) in merges.iter().enumerate() {
                apply_merge(&mut chunk, pair, (BASE + mi) as u32);
            }
            seq.extend_from_slice(&chunk);
            prev_w = w;

            let quota = (BASE + (target_vocab - BASE) * (c + 1) / CHUNKS).min(target_vocab);
            while BASE + merges.len() < quota {
                let Some((pair, cnt)) = best_pair(&seq) else {
                    break;
                };
                if cnt < MIN_FREQ {
                    break;
                }
                let id = (BASE + merges.len()) as u32;
                merges.push(pair);
                apply_merge(&mut seq, pair, id);
            }
            if BASE + merges.len() >= target_vocab {
                break;
            }
        }
        let expand = build_expand(&merges);
        Self { merges, expand }
    }

    pub(crate) fn vocab_size(&self) -> usize {
        BASE + self.merges.len()
    }

    /// Tokenize bytes → token ids by applying the merges in learned order.
    pub(crate) fn encode(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
        for (mi, &pair) in self.merges.iter().enumerate() {
            apply_merge(&mut ids, pair, (BASE + mi) as u32);
        }
        ids
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
        let expand = build_expand(&merges);
        (Self { merges, expand }, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<u8> {
        // repetitive text so merges actually form
        "the quick brown fox the lazy dog the the the quick fox jumps "
            .repeat(200)
            .into_bytes()
    }

    #[test]
    fn roundtrips_and_grows() {
        let data = corpus();
        let bpe = Bpe::learn(&data, 512);
        assert!(bpe.vocab_size() > BASE, "vocab should grow past raw bytes");
        assert!(bpe.vocab_size() <= 512);
        // encode → expand reproduces the input exactly
        let ids = bpe.encode(&data);
        let mut back = Vec::new();
        for id in &ids {
            back.extend_from_slice(bpe.expand(*id));
        }
        assert_eq!(back, data, "encode∘expand must be identity");
        // tokenization actually shortens the stream
        assert!(ids.len() < data.len());
    }

    #[test]
    fn deterministic() {
        let data = corpus();
        let a = Bpe::learn(&data, 400);
        let b = Bpe::learn(&data, 400);
        assert_eq!(a.merges, b.merges, "learning must be deterministic");
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
        // every byte value is a base token, so arbitrary input always encodes
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
