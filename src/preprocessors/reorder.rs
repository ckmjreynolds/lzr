//! Article-reorder preprocessor (idea 3, 2026-06-28): permute Wikipedia `<page>`
//! articles into a content-similarity order so the online context models / mixer /
//! `StateMaps` see locally stationary data (better-warmed → fewer bits), then restore
//! the original order on decode by sorting the articles on their in-content page
//! `<id>`.
//!
//! enwik article order is page-id-monotonic (verified on enwik8 AND enwik9), so the
//! permutation ships **free** — the decoder re-derives the original order from the
//! data, with no side-channel. (An explicit permutation would cost ~489 KB ≈ 0.008
//! bpb at enwik9.)
//!
//! Like the other preprocessors this is **enwik-structured-input specific**: it keys
//! on `  <page>\n` / `  </page>\n` / `<id>`. The contract is that the input's
//! complete `<page>` blocks are in ascending-page-`<id>` order (true for enwik8/9);
//! the decode-side inverse always restores by id-sort, so round-trip holds exactly
//! on such input. On input with fewer than two complete `<page>` blocks it is a
//! no-op. The similarity ordering runs on the **encode side only** (the decoder
//! never runs it), so it is free to be expensive and free to change without
//! touching decode — only the id-sort restore must stay fixed.
//!
//! Validated but not yet wired into [`Pipeline::default_pipeline`]: kept as
//! off-by-default infrastructure (like [`super::lz`]), so its API is dead in the
//! shipped binary until a future pipeline integration.
#![allow(dead_code)]

use super::Preprocessor;
use std::collections::HashMap;

const PAGE_OPEN: &[u8] = b"  <page>\n";
const PAGE_CLOSE: &[u8] = b"  </page>\n";

/// Reorders complete `<page>` blocks by content similarity (forward) and restores
/// the original page-id order (inverse). Carries no state — both directions derive
/// everything from the data.
pub(crate) struct Reorder;

/// First occurrence of `needle` in `hay[from..]`.
fn find_from(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= hay.len() {
        return None;
    }
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}

/// Byte spans of complete `  <page>\n` .. `  </page>\n` blocks, in input order.
fn parse_spans(input: &[u8]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0;
    while let Some(a) = find_from(input, PAGE_OPEN, i) {
        let Some(b) = find_from(input, PAGE_CLOSE, a) else {
            break;
        };
        let end = b + PAGE_CLOSE.len();
        spans.push((a, end));
        i = end;
    }
    spans
}

/// The page `<id>` — the first `<id>NNN</id>` in the block (the next one is the
/// revision id). 0 if absent (treated as the smallest, kept for determinism).
fn page_id(block: &[u8]) -> u64 {
    let Some(p) = find_from(block, b"<id>", 0) else {
        return 0;
    };
    let mut j = p + 4;
    let mut v = 0u64;
    while j < block.len() && block[j].is_ascii_digit() {
        v = v * 10 + u64::from(block[j] - b'0');
        j += 1;
    }
    v
}

/// FNV-1a over a byte run, lowercasing ASCII letters in place (`| 0x20` is correct
/// for the alphabetic run this is only ever called on).
fn hash_run(run: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &c in run {
        h = (h ^ u64::from(c | 0x20)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Term frequencies (hashed lowercase `[a-z]{3,}` words) of one block.
fn block_tf(block: &[u8]) -> HashMap<u64, u32> {
    let mut tf = HashMap::new();
    let mut start: Option<usize> = None;
    for (i, &c) in block.iter().enumerate() {
        if c.is_ascii_alphabetic() {
            start.get_or_insert(i);
        } else if let Some(s) = start.take() {
            if i - s >= 3 {
                *tf.entry(hash_run(&block[s..i])).or_insert(0) += 1;
            }
        }
    }
    if let Some(s) = start {
        if block.len() - s >= 3 {
            *tf.entry(hash_run(&block[s..])).or_insert(0) += 1;
        }
    }
    tf
}

/// Greedy nearest-neighbour chain over TF-IDF cosine: start at the highest-mass
/// block, then repeatedly jump to the most-similar unvisited block (candidates
/// found via an inverted index, so only blocks sharing a word are scored).
/// Deterministic (ties broken by smallest index).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn similarity_order(blocks: &[&[u8]]) -> Vec<usize> {
    let n = blocks.len();
    let tfs: Vec<HashMap<u64, u32>> = blocks.iter().map(|b| block_tf(b)).collect();
    let mut df: HashMap<u64, u32> = HashMap::new();
    for tf in &tfs {
        for &w in tf.keys() {
            *df.entry(w).or_insert(0) += 1;
        }
    }
    let nf = n as f32;
    let hi_df = ((0.30 * nf) as u32).max(2); // drop ~stopwords: bounds postings & idf≈0
    // L2-normalised TF-IDF vectors.
    let vecs: Vec<Vec<(u64, f32)>> = tfs
        .iter()
        .map(|tf| {
            let mut v: Vec<(u64, f32)> = tf
                .iter()
                .map(|(&w, &c)| (w, c as f32 * (nf / df[&w] as f32).ln()))
                .collect();
            let norm = v.iter().map(|&(_, x)| x * x).sum::<f32>().sqrt().max(1e-9);
            for e in &mut v {
                e.1 /= norm;
            }
            v
        })
        .collect();
    // Inverted index over discriminative words only (postings ascending by block).
    let mut inv: HashMap<u64, Vec<(u32, f32)>> = HashMap::new();
    for (j, v) in vecs.iter().enumerate() {
        for &(w, x) in v {
            if df[&w] <= hi_df {
                inv.entry(w).or_default().push((j as u32, x));
            }
        }
    }
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut cur = (0..n).max_by_key(|&j| vecs[j].len()).unwrap_or(0);
    visited[cur] = true;
    order.push(cur);
    let mut score = vec![0.0f32; n];
    let mut touched: Vec<u32> = Vec::new();
    for _ in 1..n {
        for &t in &touched {
            score[t as usize] = 0.0;
        }
        touched.clear();
        for &(w, x) in &vecs[cur] {
            if let Some(post) = inv.get(&w) {
                for &(j, xj) in post {
                    let ju = j as usize;
                    if !visited[ju] {
                        if score[ju] == 0.0 {
                            touched.push(j);
                        }
                        score[ju] = x.mul_add(xj, score[ju]);
                    }
                }
            }
        }
        let mut best = usize::MAX;
        let mut bestsc = 0.0f32;
        for &t in &touched {
            let ts = t as usize;
            if score[ts] > bestsc {
                bestsc = score[ts];
                best = ts;
            }
        }
        let nxt = if best == usize::MAX {
            visited.iter().position(|&v| !v).unwrap()
        } else {
            best
        };
        visited[nxt] = true;
        order.push(nxt);
        cur = nxt;
    }
    order
}

/// Reassemble: the bytes before the first block, then the blocks in `order`, then
/// the bytes after the last block.
fn assemble(input: &[u8], spans: &[(usize, usize)], order: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    out.extend_from_slice(&input[..spans[0].0]);
    for &j in order {
        let (a, b) = spans[j];
        out.extend_from_slice(&input[a..b]);
    }
    out.extend_from_slice(&input[spans[spans.len() - 1].1..]);
    out
}

impl Preprocessor for Reorder {
    fn forward(&self, input: &[u8]) -> Vec<u8> {
        let spans = parse_spans(input);
        if spans.len() < 2 {
            return input.to_vec();
        }
        let blocks: Vec<&[u8]> = spans.iter().map(|&(a, b)| &input[a..b]).collect();
        let order = similarity_order(&blocks);
        assemble(input, &spans, &order)
    }

    fn inverse(&self, input: &[u8]) -> Vec<u8> {
        let spans = parse_spans(input);
        if spans.len() < 2 {
            return input.to_vec();
        }
        let mut order: Vec<usize> = (0..spans.len()).collect();
        order.sort_by_key(|&j| page_id(&input[spans[j].0..spans[j].1]));
        assemble(input, &spans, &order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(id: u64, title: &str, body: &str) -> Vec<u8> {
        // Minimal enwik-shaped page: page <id> first, then a revision <id>.
        format!(
            "  <page>\n    <title>{title}</title>\n    <id>{id}</id>\n    \
             <revision>\n      <id>{}</id>\n      <text>{body}</text>\n    \
             </revision>\n  </page>\n",
            id + 1_000_000
        )
        .into_bytes()
    }

    /// inverse(forward(x)) == x for an id-sorted multi-article stream (the enwik
    /// contract): forward permutes blocks by similarity, inverse restores by id.
    #[test]
    fn reorder_roundtrips_synthetic() {
        let mut x = b"<mediawiki>\n  <siteinfo>hdr</siteinfo>\n".to_vec();
        // ids ascending == original order (the verified enwik property)
        x.extend(page(
            10,
            "Apple fruit",
            "apple orange banana fruit sweet tree",
        ));
        x.extend(page(
            11,
            "Linux kernel",
            "kernel linux unix system process memory",
        ));
        x.extend(page(
            12,
            "Banana fruit",
            "banana apple fruit yellow tropical tree",
        ));
        x.extend(page(
            13,
            "Unix system",
            "unix linux kernel system shell process",
        ));
        x.extend(b"</mediawiki>\n");
        let r = Reorder;
        let fwd = r.forward(&x);
        assert_ne!(fwd, x, "similarity order should differ from id order here");
        assert_eq!(
            r.inverse(&fwd),
            x,
            "inverse(forward) must restore the input"
        );
    }

    /// Fewer than two complete blocks → no-op both directions.
    #[test]
    fn reorder_noop_on_thin_input() {
        let r = Reorder;
        let one = page(5, "Solo", "lonely article with no sibling to reorder");
        assert_eq!(r.forward(&one), one);
        assert_eq!(r.inverse(&one), one);
        let none = b"no pages here at all".to_vec();
        assert_eq!(r.forward(&none), none);
    }

    /// Real enwik8 slice: parse a window spanning whole articles, confirm the
    /// page ids are ascending (the assumption), and that inverse(forward)==input.
    #[test]
    #[ignore = "needs assets/enwik8; round-trip on a real article-spanning slice"]
    fn reorder_roundtrips_enwik8() {
        let Ok(e8) = std::fs::read("assets/enwik8") else {
            return;
        };
        let slice = &e8[1_000_000..4_000_000];
        let spans = parse_spans(slice);
        assert!(spans.len() > 10, "expected many articles in a 3 MB slice");
        let ids: Vec<u64> = spans.iter().map(|&(a, b)| page_id(&slice[a..b])).collect();
        assert!(
            ids.windows(2).all(|w| w[0] < w[1]),
            "enwik page ids must be strictly ascending for free id-restore"
        );
        let r = Reorder;
        let fwd = r.forward(slice);
        assert_eq!(r.inverse(&fwd), slice, "enwik8 slice must round-trip");
    }
}
