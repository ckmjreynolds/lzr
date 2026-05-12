//! Byte-pair encoding for the v3 capability survey.
//!
//! Standard "fast" BPE: pre-split the corpus on letter / non-letter
//! boundaries (mirroring xml-tok's tokenizer to keep apples-to-apples
//! comparisons), aggregate to unique word frequencies, then greedily
//! merge the most-common adjacent token pair until the vocab reaches
//! the target size. Operating on unique words weighted by frequency
//! reduces per-merge cost from `O(corpus)` to `O(unique_words)`.
//!
//! Find-max is accelerated by a lazy max-heap: when a merge changes
//! a pair's count, the heap entry isn't updated in place — instead
//! a fresh `(new_count, pair)` is pushed. When popping, stale
//! entries (current count doesn't match) are skipped. Net: each
//! merge is `O(log H + affected_pairs)` instead of `O(|pair_counts|)`.
//!
//! `train` returns a sequence of `Merge` records describing each
//! merge in order. `tokenize` applies them to a byte slice, yielding
//! a `Vec<u32>` of token ids.
//!
//! Token ids 0..=255 are reserved for byte-level fallbacks. Merge
//! ids start at 256.

use std::collections::{BinaryHeap, HashMap, HashSet};

/// One BPE merge operation: a pair `(left, right)` of existing token
/// ids becomes the new token id `new`. Applied in order during
/// tokenization.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Merge {
    pub left: u32,
    pub right: u32,
    pub new: u32,
}

/// Train a BPE tokenizer on `corpus` until the vocab (including the
/// 256 base byte tokens) reaches `vocab_size`. Returns the merge
/// sequence — apply via `tokenize` to encode any byte slice.
pub(crate) fn train(corpus: &[u8], vocab_size: usize) -> Vec<Merge> {
    if vocab_size <= 256 {
        return Vec::new();
    }

    // Pre-split on letter / non-letter runs, identical to xml-tok's
    // tokenizer. Each pre-token is a sequence of byte-token ids that
    // BPE will progressively merge within.
    let mut word_freq: HashMap<Vec<u32>, u64> = HashMap::new();
    let mut p = 0;
    while p < corpus.len() {
        let is_letter = corpus[p].is_ascii_alphabetic();
        let mut q = p + 1;
        while q < corpus.len() && corpus[q].is_ascii_alphabetic() == is_letter {
            q += 1;
        }
        let bytes: Vec<u32> = corpus[p..q].iter().map(|&b| u32::from(b)).collect();
        *word_freq.entry(bytes).or_insert(0) += 1;
        p = q;
    }

    // Initial pair counts across all unique words, weighted by
    // word frequency.
    let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
    for (word, &freq) in &word_freq {
        for w in word.windows(2) {
            *pair_counts.entry((w[0], w[1])).or_insert(0) += freq;
        }
    }

    // Lazy max-heap of (count, pair). Stale entries (where the
    // current pair_counts disagrees with the heap's count) are
    // skipped on pop.
    let mut heap: BinaryHeap<(u64, (u32, u32))> = BinaryHeap::with_capacity(pair_counts.len());
    for (&pair, &count) in &pair_counts {
        heap.push((count, pair));
    }

    let mut merges: Vec<Merge> = Vec::new();
    let mut next_id: u32 = 256;
    let target = u32::try_from(vocab_size).expect("vocab_size fits u32");

    while next_id < target {
        // Pop until we find a non-stale entry.
        let (la, lb, count) = loop {
            let Some((heap_count, pair)) = heap.pop() else {
                return merges; // no more pairs
            };
            let current = pair_counts.get(&pair).copied().unwrap_or(0);
            if current == heap_count && current > 0 {
                break (pair.0, pair.1, current);
            }
            // else: stale, keep popping
        };
        if count < 2 {
            break; // no point merging singletons
        }
        let new_id = next_id;
        next_id += 1;
        merges.push(Merge {
            left: la,
            right: lb,
            new: new_id,
        });

        // Apply the merge. For each occurrence "...x la lb y..." that
        // becomes "...x M y...", update three pair counts: (x,la)
        // and (lb,y) lose `freq`, (x,M) and (M,y) gain `freq`, and
        // (la,lb) loses `freq` for each occurrence. Collect every
        // pair whose count changes; push fresh heap entries for them
        // after this merge.
        let mut changed: HashSet<(u32, u32)> = HashSet::new();
        let mut new_word_freq: HashMap<Vec<u32>, u64> = HashMap::new();
        for (word, freq) in word_freq.drain() {
            let mut out_word: Vec<u32> = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == la && word[i + 1] == lb {
                    if let Some(&prev) = out_word.last() {
                        let v = pair_counts.entry((prev, la)).or_insert(0);
                        *v = v.saturating_sub(freq);
                        changed.insert((prev, la));
                        *pair_counts.entry((prev, new_id)).or_insert(0) += freq;
                        changed.insert((prev, new_id));
                    }
                    if i + 2 < word.len() {
                        let next = word[i + 2];
                        let v = pair_counts.entry((lb, next)).or_insert(0);
                        *v = v.saturating_sub(freq);
                        changed.insert((lb, next));
                        *pair_counts.entry((new_id, next)).or_insert(0) += freq;
                        changed.insert((new_id, next));
                    }
                    let v = pair_counts.entry((la, lb)).or_insert(0);
                    *v = v.saturating_sub(freq);
                    changed.insert((la, lb));

                    out_word.push(new_id);
                    i += 2;
                } else {
                    out_word.push(word[i]);
                    i += 1;
                }
            }
            *new_word_freq.entry(out_word).or_insert(0) += freq;
        }
        word_freq = new_word_freq;

        // Push fresh heap entries for everything that changed.
        for pair in changed {
            if let Some(&c) = pair_counts.get(&pair) {
                if c > 0 {
                    heap.push((c, pair));
                }
            }
        }
    }

    merges
}

/// Tokenize `bytes` using `merges`. Splits on letter / non-letter
/// boundaries to match training, then applies the merges in order
/// within each pre-token. Returns a flat `Vec<u32>` of token ids
/// across the entire input (pre-token boundaries are implicit in
/// the token stream).
pub(crate) fn tokenize(bytes: &[u8], merges: &[Merge]) -> Vec<u32> {
    // Build a fast (pair → new_id) map for application.
    let mut merge_map: HashMap<(u32, u32), u32> = HashMap::with_capacity(merges.len());
    for m in merges {
        merge_map.insert((m.left, m.right), m.new);
    }

    let mut out: Vec<u32> = Vec::with_capacity(bytes.len() / 2);
    let mut p = 0;
    while p < bytes.len() {
        let is_letter = bytes[p].is_ascii_alphabetic();
        let mut q = p + 1;
        while q < bytes.len() && bytes[q].is_ascii_alphabetic() == is_letter {
            q += 1;
        }
        let mut word: Vec<u32> = bytes[p..q].iter().map(|&b| u32::from(b)).collect();
        apply_merges_to_word(&mut word, &merge_map, merges);
        out.extend_from_slice(&word);
        p = q;
    }
    out
}

/// Greedy-left-to-right application of merges to a single pre-token.
/// We iterate the merges *in training order* so that each merge can
/// build on the results of prior ones.
fn apply_merges_to_word(
    word: &mut Vec<u32>,
    _merge_map: &HashMap<(u32, u32), u32>,
    merges: &[Merge],
) {
    // Naive but correct: repeatedly apply each merge until no
    // further changes. Faster algorithms exist (priority-by-merge-
    // rank scanning) but for survey-scale corpora this is fine.
    for m in merges {
        let mut i = 0;
        while i + 1 < word.len() {
            if word[i] == m.left && word[i + 1] == m.right {
                word[i] = m.new;
                word.remove(i + 1);
                // Don't increment i; check if the new merge enables
                // another at the same position.
                i = i.saturating_sub(1);
            } else {
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_corpus_no_merges() {
        let merges = train(b"", 1024);
        assert!(merges.is_empty());
    }

    #[test]
    fn vocab_size_at_or_below_256_no_merges() {
        let merges = train(b"hello world hello world", 256);
        assert!(merges.is_empty());
    }

    #[test]
    fn merges_common_pair() {
        // "abxx abxx abxx ..." — (a, b) is the unambiguously most
        // frequent adjacent pair (3 per word × n_words), beating
        // (b, x) (1 per word) and (x, x) (1 per word). So the first
        // merge is (a, b) → 256.
        let mut corpus = Vec::new();
        for _ in 0..10 {
            corpus.extend_from_slice(b"ababab ");
        }
        let merges = train(&corpus, 260);
        assert!(!merges.is_empty());
        let first = merges[0];
        assert_eq!(first.left, u32::from(b'a'));
        assert_eq!(first.right, u32::from(b'b'));
        assert_eq!(first.new, 256);
    }

    #[test]
    fn tokenize_roundtrip_via_id_to_bytes() {
        // Train a small BPE and verify that decoding the tokens (by
        // expanding each merge bottom-up to its byte sequence)
        // produces the original input.
        let corpus = b"the quick brown fox jumps over the lazy dog";
        let merges = train(corpus, 280);

        // Build id → bytes lookup.
        let mut id_bytes: Vec<Vec<u8>> = (0..=255_u8).map(|b| vec![b]).collect();
        for _ in 256..280 {
            id_bytes.push(Vec::new());
        }
        for m in &merges {
            let mut bytes = id_bytes[m.left as usize].clone();
            bytes.extend(&id_bytes[m.right as usize]);
            id_bytes[m.new as usize] = bytes;
        }

        let tokens = tokenize(corpus, &merges);
        let mut reconstructed: Vec<u8> = Vec::new();
        for &t in &tokens {
            reconstructed.extend(&id_bytes[t as usize]);
        }
        assert_eq!(&reconstructed, corpus);
    }
}
