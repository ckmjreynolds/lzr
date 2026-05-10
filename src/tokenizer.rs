//! Byte-level BPE tokenizer (load + encode + decode).
//!
//! Loads a compact merge table from a binary blob (see [`bpe`] for the
//! producer side), then exposes:
//!
//! - [`Tokenizer::encode`] — bytes → tokens. Pre-tokenizes into runs of
//!   ASCII whitespace vs. non-whitespace (matching the training-time
//!   pretokenizer in [`crate::bpe`]), then applies merges greedily inside
//!   each run via lowest-rank-first.
//! - [`Tokenizer::decode`] — tokens → bytes. Direct lookup in a
//!   pre-expanded `id → bytes` table built at load time.
//!
//! Binary format (all little-endian):
//! ```text
//! u32  schema_version (= 1)
//! u32  vocab_size         (256 + num_merges)
//! u32  num_merges
//! u16 LE pairs × num_merges  (left, right) per merge in insertion order
//! ```
//!
//! Token ids fit in `u16` for any vocab ≤ 65 536. Going beyond requires
//! widening to `u32` here and in the AC/codec/model paths.

// Some accessors (e.g. `num_merges`, `decode`) are only invoked from the
// training-feature path or from external callers in tests; the
// submission build's `unused`-warning sweep flags them otherwise.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Token-id type (16-bit; supports vocab up to 65 536).
pub(crate) type Token = u16;

/// On-disk schema version.
pub(crate) const SCHEMA_VERSION: u32 = 1;

/// Loaded BPE tokenizer.
#[derive(Debug)]
pub(crate) struct Tokenizer {
    vocab_size: usize,
    /// `merges[i] = (a, b)` means token id `256 + i` was formed by merging
    /// `a` followed by `b`. Order is the order BPE produced them; lower
    /// index = applied first (lower rank).
    merges: Vec<(Token, Token)>,
    /// `merge_rank[(a, b)] = i` for fast lookup of lowest-rank pair.
    merge_rank: HashMap<(Token, Token), u32>,
    /// `vocab[id] = byte sequence that token decodes to`. Built once.
    vocab: Vec<Vec<u8>>,
}

impl Tokenizer {
    /// Parse the binary tokenizer format produced by `lzr bpe ... --bin`.
    pub(crate) fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < 12 {
            bail!("tokenizer blob too short ({} < 12 header bytes)", buf.len());
        }
        let ver = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        if ver != SCHEMA_VERSION {
            bail!("tokenizer schema version {ver} != expected {SCHEMA_VERSION}");
        }
        let vocab_size = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let num_merges = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        if vocab_size != 256 + num_merges {
            bail!(
                "vocab_size ({vocab_size}) != 256 + num_merges ({})",
                256 + num_merges
            );
        }
        let body = &buf[12..];
        if body.len() != num_merges * 4 {
            bail!(
                "tokenizer body length {} != expected {}",
                body.len(),
                num_merges * 4
            );
        }
        if vocab_size > usize::from(Token::MAX) + 1 {
            bail!(
                "vocab_size {vocab_size} exceeds Token::MAX+1 ({}); widen `Token` to u32",
                usize::from(Token::MAX) + 1
            );
        }

        let mut merges: Vec<(Token, Token)> = Vec::with_capacity(num_merges);
        for i in 0..num_merges {
            let off = i * 4;
            let a = u16::from_le_bytes(body[off..off + 2].try_into().unwrap());
            let b = u16::from_le_bytes(body[off + 2..off + 4].try_into().unwrap());
            merges.push((a, b));
        }

        let mut merge_rank: HashMap<(Token, Token), u32> = HashMap::with_capacity(num_merges);
        for (i, &pair) in merges.iter().enumerate() {
            merge_rank.insert(pair, u32::try_from(i).unwrap());
        }

        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
        for b in 0u8..=u8::MAX {
            vocab.push(vec![b]);
        }
        for &(a, b) in &merges {
            let mut combined = vocab[a as usize].clone();
            combined.extend_from_slice(&vocab[b as usize]);
            vocab.push(combined);
        }
        Ok(Self {
            vocab_size,
            merges,
            merge_rank,
            vocab,
        })
    }

    pub(crate) fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())
            .with_context(|| format!("reading tokenizer {}", path.as_ref().display()))?;
        Self::from_bytes(&bytes)
    }

    pub(crate) const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub(crate) fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// Encode bytes to tokens. Pretokenizes into whitespace/non-whitespace
    /// runs (matching the BPE trainer), then applies merges within each
    /// run greedily by lowest rank.
    pub(crate) fn encode(&self, bytes: &[u8]) -> Vec<Token> {
        let mut out: Vec<Token> = Vec::with_capacity(bytes.len() / 2);
        let mut i = 0;
        while i < bytes.len() {
            let start = i;
            let is_ws = bytes[i].is_ascii_whitespace();
            while i < bytes.len() && bytes[i].is_ascii_whitespace() == is_ws {
                i += 1;
            }
            self.encode_chunk(&bytes[start..i], &mut out);
        }
        out
    }

    /// Apply BPE merges within a single pretoken chunk (no whitespace
    /// crossing). Greedy: at each step find the lowest-rank pair in the
    /// current sequence and merge it; stop when no pair has a known merge.
    fn encode_chunk(&self, chunk: &[u8], out: &mut Vec<Token>) {
        if chunk.is_empty() {
            return;
        }
        let mut seq: Vec<Token> = chunk.iter().map(|&b| Token::from(b)).collect();
        loop {
            let mut best_rank: u32 = u32::MAX;
            let mut best_idx: usize = usize::MAX;
            for k in 0..seq.len().saturating_sub(1) {
                if let Some(&rank) = self.merge_rank.get(&(seq[k], seq[k + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = k;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            let new_id: Token = 256 + Token::try_from(best_rank).unwrap();
            seq[best_idx] = new_id;
            seq.remove(best_idx + 1);
        }
        out.extend_from_slice(&seq);
    }

    /// Decode tokens back to bytes via direct vocab lookup.
    pub(crate) fn decode(&self, tokens: &[Token]) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::with_capacity(tokens.len() * 2);
        for &t in tokens {
            out.extend_from_slice(&self.vocab[t as usize]);
        }
        out
    }

    /// Byte sequence the given token decodes to.
    pub(crate) fn token_bytes(&self, t: Token) -> &[u8] {
        &self.vocab[t as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_blob(merges: &[(u16, u16)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
        let vocab_size: u32 = u32::try_from(256 + merges.len()).unwrap();
        buf.extend_from_slice(&vocab_size.to_le_bytes());
        buf.extend_from_slice(&u32::try_from(merges.len()).unwrap().to_le_bytes());
        for &(a, b) in merges {
            buf.extend_from_slice(&a.to_le_bytes());
            buf.extend_from_slice(&b.to_le_bytes());
        }
        buf
    }

    #[test]
    fn roundtrip_no_merges() {
        let tk = Tokenizer::from_bytes(&build_blob(&[])).unwrap();
        assert_eq!(tk.vocab_size(), 256);
        let bytes = b"hello world";
        let tokens = tk.encode(bytes);
        // No merges → every byte is its own token.
        assert_eq!(tokens.len(), bytes.len());
        for (t, &b) in tokens.iter().zip(bytes.iter()) {
            assert_eq!(*t, u16::from(b));
        }
        assert_eq!(tk.decode(&tokens), bytes);
    }

    #[test]
    fn merges_apply_greedily() {
        // Merge "ab" then "abc". Encoding "abcabc" should produce
        // [abc, abc] (= [257, 257]) — abc applied before ab when possible.
        // BPE rank order: ab=256, abc=257.
        let blob = build_blob(&[(u16::from(b'a'), u16::from(b'b')), (256, u16::from(b'c'))]);
        let tk = Tokenizer::from_bytes(&blob).unwrap();
        let tokens = tk.encode(b"abcabc");
        // Greedy by lowest rank means "ab" applies first (rank 0).
        // After "ab" merges: [256, c, 256, c] → then (256, c) = "abc" merges (rank 1).
        // Result: [257, 257].
        assert_eq!(tokens, vec![257, 257]);
        assert_eq!(tk.decode(&tokens), b"abcabc");
    }

    #[test]
    fn pretokenization_doesnt_cross_whitespace() {
        // Even with a high-rank merge of (b 'a', b' '), pretokenizer
        // splits on whitespace boundaries, so no merge spans the split.
        let blob = build_blob(&[(u16::from(b'a'), u16::from(b' '))]);
        let tk = Tokenizer::from_bytes(&blob).unwrap();
        let tokens = tk.encode(b"a b");
        // Expected: ['a'][' ']['b'] — three separate runs, no cross-run merge.
        assert_eq!(tokens, vec![97, 32, 98]);
    }

    #[test]
    fn arbitrary_bytes_roundtrip() {
        let blob = build_blob(&[(u16::from(b't'), u16::from(b'h')), (256, u16::from(b'e'))]);
        let tk = Tokenizer::from_bytes(&blob).unwrap();
        let bytes: Vec<u8> = (0..=255u8).cycle().take(2048).collect();
        let tokens = tk.encode(&bytes);
        assert_eq!(tk.decode(&tokens), bytes);
    }

    #[test]
    fn rejects_wrong_schema_version() {
        let mut blob = build_blob(&[]);
        blob[0] = 99;
        assert!(Tokenizer::from_bytes(&blob).is_err());
    }
}
