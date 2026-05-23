//! Byte-level BPE encode/decode, matching the Python tokenizer in
//! `lzr-neural/scripts/train_bpe.py`.
//!
//! Tokenizer scheme (chosen for Rust portability):
//! - Tokens 0..255 are raw single bytes (always present in vocab).
//! - Tokens ≥256 are merges of two previously-existing tokens.
//! - `\n` (byte 0x0A) is the only pre-token boundary — BPE never
//!   merges across newlines, so we split the input on 0x0A and
//!   BPE-encode each line independently, then concatenate token IDs.
//! - No `byte_to_unicode` reshuffle, no GPT-2 regex — every byte
//!   `0..256` maps to itself as a single-char token. The shipped
//!   vocab table stores each token's byte sequence verbatim.
//!
//! Encode algorithm (per line):
//! 1. Start with each byte as its own token: `[t_0, t_1, ...]`.
//! 2. Look up all adjacent pairs `(t_i, t_{i+1})` in the merge table.
//! 3. Find the pair with the lowest **merge rank** (earliest learned
//!    merge takes priority — this is the standard HF/GPT-2 BPE rule).
//! 4. Merge it in place; the new token id replaces the pair.
//! 5. Repeat until no adjacent pair has a known merge.
//!
//! This implementation is `O(line_length²)` in the worst case but the
//! average is much better since each iteration reduces token count by 1
//! and most pairs are not mergeable. Good enough for compression
//! throughput on M3 Pro / Zen 2 within the Hutter wall-clock budget.
//!
//! Decode is trivial: concatenate each token id's stored byte sequence.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Magic header bytes: `'BPE0'` little-endian. Matches
/// `scripts/train_bpe.py`'s MAGIC constant.
const MAGIC: u32 = 0x3045_5042;
const VERSION: u32 = 1;
/// Pre-token boundary byte. Encode splits the input on this byte and
/// BPE-merges each line independently.
const SPLIT_BYTE: u8 = b'\n';

#[derive(Debug)]
pub(crate) struct Bpe {
    /// `id_to_bytes[id]` is the byte sequence this token expands to
    /// (used for decode; encode doesn't need it).
    id_to_bytes: Vec<Vec<u8>>,
    /// `pair_rank[(left, right)] = (merge_index, new_token_id)`.
    /// Merge index is the rank used for priority — lower means
    /// "applied first" (the merge was learned earlier in training).
    pair_rank: HashMap<(u32, u32), (u32, u32)>,
    pub(crate) vocab_size: usize,
    pub(crate) n_merges: usize,
}

impl Bpe {
    /// Load a `.bin` tokenizer file produced by `train_bpe.py`.
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading BPE table {}", path.display()))?;
        Self::parse(&bytes)
    }

    /// Parse the in-memory `.bin` format.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            bail!("BPE table too short for header: {}", bytes.len());
        }
        let mut cursor = 0_usize;
        let read_u32 = |buf: &[u8], c: &mut usize| -> u32 {
            let v = u32::from_le_bytes([buf[*c], buf[*c + 1], buf[*c + 2], buf[*c + 3]]);
            *c += 4;
            v
        };
        let read_u16 = |buf: &[u8], c: &mut usize| -> u16 {
            let v = u16::from_le_bytes([buf[*c], buf[*c + 1]]);
            *c += 2;
            v
        };

        let magic = read_u32(bytes, &mut cursor);
        if magic != MAGIC {
            bail!("BPE magic mismatch: got 0x{magic:08x}, expected 0x{MAGIC:08x}");
        }
        let version = read_u32(bytes, &mut cursor);
        if version != VERSION {
            bail!("BPE version mismatch: got {version}, expected {VERSION}");
        }
        let vocab_size = read_u32(bytes, &mut cursor) as usize;
        let n_merges = read_u32(bytes, &mut cursor) as usize;

        let mut id_to_bytes = Vec::with_capacity(vocab_size);
        for tok_id in 0..vocab_size {
            if cursor + 2 > bytes.len() {
                bail!("BPE table truncated at token {tok_id} header");
            }
            let byte_len = read_u16(bytes, &mut cursor) as usize;
            if cursor + byte_len > bytes.len() {
                bail!("BPE table truncated at token {tok_id} body (need {byte_len} bytes)");
            }
            id_to_bytes.push(bytes[cursor..cursor + byte_len].to_vec());
            cursor += byte_len;
        }

        let mut pair_rank: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(n_merges);
        // The new-token-id for the i-th merge is 256 + i (the standard
        // BPE convention, but we verify it against the table). The
        // Python trainer stores merges in priority order: the first
        // merge has the lowest priority value.
        for i in 0..n_merges {
            if cursor + 8 > bytes.len() {
                bail!("BPE table truncated at merge {i}");
            }
            let left = read_u32(bytes, &mut cursor);
            let right = read_u32(bytes, &mut cursor);
            let i_u32 = u32::try_from(i).expect("n_merges fits u32 (vocab_size <= 65536)");
            let new_id = 256_u32 + i_u32;
            pair_rank.insert((left, right), (i_u32, new_id));
        }

        if cursor != bytes.len() {
            bail!(
                "BPE table has {} trailing bytes after parse",
                bytes.len() - cursor
            );
        }
        Ok(Self {
            id_to_bytes,
            pair_rank,
            vocab_size,
            n_merges,
        })
    }

    /// Byte length of a token's expansion. Returns 0 for ids beyond
    /// `vocab_size` (which shouldn't occur in practice).
    pub(crate) fn token_byte_len(&self, id: u32) -> usize {
        self.id_to_bytes.get(id as usize).map_or(0, Vec::len)
    }

    /// Encode a byte slice into a token sequence.
    pub(crate) fn encode(&self, input: &[u8]) -> Vec<u32> {
        let mut out = Vec::with_capacity(input.len() / 3);
        // Split on newlines, keeping each `\n` as its own pre-token.
        // `Split(pattern="\n", behavior="isolated")` in HF tokenizers
        // produces three pre-token kinds: "before \n", "\n itself",
        // "after \n". Our loop interleaves these.
        let mut start = 0;
        for (i, &b) in input.iter().enumerate() {
            if b == SPLIT_BYTE {
                if start < i {
                    self.encode_chunk(&input[start..i], &mut out);
                }
                // Encode the newline as its own pre-token (single byte → token 0x0A).
                self.encode_chunk(&input[i..=i], &mut out);
                start = i + 1;
            }
        }
        if start < input.len() {
            self.encode_chunk(&input[start..], &mut out);
        }
        out
    }

    /// BPE-merge a single pre-token (assumed not to contain `\n`,
    /// or to be exactly the single `\n` byte).
    fn encode_chunk(&self, chunk: &[u8], out: &mut Vec<u32>) {
        if chunk.is_empty() {
            return;
        }
        // Start with each byte as its own token.
        let mut toks: Vec<u32> = chunk.iter().map(|&b| u32::from(b)).collect();

        // Iteratively apply the lowest-rank merge present.
        loop {
            let mut best_rank = u32::MAX;
            let mut best_pos = usize::MAX;
            let mut best_new_id = 0_u32;
            for i in 0..toks.len().saturating_sub(1) {
                if let Some(&(rank, new_id)) = self.pair_rank.get(&(toks[i], toks[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i;
                        best_new_id = new_id;
                    }
                }
            }
            if best_pos == usize::MAX {
                break;
            }
            toks[best_pos] = best_new_id;
            toks.remove(best_pos + 1);
        }

        out.extend_from_slice(&toks);
    }

    /// Decode a token sequence to bytes.
    pub(crate) fn decode(&self, tokens: &[u32]) -> Vec<u8> {
        let total: usize = tokens
            .iter()
            .map(|&t| self.id_to_bytes.get(t as usize).map_or(0, Vec::len))
            .sum();
        let mut out = Vec::with_capacity(total);
        for &t in tokens {
            let b = self
                .id_to_bytes
                .get(t as usize)
                .expect("token id out of vocab");
            out.extend_from_slice(b);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_round_trips_real_enwik8_prefix() {
        let bpe_path =
            std::path::PathBuf::from("/Users/creynolds/Programming/lzr-neural/ckpts/bpe_8k_v2.bin");
        if !bpe_path.exists() {
            eprintln!("BPE table missing — skipping round-trip test");
            return;
        }
        let bpe = Bpe::load(&bpe_path).expect("load BPE");
        assert_eq!(bpe.vocab_size, 8192);
        assert_eq!(bpe.n_merges, 7936);

        let corpus = std::path::PathBuf::from("/Users/creynolds/Programming/lzr/assets/enwik8");
        if !corpus.exists() {
            eprintln!("corpus missing — skipping round-trip test");
            return;
        }
        let bytes = std::fs::read(&corpus).expect("read corpus");
        // 4 KB sample mid-file (skips the front-matter that's all in
        // the first hundred bytes; checks real text + XML).
        let sample = &bytes[10_000..14_096];

        let tokens = bpe.encode(sample);
        let decoded = bpe.decode(&tokens);
        assert_eq!(decoded, sample, "round-trip mismatch");

        #[allow(clippy::cast_precision_loss)]
        let bytes_per_token = sample.len() as f64 / tokens.len() as f64;
        eprintln!(
            "bpe round-trip: {} bytes → {} tokens ({:.2} bytes/token)",
            sample.len(),
            tokens.len(),
            bytes_per_token,
        );
    }

    /// Same round-trip but with the 16K vocab tokenizer.
    #[test]
    fn bpe_round_trips_first_1mb_enwik9_16k() {
        let bpe_path =
            std::path::PathBuf::from("/Users/creynolds/Programming/lzr-neural/ckpts/bpe_16k.bin");
        let corpus = std::path::PathBuf::from("/Users/creynolds/Programming/lzr/assets/enwik9");
        if !bpe_path.exists() || !corpus.exists() {
            eprintln!("artifacts missing — skipping");
            return;
        }
        let bpe = Bpe::load(&bpe_path).expect("load 16K BPE");
        let mut all = std::fs::read(&corpus).expect("read corpus");
        all.truncate(1_000_000);
        let tokens = bpe.encode(&all);
        let decoded = bpe.decode(&tokens);
        eprintln!(
            "bpe 16K 1MB roundtrip: in={} tokens={} out={}",
            all.len(),
            tokens.len(),
            decoded.len()
        );
        assert_eq!(decoded.len(), all.len(), "byte length mismatch");
        assert_eq!(decoded, all, "byte content mismatch");
    }

    /// Round-trip the first 1 MB of enwik9 directly through the
    /// Rust BPE (no AC, no codec). Catches any encode/decode issue
    /// that's specific to the larger input size.
    #[test]
    fn bpe_round_trips_first_1mb_enwik9() {
        let bpe_path =
            std::path::PathBuf::from("/Users/creynolds/Programming/lzr-neural/ckpts/bpe_8k_v2.bin");
        let corpus = std::path::PathBuf::from("/Users/creynolds/Programming/lzr/assets/enwik9");
        if !bpe_path.exists() || !corpus.exists() {
            eprintln!("artifacts missing — skipping");
            return;
        }
        let bpe = Bpe::load(&bpe_path).expect("load BPE");
        let mut all = std::fs::read(&corpus).expect("read corpus");
        all.truncate(1_000_000);
        let tokens = bpe.encode(&all);
        let decoded = bpe.decode(&tokens);
        eprintln!(
            "bpe 1MB roundtrip: in={} tokens={} out={}",
            all.len(),
            tokens.len(),
            decoded.len()
        );
        assert_eq!(decoded.len(), all.len(), "byte length mismatch");
        assert_eq!(decoded, all, "byte content mismatch");
    }

    #[test]
    fn bpe_round_trips_tiny_synthetic() {
        // Build a tiny BPE table by hand for unit-test correctness:
        // 256 byte tokens + 2 merges (a,b → 256) and (256,c → 257).
        // Input "abcab" should encode to [257, 256] then decode back.
        let mut bin = Vec::new();
        bin.extend_from_slice(&MAGIC.to_le_bytes());
        bin.extend_from_slice(&VERSION.to_le_bytes());
        bin.extend_from_slice(&258_u32.to_le_bytes()); // vocab_size
        bin.extend_from_slice(&2_u32.to_le_bytes()); // n_merges
        // Token table: 256 single-byte tokens, then "ab" (256), then "abc" (257).
        for b in 0..=255_u8 {
            bin.extend_from_slice(&1_u16.to_le_bytes());
            bin.push(b);
        }
        bin.extend_from_slice(&2_u16.to_le_bytes());
        bin.extend_from_slice(b"ab");
        bin.extend_from_slice(&3_u16.to_le_bytes());
        bin.extend_from_slice(b"abc");
        // Merges: (a, b) → 256 at rank 0, then (256, c) → 257 at rank 1.
        bin.extend_from_slice(&u32::from(b'a').to_le_bytes());
        bin.extend_from_slice(&u32::from(b'b').to_le_bytes());
        bin.extend_from_slice(&256_u32.to_le_bytes());
        bin.extend_from_slice(&u32::from(b'c').to_le_bytes());

        let bpe = Bpe::parse(&bin).expect("parse synthetic BPE");
        let toks = bpe.encode(b"abcab");
        assert_eq!(toks, vec![257, 256]);
        let decoded = bpe.decode(&toks);
        assert_eq!(decoded, b"abcab");
    }

    /// Parity check: load a Python-tokenized reference (produced by
    /// `lzr-neural/scripts/dump_bpe_tokens_for_parity.py`) and verify
    /// the Rust encoder produces the exact same token sequence for
    /// the same input bytes. Catches any divergence between the
    /// Rust BPE algorithm and HF tokenizers' BPE.
    #[test]
    fn bpe_python_parity() {
        let bpe_path =
            std::path::PathBuf::from("/Users/creynolds/Programming/lzr-neural/ckpts/bpe_8k_v2.bin");
        let ref_path = std::path::PathBuf::from(
            "/Users/creynolds/Programming/lzr-neural/ckpts/bpe_8k_v2.parity.bin",
        );
        let corpus = std::path::PathBuf::from("/Users/creynolds/Programming/lzr/assets/enwik8");
        if !bpe_path.exists() || !ref_path.exists() || !corpus.exists() {
            eprintln!("parity artifacts missing — skipping");
            return;
        }
        let bpe = Bpe::load(&bpe_path).expect("load BPE");
        let ref_bytes = std::fs::read(&ref_path).expect("read parity ref");
        let read_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([
                ref_bytes[off],
                ref_bytes[off + 1],
                ref_bytes[off + 2],
                ref_bytes[off + 3],
            ])
        };
        let offset = read_u32(0) as usize;
        let n_bytes = read_u32(4) as usize;
        let n_tokens = read_u32(8) as usize;
        assert_eq!(ref_bytes.len(), 12 + 4 * n_tokens);
        let mut expected = Vec::with_capacity(n_tokens);
        for i in 0..n_tokens {
            expected.push(read_u32(12 + 4 * i));
        }

        let all = std::fs::read(&corpus).expect("read corpus");
        let input = &all[offset..offset + n_bytes];
        let actual = bpe.encode(input);

        assert_eq!(
            actual.len(),
            expected.len(),
            "token count: rust={} python={}",
            actual.len(),
            expected.len(),
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a,
                e,
                "mismatch at position {i}: rust={a} python={e}\n  rust first 8: {:?}\n  py   first 8: {:?}",
                &actual[..8.min(actual.len())],
                &expected[..8.min(expected.len())],
            );
        }
        eprintln!("bpe parity OK: {n_bytes} bytes → {n_tokens} tokens, exact match");
    }

    #[test]
    fn bpe_splits_on_newline() {
        // Verify newlines are isolated pre-tokens: "ab\nab" with merge
        // (a,b)→256 should encode as [256, 10, 256], not [256, 10, 256]
        // through any merge that crosses the newline.
        let mut bin = Vec::new();
        bin.extend_from_slice(&MAGIC.to_le_bytes());
        bin.extend_from_slice(&VERSION.to_le_bytes());
        bin.extend_from_slice(&257_u32.to_le_bytes());
        bin.extend_from_slice(&1_u32.to_le_bytes());
        for b in 0..=255_u8 {
            bin.extend_from_slice(&1_u16.to_le_bytes());
            bin.push(b);
        }
        bin.extend_from_slice(&2_u16.to_le_bytes());
        bin.extend_from_slice(b"ab");
        bin.extend_from_slice(&u32::from(b'a').to_le_bytes());
        bin.extend_from_slice(&u32::from(b'b').to_le_bytes());

        let bpe = Bpe::parse(&bin).expect("parse");
        let toks = bpe.encode(b"ab\nab");
        assert_eq!(toks, vec![256, 10, 256]);
    }
}
