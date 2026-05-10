//! Byte-level BPE tokenizer training (training-feature only).
//!
//! Trains a byte-level BPE on a corpus and emits the merge list as JSON.
//! Pre-tokenizes by alternating runs of ASCII whitespace vs. non-whitespace
//! so merges never cross those boundaries — keeps the inner loop tractable
//! on enwik9 without pulling in a regex dependency for proper Unicode
//! pre-tokenization.
//!
//! Output is for measurement only at this stage. The submission binary
//! still ingests bytes directly. If `bytes_per_token` justifies the size
//! tax, the merge list will be re-serialised into a compact binary blob
//! and embedded via `include_bytes!`.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use serde::Serialize;

#[derive(Args, Debug, Clone)]
pub(crate) struct BpeArgs {
    /// Corpus to train on (e.g. `assets/enwik9`).
    pub input: PathBuf,

    /// Target vocabulary size including the 256 base byte tokens.
    #[arg(long, default_value_t = 4096)]
    pub vocab_size: u32,

    /// Output JSON path (merges + summary metadata).
    #[arg(long, default_value = "assets/tokenizer.json")]
    pub out: PathBuf,

    /// Output binary path (compact merges-only blob for runtime loading).
    #[arg(long, default_value = "assets/tokenizer.bin")]
    pub out_bin: PathBuf,

    /// Print top-N learned tokens after training.
    #[arg(long, default_value_t = 30)]
    pub print_top: usize,

    /// Skip training; re-emit `tokenizer.bin` from an existing
    /// `tokenizer.json`. Useful when the JSON already encodes the merges
    /// you want and only the binary form is missing or stale.
    #[arg(long)]
    pub from_json: Option<PathBuf>,
}

#[derive(Serialize)]
struct TokenizerJson {
    schema_version: u32,
    corpus_path: String,
    corpus_bytes: u64,
    pretoken_count: u64,
    unique_pretokens: u64,
    vocab_size: u32,
    final_token_count: u64,
    bytes_per_token: f64,
    merges: Vec<[u32; 2]>,
}

#[allow(
    clippy::needless_pass_by_value,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::tuple_array_conversions
)]
pub(crate) fn run(args: BpeArgs) -> Result<()> {
    if let Some(json_path) = args.from_json.as_ref() {
        return repack_from_json(json_path, &args.out_bin);
    }
    let t0 = Instant::now();
    let corpus =
        std::fs::read(&args.input).with_context(|| format!("reading {}", args.input.display()))?;
    eprintln!(
        "[bpe] read {} bytes in {:.1}s",
        corpus.len(),
        t0.elapsed().as_secs_f64()
    );

    let (mut words, pretoken_count) = pretokenize(&corpus);
    eprintln!(
        "[bpe] pretokenized: {pretoken_count} pretokens, {} unique",
        words.len()
    );

    let merges = train_bpe(&mut words, args.vocab_size, true);

    let final_token_count: u64 = words.iter().map(|(w, f)| w.len() as u64 * f).sum();
    let bytes_per_token = corpus.len() as f64 / final_token_count.max(1) as f64;
    eprintln!(
        "[bpe] {} merges → vocab {}, final_tokens={final_token_count}, bytes/token={bytes_per_token:.3}",
        merges.len(),
        256 + merges.len()
    );

    let out = TokenizerJson {
        schema_version: 1,
        corpus_path: args.input.display().to_string(),
        corpus_bytes: corpus.len() as u64,
        pretoken_count,
        unique_pretokens: words.len() as u64,
        vocab_size: 256 + merges.len() as u32,
        final_token_count,
        bytes_per_token,
        merges: merges.iter().map(|&(a, b)| [a, b]).collect(),
    };
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f =
        File::create(&args.out).with_context(|| format!("creating {}", args.out.display()))?;
    f.write_all(serde_json::to_string_pretty(&out)?.as_bytes())?;
    eprintln!("[bpe] wrote {}", args.out.display());

    write_bin(&merges, &args.out_bin)?;
    eprintln!("[bpe] wrote {}", args.out_bin.display());

    if args.print_top > 0 {
        print_top_tokens(&merges, args.print_top);
    }
    Ok(())
}

/// Read an existing `tokenizer.json` and re-emit the compact
/// `tokenizer.bin` form. Skips BPE training entirely.
fn repack_from_json(json_path: &PathBuf, bin_path: &PathBuf) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct InJson {
        merges: Vec<[u32; 2]>,
    }
    let txt = std::fs::read_to_string(json_path)
        .with_context(|| format!("reading {}", json_path.display()))?;
    let parsed: InJson =
        serde_json::from_str(&txt).with_context(|| format!("parsing {}", json_path.display()))?;
    let merges: Vec<(u32, u32)> = parsed.merges.into_iter().map(|m| (m[0], m[1])).collect();
    write_bin(&merges, bin_path)?;
    eprintln!(
        "[bpe] repacked {} merges from {} → {}",
        merges.len(),
        json_path.display(),
        bin_path.display()
    );
    Ok(())
}

/// Write the compact binary tokenizer format consumed by
/// [`crate::tokenizer::Tokenizer::from_bytes`].
///
/// Layout (LE):
/// ```text
/// u32 schema_version (= 1)
/// u32 vocab_size      (256 + num_merges)
/// u32 num_merges
/// num_merges × (u16, u16) merge pairs
/// ```
#[allow(clippy::cast_possible_truncation)]
fn write_bin(merges: &[(u32, u32)], path: &PathBuf) -> Result<()> {
    use crate::tokenizer::SCHEMA_VERSION;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut buf: Vec<u8> = Vec::with_capacity(12 + merges.len() * 4);
    let vocab_size: u32 = 256 + merges.len() as u32;
    buf.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
    buf.extend_from_slice(&vocab_size.to_le_bytes());
    buf.extend_from_slice(&(merges.len() as u32).to_le_bytes());
    for &(a, b) in merges {
        let a16 = u16::try_from(a).context("merge id > 65535; widen Token to u32")?;
        let b16 = u16::try_from(b).context("merge id > 65535; widen Token to u32")?;
        buf.extend_from_slice(&a16.to_le_bytes());
        buf.extend_from_slice(&b16.to_le_bytes());
    }
    let mut f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    f.write_all(&buf)?;
    Ok(())
}

/// Split `corpus` into runs of ASCII whitespace vs. non-whitespace.
/// Returns a frequency table of unique runs (as `Vec<u32>` token streams)
/// plus the total pretoken count.
fn pretokenize(corpus: &[u8]) -> (Vec<(Vec<u32>, u64)>, u64) {
    let mut freq: HashMap<&[u8], u64> = HashMap::new();
    let mut total: u64 = 0;
    let mut i = 0;
    while i < corpus.len() {
        let start = i;
        let is_ws = corpus[i].is_ascii_whitespace();
        while i < corpus.len() && corpus[i].is_ascii_whitespace() == is_ws {
            i += 1;
        }
        *freq.entry(&corpus[start..i]).or_insert(0) += 1;
        total += 1;
    }
    let words = freq
        .into_iter()
        .map(|(bytes, f)| (bytes.iter().map(|&b| u32::from(b)).collect(), f))
        .collect();
    (words, total)
}

/// In-place BPE training. Returns the ordered list of `(left, right)`
/// merges. New token ids are `256 + merges.len()` at insertion time.
fn train_bpe(words: &mut [(Vec<u32>, u64)], vocab_size: u32, progress: bool) -> Vec<(u32, u32)> {
    let t0 = Instant::now();
    let mut merges: Vec<(u32, u32)> = Vec::new();
    let mut next_id: u32 = 256;

    while next_id < vocab_size {
        let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
        for (word, f) in words.iter() {
            for w in word.windows(2) {
                *pair_counts.entry((w[0], w[1])).or_insert(0) += *f;
            }
        }
        let Some((&best_pair, &best_count)) = pair_counts.iter().max_by_key(|&(_, &c)| c) else {
            break;
        };
        if best_count < 2 {
            break;
        }

        let new_id = next_id;
        next_id += 1;
        merges.push(best_pair);

        for (word, _) in words.iter_mut() {
            if word.len() < 2 {
                continue;
            }
            let mut buf: Vec<u32> = Vec::with_capacity(word.len());
            let mut k = 0;
            while k < word.len() {
                if k + 1 < word.len() && word[k] == best_pair.0 && word[k + 1] == best_pair.1 {
                    buf.push(new_id);
                    k += 2;
                } else {
                    buf.push(word[k]);
                    k += 1;
                }
            }
            *word = buf;
        }

        if progress && (merges.len() <= 32 || merges.len() % 256 == 0 || next_id == vocab_size) {
            eprintln!(
                "[bpe] merge {:>5} ({:>5},{:>5}) freq={best_count:>10} elapsed={:.1}s",
                merges.len(),
                best_pair.0,
                best_pair.1,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    merges
}

fn print_top_tokens(merges: &[(u32, u32)], n: usize) {
    let mut vocab: Vec<Vec<u8>> = (0..=u8::MAX).map(|b| vec![b]).collect();
    for &(a, b) in merges {
        let mut combo = vocab[a as usize].clone();
        combo.extend_from_slice(&vocab[b as usize]);
        vocab.push(combo);
    }
    eprintln!("[bpe] first {} learned tokens:", n.min(merges.len()));
    for (i, _) in merges.iter().take(n).enumerate() {
        let id = 256 + i;
        let bytes = &vocab[id];
        eprintln!("  id {id:>5}: {:?}", String::from_utf8_lossy(bytes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_most_frequent_pair_first() {
        let mut words: Vec<(Vec<u32>, u64)> = vec![(vec![97, 98, 97, 98], 1), (vec![97, 98], 2)];
        let merges = train_bpe(&mut words, 257, false);
        assert_eq!(merges, vec![(97, 98)]);
        assert_eq!(words[0].0, vec![256, 256]);
        assert_eq!(words[1].0, vec![256]);
    }

    #[test]
    fn stops_when_no_pair_repeats() {
        let mut words: Vec<(Vec<u32>, u64)> = vec![(vec![97, 98, 99, 100], 1)];
        let merges = train_bpe(&mut words, 4096, false);
        assert!(merges.is_empty());
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn pretokenize_alternates_runs() {
        let (words, total) = pretokenize(b"hi  world\n  foo");
        assert_eq!(total, 5);
        let strings: Vec<Vec<u8>> = words
            .iter()
            .map(|(w, _)| w.iter().map(|&t| t as u8).collect())
            .collect();
        for needle in [b"hi" as &[u8], b"  ", b"world", b"\n  ", b"foo"] {
            assert!(strings.iter().any(|s| s == needle), "missing {needle:?}");
        }
    }
}
