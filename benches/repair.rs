//! Throughput benchmarks for the internal capped Re-Pair tokenizer.
//!
//! The `tokenizers` module is `pub(crate)`; the `bench-internals` feature widens it (and
//! `Tokenize`/`RepairTokenizer`/the cost criteria) to `pub` so this external bench crate can reach
//! them, mirroring `adler32`/`uleb128`.
//!
//! Run with: `cargo bench --features bench-internals --bench repair`

use divan::{Bencher, black_box, counter::BytesCount};
use lzr::tokenizers::{Entropy, Frequency, RepairTokenizer, Tokenize};

fn main() {
    divan::main();
}

/// A corpus file under the crate root, or `None` if absent (bench is skipped).
fn corpus(rel: &str) -> Option<Vec<u8>> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).ok()
}

/// Tokenize the whole corpus with the default (entropy-cost) criterion.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn tokenize(bencher: Bencher<'_, '_>) {
    let Some(data) = corpus("corpora/large/bible.txt") else {
        return;
    };
    bencher.counter(BytesCount::of_slice(&data)).bench(|| RepairTokenizer::default().forward(black_box(&data)));
}

/// Detokenize a pre-tokenized corpus; throughput is reported in original bytes.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn detokenize(bencher: Bencher<'_, '_>) {
    let Some(data) = corpus("corpora/large/bible.txt") else {
        return;
    };
    let tokens = RepairTokenizer::default().forward(&data);
    // The `Result` is the bench's return value (divan black-boxes it), so no `unwrap` is needed.
    bencher.counter(BytesCount::of_slice(&data)).bench(|| RepairTokenizer::default().inverse(black_box(&tokens)));
}

/// Compare the two merge criteria head-to-head on tokenize throughput.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn tokenize_frequency(bencher: Bencher<'_, '_>) {
    let Some(data) = corpus("corpora/large/bible.txt") else {
        return;
    };
    let tokenizer = RepairTokenizer::new(Frequency);
    bencher.counter(BytesCount::of_slice(&data)).bench(|| tokenizer.forward(black_box(&data)));
}

/// The entropy criterion (the shipped default), benched alongside `tokenize_frequency`.
#[divan::bench(sample_count = 1, sample_size = 1)]
fn tokenize_entropy(bencher: Bencher<'_, '_>) {
    let Some(data) = corpus("corpora/large/bible.txt") else {
        return;
    };
    let tokenizer = RepairTokenizer::new(Entropy::default());
    bencher.counter(BytesCount::of_slice(&data)).bench(|| tokenizer.forward(black_box(&data)));
}
