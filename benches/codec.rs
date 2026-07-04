//! Throughput benchmarks for the public `compress`/`decompress` round trip.
//!
//! Benches the in-process library API rather than the CLI subprocess, whose
//! timing is dominated by process spawn and file I/O.
//!
//! Run with: `cargo bench --features bench-internals --bench codec`

use divan::{Bencher, black_box, counter::BytesCount};

fn main() {
    divan::main();
}

/// A corpus file under the crate root, or `None` if absent (bench is skipped).
fn corpus(rel: &str) -> Option<Vec<u8>> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).ok()
}

#[divan::bench(sample_count = 1, sample_size = 1)]
fn compress(bencher: Bencher<'_, '_>) {
    let Some(data) = corpus("corpora/large/bible.txt") else {
        return;
    };
    bencher.counter(BytesCount::of_slice(&data)).bench(|| lzr::compress(black_box(&data)));
}

#[divan::bench(sample_count = 1, sample_size = 1)]
fn decompress(bencher: Bencher<'_, '_>) {
    let Some(data) = corpus("corpora/large/bible.txt") else {
        return;
    };
    let packed = lzr::compress(&data);
    // Throughput is reported in original bytes. The `Result` is the bench's return
    // value (divan black-boxes it), so no `unwrap` is needed in bench code.
    bencher.counter(BytesCount::of_slice(&data)).bench(|| lzr::decompress(black_box(&packed)));
}
