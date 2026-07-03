//! Throughput benchmarks for the internal Adler-32 checksum.
//!
//! The `adler32` module is `pub(crate)`; the `bench-internals` feature widens it
//! to `pub` (via the `visibility` crate) so this external bench crate can reach it.
//!
//! Run with: `cargo bench --features bench-internals --bench adler32`

use divan::{Bencher, black_box, counter::BytesCount};
use lzr::adler32::Adler32;

fn main() {
    divan::main();
}

/// Input sizes in bytes.
const SIZES: &[usize] = &[1 << 10, 1 << 16, 1 << 20];

/// Deterministic English-like text of exactly `min_bytes` bytes.
fn text(min_bytes: usize) -> Vec<u8> {
    // lipsum counts words, not bytes; average English word is ~5 chars + space.
    let mut words = min_bytes / 5 + 1;
    loop {
        let mut bytes = lipsum::lipsum(words).into_bytes();
        if bytes.len() >= min_bytes {
            bytes.truncate(min_bytes);
            return bytes;
        }
        words *= 2;
    }
}

#[divan::bench(args = SIZES)]
fn checksum(bencher: Bencher<'_, '_>, size: usize) {
    let data = text(size);
    bencher.counter(BytesCount::of_slice(&data)).bench(|| {
        let mut ck = Adler32::new();
        ck.update(black_box(&data));
        ck.checksum()
    });
}
