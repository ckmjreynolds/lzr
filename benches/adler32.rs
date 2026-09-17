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

/// A deterministic buffer of exactly `n` bytes.
///
/// An odd-constant LCG walk (the same spread trick the `uleb128` bench uses) fills
/// the whole `0..=255` range without pulling in a text-generation dependency.
fn bytes(n: usize) -> Vec<u8> {
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            ((state >> 32) & 0xFF) as u8
        })
        .collect()
}

#[divan::bench(args = SIZES)]
fn checksum(bencher: Bencher<'_, '_>, size: usize) {
    let data = bytes(size);
    bencher.counter(BytesCount::of_slice(&data)).bench(|| {
        let mut ck = Adler32::new();
        ck.update(black_box(&data));
        ck.checksum()
    });
}
