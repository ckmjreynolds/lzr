//! Encode/decode throughput benchmarks for the internal ULEB128 codec.
//!
//! The `uleb128` module is `pub(crate)`; the `bench-internals` feature widens it
//! to `pub` (via the `visibility` crate) so this external bench crate can reach it.
//!
//! Run with: `cargo bench --features bench-internals --bench uleb128`

use divan::{Bencher, black_box, counter::ItemsCount};
use lzr::uleb128::{decode_uleb128_u64, encode_uleb128_u64};

fn main() {
    divan::main();
}

/// Number of values encoded/decoded per iteration.
const COUNT: usize = 10_000;

/// A deterministic spread of `u64` values covering all ULEB128 byte-lengths.
///
/// Multiplying an index by a large odd constant walks the full 64-bit range, so
/// the sample exercises 1- through 9-byte encodings rather than only small values.
fn sample_values() -> Vec<u64> {
    (0..COUNT as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect()
}

#[divan::bench]
fn encode(bencher: Bencher<'_, '_>) {
    let values = sample_values();
    bencher.counter(ItemsCount::new(values.len())).bench(|| {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in &values {
            encode_uleb128_u64(black_box(v), &mut out);
        }
        out
    });
}

#[divan::bench]
fn decode(bencher: Bencher<'_, '_>) {
    let values = sample_values();
    let mut buf = Vec::new();
    for &v in &values {
        encode_uleb128_u64(v, &mut buf);
    }

    bencher.counter(ItemsCount::new(values.len())).bench(|| {
        let mut pos = 0;
        let mut acc = 0u64;
        while pos < buf.len() {
            match decode_uleb128_u64(black_box(&buf), &mut pos) {
                Ok(v) => acc = acc.wrapping_add(v),
                Err(_) => break,
            }
        }
        acc
    });
}
