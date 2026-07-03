//! Encode/decode throughput benchmarks for the internal ULEB128 codec.
//!
//! The `uleb128` module is `pub(crate)`; the `bench-internals` feature widens it
//! to `pub` (via the `visibility` crate) so this external bench crate can reach it.
//!
//! Run with: `cargo bench --features bench-internals --bench uleb128`

use arbitrary_int::u15;
use divan::{Bencher, black_box, counter::ItemsCount};

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

/// A deterministic spread of [`u15`] values covering both 1- and 2-byte encodings.
///
/// The same odd-constant walk, masked to 15 bits, spreads values across the whole
/// `0..=0x7FFF` range so roughly `127/32768` fall in the 1-byte form and the rest
/// in the 2-byte form — a spread biased toward the (slower) 2-byte path.
fn sample_u15_values() -> Vec<u15> {
    (0..COUNT as u64).map(|i| u15::new((i.wrapping_mul(0x9E37_79B9_7F4A_7C15) & 0x7FFF) as u16)).collect()
}

#[divan::bench]
fn encode_u64(bencher: Bencher<'_, '_>) {
    let values = sample_values();
    // Allocate the output buffer in divan's (untimed) input setup so the timed
    // closure measures encoding only, not the per-iteration allocation.
    bencher.counter(ItemsCount::new(values.len())).with_inputs(|| Vec::with_capacity(values.len() * 2)).bench_values(
        |mut out| {
            for &v in &values {
                lzr::uleb128::encode_u64(black_box(v), &mut out);
            }
            out
        },
    );
}

#[divan::bench]
fn decode_u64(bencher: Bencher<'_, '_>) {
    let values = sample_values();
    let mut buf = Vec::new();
    for &v in &values {
        lzr::uleb128::encode_u64(v, &mut buf);
    }

    bencher.counter(ItemsCount::new(values.len())).bench(|| {
        let mut pos = 0;
        let mut acc = 0u64;
        while pos < buf.len() {
            match lzr::uleb128::decode_u64(black_box(&buf), &mut pos) {
                Ok(v) => acc = acc.wrapping_add(v),
                Err(_) => break,
            }
        }
        acc
    });
}

#[divan::bench]
fn encode_u15(bencher: Bencher<'_, '_>) {
    let values = sample_u15_values();
    // Allocate the output buffer in divan's (untimed) input setup so the timed
    // closure measures encoding only, not the per-iteration allocation.
    bencher.counter(ItemsCount::new(values.len())).with_inputs(|| Vec::with_capacity(values.len() * 2)).bench_values(
        |mut out| {
            for &v in &values {
                lzr::uleb128::encode_u15(black_box(v), &mut out);
            }
            out
        },
    );
}

#[divan::bench]
fn decode_u15(bencher: Bencher<'_, '_>) {
    let values = sample_u15_values();
    let mut buf = Vec::new();
    for &v in &values {
        lzr::uleb128::encode_u15(v, &mut buf);
    }

    bencher.counter(ItemsCount::new(values.len())).bench(|| {
        let mut pos = 0;
        let mut acc = 0u32;
        while pos < buf.len() {
            match lzr::uleb128::decode_u15(black_box(&buf), &mut pos) {
                Ok(v) => acc = acc.wrapping_add(u32::from(v.value())),
                Err(_) => break,
            }
        }
        acc
    });
}
