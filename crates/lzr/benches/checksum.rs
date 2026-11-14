//! Checksum throughput.

use divan::counter::BytesCount;
use divan::{Bencher, black_box};

fn main() {
    divan::main();
}

/// Pseudo-random bytes, so the input is neither all zeros nor trivially periodic.
fn input(len: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_u32;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[0]
        })
        .collect()
}

#[divan::bench(args = [64, 4096, 1 << 20])]
fn adler32(bencher: Bencher<'_, '_>, len: usize) {
    let data = input(len);
    bencher.counter(BytesCount::of_slice(&data)).bench(|| lzr::__bench::adler32(black_box(&data)));
}
