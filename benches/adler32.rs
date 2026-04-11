#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lipsum::lipsum;
use lzr::_bench::Adler32;

const SIZE: usize = 1_048_576;

fn bench_compute(c: &mut Criterion) {
    let data = lipsum_bytes(SIZE);
    let mut group = c.benchmark_group("checksum");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("compute", |b| {
        b.iter(|| {
            let mut ck = Adler32::new();
            ck.update(&data);
            ck.checksum()
        });
    });
    group.finish();
}

/// Generates `min_bytes` of deterministic lipsum text.
fn lipsum_bytes(min_bytes: usize) -> Vec<u8> {
    // lipsum counts words, not bytes; average English word is ~5 chars + space.
    let words = min_bytes / 5 + 1;
    let text = lipsum(words);
    text.into_bytes()[..min_bytes].to_vec()
}

criterion_group!(benches, bench_compute);
criterion_main!(benches);
