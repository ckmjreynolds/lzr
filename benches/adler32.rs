#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lzr::bench::{Adler32, lipsum_bytes};

const SIZE: usize = 65_536;

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

criterion_group!(benches, bench_compute);
criterion_main!(benches);
