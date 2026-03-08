#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const CORPUS: &str = "corpora/large/bible.txt";

fn checksum(c: &mut Criterion) {
    let data = std::fs::read(CORPUS).unwrap();

    let mut group = c.benchmark_group("checksum");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("compute", |b| {
        b.iter(|| {
            let mut adler = lzr::adler32::Adler32::new();
            adler.update(&data);
            std::hint::black_box(&adler.checksum());
        });
    });
    group.finish();
}

criterion_group!(benches, checksum);
criterion_main!(benches);
