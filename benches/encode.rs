#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use lzr::options::EncodeOptions;

const CORPUS: &str = "corpora/large/bible.txt";

fn encode(c: &mut Criterion) {
    let data = std::fs::read(CORPUS).unwrap();

    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut input = data.as_slice();
            let mut output = Vec::with_capacity(data.len());
            lzr::encode::encode(&mut input, &mut output, &EncodeOptions::default()).unwrap();
            std::hint::black_box(&output);
        });
    });
    group.finish();
}

criterion_group!(benches, encode);
criterion_main!(benches);
