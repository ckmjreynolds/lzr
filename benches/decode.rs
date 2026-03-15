#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use lzr::options::EncodeOptions;

const CORPUS: &str = "corpora/large/bible.txt";

fn decode(c: &mut Criterion) {
    let data = std::fs::read(CORPUS).unwrap();

    let mut compressed = Vec::with_capacity(data.len());
    lzr::encode::encode(&mut data.as_slice(), &mut compressed, &EncodeOptions::default()).unwrap();

    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Bytes(compressed.len() as u64));
    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut input = compressed.as_slice();
            let mut output = Vec::with_capacity(compressed.len());
            lzr::decode::decode(&mut input, &mut output).unwrap();
            std::hint::black_box(&output);
        });
    });
    group.finish();
}

criterion_group!(benches, decode);
criterion_main!(benches);
