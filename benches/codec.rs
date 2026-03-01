#![allow(missing_docs)]

use std::io::{self, Cursor};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const CORPUS: &str = "corpora/large/bible.txt";

fn encode(c: &mut Criterion) {
    let data = std::fs::read(CORPUS).expect("failed to read corpus file");

    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("encode", |b| {
        b.iter(|| {
            lzr::encoder::encode(Cursor::new(&data), io::sink()).unwrap();
        });
    });
    group.finish();
}

fn decode(c: &mut Criterion) {
    let data = std::fs::read(CORPUS).expect("failed to read corpus file");
    let mut compressed = Vec::new();
    lzr::encoder::encode(Cursor::new(&data), &mut compressed).expect("failed to pre-compress corpus");

    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("decode", |b| {
        b.iter(|| {
            lzr::decoder::decode(Cursor::new(&compressed), io::sink()).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, encode, decode);
criterion_main!(benches);
