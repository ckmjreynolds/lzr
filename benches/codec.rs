#![allow(missing_docs)]

use std::io::Cursor;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const CORPUS: &str = "corpora/large/bible.txt";

fn encode(c: &mut Criterion) {
    let data = std::fs::read(CORPUS).expect("failed to read corpus file");
    let mut out = Vec::with_capacity(data.len() + 8);

    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("encode", |b| {
        b.iter(|| {
            out.clear();
            lzr::encoder::encode(Cursor::new(&data), &mut out).unwrap();
            std::hint::black_box(&out);
        });
    });
    group.finish();
}

fn decode(c: &mut Criterion) {
    let data = std::fs::read(CORPUS).expect("failed to read corpus file");
    let mut compressed = Vec::new();
    lzr::encoder::encode(Cursor::new(&data), &mut compressed).expect("failed to pre-compress corpus");
    let mut out = Vec::with_capacity(data.len());

    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("decode", |b| {
        b.iter(|| {
            out.clear();
            lzr::decoder::decode(Cursor::new(&compressed), &mut out).unwrap();
            std::hint::black_box(&out);
        });
    });
    group.finish();
}

criterion_group!(benches, encode, decode);
criterion_main!(benches);
