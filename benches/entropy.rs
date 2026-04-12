#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lzr::bench::{Decoder, Encoder, Model, lipsum_bytes};

const SIZE: usize = 65_536;

fn bench_encode(c: &mut Criterion) {
    let data = lipsum_bytes(SIZE);
    let mut group = c.benchmark_group("entropy");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut enc = Encoder::new();
            let mut model = Model::new();
            let mut compressed = Vec::with_capacity(data.len());
            for &byte in &data {
                enc.encode(&mut model, byte, &mut compressed);
            }
            enc.finish(&mut compressed);
            compressed
        });
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let data = lipsum_bytes(SIZE);

    // Pre-encode so the decode benchmark only measures decoding.
    let mut enc = Encoder::new();
    let mut model = Model::new();
    let mut compressed = Vec::new();
    for &byte in &data {
        enc.encode(&mut model, byte, &mut compressed);
    }
    enc.finish(&mut compressed);

    let mut group = c.benchmark_group("entropy");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut dec = Decoder::new();
            let mut model = Model::new();
            let mut input: &[u8] = &compressed;
            let mut decoded = Vec::with_capacity(data.len());
            for _ in 0..data.len() {
                decoded.push(dec.decode(&mut model, &mut input));
            }
            decoded
        });
    });
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
