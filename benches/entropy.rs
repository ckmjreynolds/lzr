#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lzr::_bench::{EntropyDecoder, EntropyEncoder, EntropyModel, lipsum_bytes};

const SIZE: usize = 1_048_576;

fn bench_encode(c: &mut Criterion) {
    let data = lipsum_bytes(SIZE);
    let mut group = c.benchmark_group("entropy");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            let mut model = EntropyModel::new();
            let mut encoder = EntropyEncoder::new();
            for &byte in &data {
                encoder.encode(byte, &mut model, &mut output);
            }
            encoder.finish(&mut output);
            output
        });
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let data = lipsum_bytes(SIZE);
    let mut encoded = Vec::new();
    let mut model = EntropyModel::new();
    let mut encoder = EntropyEncoder::new();
    for &byte in &data {
        encoder.encode(byte, &mut model, &mut encoded);
    }
    encoder.finish(&mut encoded);

    let mut group = c.benchmark_group("entropy");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut input = encoded.as_slice();
            let mut decoder = EntropyDecoder::new(&mut input);
            let mut model = EntropyModel::new();
            let mut decoded = Vec::with_capacity(data.len());
            for _ in 0..data.len() {
                decoded.push(decoder.decode(&mut model, &mut input));
            }
            decoded
        });
    });
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
