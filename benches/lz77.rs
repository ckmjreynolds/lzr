#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lzr::bench::{Lz77Decoder, Lz77Encoder, lipsum_bytes};

const SIZE: usize = 65_536;

fn bench_encode(c: &mut Criterion) {
    let data = lipsum_bytes(SIZE);
    let mut group = c.benchmark_group("lz77");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut enc = Lz77Encoder::new(&data);
            let mut tokens = Vec::new();
            while let Some((seq, literals)) = enc.next_token() {
                tokens.push((seq, literals));
            }
            tokens
        });
    });
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let data = lipsum_bytes(SIZE);

    // Pre-encode so the decode benchmark only measures decoding.
    let mut enc = Lz77Encoder::new(&data);
    let mut tokens = Vec::new();
    while let Some((seq, literals)) = enc.next_token() {
        tokens.push((seq, literals));
    }

    let mut group = c.benchmark_group("lz77");
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut dec = Lz77Decoder::new();
            let mut output = Vec::with_capacity(data.len());
            for (seq, literals) in &tokens {
                dec.decode(seq, literals, &mut output);
            }
            output
        });
    });
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
