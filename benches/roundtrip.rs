#![allow(missing_docs)]

use std::io::{Cursor, Read, Write};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lzr::{Reader, Writer};

const SIZE_64K: usize = 65_536;
const SIZE_1M: usize = 1_048_576;

fn generate_data(size: usize) -> Vec<u8> {
    let base = b"The quick brown fox jumps over the lazy dog. \
                 Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                 Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        data.extend_from_slice(base);
    }
    data.truncate(size);
    data
}

fn bench_write(c: &mut Criterion) {
    let data_64k = generate_data(SIZE_64K);
    let data_1m = generate_data(SIZE_1M);

    let mut group = c.benchmark_group("writer");

    group.throughput(Throughput::Bytes(data_64k.len() as u64));
    group.bench_function("write_64kb", |b| {
        b.iter(|| {
            let buf = Vec::with_capacity(data_64k.len());
            let mut w = Writer::new(buf).unwrap();
            w.write_all(&data_64k).unwrap();
            w.seal().unwrap()
        });
    });

    group.throughput(Throughput::Bytes(data_1m.len() as u64));
    group.bench_function("write_1mb", |b| {
        b.iter(|| {
            let buf = Vec::with_capacity(data_1m.len());
            let mut w = Writer::new(buf).unwrap();
            w.write_all(&data_1m).unwrap();
            w.seal().unwrap()
        });
    });

    group.finish();
}

fn bench_read(c: &mut Criterion) {
    let data_64k = generate_data(SIZE_64K);
    let data_1m = generate_data(SIZE_1M);

    // Pre-compress.
    let compressed_64k = {
        let buf = Vec::new();
        let mut w = Writer::new(buf).unwrap();
        w.write_all(&data_64k).unwrap();
        w.seal().unwrap()
    };
    let compressed_1m = {
        let buf = Vec::new();
        let mut w = Writer::new(buf).unwrap();
        w.write_all(&data_1m).unwrap();
        w.seal().unwrap()
    };

    let mut group = c.benchmark_group("reader");

    group.throughput(Throughput::Bytes(data_64k.len() as u64));
    group.bench_function("read_64kb", |b| {
        b.iter(|| {
            let cursor = Cursor::new(&compressed_64k);
            let mut r = Reader::new(cursor).unwrap();
            let mut output = Vec::with_capacity(data_64k.len());
            r.read_to_end(&mut output).unwrap();
            output
        });
    });

    group.throughput(Throughput::Bytes(data_1m.len() as u64));
    group.bench_function("read_1mb", |b| {
        b.iter(|| {
            let cursor = Cursor::new(&compressed_1m);
            let mut r = Reader::new(cursor).unwrap();
            let mut output = Vec::with_capacity(data_1m.len());
            r.read_to_end(&mut output).unwrap();
            output
        });
    });

    group.finish();
}

criterion_group!(benches, bench_write, bench_read);
criterion_main!(benches);
