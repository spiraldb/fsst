#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fsst::{Compressor, Symbol};

fn bench1(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress-overhead");
    group.bench_function("compress_word_fast", |b| {
        // We create a symbol table and an input that will execute exactly one iteration,
        // in the fast compress_word pathway.
        let mut compressor = Compressor::default();
        compressor.insert(Symbol::from_slice(&[
            b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h',
        ]));

        b.iter(|| {
            let _ = std::hint::black_box(compressor.compress(b"abcdefgh"));
        });
    });

    group.bench_function("compress_word_slow", |b| {
        // We create a symbol table and an input that will execute exactly one iteration,
        // but it misses the compress_word and needs to go on the slow path.
        let mut compressor = Compressor::default();
        compressor.insert(Symbol::from_slice(&[b'a', b'b', b'c', b'd', 0, 0, 0, 0]));

        b.iter(|| {
            let _ = std::hint::black_box(compressor.compress(b"abcd"));
        });
    });
    group.finish();

    let mut group = c.benchmark_group("cf=2");
    let test_string = b"ababababababababababababababab";
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = Compressor::default();
        assert!(compressor.insert(Symbol::from_slice(&[b'a', b'b', 0, 0, 0, 0, 0, 0])));

        b.iter_with_large_drop(|| {
            // We expect this to have
            std::hint::black_box(compressor.compress(test_string))
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cf=4");
    let test_string = b"abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd";
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = Compressor::default();
        assert!(compressor.insert(Symbol::from_slice(&[b'a', b'b', b'c', b'd', 0, 0, 0, 0])));

        b.iter_with_large_drop(|| {
            // We expect this to have
            std::hint::black_box(compressor.compress(test_string))
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cf=8");
    let test_string = b"abcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefgh";
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = Compressor::default();
        assert!(compressor.insert(Symbol::from_slice(&[
            b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h'
        ])));

        b.iter_with_large_drop(|| {
            // We expect this to have
            std::hint::black_box(compressor.compress(test_string))
        })
    });
    group.finish();
}

criterion_group!(bench_toy, bench1);
criterion_main!(bench_toy);
