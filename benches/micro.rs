#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fsst::{Compressor, Symbol};

fn bench1(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress-overhead");
    group.bench_function("compress-word", |b| {
        let mut compressor = Compressor::default();
        compressor.insert(Symbol::from_u8(b'a'));

        let mut output = [0u8, 0u8];

        b.iter(|| unsafe {
            compressor.compress_word('a' as u64, output.as_mut_ptr());
        });
    });

    // Reusable memory to hold outputs
    let mut output_buf: Vec<u8> = Vec::with_capacity(1_024);

    group.throughput(Throughput::Bytes(8u64));
    group.bench_function("compress_fastpath", |b| {
        // We create a symbol table and an input that will execute exactly one iteration,
        // in the fast compress_word pathway.
        let mut compressor = Compressor::default();
        compressor.insert(Symbol::from_slice(b"abcdefgh"));

        b.iter(|| unsafe {
            compressor.compress_into(b"abcdefgh", &mut output_buf);
        });
    });

    group.throughput(Throughput::Bytes(4u64));
    group.bench_function("compress_slowpath", |b| {
        // We create a symbol table and an input that will execute exactly one iteration,
        // but it misses the compress_word and needs to go on the slow path.
        let mut compressor = Compressor::default();
        compressor.insert(Symbol::from_slice(&[b'a', b'b', b'c', b'd', 0, 0, 0, 0]));

        b.iter(|| unsafe {
            compressor.compress_into(b"abcd", &mut output_buf);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("cf=1");
    let test_string = b"aaaaaaaa";
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = Compressor::default();
        assert!(compressor.insert(Symbol::from_u8(b'a')));

        b.iter(|| unsafe {
            compressor.compress_into(test_string, &mut output_buf);
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cf=2");
    let test_string = b"ababababababababababababababab";
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = Compressor::default();
        assert!(compressor.insert(Symbol::from_slice(&[b'a', b'b', 0, 0, 0, 0, 0, 0])));

        b.iter(|| unsafe {
            compressor.compress_into(test_string, &mut output_buf);
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cf=4");
    let test_string = b"abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd";
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = Compressor::default();
        assert!(compressor.insert(Symbol::from_slice(&[b'a', b'b', b'c', b'd', 0, 0, 0, 0])));

        b.iter(|| unsafe {
            compressor.compress_into(test_string, &mut output_buf);
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cf=8");
    let test_string = b"abcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefgh";
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = Compressor::default();
        assert!(compressor.insert(Symbol::from_slice(b"abcdefgh")));

        b.iter(|| unsafe {
            compressor.compress_into(test_string, &mut output_buf);
        })
    });
    group.finish();

    let _ = std::hint::black_box(output_buf);
}

criterion_group!(bench_toy, bench1);
criterion_main!(bench_toy);
