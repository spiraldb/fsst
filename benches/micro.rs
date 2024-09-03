#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use fsst::{CompressorBuilder, Symbol};

fn one_megabyte(seed: &[u8]) -> Vec<u8> {
    seed.iter().copied().cycle().take(1024 * 1024).collect()
}

fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress-overhead");
    group.bench_function("compress-word", |b| {
        let mut compressor = CompressorBuilder::new();
        compressor.insert(Symbol::from_u8(b'a'), 1);
        let compressor = compressor.build();

        let mut output = [0u8, 0u8];

        b.iter(|| unsafe {
            compressor.compress_word('a' as u64, output.as_mut_ptr());
        });
    });

    // Reusable memory to hold outputs
    let mut output_buf: Vec<u8> = Vec::with_capacity(1_024 * 1024 * 2);

    group.bench_function("compress-hashtab", |b| {
        // We create a symbol table and an input that will execute exactly one iteration,
        // in the fast compress_word pathway.
        let mut compressor = CompressorBuilder::new();
        compressor.insert(Symbol::from_slice(b"abcdefgh"), 8);
        let compressor = compressor.build();

        b.iter(|| unsafe {
            compressor.compress_into(
                b"abcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefghabcdefgh",
                &mut output_buf,
            );
        });
    });

    group.bench_function("compress-twobytes", |b| {
        // We create a symbol table and an input that will execute exactly one iteration,
        // in the fast compress_word pathway.
        let mut compressor = CompressorBuilder::new();
        compressor.insert(Symbol::from_slice(&[b'a', b'b', 0, 0, 0, 0, 0, 0]), 8);
        let compressor = compressor.build();

        b.iter(|| unsafe {
            compressor.compress_into(b"abababababababab", &mut output_buf);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("cf=1");
    let test_string = one_megabyte(b"aaaaaaaa");
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = CompressorBuilder::new();
        assert!(compressor.insert(Symbol::from_u8(b'a'), 1));
        let compressor = compressor.build();

        b.iter(|| unsafe {
            compressor.compress_into(&test_string, &mut output_buf);
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cf=2");
    let test_string = one_megabyte(b"ab");

    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = CompressorBuilder::new();
        // This outputs two codes for every 4 bytes of text.
        assert!(compressor.insert(Symbol::from_slice(&[b'a', 0, 0, 0, 0, 0, 0, 0]), 1));
        assert!(compressor.insert(Symbol::from_slice(&[b'b', b'a', b'b', 0, 0, 0, 0, 0]), 3));
        let compressor = compressor.build();

        b.iter(|| unsafe {
            compressor.compress_into(&test_string, &mut output_buf);
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cf=4");
    let test_string = one_megabyte(b"abcd");
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = CompressorBuilder::new();
        assert!(compressor.insert(Symbol::from_slice(&[b'a', b'b', b'c', b'd', 0, 0, 0, 0]), 4));
        let compressor = compressor.build();

        b.iter(|| unsafe {
            compressor.compress_into(&test_string, &mut output_buf);
        })
    });
    group.finish();

    let mut group = c.benchmark_group("cf=8");
    let test_string = one_megabyte(b"abcdefgh");
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = CompressorBuilder::new();
        assert!(compressor.insert(Symbol::from_slice(b"abcdefgh"), 8));
        let compressor = compressor.build();

        b.iter(|| unsafe {
            compressor.compress_into(&test_string, &mut output_buf);
        })
    });

    group.bench_function("decompress", |b| {
        let mut compressor = CompressorBuilder::new();
        assert!(compressor.insert(Symbol::from_slice(b"abcdefgh"), 8));
        let compressor = compressor.build();
        let compressed = compressor.compress(&test_string);

        let decompressor = compressor.decompressor();

        b.iter(|| decompressor.decompress(&compressed))
    });
    group.finish();

    let _ = std::hint::black_box(output_buf);
}

criterion_group!(bench_micro, bench_compress);
criterion_main!(bench_micro);
