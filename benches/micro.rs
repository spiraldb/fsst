#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use fsst::{CompressorBuilder, Symbol};

fn one_megabyte(seed: &[u8]) -> Vec<u8> {
    seed.iter().copied().cycle().take(1024 * 1024).collect()
}

fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress-overhead");
    // Reusable memory to hold outputs
    let mut output_buf: Vec<u8> = Vec::with_capacity(12);

    // We create a symbol table that requires probing the hash table to perform
    // decompression.
    group.bench_function("compress-hashtab", |b| {
        let mut compressor = CompressorBuilder::new();
        compressor.insert(Symbol::from_slice(b"abcdefgh"), 8);
        let compressor = compressor.build();

        let word = u64::from_le_bytes([b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']);
        b.iter(|| unsafe { compressor.compress_word(word, output_buf.as_mut_ptr()) });
    });

    // We create a symbol table that is able to short-circuit the decompression
    group.bench_function("compress-twobytes", |b| {
        let mut compressor = CompressorBuilder::new();
        compressor.insert(Symbol::from_slice(&[b'a', b'b', 0, 0, 0, 0, 0, 0]), 2);
        let compressor = compressor.build();

        let word = u64::from_le_bytes([b'a', b'b', 0, 0, 0, 0, 0, 0]);
        b.iter(|| unsafe { compressor.compress_word(word, output_buf.as_mut_ptr()) });
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
