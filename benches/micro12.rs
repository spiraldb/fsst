//! Micro-benchmarks for 12-bit FSST compression and decompression.
#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use fsst::Symbol;
use fsst::fsst12::CompressorBuilder12;

fn one_megabyte(seed: &[u8]) -> Vec<u8> {
    seed.iter().copied().cycle().take(1024 * 1024).collect()
}

fn bench_compress_12(c: &mut Criterion) {
    let mut output_buf: Vec<u8> = Vec::with_capacity(8 * 1024 * 1024);

    // cf=2: 2-byte symbol compresses 2 input bytes into one 12-bit code.
    let mut group = c.benchmark_group("fsst12/cf=1.33");
    let test_string = one_megabyte(b"ab");
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = CompressorBuilder12::new();
        assert!(compressor.insert(Symbol::from_slice(&[b'a', b'b', 0, 0, 0, 0, 0, 0]), 2));
        let compressor = compressor.build();

        b.iter(|| unsafe {
            compressor.compress_into(&test_string, &mut output_buf);
        })
    });
    group.finish();

    // cf=4: 4-byte symbol → one 12-bit code per 4 input bytes.
    let mut group = c.benchmark_group("fsst12/cf=2.67");
    let test_string = one_megabyte(b"abcd");
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = CompressorBuilder12::new();
        assert!(compressor.insert(Symbol::from_slice(&[b'a', b'b', b'c', b'd', 0, 0, 0, 0]), 4));
        let compressor = compressor.build();

        b.iter(|| unsafe {
            compressor.compress_into(&test_string, &mut output_buf);
        })
    });
    group.finish();

    // cf=8: 8-byte symbol → one 12-bit code per 8 input bytes.
    let mut group = c.benchmark_group("fsst12/cf=5.33");
    let test_string = one_megabyte(b"abcdefgh");
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("compress", |b| {
        let mut compressor = CompressorBuilder12::new();
        assert!(compressor.insert(Symbol::from_slice(b"abcdefgh"), 8));
        let compressor = compressor.build();

        b.iter(|| unsafe {
            compressor.compress_into(&test_string, &mut output_buf);
        })
    });

    group.bench_function("decompress", |b| {
        let mut compressor = CompressorBuilder12::new();
        assert!(compressor.insert(Symbol::from_slice(b"abcdefgh"), 8));
        let compressor = compressor.build();
        let compressed = compressor.compress(&test_string);

        let decompressor = compressor.decompressor();
        b.iter(|| decompressor.decompress(&compressed))
    });
    group.finish();

    // All identity: no multi-byte symbols, every byte is its own 12-bit code.
    let mut group = c.benchmark_group("fsst12/all-identity");
    let test_string = one_megabyte(b"abcdefgh");
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    let identity_compressor = CompressorBuilder12::new().build();
    group.bench_function("compress", |b| {
        b.iter(|| unsafe {
            identity_compressor.compress_into(&test_string, &mut output_buf);
        })
    });

    let compressed_identity = identity_compressor.compress(&test_string);
    let identity_decompressor = identity_compressor.decompressor();
    group.bench_function("decompress", |b| {
        b.iter(|| identity_decompressor.decompress(&compressed_identity))
    });
    group.finish();

    let _ = std::hint::black_box(output_buf);
}

fn bench_decompress_short_12(c: &mut Criterion) {
    let mut compressor = CompressorBuilder12::new();
    assert!(compressor.insert(Symbol::from_slice(b"abcdefgh"), 8));
    let compressor = compressor.build();
    let decompressor = compressor.decompressor();

    let short_64 = b"abcdefgh"
        .iter()
        .copied()
        .cycle()
        .take(64)
        .collect::<Vec<_>>();
    let compressed_64 = compressor.compress(&short_64);
    let mut decoded_64 =
        Vec::with_capacity(decompressor.max_decompression_capacity(&compressed_64) + 7);

    let mut group = c.benchmark_group("fsst12/decompress-short/8b-64b");
    group.throughput(Throughput::Bytes(short_64.len() as u64));
    group.bench_function("decompress-into-reuse", |b| {
        b.iter(|| {
            let len = decompressor.decompress_into(&compressed_64, decoded_64.spare_capacity_mut());
            unsafe { decoded_64.set_len(len) };
            let _ = std::hint::black_box(&decoded_64);
            decoded_64.clear();
        })
    });
    group.finish();

    let short_128 = b"abcdefgh"
        .iter()
        .copied()
        .cycle()
        .take(128)
        .collect::<Vec<_>>();
    let compressed_128 = compressor.compress(&short_128);
    let mut decoded_128 =
        Vec::with_capacity(decompressor.max_decompression_capacity(&compressed_128) + 7);

    let mut group = c.benchmark_group("fsst12/decompress-short/16b-128b");
    group.throughput(Throughput::Bytes(short_128.len() as u64));
    group.bench_function("decompress-into-reuse", |b| {
        b.iter(|| {
            let len =
                decompressor.decompress_into(&compressed_128, decoded_128.spare_capacity_mut());
            unsafe { decoded_128.set_len(len) };
            let _ = std::hint::black_box(&decoded_128);
            decoded_128.clear();
        })
    });
    group.finish();
}

criterion_group!(bench_micro12, bench_compress_12, bench_decompress_short_12,);
criterion_main!(bench_micro12);
