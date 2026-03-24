#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use fsst::{CompressorBuilder, Symbol};

fn one_megabyte(seed: &[u8]) -> Vec<u8> {
    seed.iter().copied().cycle().take(1024 * 1024).collect()
}

fn bench_decompress_short(c: &mut Criterion) {
    let mut compressor = CompressorBuilder::new();
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
    assert_eq!(compressed_64.len(), 8);
    let mut decoded_64 =
        Vec::with_capacity(decompressor.max_decompression_capacity(&compressed_64) + 7);

    let mut group = c.benchmark_group("decompress-short/8b-64b");
    group.throughput(Throughput::Bytes(short_64.len() as u64));
    group.bench_function("decompress-into-reuse", |b| {
        b.iter(|| {
            let len = decompressor.decompress_into(&compressed_64, decoded_64.spare_capacity_mut());
            // SAFETY: decompress_into initializes exactly len bytes.
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
    assert_eq!(compressed_128.len(), 16);
    let mut decoded_128 =
        Vec::with_capacity(decompressor.max_decompression_capacity(&compressed_128) + 7);

    let mut group = c.benchmark_group("decompress-short/16b-128b");
    group.throughput(Throughput::Bytes(short_128.len() as u64));
    group.bench_function("decompress-into-reuse", |b| {
        b.iter(|| {
            let len =
                decompressor.decompress_into(&compressed_128, decoded_128.spare_capacity_mut());
            // SAFETY: decompress_into initializes exactly len bytes.
            unsafe { decoded_128.set_len(len) };
            let _ = std::hint::black_box(&decoded_128);
            decoded_128.clear();
        })
    });
    group.finish();
}

fn bench_decompress_escape_heavy(c: &mut Criterion) {
    let test_string = one_megabyte(b"abcdefgh");

    let mut compressor = CompressorBuilder::new();
    assert!(compressor.insert(Symbol::from_slice(b"abcdefgh"), 8));
    let compressor = compressor.build();
    let compressed_cf8 = compressor.compress(&test_string);
    let decompressor_cf8 = compressor.decompressor();
    let mut decoded_cf8 =
        Vec::with_capacity(decompressor_cf8.max_decompression_capacity(&compressed_cf8) + 7);

    // Empty symbol table forces every input byte to be encoded as ESCAPE + raw byte.
    let escape_compressor = CompressorBuilder::new().build();
    let compressed_escape = escape_compressor.compress(&test_string);
    let escape_decompressor = escape_compressor.decompressor();
    let mut decoded_escape =
        Vec::with_capacity(escape_decompressor.max_decompression_capacity(&compressed_escape) + 7);

    let mut group = c.benchmark_group("decompress-regimes/1mb");
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    group.bench_function("decompress-into-reuse-cf8", |b| {
        b.iter(|| {
            let len =
                decompressor_cf8.decompress_into(&compressed_cf8, decoded_cf8.spare_capacity_mut());
            // SAFETY: decompress_into initializes exactly len bytes.
            unsafe { decoded_cf8.set_len(len) };
            let _ = std::hint::black_box(&decoded_cf8);
            decoded_cf8.clear();
        })
    });
    group.bench_function("decompress-into-reuse-all-escape", |b| {
        b.iter(|| {
            let len = escape_decompressor
                .decompress_into(&compressed_escape, decoded_escape.spare_capacity_mut());
            // SAFETY: decompress_into initializes exactly len bytes.
            unsafe { decoded_escape.set_len(len) };
            let _ = std::hint::black_box(&decoded_escape);
            decoded_escape.clear();
        })
    });
    group.finish();
}

fn bench_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress-overhead");
    // Reusable memory to hold outputs
    let mut output_buf: Vec<u8> = Vec::with_capacity(8 * 1024 * 1024);

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

    let mut group = c.benchmark_group("all-escape");
    let test_string = one_megabyte(b"abcdefgh");
    group.throughput(Throughput::Bytes(test_string.len() as u64));
    // Empty symbol table: every byte becomes ESCAPE + raw byte.
    let escape_compressor = CompressorBuilder::new().build();
    group.bench_function("compress", |b| {
        b.iter(|| unsafe {
            escape_compressor.compress_into(&test_string, &mut output_buf);
        })
    });

    let compressed_escape = escape_compressor.compress(&test_string);
    let escape_decompressor = escape_compressor.decompressor();
    group.bench_function("decompress", |b| {
        b.iter(|| escape_decompressor.decompress(&compressed_escape))
    });
    group.finish();

    let _ = std::hint::black_box(output_buf);
}

criterion_group!(
    bench_micro,
    bench_compress,
    bench_decompress_short,
    bench_decompress_escape_heavy
);
criterion_main!(bench_micro);
