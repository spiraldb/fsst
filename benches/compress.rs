//! Benchmarks for FSST compression, decompression, and symbol table training.
#![allow(missing_docs)]
use core::str;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use fsst::{Compressor, ESCAPE_CODE};

const CORPUS: &str = include_str!("dracula.txt");
const TEST: &str = "I found my smattering of German very useful here";

fn bench_fsst(c: &mut Criterion) {
    let mut group = c.benchmark_group("fsst");
    group.bench_function("train", |b| {
        let corpus = CORPUS.as_bytes();
        b.iter(|| black_box(Compressor::train(black_box(corpus))));
    });

    let compressor = Compressor::train(CORPUS);
    let plaintext = TEST.as_bytes();

    let compressed = compressor.compress(plaintext);
    let escape_count = compressed.iter().filter(|b| **b == ESCAPE_CODE).count();
    let ratio = (plaintext.len() as f64) / (compressed.len() as f64);
    println!(
        "Escapes = {escape_count}/{}, compression_ratio = {ratio}",
        compressed.len()
    );

    let decompressor = compressor.decompressor();
    let decompressed = decompressor.decompress(&compressed);
    let decompressed = str::from_utf8(&decompressed).unwrap();

    group.throughput(Throughput::Elements(1));
    group.bench_function("compress-word", |b| {
        let mut out = vec![0u8; 8];
        let out_ptr = out.as_mut_ptr();
        let front = &TEST.as_bytes()[0..8];
        let word = u64::from_le_bytes(front.try_into().unwrap());

        b.iter(|| black_box(unsafe { compressor.compress_word(word, out_ptr) }));
    });

    group.throughput(Throughput::Bytes(CORPUS.len() as u64));
    group.bench_function("compress-single", |b| {
        b.iter(|| black_box(compressor.compress(black_box(CORPUS.as_bytes()))));
    });

    group.throughput(Throughput::Bytes(decompressed.len() as u64));
    group.bench_function("decompress-single", |b| {
        b.iter(|| black_box(decompressor.decompress(black_box(&compressed))));
    });
}

criterion_group!(compress_bench, bench_fsst);
criterion_main!(compress_bench);
