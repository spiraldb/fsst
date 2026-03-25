//! Benchmarks for 12-bit FSST compression, decompression, and training.
//!
//! Uses the same dbtext datasets as the 8-bit benchmarks for direct comparison.
#![allow(missing_docs, clippy::unwrap_in_result)]

use core::str;
use std::{
    error::Error,
    fs::{self, DirBuilder, File},
    io::{Read, Write},
    path::Path,
};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use curl::easy::Easy;
use fsst::Compressor;
use fsst::fsst12::Compressor12;

fn download_dataset(url: &str, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
    let target = path.as_ref();

    let mut dir_builder = DirBuilder::new();
    dir_builder.recursive(true);

    dir_builder.create(target.parent().unwrap())?;

    if target.exists() {
        return Ok(());
    }

    let mut handle = Easy::new();

    let mut buffer = Vec::new();
    handle.url(url)?;
    {
        let mut transfer = handle.transfer();
        transfer.write_function(|data| {
            buffer.extend_from_slice(data);
            Ok(data.len())
        })?;
        transfer.perform()?;
    }

    let mut output = File::create(target)?;
    match output.write_all(&buffer) {
        Ok(()) => {}
        Err(err) => {
            fs::remove_file(target).unwrap();
            return Err(Box::new(err));
        }
    }

    Ok(())
}

#[allow(clippy::use_debug)]
fn bench_dbtext_12(c: &mut Criterion) {
    fn run_dataset_bench(name: &str, url: &str, path: &str, c: &mut Criterion) {
        let group_name = format!("fsst12/{name}");
        let mut group = c.benchmark_group(&group_name);
        download_dataset(url, path).unwrap();

        let mut buf = Vec::new();
        {
            let mut file = File::open(path).unwrap();
            file.read_to_end(&mut buf).unwrap();
        }

        group.bench_function("train-and-compress", |b| {
            b.iter_with_large_drop(|| {
                let compressor = Compressor12::train(&vec![&buf]);
                compressor.compress_bulk(std::hint::black_box(&vec![&buf]))
            });
        });

        let compressor = Compressor12::train(&vec![&buf]);
        let mut buffer = Vec::with_capacity(200 * 1024 * 1024);
        group.throughput(Throughput::Bytes(buf.len() as u64));
        group.bench_function("compress-only", |b| {
            b.iter(|| unsafe { compressor.compress_into(&buf, &mut buffer) });
        });

        unsafe {
            compressor.compress_into(&buf, &mut buffer);
        };
        let decompressor = compressor.decompressor();
        group.bench_function("decompress", |b| {
            b.iter_with_large_drop(|| decompressor.decompress(&buffer));
        });

        group.finish();

        // Report compression factor.
        let uncompressed_size = buf.len();
        let compressed = compressor.compress_bulk(&vec![&buf]);
        let compressed_size = compressed.iter().map(|l| l.len()).sum::<usize>();
        let cf = (uncompressed_size as f64) / (compressed_size as f64);
        println!(
            "12-bit: compressed {name} {uncompressed_size} => {compressed_size}B (compression factor {cf:.2}:1)"
        )
    }

    run_dataset_bench(
        "dbtext/wikipedia",
        "https://raw.githubusercontent.com/cwida/fsst/4e188a/paper/dbtext/wikipedia",
        "benches/data/wikipedia",
        c,
    );

    run_dataset_bench(
        "dbtext/l_comment",
        "https://raw.githubusercontent.com/cwida/fsst/4e188a/paper/dbtext/l_comment",
        "benches/data/l_comment",
        c,
    );

    run_dataset_bench(
        "dbtext/urls",
        "https://raw.githubusercontent.com/cwida/fsst/4e188a/paper/dbtext/urls",
        "benches/data/urls",
        c,
    );
}

/// Head-to-head comparison: 8-bit vs 12-bit on the same dataset.
#[allow(clippy::use_debug)]
fn bench_head_to_head(c: &mut Criterion) {
    fn run_comparison(name: &str, path: &str, c: &mut Criterion) {
        let mut buf = Vec::new();
        {
            if let Ok(mut file) = File::open(path) {
                file.read_to_end(&mut buf).unwrap();
            } else {
                return; // dataset not downloaded yet
            }
        }

        let group_name = format!("head-to-head/{name}");
        let mut group = c.benchmark_group(&group_name);
        group.throughput(Throughput::Bytes(buf.len() as u64));

        // 8-bit
        let compressor8 = Compressor::train(&vec![&buf]);
        let mut buffer8 = Vec::with_capacity(buf.len() * 2);
        group.bench_function("8bit-compress", |b| {
            b.iter(|| unsafe { compressor8.compress_into(&buf, &mut buffer8) });
        });

        unsafe { compressor8.compress_into(&buf, &mut buffer8) };
        let decompressor8 = compressor8.decompressor();
        group.bench_function("8bit-decompress", |b| {
            b.iter_with_large_drop(|| decompressor8.decompress(&buffer8));
        });

        // 12-bit
        let compressor12 = Compressor12::train(&vec![&buf]);
        let mut buffer12 = Vec::with_capacity(buf.len() * 2);
        group.bench_function("12bit-compress", |b| {
            b.iter(|| unsafe { compressor12.compress_into(&buf, &mut buffer12) });
        });

        unsafe { compressor12.compress_into(&buf, &mut buffer12) };
        let decompressor12 = compressor12.decompressor();
        group.bench_function("12bit-decompress", |b| {
            b.iter_with_large_drop(|| decompressor12.decompress(&buffer12));
        });

        group.finish();

        // Report comparison.
        let raw = buf.len();
        let c8 = buffer8.len();
        let c12 = buffer12.len();
        println!(
            "{name}: 8-bit {raw}=>{c8}B ({:.2}:1)  12-bit {raw}=>{c12}B ({:.2}:1)",
            raw as f64 / c8 as f64,
            raw as f64 / c12 as f64,
        );
    }

    run_comparison("wikipedia", "benches/data/wikipedia", c);
    run_comparison("l_comment", "benches/data/l_comment", c);
    run_comparison("urls", "benches/data/urls", c);
}

criterion_group!(bench_compress12, bench_dbtext_12, bench_head_to_head);
criterion_main!(bench_compress12);
