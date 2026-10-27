//! Benchmarks for FSST compression, decompression, and symbol table training.
//!
//! We use the dbtext data at https://github.com/cwida/fsst/tree/master/paper/dbtext
#![allow(missing_docs, clippy::unwrap_in_result)]
use core::str;
use std::{
    error::Error,
    fs::{self, DirBuilder, File},
    io::{Read, Write},
    path::Path,
};

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main};

use curl::easy::Easy;
use fsst::Compressor;

fn download_dataset(url: &str, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
    let target = path.as_ref();

    let mut dir_builder = DirBuilder::new();
    dir_builder.recursive(true);

    dir_builder.create(target.parent().unwrap())?;

    // Avoid downloading the file twice.
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
            // cleanup in case of failure
            fs::remove_file(target).unwrap();

            return Err(Box::new(err));
        }
    }

    Ok(())
}

fn bench_bulk<const LANES: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    compressor: &Compressor,
    lines: &[&[u8]],
    total_len: usize,
) {
    let mut output: Vec<u8> = Vec::with_capacity(total_len * 2);
    let mut offsets: Vec<u64> = Vec::with_capacity(lines.len());

    group.bench_function(format!("compress-bulk-into/{LANES}"), |b| {
        b.iter(|| {
            output.clear();
            offsets.clear();
            compressor.compress_bulk_into::<LANES>(lines, &mut output, &mut offsets);
        });
    });
}

fn run_bench(name: &str, buf: &[u8], c: &mut Criterion) {
    let mut group = c.benchmark_group(name);

    group.bench_function("train-and-compress", |b| {
        b.iter_with_large_drop(|| {
            let compressor = Compressor::train(&vec![&buf]);
            compressor.compress_bulk(std::hint::black_box(&vec![&buf]))
        });
    });

    let compressor = Compressor::train(&vec![&buf]);
    let mut buffer = Vec::with_capacity(buf.len() * 2);
    group.throughput(Throughput::Bytes(buf.len() as u64));
    group.bench_function("compress-only", |b| {
        b.iter(|| unsafe { compressor.compress_into(buf, buffer.spare_capacity_mut()) });
    });

    // Row-wise compression: each line is compressed on its own, and the end offset of each
    // value is recorded. This is the shape of a string column, and the baseline for bulk APIs.
    let lines: Vec<&[u8]> = buf.split(|&b| b == b'\n').collect();
    let mut output: Vec<u8> = Vec::with_capacity(buf.len() * 2);
    let mut offsets: Vec<u64> = Vec::with_capacity(lines.len());

    group.bench_function("compress-rows", |b| {
        b.iter(|| {
            output.clear();
            offsets.clear();

            for line in &lines {
                // SAFETY: a value compresses to at most twice its length, and the lines together
                // are shorter than `buf`, so the capacity of `output` is sufficient.
                unsafe {
                    let length = compressor.compress_into(line, output.spare_capacity_mut());
                    output.set_len(output.len() + length);
                }

                offsets.push(output.len() as u64);
            }
        });
    });

    // Row-wise bulk compression: values are compressed independently of one another, which is
    // what lets `compress_bulk_into` run several cursors at once. The best lane count depends on
    // the machine, so measure the two likely candidates. `compress-rows` is the one-lane baseline.
    bench_bulk::<4>(&mut group, &compressor, &lines, buf.len());
    bench_bulk::<8>(&mut group, &compressor, &lines, buf.len());

    unsafe {
        let length = compressor.compress_into(buf, buffer.spare_capacity_mut());
        buffer.set_len(length);
    };
    let decompressor = compressor.decompressor();
    group.bench_function("decompress", |b| {
        b.iter_with_large_drop(|| decompressor.decompress(&buffer));
    });

    group.finish();

    // Report the compression factor for this dataset.
    let uncompressed_size = buf.len();
    let compressor = Compressor::train(&vec![&buf]);

    let compressed = compressor.compress_bulk(&vec![&buf]);
    let compressed_size = compressed.iter().map(|l| l.len()).sum::<usize>();
    let cf = (uncompressed_size as f64) / (compressed_size as f64);
    println!(
        "compressed {name} {uncompressed_size} => {compressed_size}B (compression factor {cf:.2}:1)"
    )
}

#[allow(clippy::use_debug)]
fn bench_dbtext(c: &mut Criterion) {
    fn run_dataset_bench(name: &str, url: &str, path: &str, c: &mut Criterion) {
        download_dataset(url, path).unwrap();

        let mut buf = Vec::new();
        File::open(path).unwrap().read_to_end(&mut buf).unwrap();

        run_bench(name, &buf, c);
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

/// For small corpus, there can be a pruning benefit.
fn bench_small_input(c: &mut Criterion) {
    let mut buf = vec![b'a'; 500];
    for b in 200u8..210 {
        buf.extend(std::iter::repeat_n(b, 5));
    }

    run_bench("small-input", &buf, c);
}

criterion_group!(compress_bench, bench_dbtext, bench_small_input);
criterion_main!(compress_bench);
