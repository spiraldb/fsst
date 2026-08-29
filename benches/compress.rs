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

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

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

    // Row-wise bulk compression: values are compressed independently of one another, which is
    // what lets `compress_bulk_into` run several cursors at once.
    let lines: Vec<&[u8]> = buf.split(|&b| b == b'\n').collect();
    let mut bulk_output: Vec<u8> = Vec::with_capacity(buf.len() * 2);
    let mut bulk_offsets: Vec<u64> = Vec::with_capacity(lines.len());
    group.bench_function("compress-bulk-into", |b| {
        b.iter(|| {
            bulk_output.clear();
            bulk_offsets.clear();
            compressor.compress_bulk_into(&lines, &mut bulk_output, &mut bulk_offsets);
        });
    });

    group.bench_function("compress-bulk", |b| {
        b.iter_with_large_drop(|| compressor.compress_bulk(std::hint::black_box(&lines)));
    });

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
